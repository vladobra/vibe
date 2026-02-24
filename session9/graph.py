"""RAG agent graph for LangGraph Studio.

This module exports a compiled LangGraph `graph` that Studio can connect to.
"""
import os
import time
from typing import List

from typing_extensions import TypedDict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langgraph.graph import END, StateGraph, START

# ── Config ──────────────────────────────────────────────────────────
local_llm = "adrienbrault/phi3-medium-128k:q4_K_M"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = FAISS.load_local(
    "./faisss", embeddings=embeddings, allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 12})

# ── LLM ─────────────────────────────────────────────────────────────
llm = ChatOllama(model=local_llm, format="json", temperature=0.7)

# ── Chains ──────────────────────────────────────────────────────────
# Retrieval grader
retrieval_grader = (
    PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
        input_variables=["question", "document"],
    )
    | llm
    | JsonOutputParser()
)

# RAG chain
rag_chain = (
    PromptTemplate(
        template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:""",
        input_variables=["question", "context"],
    )
    | llm
    | StrOutputParser()
)

# Hallucination grader
hallucination_grader = (
    PromptTemplate(
        template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "documents"],
    )
    | llm
    | JsonOutputParser()
)

# Answer grader
answer_grader = (
    PromptTemplate(
        template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
    Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question}
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "question"],
    )
    | llm
    | JsonOutputParser()
)

# Question rewriter
question_rewriter = (
    PromptTemplate(
        template="""You are a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the initial and formulate an improved question. \n
     Here is the initial question: \n\n "{question}". \n Do not return any reasoning, just give the improved question only. \n Improved question with no preamble: \n """,
        input_variables=["generation", "question"],
    )
    | llm
    | StrOutputParser()
)


# ── Graph State ─────────────────────────────────────────────────────
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]


# ── Nodes ───────────────────────────────────────────────────────────
def retrieve(state):
    print("---RETRIEVE---")
    documents = retriever.invoke(state["question"])
    return {"documents": documents, "question": state["question"]}


def generate(state):
    print("---GENERATE---")
    generation = rag_chain.invoke({"context": state["documents"], "question": state["question"]})
    return {"documents": state["documents"], "question": state["question"], "generation": generation}


def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    filtered_docs = []
    for d in state["documents"]:
        score = retrieval_grader.invoke({"question": state["question"], "document": d.page_content})
        if score["score"] == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    return {"documents": filtered_docs, "question": state["question"]}


def transform_query(state):
    print("---TRANSFORM QUERY---")
    better_question = question_rewriter.invoke({"question": state["question"]})
    print("============== " + better_question)
    return {"documents": state["documents"], "question": better_question}


# ── Edges ───────────────────────────────────────────────────────────
def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    if not state["documents"]:
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    print("---CHECK HALLUCINATIONS---")
    score = hallucination_grader.invoke({"documents": state["documents"], "generation": state["generation"]})
    if score["score"] == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": state["question"], "generation": state["generation"]})
        if score["score"] == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


# ── Build Graph ─────────────────────────────────────────────────────
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"transform_query": "transform_query", "generate": "generate"},
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {"not supported": "generate", "useful": END, "not useful": "transform_query"},
)

# This is what Studio imports
graph = workflow.compile()
