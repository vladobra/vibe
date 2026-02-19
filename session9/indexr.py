import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.stores import InMemoryBaseStore as InMemoryStore

from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever

from langchain_ollama import OllamaEmbeddings

local_llm = "adrienbrault/phi3-medium-128k:q4_K_M"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

urls = [
    "https://vladobra.com",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    "http://artscene.textfiles.com/ansimusic/information/ansimtech.txt",
    "http://web.textfiles.com/computers/llaptop.txt",
    "https://www.rfc-editor.org/rfc/rfc223.txt",
    "http://web.textfiles.com/computers/serv-u.txt",
    "https://www.rfc-editor.org/rfc/rfc238.txt",
]

start_time = time.time()

docs_nested = [WebBaseLoader(url).load() for url in urls]
docs_list = [d for sub in docs_nested for d in sub]

child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
split_docs = child_splitter.split_documents(docs_list)

# Vectorstore built from chunks so each text fits the embedding model context
vectorstore = FAISS.from_documents(split_docs, embeddings)

# Parent doc storage
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

retriever.add_documents(docs_list)

vectorstore.save_local("./faisss")

print(f"Done in {time.time() - start_time:.2f}s")