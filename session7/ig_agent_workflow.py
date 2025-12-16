import os
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from langchain_openai import ChatOpenAI

from instagrapi import Client


os.environ["OPENAI_API_KEY"] = ""
# Define LLM instance
#llm = Ollama(model="phi4:14b", temperature=0.7)
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Define State Class
class AgentState(BaseModel):
    topic: str
    generated_text: str = ""
    markdown_text: str = ""
    approved: bool = False

# Define LLM Prompt Template
# Updated prompt template to instruct adding Instagram-friendly emoticons and style
create_prompt = PromptTemplate.from_template(
    """Write a short (50-100 words), engaging, and informative text suitable for an Instagram post on the following topic. Add relevant emoticons or emojis to make the text attractive and Instagram-friendly.

Topic:
{topic}

Instagram Post with Emojis:""")

# Updated Prompt to convert plain text into formatted Markdown with Title
markdown_prompt = PromptTemplate.from_template(
    """Convert the following informative text into the instagram friendly format. Preserve the original language of the text.

    Add a relevant, concise title at the top and organize content clearly, such as using appropriate headings, bullet points, or numbered lists if applicable.

    IMPORTANT: Do NOT wrap your text in ``` delimiters.

    Text:
    {generated_text}

    Markdown formatted text with title:""")

def create_short_text(state: AgentState):
    response = llm.invoke(create_prompt.format(topic=state.topic))
    state.generated_text = response.content.strip()
    print("\nGenerated Text:\n---")
    print(state.generated_text)
    print("---\n")
    return state

def ask_approval(state: AgentState):
    human_input = input("Do you approve this text? (yes/no): ").strip().lower()
    state.approved = human_input in ["yes", "y"]
    return state

def postit(txt):

    username = ""
    password = ""


    cl = Client()
    cl.login(username, password)

    photo_path = '1.png'
    caption = txt

    media = cl.photo_upload(path=photo_path, caption=caption)
    #print(media)

def convert_to_markdown(state: AgentState):
    # Generate markdown formatted text using the LLM
    formatted_markdown = llm.invoke(markdown_prompt.format(generated_text=state.generated_text)).content.strip()

    # Clean possible markdown delimiters if LLM still includes them
    if formatted_markdown.startswith("```markdown"):
        formatted_markdown = formatted_markdown[len("```markdown"):].strip()
    if formatted_markdown.startswith("```"):
        formatted_markdown = formatted_markdown[3:].strip()
    if formatted_markdown.endswith("```"):
        formatted_markdown = formatted_markdown[:-3].strip()

    # Set the cleaned markdown text
    state.markdown_text = formatted_markdown

    # Save generated markdown to a .md file
    #filename = f"{state.topic.replace(' ', '_')}.md"
    filename = "out.md"
    with open(filename, 'w') as f:
        f.write(state.markdown_text)

    print("\nGenerated Markdown Content:\n---")
    print(state.markdown_text)
    postit(state.markdown_text)
    print(f"---\nMarkdown content saved in '{filename}'. Workflow Finished Successfully!")
    return state

# Setup Langgraph StateGraph
workflow = StateGraph(AgentState)

workflow.add_node("create_short_text", create_short_text)
workflow.add_node("ask_approval", ask_approval)
workflow.add_node("convert_to_markdown", convert_to_markdown)

# Start the workflow by generating short text
workflow.set_entry_point("create_short_text")

# Define transitions
workflow.add_edge("create_short_text", "ask_approval")
# Conditional approval check
workflow.add_conditional_edges(
    "ask_approval",
    lambda state: "convert_to_markdown" if state.approved else "create_short_text",
)

# Markdown conversion then ends the workflow
workflow.add_edge("convert_to_markdown", END)

# Compile workflow
app = workflow.compile()

# Execute from Command Line
if __name__ == "__main__":
    print("### Topic to Markdown Agent ###")
    topic = input("Enter your topic: ")
    initial_state = AgentState(topic=topic)
    final_state = app.invoke(initial_state)