from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

''' export OPENAI_API_KEY="your-api-key"  '''

@tool
def multiply(a: int, b: int) -> int:
    """This tool multiplies two numbers."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """This tool adds two numbers and returns the sum."""
    return a + b


llm = ChatOpenAI(model="gpt-4o-mini", api_key="")
llm_tools = llm.bind_tools([multiply, add])   # exposing the tools to the model

resp = llm_tools.invoke("ako imam 5 jabuka i drug me da 2 koliko imam")
print(resp.tool_calls or resp.content)   # model may emit a tool call with arguments
print(resp)