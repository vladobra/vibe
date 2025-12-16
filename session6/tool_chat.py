from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage

@tool
def multiply(a: int, b: int) -> int:
    """This tool multiplies two numbers."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """This tool adds two numbers and returns the sum."""
    return a + b


llm = ChatOpenAI(model="gpt-4o-mini", api_key="")  # uses OPENAI_API_KEY env var
llm_tools = llm.bind_tools([multiply, add])

tools_by_name = {t.name: t for t in [multiply, add]}

prompt = "ako dragan mijatovic ima 5 jabuka i drug mu da 30 a Sava ima 3 puta vise jabuka od dragana koliko imaju zajedno dragan i sava."

messages = [HumanMessage(content=prompt)]

# keep running until the model stops requesting tools
while True:
    ai_msg = llm_tools.invoke(messages)
    messages.append(ai_msg)

    print("####################")
    print(ai_msg)
    print("####################")

    if not ai_msg.tool_calls:
        print(ai_msg.content)   # final natural-language answer
        break

    # execute each requested tool call
    for call in ai_msg.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]

        tool_result = tools_by_name[tool_name].invoke(tool_args)

        # send tool result back to the model
        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=call["id"],
            )
        )

    # If you just want the numeric result without another model step:
    print(tool_result)