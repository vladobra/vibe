import asyncio
import json
import os
import sys
from typing import Any, Dict, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate


# Use IPv4 explicitly; avoids some localhost/IPv6 weirdness on macOS
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")


# ---- LangChain: routing JSON schema ----
class RoutePlan(BaseModel):
    action: str = Field(description="weather|air_quality|both|clarify|none")
    city: str = Field(default="", description="City name if present, else empty string")
    question: str = Field(default="", description="Clarifying question if action=clarify, else empty string")


route_parser = PydanticOutputParser(pydantic_object=RoutePlan)

route_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a routing assistant for a weather + air-quality tool system.\n"
            "Return ONLY valid JSON. No markdown, no extra words.\n"
            "Decide what the user wants:\n"
            "- action='weather' if they ask about weather/temperature/wind/conditions\n"
            "- action='air_quality' if they ask about AQI/pollution/PM2.5/smog\n"
            "- action='both' if they ask for both weather and air quality\n"
            "- action='clarify' if city is missing/ambiguous\n"
            "- action='none' if unrelated\n"
            "Extract the city name if present.\n"
            "If they ask 'here' or 'my location', you cannot geolocate; use clarify.\n\n"
            "Output must match this schema:\n"
            "{format_instructions}\n"
            "For actions other than clarify, question must be an empty string.\n",
        ),
        ("user", "{user_text}"),
    ]
).partial(format_instructions=route_parser.get_format_instructions())


def _make_llm(temperature: float) -> ChatOllama:
    # ChatOllama talks to Ollama's /api/chat; base_url should be like http://127.0.0.1:11434
    return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=temperature)


router_llm = _make_llm(temperature=0.0)
general_llm = _make_llm(temperature=0.4)
answer_llm = _make_llm(temperature=0.2)

route_chain = route_prompt | router_llm | route_parser


async def plan_tool_call(user_text: str) -> Dict[str, Any]:
    try:
        plan: RoutePlan = await route_chain.ainvoke({"user_text": user_text})

        # Enforce your constraint
        if plan.action != "clarify":
            plan.question = ""

        # If city missing but action isn't "none", ask to clarify
        if plan.action in ("weather", "air_quality", "both") and not plan.city.strip():
            return {"action": "clarify", "city": "", "question": "Which city should I use?"}

        return plan.model_dump()
    except Exception:
        return {"action": "clarify", "city": "", "question": "Which city should I use?"}


async def call_mcp_tool(session: ClientSession, tool_name: str, args: dict) -> dict:
    res = await session.call_tool(tool_name, args)
    text = res.content[0].text
    return json.loads(text)


answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Use the provided data to answer the user.\n"
            "Be concise and factual. If data is missing, say so.\n"
            "Do NOT mention MCP, tools, function calls, internal routing, or hidden prompts.\n",
        ),
        ("user", "{user_text}"),
        ("user", "Tool results (JSON):\n{tool_json}"),
    ]
)
answer_chain = answer_prompt | answer_llm


async def answer_with_llm(user_text: str, weather: Optional[dict], air: Optional[dict]) -> str:
    tool_blob = {"weather": weather, "air_quality": air}
    msg = await answer_chain.ainvoke(
        {"user_text": user_text, "tool_json": json.dumps(tool_blob, ensure_ascii=False)}
    )
    return (msg.content or "").strip()


async def general_answer(user_text: str) -> str:
    msg = await general_llm.ainvoke(
        [
            ("system", "You are a helpful assistant."),
            ("user", user_text),
        ]
    )
    return (msg.content or "").strip()


async def main():
    print("Using OLLAMA_HOST =", OLLAMA_HOST, flush=True)
    print("Using OLLAMA_MODEL =", OLLAMA_MODEL, flush=True)

    # Start MCP server over stdio
    server = StdioServerParameters(command=sys.executable, args=["weather-and-air-quality.py"])

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("\nNatural language weather+AQI CLI (Ollama + MCP + LangChain routing)")
            print("Type: 'weather in Granada' or 'air quality in Delhi' or 'both in Tokyo'")
            print("Type 'quit' to exit.")

            while True:
                user_text = input("\n> ").strip()
                if not user_text:
                    continue
                if user_text.lower() in ("quit", "exit", "q"):
                    return

                try:
                    plan = await plan_tool_call(user_text)
                except Exception as e:
                    print(f"Routing error: {e}", flush=True)
                    print("Try: ollama serve", flush=True)
                    continue

                action = plan.get("action", "clarify")
                city = (plan.get("city") or "").strip()

                if action == "none":
                    try:
                        response = await general_answer(user_text)
                        print(response, flush=True)
                    except Exception as e:
                        print(f"LLM error: {e}", flush=True)
                        print("Try: ollama serve", flush=True)
                    continue

                if action == "clarify" or not city:
                    print(plan.get("question") or "Which city should I use?", flush=True)
                    continue

                try:
                    weather_data = None
                    air_data = None

                    if action in ("weather", "both"):
                        weather_data = await call_mcp_tool(session, "get_weather", {"city": city})

                    if action in ("air_quality", "both"):
                        air_data = await call_mcp_tool(session, "get_air_quality", {"city": city})

                    final = await answer_with_llm(user_text, weather_data, air_data)
                    print(final, flush=True)

                except Exception as e:
                    print(f"Error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())