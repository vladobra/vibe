import os
from crewai import Agent, Task, Crew, Process

# Set your OpenAI key here or via environment variable


os.environ["OPENAI_API_KEY"] = "...."

MODEL_NAME = "gpt-5-mini"  # or whatever your OpenAI deployment name is

# --------- Agents --------- #

agent1 = Agent(
    role="AI Writer",
    goal="Write a concise text about AI in no more than 100 words.",
    backstory="You are a knowledgeable writer who explains AI clearly and briefly.",
    model=MODEL_NAME,
    verbose=True,
)

agent2 = Agent(
    role="Odd Word Counter",
    goal="Check if the word count of a given text is odd and say 'odd' if it is.",
    backstory=(
        "You receive a text, carefully count its words, and only output 'odd' "
        "when the word count is odd."
    ),
    model=MODEL_NAME,
    verbose=True,
)

agent3 = Agent(
    role="Even Word Counter",
    goal="Check if the word count of a given text is even and say 'even' if it is.",
    backstory=(
        "You receive a text, carefully count its words, and only output 'even' "
        "when the word count is even."
    ),
    model=MODEL_NAME,
    verbose=True,
)

# --------- Task 1: create up to 100-word AI text --------- #

task1 = Task(
    description=(
        "Write a single paragraph about artificial intelligence, "
        "with a maximum of 100 words. "
        "Use clear language and avoid bullet points. "
        "Return only the paragraph text."
    ),
    expected_output="A single paragraph of at most 100 words describing AI.",
    agent=agent1,
)


def main():
    # 1. Run Agent 1 to create the AI text
    crew1 = Crew(
        agents=[agent1],
        tasks=[task1],
        process=Process.sequential,
    )
    result1 = crew1.kickoff()

    # In most CrewAI versions, result1 is already a string with the output
    text = str(result1).strip()

    # 2. Count words in Python to decide which agent runs next
    word_count = len(text.split())
    print(f"\nAgent1 output ({word_count} words):\n{text}\n")

    if word_count % 2 == 1:
        # --------- Task 2: run Agent 2 if odd --------- #
        task2 = Task(
            description=(
                "You are given the following text:\n\n"
                f"{text}\n\n"
                "1. Count how many words are in this text.\n"
                "2. If the count is odd, respond with exactly:\n"
                "odd\n"
                "Use only that single word, in lowercase, with no punctuation "
                "or explanation."
            ),
            expected_output="The single word: odd",
            agent=agent2,
        )

        crew2 = Crew(
            agents=[agent2],
            tasks=[task2],
            process=Process.sequential,
        )
        result2 = crew2.kickoff()
        print("Agent2 result:", str(result2).strip())

    else:
        # --------- Task 3: run Agent 3 if even --------- #
        task3 = Task(
            description=(
                "You are given the following text:\n\n"
                f"{text}\n\n"
                "1. Count how many words are in this text.\n"
                "2. If the count is even, respond with exactly:\n"
                "even\n"
                "Use only that single word, in lowercase, with no punctuation "
                "or explanation."
            ),
            expected_output="The single word: even",
            agent=agent3,
        )

        crew3 = Crew(
            agents=[agent3],
            tasks=[task3],
            process=Process.sequential,
        )
        result3 = crew3.kickoff()
        print("Agent3 result:", str(result3).strip())


if __name__ == "__main__":
    main()