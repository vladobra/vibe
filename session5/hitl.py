import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput
from pydantic import BaseModel
from typing import List

os.environ["SERPER_API_KEY"] = "4e9d2876031c3e1bd26fd69b56bd5e930a42255b"  # serper.dev API key
os.environ["OPENAI_API_KEY"] = "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"

# Loading Tools
search_tool = SerperDevTool()

class EventOutput(BaseModel):
    events: List[str]

# Define a condition function for the conditional task
# If false, the task will be skipped, if true, then execute the task.
def is_data_missing(output: TaskOutput) -> bool:
    print(".........", output.pydantic.events)
    return len(output.pydantic.events) < 10  # this will skip this task

# Define your agents with roles, goals, tools, and additional attributes
researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI and data science',
    backstory=(
        "You are a Senior Research Analyst at a leading tech think tank. "
        "Your expertise lies in identifying emerging trends and technologies in AI and data science. "
        "You have a knack for dissecting complex data and presenting actionable insights."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[search_tool]
)
writer = Agent(
    role='Tech Content Strategist',
    goal='Craft compelling content on tech advancements',
    backstory=(
        "You are a renowned Tech Content Strategist, known for your insightful and engaging articles on technology and innovation. "
        "With a deep understanding of the tech industry, you transform complex concepts into compelling narratives."
    ),
    verbose=True,
    allow_delegation=True,
    tools=[search_tool],
    cache=False,  # Disable cache for this agent
)

# Create tasks for your agents
task1 = Task(
    description=(
        "Conduct a comprehensive analysis of the latest advancements in AI in 2025. "
        "Identify key trends, breakthrough technologies, and potential industry impacts. "
        "Compile your findings in a detailed report. "
        "Make sure to check with a human if the draft is good before finalizing your answer."
    ),
    expected_output='A comprehensive full report of the list of 10 events on the latest AI advancements in 2025, leave nothing out',
    output_pydantic=EventOutput,
    agent=researcher,
    human_input=True
)

task2 = Task(
    description=(
        "Using the insights from the researcher\'s report, develop an engaging blog post that highlights the most significant AI advancements. "
        "Your post should be informative yet accessible, catering to a tech-savvy audience. "
        "Aim for a narrative that captures the essence of these breakthroughs and their implications for the future."
    ),
    expected_output='A compelling 3 paragraphs blog post formatted as markdown about the latest AI advancements in 2025',
    agent=writer,
    human_input=True,
    condition=is_data_missing,
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=True,
    memory=True,
    planning=True  # Enable planning feature for the crew
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)