# Import base packages
import os
import dotenv
from composio_langchain import Action, App, ComposioToolSet
from crewai import Agent, Crew, Process, Task
from datetime import datetime
from model_zoo import crewai_llm_groq as llm


# Load environment variables from the .env file
dotenv.load_dotenv()

# Initialize the ComposioToolSet
toolset = ComposioToolSet(api_key=os.environ["COMPOSIO_API_KEY"])

# Get the Gmail tools from the ComposioToolSet
gmail_tools = toolset.get_tools(apps=[App.GMAIL])

# Define the Email Fetcher Agent
email_fetcher_agent = Agent(
role="Email Fetcher Agent",
goal="Fetch recent newsletter emails from the inbox sent by bytebytego.",
verbose=True,
memory=True,
backstory=f"You are an expert in retrieving and organizing email content. Today's date is {datetime.now().strftime('%B %d, %Y')}.",
llm=llm,
allow_delegation=False,
tools=gmail_tools,
)

# Define the Summarizer Agent
summarizer_agent = Agent(
role="Summarizer Agent",
goal="Summarize the content of newsletter emails.",
verbose=True,
memory=True,
backstory=f"You are an expert in analyzing and summarizing complex information. Today's date is {datetime.now().strftime('%B %d, %Y')}.",
llm=llm,
allow_delegation=False,
tools=[],
)

# Define the Email Sender Agent
email_sender_agent = Agent(
role="Email Sender Agent",
goal="Send the summarized newsletter content via email to vishalvatsalya7@gmail.com",
verbose=True,
memory=True,
backstory=f"You are an expert in composing and sending emails. Today's date is {datetime.now().strftime('%B %d, %Y')}.",
llm=llm,
allow_delegation=False,
tools=gmail_tools,
)

# Define the task for fetching emails
fetch_emails_task = Task(
description="Fetch the most recent newsletter emails from the inbox sent by bytebytego.",
expected_output="A detailed list of recent newsletter emails with their content.",
tools=gmail_tools,
agent=email_fetcher_agent,
)

# Define the task for summarizing emails
summarize_emails_task = Task(
description="Summarize the content of the fetched newsletter emails.",
expected_output="A comprehensive summary of the newsletter emails.",
agent=summarizer_agent,
context=[fetch_emails_task],
)

# Define the task for sending the summary email
send_summary_task = Task(
description="Compose and send an email containing the summarized newsletter content.",
expected_output="Confirmation that the summary email has been sent.",
tools=gmail_tools,
agent=email_sender_agent,
context=[summarize_emails_task],
)

# Define the crew with the agents and tasks
crew = Crew(
agents=[email_fetcher_agent, summarizer_agent, email_sender_agent],
tasks=[fetch_emails_task, summarize_emails_task, send_summary_task],
process=Process.sequential,
)

# Kickoff the process and print the result
result = crew.kickoff()
print("Newsletter Summary Process Completed:")
print(result)