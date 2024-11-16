from composio_openai import ComposioToolSet, App, Action
from model_zoo import azure_openai_client as openai_client

composio_toolset = ComposioToolSet()

tools = composio_toolset.get_tools(actions=[Action.GITHUB_ACTIVITY_STAR_REPO_FOR_AUTHENTICATED_USER])

task = "Star a repo microsoft/autogen on GitHub"

response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    tools=tools,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": task},
    ],
)

result = composio_toolset.handle_tool_calls(response)
print(result)
