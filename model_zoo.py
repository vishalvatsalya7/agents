import os
from dotenv import load_dotenv
load_dotenv()
from llama_index.llms.azure_openai import AzureOpenAI as llama_az_openai
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from crewai import LLM

# langchain
langchain_llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    model="gpt-4o-mini",
    temperature=0.1,
    max_retries=2,
    api_version=os.environ["AZURE_OPENAI_API_VERSION"]
)

# llamaindex chat
llama_index_llm = llama_az_openai(
    engine="gpt-4o-mini",
    model="gpt-4o-mini",
    temperature=0.0,
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

# llamaindex embedding
llama_index_embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_EMBED_ENDPOINT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

# openai/azureopenai chat
azure_openai_client = AzureOpenAI(
    azure_deployment='gpt-4o-mini',
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)

# crewai azure chat
crewai_llm = LLM(
    model="azure/gpt-4o-mini",
    temperature=0.0,
    base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"]
)

# crewai groq chat
crewai_llm_groq = LLM(
    model='groq/llama-3.1-8b-instant',
    api_key=os.environ["GROQ_API_KEY"]
)