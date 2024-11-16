import os
from dotenv import load_dotenv
load_dotenv()
from llama_index.llms.azure_openai import AzureOpenAI as llama_az_openai
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from openai import AzureOpenAI

llama_index_llm = llama_az_openai(
    engine="gpt-4o-mini",
    model="gpt-4o-mini",
    temperature=0.0,
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

llama_index_embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_EMBED_ENDPOINT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

azure_openai_client = AzureOpenAI(
    azure_deployment='gpt-4o-mini',
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)
