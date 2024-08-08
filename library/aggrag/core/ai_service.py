from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from library.aggrag.core.config import settings, AzureOpenAIModelNames, AzureOpenAIModelEngines

rag_temperature = 0.1


class AzureAIService:
    def __init__(self, model=None, deployment_name=None, api_key=None, azure_endpoint=None, api_version=None, embed_model=None):
        self.llm = AzureOpenAI(
            model=model or AzureOpenAIModelNames.gpt_35_turbo_16k.value,
            deployment_name=deployment_name or AzureOpenAIModelEngines.gpt_35_turbo_16k.value,
            api_key=api_key or settings.AZURE_OPENAI_KEY,
            azure_endpoint=azure_endpoint or settings.AZURE_API_BASE,
            api_version=api_version or settings.OPENAI_API_VERSION,
            temperature=rag_temperature
        )
        self.embed_model = AzureOpenAIEmbedding(
            model = embed_model or AzureOpenAIModelNames.text_embedding_ada_002.value,
            deployment_name = deployment_name or AzureOpenAIModelEngines.text_embedding_ada_002.value,
            api_key=api_key or settings.AZURE_OPENAI_KEY,
            azure_endpoint = azure_endpoint or settings.AZURE_API_BASE,
            api_version = api_version or settings.OPENAI_API_VERSION
        )
        self.model_name = self.llm.model
