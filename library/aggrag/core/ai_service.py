from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


from library.aggrag.core.config import (
    settings,
    AzureOpenAIModelNames,
    AzureOpenAIModelEngines,
    TogetherLLMModelNames,
    ReplicateModelNames
)
from llama_index.llms.replicate import Replicate
from llama_index.llms.together import TogetherLLM
from llama_index.llms.openai import OpenAI
from library.aggrag.core.config import ai_services_config
from library.aggrag.core.config import settings, AzureOpenAIModelNames, AzureOpenAIModelEngines, OpenAIModelNames

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


class ReplicateAIService:
    def __init__(self, model=None, embed_model=None):

        self.llm = Replicate(
            model=model or ReplicateModelNames.meta_llama_3_70b_instruct.value,
            temperature=0.1,
            # context_window=32,
        )



class TogetherAIService:
    def __init__(self, model=None, embed_model=None):
        self.llm = TogetherLLM(
            model=model or TogetherLLMModelNames.mixtral_8x7b_instruct.value,
            api_key=settings.TOGETHER_API_KEY,
        )



class OpenAIService:
    def __init__(self, model=None, deployment_name=None, api_key=None, azure_endpoint=None, api_version=None, embed_model=None):

        self.llm = OpenAI(
            model=model or OpenAIModelNames.gpt_4_turbo.value,
            api_key=settings.OPENAI_API_KEY,
        )
        
        self.embed_model = OpenAIEmbedding(
            model = embed_model or OpenAIModelNames.text_embedding_ada_002.value,
            api_key=api_key or settings.OPENAI_API_KEY)



class AIServiceFactory:
    @staticmethod
    def get_ai_service(ai_service, llm_model=None, embed_model=None):
        if ai_service not in ai_services_config.keys():
            raise ValueError(f"Unsupported AI service: {ai_service}")
        # Check if the model is valid for the selected service
        # model_names = [model['model_name'] for model in ai_services_config.get(ai_service, {}).values()]
        # if llm_model not in model_names:
        #     raise ValueError(f"Model '{llm_model}' is not available for service '{ai_service}'")


        
        # Return the appropriate AI service instance
        if ai_service == "AzureOpenAI":
            return AzureAIService(model=llm_model, embed_model=embed_model)
        elif ai_service == "Replicate":
            return ReplicateAIService(model=llm_model, embed_model=embed_model)
        elif ai_service == "Together":
            return TogetherAIService(model=llm_model, embed_model=embed_model)
        elif ai_service == "OpenAI":
            return OpenAIService(model=llm_model, embed_model=embed_model)
        else:
            raise ValueError(f"Unsupported AI service: {ai_service}")
