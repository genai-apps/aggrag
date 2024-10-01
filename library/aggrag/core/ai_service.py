from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


from library.aggrag.core.config import (
    settings,
    AzureOpenAIModelNames,
    AzureOpenAIModelEngines,
    TogetherLLMModelNames,
    ReplicateModelNames,
    AI_SERVICES_CONFIG
)
from llama_index.llms.replicate import Replicate
from llama_index.llms.together import TogetherLLM
from llama_index.llms.openai import OpenAI
from library.aggrag.core.config import ai_services_config
from library.aggrag.core.config import settings, AzureOpenAIModelNames, AzureOpenAIModelEngines, OpenAIModelNames, AnthropicModelNames
from llama_index.llms.anthropic import Anthropic


rag_temperature = 0.1


class AzureAIService:

    def initialize_llm_model(**kwargs):
        try:
            service_config = AI_SERVICES_CONFIG.AzureOpenAI
            if not service_config:
                raise ValueError("AzureOpenAI configuration not found in AI_SERVICES_CONFIG.")

            llm_model = kwargs.get("llm_model")
            if llm_model:
                model_config = service_config.chat_models.get(llm_model)
                if not model_config:
                    raise ValueError(f"Configuration not found for LLM model '{llm_model}' in AzureOpenAI chat_models.")
                model = model_config.model_name
                deployment_name = model_config.deployment_name
            else:
                model = AzureOpenAIModelNames.gpt_35_turbo.value
                deployment_name = AzureOpenAIModelEngines.gpt_35_turbo.value

            temperature = kwargs.get("temperature", 0.1)
            api_key = kwargs.get("api_key", settings.AZURE_OPENAI_KEY)
            azure_endpoint = kwargs.get("azure_endpoint", settings.AZURE_API_BASE)
            api_version = kwargs.get("api_version", settings.OPENAI_API_VERSION)

            llm = AzureOpenAI(
                model=model,
                deployment_name=deployment_name,
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                temperature=temperature,
            )

            return llm
        except Exception as e:
            print(f"Error initializing LLM model: {e}")
            raise


    def initialize_embed_model(**kwargs):
        try:
            service_config = AI_SERVICES_CONFIG.AzureOpenAI
            if not service_config:
                raise ValueError("AzureOpenAI configuration not found in AI_SERVICES_CONFIG.")

            embed_model = kwargs.get("embed_model")
            if embed_model:
                model_config = service_config.embed_models.get(embed_model)
                if not model_config:
                    raise ValueError(f"Configuration not found for embedding model '{embed_model}' in AzureOpenAI embed_models.")
                embed_model = model_config.model_name
                deployment_name = model_config.deployment_name
            else:
                embed_model = AzureOpenAIModelNames.text_embedding_ada_002.value
                deployment_name = AzureOpenAIModelEngines.text_embedding_ada_002.value

            api_key = kwargs.get("api_key", settings.AZURE_OPENAI_KEY)
            azure_endpoint = kwargs.get("azure_endpoint", settings.AZURE_API_BASE)
            api_version = kwargs.get("api_version", settings.OPENAI_API_VERSION)

            embed_model = AzureOpenAIEmbedding(
                model=embed_model,
                deployment_name=deployment_name,
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
            )
            return embed_model
        except Exception as e:
            print(f"Error initializing AzureOpenAI embedding model: {e}")
            raise

class ReplicateAIService:
    def initialize_llm_model(**kwargs):
        try:
            service_config = AI_SERVICES_CONFIG.Replicate
            if not service_config:
                raise ValueError("Replicate configuration not found in AI_SERVICES_CONFIG.")

            llm_model = kwargs.get("llm_model")
            if llm_model:
                model_config = service_config.chat_models.get(llm_model)
                if not model_config:
                    raise ValueError(f"Configuration not found for LLM model '{llm_model}' in Replicate chat_models.")
                model = model_config.model_name
            else:
                model = ReplicateModelNames.meta_llama_3_70b_instruct.value

            llm = Replicate(
                model=model,
                temperature=kwargs.get("temperature", 0.1),
            )

            return llm
        except Exception as e:
            print(f"Error initializing LLM model: {e}")
            raise



class TogetherAIService:
    def initialize_llm_model(**kwargs):
        try:
            service_config = AI_SERVICES_CONFIG.Together
            if not service_config:
                raise ValueError("Together configuration not found in AI_SERVICES_CONFIG.")

            llm_model = kwargs.get("llm_model")
            if llm_model:
                model_config = service_config.chat_models.get(llm_model)
                if not model_config:
                    raise ValueError(f"Configuration not found for LLM model '{llm_model}' in Together chat_models.")
                model = model_config.model_name
            else:
                model = TogetherLLMModelNames.mixtral_8x7b_instruct.value

            api_key = kwargs.get("api_key", settings.TOGETHER_API_KEY)
            temperature = kwargs.get("temperature", 0.1)


            llm = TogetherLLM(
                model=model,
                api_key=api_key,
                temperature=temperature
            )

            return llm
        except Exception as e:
            print(f"Error initializing LLM model: {e}")
            raise


class OpenAIService:
    def initialize_llm_model(**kwargs):
        try:
            service_config = AI_SERVICES_CONFIG.OpenAI
            if not service_config:
                raise ValueError("OpenAI configuration not found in AI_SERVICES_CONFIG.")

            llm_model = kwargs.get("llm_model")
            if llm_model:
                model_config = service_config.chat_models.get(llm_model)
                if not model_config:
                    raise ValueError(f"Configuration not found for LLM model '{llm_model}' in OpenAI chat_models.")
                model = model_config.model_name
            else:
                model = OpenAIModelNames.gpt_35_turbo.value

            api_key = kwargs.get("api_key", settings.OPENAI_API_KEY)
            temperature = kwargs.get("temperature", 0.1)

            llm = OpenAI(
                model=model,
                api_key=api_key,
                temperature=temperature
            )

            return llm
        except Exception as e:
            print(f"Error initializing LLM model: {e}")
            raise

    def initialize_embed_model(**kwargs):
        try:
            service_config = AI_SERVICES_CONFIG.OpenAI
            if not service_config:
                raise ValueError("OpenAI configuration not found in AI_SERVICES_CONFIG.")

            embed_model = kwargs.get("embed_model")
            if embed_model:
                model_config = service_config.embed_models.get(embed_model)
                if not model_config:
                    raise ValueError(f"Configuration not found for embedding model '{embed_model}' in OpenAI embed_models.")
                model = model_config.model_name
            else:
                model = OpenAIModelNames.text_embedding_ada_002.value

            api_key = kwargs.get("api_key", settings.OPENAI_API_KEY)

            embed_model = OpenAIEmbedding(
                model=model,
                api_key=api_key,
            )

            return embed_model
        except Exception as e:
            print(f"Error initializing embedding model: {e}")
            raise


class AnthropicAIService:
    def initialize_llm_model(**kwargs):
        try:
            service_config = AI_SERVICES_CONFIG.Anthropic
            if not service_config:
                raise ValueError("Anthropic configuration not found in AI_SERVICES_CONFIG.")
            llm_model = kwargs.get("llm_model")
            if llm_model:
                model_config = service_config.chat_models.get(llm_model)
                if not model_config:
                    raise ValueError(f"Configuration not found for LLM model '{llm_model}' in Anthropic chat_models.")
                model = model_config.model_name
            else:
                model = AnthropicModelNames.claude_3_sonnet_20240229.value

            api_key = kwargs.get("api_key", settings.ANTHROPIC_API_KEY)
            temperature = kwargs.get("temperature", 0.1)

            llm = Anthropic(
                model=model,
                api_key=api_key,
                temperature=temperature
            )

            return llm
        except Exception as e:
            print(f"Error initializing LLM model: {e}")
            raise

class AIServiceFactory:

    service_map = {
        "AzureOpenAI": AzureAIService,
        "Replicate": ReplicateAIService,
        "Together": TogetherAIService,
        "OpenAI": OpenAIService,
        "Anthropic": AnthropicAIService,
    }

    def create_llm_model(**kwargs):
        ai_service = kwargs['ai_service']
        return AIServiceFactory.service_map[ai_service].initialize_llm_model(**kwargs)

    def create_embed_model(**kwargs):
        embed_ai_service = kwargs['embed_ai_service']
        return AIServiceFactory.service_map[embed_ai_service].initialize_embed_model(**kwargs)
