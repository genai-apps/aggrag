import os
import json
import dotenv
import logging
from enum import Enum
import os
from typing import Optional

from pydantic_settings import BaseSettings
logger = logging.getLogger(__name__)

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)
from dotenv import dotenv_values
config = dotenv_values(".env")


try:
    ai_services_config_str = config['AI_SERVICES_CONFIG']
    
    if ai_services_config_str is None or ai_services_config_str.strip() == "":
        raise ValueError("AI_SERVICES_CONFIG is not set or is empty.")

    ai_services_config_str = ai_services_config_str.strip()
    ai_services_config = json.loads(ai_services_config_str)

    # Accessing Each service
    azure_service = ai_services_config.get("AzureOpenAI", {})
    nemo_service = ai_services_config.get("Nvidia", {})
    openai_service =  ai_services_config.get("OpenAI", {})
    together_ai_service = ai_services_config.get("Together", {})
    replicate_ai_service = ai_services_config.get("Replicate", {})


    # List of all AI services
    all_ai_services = list(ai_services_config.keys())  


except json.JSONDecodeError as e:
    logger.error(f"Error parsing JSON: {e}")
    raise Exception(f"Please provide a valid json in AI_SERVICES_CONFIG, error while parsing json --> {e}")
except ValueError as ve:
    logger.error(f"ValueError: {ve}")
    raise
except Exception as e:
    logger.exception("An unexpected error occurred")


class OpenAIModelNames(Enum):
    """
    Enum representing different OpenAI model names, each identified by a unique model string.
    """
    gpt_35_turbo = openai_service.get('chat_models').get("gpt-35-turbo", {}).get("model_name", None)
    gpt_35_turbo_16k = openai_service.get('chat_models').get("gpt-35-turbo-16k", {}).get("model_name", None)
    gpt_4_turbo = openai_service.get('chat_models').get("gpt-4-turbo", {}).get("model_name", None)
    gpt_4_32k = openai_service.get('chat_models').get("gpt-4-32k", {}).get("model_name", None)
    text_embedding_ada_002 = openai_service.get("embed_models").get("text-embedding-ada-002", {}).get("model_name", None)
    gpt_4o = openai_service.get('chat_models').get("gpt-4o", {}).get("model_name", None)


class OpenAIModelEngines(Enum):
    """
    Enum representing different OpenAI model engines on Azure, each identified by a unique deployment string.
    """
    gpt_35_turbo = azure_service.get('chat_models').get("gpt-35-turbo", {}).get("deployment_name", None)
    text_embedding_ada_002 = azure_service.get('chat_models').get("text-embedding-ada-002", {}).get("model_name", None)


class AzureOpenAIModelEngines(Enum):
    """
    Enum representing different OpenAI model engines on Azure, each identified by a unique deployment string.
    """
    gpt_35_turbo = azure_service.get('chat_models').get("gpt-35-turbo", {}).get("deployment_name", None)
    gpt_35_turbo_16k = azure_service.get('chat_models').get("gpt-35-turbo-16k", {}).get("deployment_name", None)
    gpt_4_turbo = azure_service.get('chat_models').get("gpt-4-turbo", {}).get("deployment_name", None)
    gpt_4_32k = azure_service.get('chat_models').get("gpt-4-32k", {}).get("deployment_name", None)
    text_embedding_ada_002 = azure_service.get('embed_models').get("text-embedding-ada-002", {}).get("deployment_name", None)
    gpt_4o = azure_service.get('chat_models').get("gpt-4o", {}).get("deployment_name", None)

class AzureOpenAIModelNames(Enum):
    """
    Enum representing different OpenAI model names on Azure, each identified by a unique deployment string.
    """
    gpt_35_turbo_16k = azure_service.get('chat_models').get("gpt-35-turbo-16k", {}).get("model_name", None)
    gpt_4_32k = azure_service.get('chat_models').get("gpt-4-32k", {}).get("model_name", None)
    gpt_4_turbo = azure_service.get('chat_models').get("gpt-4-turbo", {}).get("model_name", None)
    text_embedding_ada_002 = azure_service.get('embed_models').get("text-embedding-ada-002", {}).get("model_name", None)
    gpt_4o = azure_service.get('chat_models').get("gpt-4o", {}).get("model_name", None)



class NemoModelNames(Enum):
    """
    Enum representing NVIDIA's Nemo model names, each corresponding to a specific model configuration.
    """
    mixtral_8x7b = nemo_service.get('chat_models', {}).get("mixtral_8x7b", {}).get("model_name", None)
    llama2_13b = nemo_service.get('chat_models', {}).get("playground_llama2_13b", {}).get("model_name", None)
    llama2_70b = nemo_service.get('chat_models', {}).get("playground_llama2_70b", {}).get("model_name", None)
    nvolveqa_40k = nemo_service.get('chat_models', {}).get("nvolveqa_40k", {}).get("model_name", None)
    ai_embed_qa_4 = nemo_service.get('chat_models', {}).get("ai-embed-qa-4", {}).get("model_name", None)


class OpenAIModelNames(Enum):
    """
    Enum representing different OpenAI model names, each identified by a unique deployment string.
    """
    gpt_35_turbo = openai_service.get('chat_models').get("gpt-35-turbo", {}).get("model_name", None)
    gpt_35_turbo_16k = openai_service.get('chat_models').get("gpt-35-turbo-16k", {}).get("model_name", None)
    gpt_4_turbo = openai_service.get('chat_models').get("gpt-4-turbo", {}).get("model_name", None)
    gpt_4_32k = openai_service.get('chat_models').get("gpt-4-32k", {}).get("model_name", None)
    text_embedding_ada_002 = openai_service.get('embed_models').get("text-embedding-ada-002", {}).get("model_name", None)
    gpt_4o = openai_service.get('chat_models').get("gpt-4o", {}).get("model_name", None)
    text_embedding_ada_003 = openai_service.get('chat_models').get("text-embedding-ada-003", {}).get("model_name", None)


class TogetherLLMModelNames(Enum):
    """
    Enum representing different model names for TogetherLLM, each identified by a unique model string.
    """
    mixtral_8x7b_instruct = together_ai_service.get("chat_models", {}).get("mixtral_8x7b_instruct", {}).get("model_name", None)
    # Add more models as needed

class ReplicateModelNames(Enum):
    """
    Enum representing different model names for Replicate, each identified by a unique model string.
    """
    meta_llama_3_70b_instruct = replicate_ai_service.get("chat_models", {}).get("meta_llama_3_70b_instruct", {}).get("model_name", None)
    # Add more models as needed

logger.info(f"NemoModelnames: {NemoModelNames.__members__}")
logger.info(f"AzureOpenAIModelEngines:  {AzureOpenAIModelEngines.__members__}")
logger.info(f"AzureOpenAIModelNames:  {AzureOpenAIModelNames.__members__}")
logger.info(f"AzureOpenAIModelEngines:  {OpenAIModelNames.__members__}")
logger.info(f"TogetherLLMModelNames {TogetherLLMModelNames.__members__}")
logger.info(f"ReplicateModelNames: {ReplicateModelNames.__members__}")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Settings(BaseSettings):

    """
    Settings for the application, loading configuration from environment variables and providing default values.
    """
    
    TOGETHER_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_VERSION: str = '2024-02-15-preview'
    AZURE_OPENAI_KEY: Optional[str] = None
    AZURE_API_BASE: str
    REACT_APP_API_URL:str
    REPLICATE_API_TOKEN: Optional[str] = None
    LLAMA_CLOUD_API_KEY: Optional[str] = None
    # Log
    LOGGING_LEVEL: str = 'INFO'

    # def chat_file_temp_file_directory(self, user_id: int, chat_id: int) -> str:
    #     return os.path.join(f'../tmp/{user_id}/{chat_id}/raw_docs/')

    # def html_files_temp_file_directory(self, user_id: int, chat_id: int) -> str:
    #     return os.path.join(f'../tmp/{user_id}/{chat_id}/html_files/')
    
    class Config:
        env_file = f'{BASE_DIR}/.env'


settings = Settings()
