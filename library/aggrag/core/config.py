import os
import json
import dotenv
import logging
from enum import Enum
import os

from pydantic_settings import BaseSettings
logger = logging.getLogger(__name__)

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)


try:
    ai_services_config_str = os.getenv('AI_SERVICES_CONFIG')
    
    if ai_services_config_str is None or ai_services_config_str.strip() == "":
        raise ValueError("AI_SERVICES_CONFIG is not set or is empty.")

    ai_services_config_str = ai_services_config_str.strip()
    ai_services_config = json.loads(ai_services_config_str)

    # Accessing Each service
    azure_service = ai_services_config.get("AzureAIService", {})
    nemo_service = ai_services_config.get("NemoAIService", {})
    openai_service =  ai_services_config.get("OpenAIService", {})


except json.JSONDecodeError as e:
    logger.error(f"Error parsing JSON: {e}")
except ValueError as ve:
    logger.error(f"ValueError: {ve}")
except Exception as e:
    logger.exception("An unexpected error occurred")


class AzureOpenAIModelEngines(Enum):
    """
    Enum representing different OpenAI model engines on Azure, each identified by a unique deployment string.
    """
    gpt_35_turbo = azure_service.get("gpt-35-turbo", {}).get("deployment_name", None)
    gpt_35_turbo_16k = azure_service.get("gpt-35-turbo-16k", {}).get("deployment_name", None)
    gpt_4_turbo = azure_service.get("gpt-4-turbo", {}).get("deployment_name", None)
    gpt_4_32k = azure_service.get("gpt-4-32k", {}).get("deployment_name", None)
    text_embedding_ada_002 = azure_service.get("text-embedding-ada-002", {}).get("deployment_name", None)
    gpt_4o = azure_service.get("gpt-4o", {}).get("deployment_name", None)

class AzureOpenAIModelNames(Enum):
    """
    Enum representing different OpenAI model names on Azure, each identified by a unique deployment string.
    """
    gpt_35_turbo_16k = azure_service.get("gpt-35-turbo-16k", {}).get("model_name", None)
    gpt_4_32k = azure_service.get("gpt-4-32k", {}).get("model_name", None)
    gpt_4_turbo = azure_service.get("gpt-4-turbo", {}).get("model_name", None)
    text_embedding_ada_002 = azure_service.get("text-embedding-ada-002", {}).get("model_name", None)
    gpt_4o = azure_service.get("gpt-4o", {}).get("model_name", None)



class NemoModelnames(Enum):
    """
    Enum representing NVIDIA's Nemo model names, each corresponding to a specific model configuration.
    """
    mixtral_8x7b = nemo_service.get("mixtral_8x7b", {}).get("model_name", None)
    llama2_13b = nemo_service.get("playground_llama2_13b", {}).get("model_name", None)
    llama2_70b = nemo_service.get("playground_llama2_70b", {}).get("model_name", None)
    nvolveqa_40k = nemo_service.get("nvolveqa_40k", {}).get("model_name", None)
    ai_embed_qa_4 = nemo_service.get("ai-embed-qa-4", {}).get("model_name", None)


class OpenAIModelNames(Enum):
    """
    Enum representing different OpenAI model names, each identified by a unique deployment string.
    """
    gpt_35_turbo = openai_service.get("gpt-35-turbo", {}).get("model_name", None)
    gpt_35_turbo_16k = openai_service.get("gpt-35-turbo-16k", {}).get("model_name", None)
    gpt_4_turbo = openai_service.get("gpt-4-turbo", {}).get("model_name", None)
    gpt_4_32k = openai_service.get("gpt-4-32k", {}).get("model_name", None)
    text_embedding_ada_002 = openai_service.get("text-embedding-ada-002", {}).get("model_name", None)
    gpt_4o = openai_service.get("gpt-4o", {}).get("model_name", None)


logger.info(f"NemoModelnames: {NemoModelnames.__members__}")
logger.info(f"AzureOpenAIModelEngines:  {AzureOpenAIModelEngines.__members__}")
logger.info(f"AzureOpenAIModelNames:  {AzureOpenAIModelNames.__members__}")
logger.info(f"AzureOpenAIModelEngines:  {OpenAIModelNames.__members__}")



BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Settings(BaseSettings):

    """
    Settings for the application, loading configuration from environment variables and providing default values.
    """
    

    OPENAI_API_KEY: str = ''
    OPENAI_API_VERSION: str = '2024-02-15-preview'
    AZURE_OPENAI_KEY: str = ''
    AZURE_API_BASE: str = ''
    REACT_APP_API_URL:str 
    # Log
    LOGGING_LEVEL: str = 'INFO'

    def chat_file_temp_file_directory(self, user_id: int, chat_id: int) -> str:
        return os.path.join(f'../tmp/{user_id}/{chat_id}/raw_docs/')

    def html_files_temp_file_directory(self, user_id: int, chat_id: int) -> str:
        return os.path.join(f'../tmp/{user_id}/{chat_id}/html_files/')
    
    class Config:
        env_file = f'{BASE_DIR}/.env'


settings = Settings()
