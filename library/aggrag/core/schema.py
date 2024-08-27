from pydantic import BaseModel, Field, model_validator
from typing import Optional, Any, Dict, Union
from library.aggrag.core.config import settings, AzureOpenAIModelNames, AzureOpenAIModelEngines, OpenAIModelNames

from library.aggrag.ragstore import Raptor, Base, SubQA, MetaLlama, MetaLang, TableBase
from pydantic import BaseModel
from library.aggrag.core.config import ai_services_config, all_ai_services

from typing import Literal
from library.aggrag.prompts import (
    DEFAULT_CONTEXT_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    CHAT_REFINE_PROMPT_TMPL_MSGS_CONTENT,
    INDEX_TEXT_QA_PROMPT_TMPL_MSGS_CONTENT,
    SUBQ_TEXT_QA_PROMPT_TMPL_MSGS_CONTENT,
    DEFAULT_OPENAI_SUB_QUESTION_PROMPT_TMPL,
    INDEX_TEXT_QA_SYSTEM_PROMPT_CONTENT,
    SUBQ_TEXT_QA_SYSTEM_PROMPT_CONTENT,
    SUMMARY_PROMPT,
    DEFAULT_TABLEBASE_PROMPT
)


class BaseRagSetting(BaseModel):
    ai_service: Optional[str] = 'AzureOpenAI'
    embed_ai_service: Optional[str] = 'AzureOpenAI'
    chunk_size: Optional[int] = 512
    llm_model: Optional[str] = None
    llm_deployment: Optional[AzureOpenAIModelEngines] = None
    embed_model: Optional[str] = None
    embed_deployment: Optional[AzureOpenAIModelEngines] = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    context_prompt: str = DEFAULT_CONTEXT_PROMPT
    temperature: float = 0.1
    index_name: str = "base_index"

    @model_validator(mode='before')
    def input_validation(cls, values):

        ai_service = values.get('ai_service')
        embed_ai_service = values.get('embed_ai_service')
        llm_model = values.get('llm_model')
        embed_model = values.get('embed_model')

        if ai_service and ai_service not in all_ai_services :
            raise ValueError(f"Invalid AI service '{ai_service}'. Expected one of {all_ai_services}.")
        
        if embed_ai_service and  embed_ai_service not in all_ai_services:
            raise ValueError(f"Invalid Embed AI service '{embed_ai_service}'. Expected one of {all_ai_services}.")

        if llm_model and ai_service is None:
            raise ValueError(f"Please provide ai_service as well while providing a llm_model or just opt out llm_model to use the default service and model")

        if embed_model and embed_ai_service is None:
            raise ValueError(f"Please provide embed_ai_service as well while providing a embed_model or just opt out embed_model to use the default service and model")


        expected_llm_models = [
            model['model_name'] 
            for model in ai_services_config.get(ai_service, {}).get('chat_models', {}).values()  # Access models in chat_models
        ]

        if llm_model and llm_model not in expected_llm_models:
            raise ValueError(f"Invalid model '{llm_model}' for ai_service '{ai_service}'. Expected one of {expected_llm_models}")


        expected_embed_models = [
            model_info['model_name'] 
            for model_info in ai_services_config.get(embed_ai_service, {}).get('embed_models', {}).values()  # Access models in embed_models
        ]


        if embed_model and embed_model not in expected_embed_models:
                raise ValueError(f"Invalid model '{llm_model}' for service '{ai_service}'. Expected a type of embedding model from the list {expected_embed_models}")

        return values
    # class Config:
    #     extra = 'forbid'

class MetaLlamaRagSetting(BaseModel):
    ai_service: Optional[str] = 'AzureOpenAI'
    embed_ai_service: Optional[str] = 'AzureOpenAI'
    chunk_size: Optional[int] = 512
    llm_model: Optional[str] = None
    llm_deployment: Optional[AzureOpenAIModelEngines] = None
    embed_model: Optional[str] = None
    embed_deployment: Optional[AzureOpenAIModelEngines] = None
    metadata_json_schema: Optional[str] = Field(
        default=None, description="A JSON schema for the system prompt."
    )
    temperature: float = 0.1
    index_name: str = "meta_llama_index"

    @model_validator(mode='before')
    def input_validation(cls, values):

        ai_service = values.get('ai_service')
        embed_ai_service = values.get('embed_ai_service')
        llm_model = values.get('llm_model')
        embed_model = values.get('embed_model')

        if ai_service and ai_service not in all_ai_services :
            raise ValueError(f"Invalid AI service '{ai_service}'. Expected one of {all_ai_services}.")
        
        if embed_ai_service and  embed_ai_service not in all_ai_services:
            raise ValueError(f"Invalid Embed AI service '{embed_ai_service}'. Expected one of {all_ai_services}.")

        if llm_model and ai_service is None:
            raise ValueError(f"Please provide ai_service as well while providing a llm_model or just opt out llm_model to use the default service and model")

        if embed_model and embed_ai_service is None:
            raise ValueError(f"Please provide embed_ai_service as well while providing a embed_model or just opt out embed_model to use the default service and model")


        expected_llm_models = [
            model['model_name'] 
            for model in ai_services_config.get(ai_service, {}).get('chat_models', {}).values()  # Access models in chat_models
        ]

        if llm_model and llm_model not in expected_llm_models:
            raise ValueError(f"Invalid model '{llm_model}' for ai_service '{ai_service}'. Expected one of {expected_llm_models}")


        expected_embed_models = [
            model_info['model_name'] 
            for model_info in ai_services_config.get(embed_ai_service, {}).get('embed_models', {}).values()  # Access models in embed_models
        ]


        if embed_model and embed_model not in expected_embed_models:
                raise ValueError(f"Invalid model '{llm_model}' for service '{ai_service}'. Expected a type of embedding model from the list {expected_embed_models}")

        return values
    # class Config:
    #     extra = 'forbid'

class MetaLangRagSetting(BaseModel):
    ai_service: Optional[str] = 'AzureOpenAI'
    embed_ai_service: Optional[str] = 'AzureOpenAI'
    chunk_size: Optional[int] = 512
    llm_model: Optional[str] = None
    llm_deployment: Optional[AzureOpenAIModelEngines] = None
    embed_model: Optional[str] = None
    embed_deployment: Optional[AzureOpenAIModelEngines] = None
    metadata_json_schema: Optional[str] = Field(
        default=None, description="A JSON schema for the system prompt."
    )
    temperature: float = 0.1
    index_name: str = "meta_lang_index"

    @model_validator(mode='before')
    def input_validation(cls, values):

        ai_service = values.get('ai_service')
        embed_ai_service = values.get('embed_ai_service')
        llm_model = values.get('llm_model')
        embed_model = values.get('embed_model')

        if ai_service and ai_service not in all_ai_services :
            raise ValueError(f"Invalid AI service '{ai_service}'. Expected one of {all_ai_services}.")
        
        if embed_ai_service and  embed_ai_service not in all_ai_services:
            raise ValueError(f"Invalid Embed AI service '{embed_ai_service}'. Expected one of {all_ai_services}.")

        if llm_model and ai_service is None:
            raise ValueError(f"Please provide ai_service as well while providing a llm_model or just opt out llm_model to use the default service and model")

        if embed_model and embed_ai_service is None:
            raise ValueError(f"Please provide embed_ai_service as well while providing a embed_model or just opt out embed_model to use the default service and model")


        expected_llm_models = [
            model['model_name'] 
            for model in ai_services_config.get(ai_service, {}).get('chat_models', {}).values()  # Access models in chat_models
        ]

        if llm_model and llm_model not in expected_llm_models:
            raise ValueError(f"Invalid model '{llm_model}' for ai_service '{ai_service}'. Expected one of {expected_llm_models}")


        expected_embed_models = [
            model_info['model_name'] 
            for model_info in ai_services_config.get(embed_ai_service, {}).get('embed_models', {}).values()  # Access models in embed_models
        ]


        if embed_model and embed_model not in expected_embed_models:
                raise ValueError(f"Invalid model '{llm_model}' for service '{ai_service}'. Expected a type of embedding model from the list {expected_embed_models}")

        return values
    # class Config:
    #     extra = 'forbid'


class SubQARagSetting(BaseModel):
    ai_service: Optional[str] = 'AzureOpenAI'
    embed_ai_service: Optional[str] = 'AzureOpenAI'
    chunk_size: Optional[int] = 512
    llm_model: Optional[str] = None
    llm_deployment: Optional[AzureOpenAIModelEngines] = None
    embed_model: Optional[str] = None
    embed_deployment: Optional[AzureOpenAIModelEngines] = None
    CHAT_REFINE_PROMPT_TMPL_MSGS_CONTENT: str = CHAT_REFINE_PROMPT_TMPL_MSGS_CONTENT
    INDEX_TEXT_QA_PROMPT_TMPL_MSGS_CONTENT: str = INDEX_TEXT_QA_PROMPT_TMPL_MSGS_CONTENT
    SUBQ_TEXT_QA_PROMPT_TMPL_MSGS_CONTENT: str = SUBQ_TEXT_QA_PROMPT_TMPL_MSGS_CONTENT 
    DEFAULT_OPENAI_SUB_QUESTION_PROMPT_TMPL: str = DEFAULT_OPENAI_SUB_QUESTION_PROMPT_TMPL  
    INDEX_TEXT_QA_SYSTEM_PROMPT_CONTENT: str = INDEX_TEXT_QA_SYSTEM_PROMPT_CONTENT
    SUBQ_TEXT_QA_SYSTEM_PROMPT_CONTENT: str = SUBQ_TEXT_QA_SYSTEM_PROMPT_CONTENT
    index_name: str = "subqa_index"
    temperature: float = 0.2

    @model_validator(mode='before')
    def input_validation(cls, values):

        ai_service = values.get('ai_service')
        embed_ai_service = values.get('embed_ai_service')
        llm_model = values.get('llm_model')
        embed_model = values.get('embed_model')

        if ai_service and ai_service not in all_ai_services :
            raise ValueError(f"Invalid AI service '{ai_service}'. Expected one of {all_ai_services}.")
        
        if embed_ai_service and  embed_ai_service not in all_ai_services:
            raise ValueError(f"Invalid Embed AI service '{embed_ai_service}'. Expected one of {all_ai_services}.")

        if llm_model and ai_service is None:
            raise ValueError(f"Please provide ai_service as well while providing a llm_model or just opt out llm_model to use the default service and model")

        if embed_model and embed_ai_service is None:
            raise ValueError(f"Please provide embed_ai_service as well while providing a embed_model or just opt out embed_model to use the default service and model")


        expected_llm_models = [
            model['model_name'] 
            for model in ai_services_config.get(ai_service, {}).get('chat_models', {}).values()  # Access models in chat_models
        ]

        if llm_model and llm_model not in expected_llm_models:
            raise ValueError(f"Invalid model '{llm_model}' for ai_service '{ai_service}'. Expected one of {expected_llm_models}")


        expected_embed_models = [
            model_info['model_name'] 
            for model_info in ai_services_config.get(embed_ai_service, {}).get('embed_models', {}).values()  # Access models in embed_models
        ]


        if embed_model and embed_model not in expected_embed_models:
                raise ValueError(f"Invalid model '{llm_model}' for service '{ai_service}'. Expected a type of embedding model from the list {expected_embed_models}")

        return values
    # class Config:
    #     extra = 'forbid'

class RaptorRagSetting(BaseModel):
    ai_service: Optional[str] = 'AzureOpenAI'
    embed_ai_service: Optional[str] = 'AzureOpenAI'
    chunk_size: Optional[int] = 512
    llm_model: Optional[str] = None
    llm_deployment: Optional[AzureOpenAIModelEngines] = None
    embed_model: Optional[str] = None
    embed_deployment: Optional[AzureOpenAIModelEngines] = None
    summary_prompt: str = SUMMARY_PROMPT
    temperature: float = 0.3
    index_name: str = "raptor_index"

    @model_validator(mode='before')
    def input_validation(cls, values):

        ai_service = values.get('ai_service')
        embed_ai_service = values.get('embed_ai_service')
        llm_model = values.get('llm_model')
        embed_model = values.get('embed_model')

        if ai_service and ai_service not in all_ai_services :
            raise ValueError(f"Invalid AI service '{ai_service}'. Expected one of {all_ai_services}.")
        
        if embed_ai_service and  embed_ai_service not in all_ai_services:
            raise ValueError(f"Invalid Embed AI service '{embed_ai_service}'. Expected one of {all_ai_services}.")

        if llm_model and ai_service is None:
            raise ValueError(f"Please provide ai_service as well while providing a llm_model or just opt out llm_model to use the default service and model")

        if embed_model and embed_ai_service is None:
            raise ValueError(f"Please provide embed_ai_service as well while providing a embed_model or just opt out embed_model to use the default service and model")


        expected_llm_models = [
            model['model_name'] 
            for model in ai_services_config.get(ai_service, {}).get('chat_models', {}).values()  # Access models in chat_models
        ]

        if llm_model and llm_model not in expected_llm_models:
            raise ValueError(f"Invalid model '{llm_model}' for ai_service '{ai_service}'. Expected one of {expected_llm_models}")


        expected_embed_models = [
            model_info['model_name'] 
            for model_info in ai_services_config.get(embed_ai_service, {}).get('embed_models', {}).values()  # Access models in embed_models
        ]


        if embed_model and embed_model not in expected_embed_models:
                raise ValueError(f"Invalid model '{llm_model}' for service '{ai_service}'. Expected a type of embedding model from the list {expected_embed_models}")

        return values
    # class Config:
    #     extra = 'forbid'

class TableBaseRagSetting(BaseModel):
    ai_service: Optional[str] = 'AzureOpenAI'
    embed_ai_service: Optional[str] = 'AzureOpenAI'
    chunk_size: Optional[int] = 512
    llm_model: Optional[str] = None
    llm_deployment: Optional[AzureOpenAIModelEngines] = None
    embed_model: Optional[str] = None
    embed_deployment: Optional[AzureOpenAIModelEngines] = None
    engine_prompt: str = DEFAULT_TABLEBASE_PROMPT
    temperature: float = 0.1
    index_name: str = "tableBase_index"
    # class Config:
    #     extra = 'forbid'
    @model_validator(mode='before')
    def input_validation(cls, values):

        ai_service = values.get('ai_service')
        embed_ai_service = values.get('embed_ai_service')
        llm_model = values.get('llm_model')
        embed_model = values.get('embed_model')

        if ai_service and ai_service not in all_ai_services :
            raise ValueError(f"Invalid AI service '{ai_service}'. Expected one of {all_ai_services}.")
        
        if embed_ai_service and  embed_ai_service not in all_ai_services:
            raise ValueError(f"Invalid Embed AI service '{embed_ai_service}'. Expected one of {all_ai_services}.")

        if llm_model and ai_service is None:
            raise ValueError(f"Please provide ai_service as well while providing a llm_model or just opt out llm_model to use the default service and model")

        if embed_model and embed_ai_service is None:
            raise ValueError(f"Please provide embed_ai_service as well while providing a embed_model or just opt out embed_model to use the default service and model")


        expected_llm_models = [
            model['model_name'] 
            for model in ai_services_config.get(ai_service, {}).get('chat_models', {}).values()  # Access models in chat_models
        ]

        if llm_model and llm_model not in expected_llm_models:
            raise ValueError(f"Invalid model '{llm_model}' for ai_service '{ai_service}'. Expected one of {expected_llm_models}")


        expected_embed_models = [
            model_info['model_name'] 
            for model_info in ai_services_config.get(embed_ai_service, {}).get('embed_models', {}).values()  # Access models in embed_models
        ]


        if embed_model and embed_model not in expected_embed_models:
                raise ValueError(f"Invalid model '{llm_model}' for service '{ai_service}'. Expected a type of embedding model from the list {expected_embed_models}")

        return values
    # class Config:
    #     extra = 'forbid'


class RagStoreSettings(BaseModel):
    base_rag_setting: Optional[BaseRagSetting] = None
    raptor_rag_setting: Optional[RaptorRagSetting] = None
    subqa_rag_setting: Optional[SubQARagSetting] = None
    meta_llama_rag_setting: Optional[MetaLlamaRagSetting] = None
    meta_lang_rag_setting: Optional[MetaLangRagSetting] = None
    tableBase_rag_setting: Optional[TableBaseRagSetting] = None


class RagStore(BaseModel):
    base: Optional[Base] 
    raptor: Optional[Raptor] = None
    subqa: Optional[SubQA] = None
    meta_llama: Optional[MetaLlama] = None
    meta_lang: Optional[MetaLang] = None
    tableBase: Optional[TableBase] = None

    class Config:
        arbitrary_types_allowed = True


class RagStoreBool(BaseModel):
    base: Optional[bool] = False
    raptor: Optional[bool] = False
    subqa: Optional[bool] = False
    meta_llama: Optional[bool] = False
    meta_lang: Optional[bool] = False
    tableBase: Optional[bool] = False


class UserConfig(BaseModel):
    usecase_name: str = 'introductory_questions'
    iteration: str = 'iteration 1'
    upload_type: str ='pdf'