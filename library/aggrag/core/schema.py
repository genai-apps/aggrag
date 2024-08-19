from pydantic import BaseModel, Field
from typing import Optional, Any, Dict
from library.aggrag.core.config import (
    settings,
    AzureOpenAIModelNames,
    AzureOpenAIModelEngines,
)

from library.aggrag.ragstore import Raptor, Base, SubQA, MetaLlama, MetaLang, TableBase
from pydantic import BaseModel

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
    ai_service: Literal["AzureOpenAI", "OpenAI", "NVIDIA"] = "AzureOpenAI"
    chunk_size: int = 512
    llm_model: AzureOpenAIModelNames = AzureOpenAIModelNames.gpt_35_turbo_16k
    llm_deployment: AzureOpenAIModelEngines = AzureOpenAIModelEngines.gpt_35_turbo_16k
    embed_model: AzureOpenAIModelNames = AzureOpenAIModelNames.text_embedding_ada_002
    embed_deployment: AzureOpenAIModelEngines = (
        AzureOpenAIModelEngines.text_embedding_ada_002
    )
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    context_prompt: str = DEFAULT_CONTEXT_PROMPT
    temperature: float = 0.1
    index_name: str = "base_index"
    # class Config:
    #     extra = 'forbid'


class MetaLlamaRagSetting(BaseModel):
    ai_service: Literal["AzureOpenAI", "OpenAI", "NVIDIA"] = "AzureOpenAI"
    chunk_size: int = 512
    llm_model: AzureOpenAIModelNames = AzureOpenAIModelNames.gpt_35_turbo_16k
    llm_deployment: AzureOpenAIModelEngines = AzureOpenAIModelEngines.gpt_35_turbo_16k
    embed_model: AzureOpenAIModelNames = AzureOpenAIModelNames.text_embedding_ada_002
    embed_deployment: AzureOpenAIModelEngines = (
        AzureOpenAIModelEngines.text_embedding_ada_002
    )
    metadata_json_schema: Optional[str] = Field(
        default=None, description="A JSON schema for the system prompt."
    )
    temperature: float = 0.1
    index_name: str = "meta_llama_index"
    # class Config:
    #     extra = 'forbid'


class MetaLangRagSetting(BaseModel):
    ai_service: Literal["AzureOpenAI", "OpenAI", "NVIDIA"] = "AzureOpenAI"
    chunk_size: int = 512
    llm_model: AzureOpenAIModelNames = AzureOpenAIModelNames.gpt_35_turbo_16k
    llm_deployment: AzureOpenAIModelEngines = AzureOpenAIModelEngines.gpt_35_turbo_16k
    embed_model: AzureOpenAIModelNames = AzureOpenAIModelNames.text_embedding_ada_002
    embed_deployment: AzureOpenAIModelEngines = (
        AzureOpenAIModelEngines.text_embedding_ada_002
    )
    metadata_json_schema: Optional[str] = Field(
        default=None, description="A JSON schema for the system prompt."
    )
    temperature: float = 0.1
    index_name: str = "meta_lang_index"
    # class Config:
    #     extra = 'forbid'


class SubQARagSetting(BaseModel):
    ai_service: Literal["AzureOpenAI", "OpenAI", "NVIDIA"] = "AzureOpenAI"
    chunk_size: int = 513
    llm_model: AzureOpenAIModelNames = AzureOpenAIModelNames.gpt_35_turbo_16k
    llm_deployment: AzureOpenAIModelEngines = AzureOpenAIModelEngines.gpt_35_turbo_16k
    embed_model: AzureOpenAIModelNames = AzureOpenAIModelNames.text_embedding_ada_002
    embed_deployment: AzureOpenAIModelEngines = (
        AzureOpenAIModelEngines.text_embedding_ada_002
    )
    CHAT_REFINE_PROMPT_TMPL_MSGS_CONTENT: str = CHAT_REFINE_PROMPT_TMPL_MSGS_CONTENT
    INDEX_TEXT_QA_PROMPT_TMPL_MSGS_CONTENT: str = INDEX_TEXT_QA_PROMPT_TMPL_MSGS_CONTENT
    SUBQ_TEXT_QA_PROMPT_TMPL_MSGS_CONTENT: str = SUBQ_TEXT_QA_PROMPT_TMPL_MSGS_CONTENT
    DEFAULT_OPENAI_SUB_QUESTION_PROMPT_TMPL: str = (
        DEFAULT_OPENAI_SUB_QUESTION_PROMPT_TMPL
    )
    INDEX_TEXT_QA_SYSTEM_PROMPT_CONTENT: str = INDEX_TEXT_QA_SYSTEM_PROMPT_CONTENT
    SUBQ_TEXT_QA_SYSTEM_PROMPT_CONTENT: str = SUBQ_TEXT_QA_SYSTEM_PROMPT_CONTENT
    index_name: str = "subqa_index"
    temperature: float = 0.2
    # class Config:
    #     extra = 'forbid'


class RaptorRagSetting(BaseModel):
    ai_service: Literal["AzureOpenAI", "OpenAI", "NVIDIA"] = "AzureOpenAI"
    chunk_size: int = 514
    llm_model: AzureOpenAIModelNames = AzureOpenAIModelNames.gpt_35_turbo_16k
    llm_deployment: AzureOpenAIModelEngines = AzureOpenAIModelEngines.gpt_35_turbo_16k
    embed_model: AzureOpenAIModelNames = AzureOpenAIModelNames.text_embedding_ada_002
    embed_deployment: AzureOpenAIModelEngines = (
        AzureOpenAIModelEngines.text_embedding_ada_002
    )
    summary_prompt: str = SUMMARY_PROMPT
    temperature: float = 0.3
    index_name: str = "raptor_index"
    # class Config:
    #     extra = 'forbid'

class TableBaseRagSetting(BaseModel):
    ai_service: Literal["AzureOpenAI", "OpenAI", "NVIDIA"] = "AzureOpenAI"
    chunk_size: int = 512
    llm_model: AzureOpenAIModelNames = AzureOpenAIModelNames.gpt_35_turbo_16k
    llm_deployment: AzureOpenAIModelEngines = AzureOpenAIModelEngines.gpt_35_turbo_16k
    embed_model: AzureOpenAIModelNames = AzureOpenAIModelNames.text_embedding_ada_002
    embed_deployment: AzureOpenAIModelEngines = (
        AzureOpenAIModelEngines.text_embedding_ada_002
    )
    engine_prompt: str = DEFAULT_TABLEBASE_PROMPT
    temperature: float = 0.1
    index_name: str = "tableBase_index"
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
    usecase_name: str = "introductory_questions"
    iteration: str = "iteration 1"
    upload_type: str = "pdf"
