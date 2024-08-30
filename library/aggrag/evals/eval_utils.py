from langchain_openai import AzureChatOpenAI
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings

from library.aggrag.core.config import settings, AzureOpenAIModelNames, AzureOpenAIModelEngines
from deepeval.models.base_model import DeepEvalBaseLLM
from library.aggrag.core.config import settings, AzureOpenAIModelEngines


class AzureOpenAI(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom Azure OpenAI Model"

'''
llm_openai = ChatOpenAI (
    model="gpt-4o",
    api_key=settings.OPENAI_API_KEY,
    temperature=0.2
)
'''


def load_llm_for_deepeval():
    custom_model = AzureChatOpenAI(
        openai_api_version=settings.OPENAI_API_VERSION,
        azure_deployment=AzureOpenAIModelEngines.gpt_4o.value,
        azure_endpoint=settings.AZURE_API_BASE,
        openai_api_key=settings.AZURE_OPENAI_KEY,
    )
    judge_llm = AzureOpenAI(model=custom_model)
    return judge_llm


def load_llms_for_ragas():
    llm = AzureChatOpenAI(
        azure_deployment=AzureOpenAIModelEngines.gpt_4o.value,
        openai_api_version=settings.OPENAI_API_VERSION,
        api_key=settings.AZURE_OPENAI_KEY,
        azure_endpoint=settings.AZURE_API_BASE,
        validate_base_url=False,
    )

    embed_llm = AzureOpenAIEmbeddings(
        api_key=settings.AZURE_OPENAI_KEY,
        azure_endpoint=settings.AZURE_API_BASE,
        openai_api_version=settings.OPENAI_API_VERSION,
        azure_deployment=AzureOpenAIModelEngines.text_embedding_ada_002.value,
        model=AzureOpenAIModelNames.text_embedding_ada_002.value,
    )
    return llm, embed_llm