import logging
import asyncio
import os


from llama_index.core import SimpleDirectoryReader
from typing import Optional

from library.aggrag.core.config import AzureOpenAIModelNames, AzureOpenAIModelEngines
from library.aggrag.core.ai_service import AIServiceFactory

from library.aggrag.ragstore import SubQA, Base, Raptor, MetaLlama, MetaLang, TableBase

from library.aggrag.core.schema import RagStoreBool, RagStore, RagStoreSettings, BaseRagSetting, RaptorRagSetting, SubQARagSetting, MetaLlamaRagSetting, MetaLangRagSetting, TableBaseRagSetting

logger = logging.getLogger(__name__)


class AggRAG:
    def __init__(self, 
                 DATA_DIR: str, 
                 ragstore_bool: RagStoreBool, 
                 usecase_name:  Optional[str] = None, 
                 iteration:  Optional[str] = None, 
                 upload_type: Optional[str] = None, 
                 ragstore_settings: Optional[RagStoreSettings] = None):

        self.documents = None
        self.table_docs = None
        self.index = None
        from library.aggrag.core.schema import UserConfig

        self.usecase_name = usecase_name or UserConfig.usecase_name
        self.iteration = iteration or UserConfig.iteration

        self.BASE_DIR=os.path.join("/configurations",self.usecase_name, self.iteration)  
        self.DATA_DIR = os.path.join(self.BASE_DIR,'raw_docs')
        self.PERSIST_DIR = os.path.join(self.BASE_DIR,'index')
        self.ragstore_settings = ragstore_settings or RagStoreSettings()

        self.upload_type = upload_type
        if self.upload_type == 'url':
            self.DATA_DIR = os.path.join(DATA_DIR, 'html_files')
            logger.info(f"Data directory: {self.DATA_DIR}")

        elif self.upload_type == 'doc' or self.upload_type == 'pdf':
            self.DATA_DIR = os.path.join(DATA_DIR, 'raw_docs')
            logger.info(f"Data directory: {self.DATA_DIR}")


        def create_ragstore(ragstore_bool: RagStoreBool) -> RagStore:
            """
            Create a RagStore instance based on the boolean values in RagStoreBool.

            This function initializes the RagStore with Base, Raptor, and SubQA components
            if their corresponding boolean values in ragstore_bool are set to True. It uses
            the provided usecase_name, iteration, and ragstore_settings to configure each component.

            Args:
                ragstore_bool (RagStoreBool): A boolean configuration indicating which components to initialize.

            Returns:
                RagStore: An instance of RagStore with the specified components initialized.
            """
            return RagStore(
                base=Base(
                    usecase_name=self.usecase_name,
                    iteration=self.iteration,
                    base_rag_setting=self.ragstore_settings.base_rag_setting if self.ragstore_settings.base_rag_setting else BaseRagSetting(),
                    llm=AIServiceFactory.get_ai_service(
                            ai_service=self.ragstore_settings.base_rag_setting.ai_service,
                            llm_model=self.ragstore_settings.base_rag_setting.llm_model,
                        ).llm,
                    embed_model=AIServiceFactory.get_ai_service(
                            ai_service=self.ragstore_settings.base_rag_setting.embed_ai_service,
                            embed_model=self.ragstore_settings.base_rag_setting.embed_model,
                        ).embed_model
                ) if ragstore_bool.base else None,

                subqa=SubQA(
                    usecase_name=self.usecase_name,
                    iteration=self.iteration,
                    DATA_DIR=self.DATA_DIR,
                    subqa_rag_setting=self.ragstore_settings.subqa_rag_setting if self.ragstore_settings.subqa_rag_setting else SubQARagSetting(),
                    llm=AIServiceFactory.get_ai_service(
                            ai_service=self.ragstore_settings.subqa_rag_setting.ai_service,
                            llm_model=self.ragstore_settings.subqa_rag_setting.llm_model,
                        ).llm,
                    embed_model=AIServiceFactory.get_ai_service(
                            ai_service=self.ragstore_settings.subqa_rag_setting.embed_ai_service,
                            embed_model=self.ragstore_settings.subqa_rag_setting.embed_model,
                        ).embed_model
                ) if ragstore_bool.subqa else None,

                raptor=Raptor(
                    usecase_name=self.usecase_name,
                    iteration=self.iteration,
                    DATA_DIR=self.DATA_DIR,
                    raptor_rag_setting=self.ragstore_settings.raptor_rag_setting if self.ragstore_settings.raptor_rag_setting else RaptorRagSetting(),
                    llm=AIServiceFactory.get_ai_service(
                            ai_service=self.ragstore_settings.raptor_rag_setting.ai_service,
                            llm_model=self.ragstore_settings.raptor_rag_setting.llm_model,
                        ).llm,
                    embed_model=AIServiceFactory.get_ai_service(
                            ai_service=self.ragstore_settings.raptor_rag_setting.embed_ai_service,
                            embed_model=self.ragstore_settings.raptor_rag_setting.embed_model,
                        ).embed_model
                ) if ragstore_bool.raptor else None,

                meta_llama=MetaLlama(
                    usecase_name=self.usecase_name,
                    iteration=self.iteration,
                    DATA_DIR=self.DATA_DIR,
                    meta_llama_rag_setting=self.ragstore_settings.meta_llama_rag_setting if self.ragstore_settings.meta_llama_rag_setting else MetaLlamaRagSetting(),
                    llm=AIServiceFactory.get_ai_service(
                            ai_service=self.ragstore_settings.meta_llama_rag_setting.ai_service,
                            llm_model=self.ragstore_settings.meta_llama_rag_setting.llm_model,
                        ).llm,
                    embed_model=AIServiceFactory.get_ai_service(
                            ai_service=self.ragstore_settings.meta_llama_rag_setting.embed_ai_service,
                            embed_model=self.ragstore_settings.meta_llama_rag_setting.embed_model,
                        ).embed_model                        
                ) if ragstore_bool.meta_llama else None,

                meta_lang=MetaLang(
                    usecase_name=self.usecase_name,
                    iteration=self.iteration,
                    DATA_DIR=self.DATA_DIR,
                    meta_lang_rag_setting=self.ragstore_settings.meta_lang_rag_setting if self.ragstore_settings.meta_lang_rag_setting else MetaLangRagSetting(),
                    llm=AIServiceFactory.get_ai_service(
                            ai_service=self.ragstore_settings.meta_lang_rag_setting.ai_service,
                            llm_model=self.ragstore_settings.meta_lang_rag_setting.llm_model,
                        ).llm,
                    embed_model=AIServiceFactory.get_ai_service(
                            ai_service=self.ragstore_settings.meta_lang_rag_setting.embed_ai_service,
                            embed_model=self.ragstore_settings.meta_lang_rag_setting.embed_model,
                        ).embed_model  
                ) if ragstore_bool.meta_lang else None,

                tableBase=TableBase(
                    usecase_name=self.usecase_name,
                    iteration=self.iteration,
                    DATA_DIR=self.DATA_DIR,
                    tableBase_rag_setting=self.ragstore_settings.tableBase_rag_setting if self.ragstore_settings.tableBase_rag_setting else TableBaseRagSetting()
                ) if ragstore_bool.tableBase else None,

            )

        self.ragstore: RagStore = create_ragstore(ragstore_bool)

    def documents_loader(self, DIR=None):
        """
        TODO: Include transformations / chunking here or keep this base
        Option1: keep base documents loader here and use transformations/chunking in the ragstore
        Option2: use common transformations/chunking for all rags in the ragostre.
        """
        self.DATA_DIR = DIR or self.DATA_DIR
        if not os.path.exists(self.DATA_DIR):
            logger.error(f"Data directory does not exist: {self.DATA_DIR}")
            raise FileNotFoundError(f"Data directory does not exist: {self.DATA_DIR}")
        if self.ragstore.tableBase:
            self.table_docs = self.ragstore.tableBase.documents_loader(self.DATA_DIR)
        self.documents = SimpleDirectoryReader(self.DATA_DIR, recursive=True, exclude_hidden=True).load_data()
        return self.documents

    async def create_all_index_async(self, documents, exclude=[], include=[]):
        """
        Using multiprocessing might cause more overhead then only running asyncio in case of creating index.
        TODO: Add async for loop to create multiple index may be using multiprocessing or only async
        It will be a good idea however to use different embedding/llm models for each rag to avoid "rush" initially.
        """
        if not documents:
            documents = self.documents

        # Initialize tasks for creating indexes for each RAG in the RagStore
        tasks = []
        if self.ragstore.base and self.ragstore.base.name not in exclude:
            tasks.append(self.ragstore.base.create_index_async(documents))

        if self.ragstore.raptor and self.ragstore.raptor.name not in exclude:
            tasks.append(self.ragstore.raptor.create_index_async(documents))

        if self.ragstore.subqa and self.ragstore.subqa.name not in exclude:
            tasks.append(self.ragstore.subqa.create_index_async(documents))

        if self.ragstore.meta_llama and self.ragstore.meta_llama.name not in exclude:
            tasks.append(self.ragstore.meta_llama.create_index_async(documents))        

        if self.ragstore.meta_lang and self.ragstore.meta_lang.name not in exclude:
            tasks.append(self.ragstore.meta_lang.create_index_async(documents)) 
        
        if self.ragstore.tableBase and self.ragstore.tableBase.name not in exclude:
            tasks.append(self.ragstore.tableBase.create_index_async(self.table_docs)) 



        # Wait for all index creation tasks to complete
        indexes = await asyncio.gather(*tasks)
        return indexes

    async def retrieve_all_index_async(self):
        """ TODO:
        - Retrieve index for the all the registered rags
        """

        # Initialize tasks for creating indexes for each RAG in the RagStore
        tasks = []
        if self.ragstore.base:
            tasks.append(self.ragstore.base.retrieve_index_async())

        if self.ragstore.raptor:
            tasks.append(self.ragstore.raptor.retrieve_index_async())

        if self.ragstore.subqa:
            tasks.append(self.ragstore.subqa.retrieve_index_async())

        if self.ragstore.meta_llama:
            tasks.append(self.ragstore.meta_llama.retrieve_index_async())

        if self.ragstore.meta_lang:
            tasks.append(self.ragstore.meta_lang.retrieve_index_async())

        if self.ragstore.tableBase:
            tasks.append(self.ragstore.tableBase.retrieve_index_async())

        # Wait for all index creation tasks to complete
        indexes = await asyncio.gather(*tasks)
        logger.info(f"Retrieved indexes: {indexes}")
        return indexes

    async def load_chat_engines(self):
        """ TODO: Load chat engines for all ragstores"""
        # Initialize tasks for creating indexes for each RAG in the RagStore
        tasks = []
        if self.ragstore.base:
            tasks.append(self.ragstore.base.get_chat_engine())
        if self.ragstore.raptor:
            tasks.append(self.ragstore.raptor.get_chat_engine())
        if self.ragstore.subqa:
            tasks.append(self.ragstore.subqa.get_chat_engine())
        if self.ragstore.tableBase:
            tasks.append(self.ragstore.tableBase.get_chat_engine())

        chat_engine = await asyncio.gather(*tasks)

        return chat_engine

    async def ragstore_chat(self, query, streaming=False):

        """
        Conducts asynchronous chat sessions using all configured RAG store components based on the given query.

        This method manages parallel chat interactions across different RAG configurations (like base, raptor, subqa)
        based on the provided query. It can handle both streaming and non-streaming chat modes, optionally incorporating
        historical chat data if required.

        Parameters:
            query (str): The chat query or prompt to be processed by the chat engines.

        Returns:
            list: A list of responses from each active RAG configuration, formatted according to the respective
                configuration's response structure.

        Notes:
            - The method initializes and potentially waits for chat engines if they are not already active.
            - Responses are gathered asynchronously to optimize performance and response time.
        """

        """ TODO: Run parallel chats with all rags in the ragstore"""
        try:
            _evaluation = False  # to come from the settings
            # Initialize tasks for creating indexes for each RAG in the RagStore
            chat_history=None
            tasks = []

            if self.ragstore.base and streaming:
                tasks.append(self.ragstore.base.astream_chat(query, chat_history, _evaluation))
            elif self.ragstore.base:
                tasks.append(self.ragstore.base.achat(query, chat_history, _evaluation))

            if self.ragstore.raptor:
                if not self.ragstore.raptor.chat_engine:
                    await self.ragstore.raptor.get_chat_engine()

                if self.ragstore.raptor.chat_engine:
                    tasks.append(self.ragstore.raptor.achat(query, chat_history, _evaluation))

            if self.ragstore.subqa and streaming:
                tasks.append(self.ragstore.subqa.astream_chat(query, chat_history))
            elif self.ragstore.subqa:
                tasks.append(self.ragstore.subqa.achat(query, chat_history, _evaluation))

            if self.ragstore.meta_llama and streaming:
                tasks.append(self.ragstore.meta_llama.astream_chat(query, chat_history))
            elif self.ragstore.meta_llama:
                tasks.append(self.ragstore.meta_llama.achat(query, chat_history, _evaluation))


            if self.ragstore.meta_lang and streaming:
                tasks.append(self.ragstore.meta_lang.astream_chat(query, chat_history))
            elif self.ragstore.meta_lang:
                tasks.append(self.ragstore.meta_lang.achat(query, chat_history, _evaluation))        
    
            if self.ragstore.tableBase:
                if not self.ragstore.tableBase.chat_engine:
                    await self.ragstore.tableBase.get_chat_engine()

                if self.ragstore.tableBase.chat_engine:
                    tasks.append(self.ragstore.tableBase.achat(query, chat_history, _evaluation))


            response = await asyncio.gather(*tasks)

            logger.debug(f"Response received: {response}")
            return response
        except Exception as e:
            logger.error(f"Error in ragstorechat: {str(e)}")
            raise 
