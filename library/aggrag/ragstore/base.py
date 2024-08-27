import logging
import os
import time

from typing import Optional

from datasets import Dataset
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.node_parser import HTMLNodeParser

from library.aggrag.core.ai_service import ReplicateAIService, TogetherAIService, AIServiceFactory


from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from ragas.metrics import (
    answer_relevancy
)

from library.aggrag.evals.llm_evaluator import Evaluator
from library.aggrag.core.utils import get_time_taken
from library.aggrag.core.config import settings

metrics = [
        answer_relevancy,
        ]

parser = SimpleNodeParser.from_defaults(
    chunk_size=512,
    include_prev_next_rel=False,
)

logger = logging.getLogger(__name__)





BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Base:

    """
    Base class for handling document loading, index creation and retrieval, and chat functionalities
    for a specific configuration of RAG.

    Attributes:
        name (str): Identifier name for the version of the base configuration.
        llm (AzureOpenAI): Configured Azure Language Model for generating responses.
        embed_model (AzureOpenAIEmbedding): Configured embedding model for document indexing.
        documents (list): Loaded documents for processing.
        index_name (str): Name of the created index for document vectors.
        index (VectorStoreIndex): Index object containing document vectors.
        usecase_name (str): User identifier for session-specific data handling.
        iteration (str): Session identifier.
        DATA_DIR (str): Base directory for storing data.
        PERSIST_DIR (str): Directory for persisting index data.
        CHAT_DIR (str): Directory for storing chat data.
        upload_type (str): Type of documents being uploaded ('url', 'doc', 'pdf').
        chat_engine (ChatEngine): Chat engine for handling chat interactions.
    """


    def __init__(self,
                 usecase_name: str, 
                 iteration: str,
                 DATA_DIR: Optional[str] = None,
                 upload_type: Optional[str] = None,
                 base_rag_setting= None,
                 llm: str = None,
                 embed_model: str = None):
        
        """
        Initializes a base configuration for RAG with given parameters, setting up directories and logging essential information.

        Parameters:
            usecase_name (str, optional): Identifier for the user.
            iteration (str, optional): Identifier for the session.
            upload_type (str, optional): Type of documents being uploaded ('url', 'doc', 'pdf').
            DATA_DIR (str, optional): Directory where data will be stored and processed.
            llm (AzureOpenAI, optional): Configured Azure Language Model.
            embed_model (AzureOpenAIEmbedding, optional): Configured embedding model.
        """
        self.name = 'base v2'

        from library.aggrag.core.schema import BaseRagSetting

        # TODO: Ingest variables wherever required
        # This will be used in later phases of developemnt
        self.ai_service = base_rag_setting.ai_service 
        self.chunk_size = base_rag_setting.chunk_size

        self.system_prompt=base_rag_setting.system_prompt
        self.context_prompt=base_rag_setting.context_prompt
        
        self.llm = llm
        self.embed_model = embed_model

        self.documents = None
        self.index_name = base_rag_setting.index_name or "base_index"
        
        self.index = None
        from library.aggrag.core.schema import UserConfig

        self.usecase_name=usecase_name or UserConfig.usecase_name
        self.iteration=iteration or UserConfig.iteration
        
        self.BASE_DIR=os.path.join("configurations",self.usecase_name, self.iteration)  
        self.DATA_DIR = os.path.join(self.BASE_DIR,'raw_docs')
        self.PERSIST_DIR = os.path.join(self.BASE_DIR,'index')


        
        self.upload_type = upload_type
        self.model_name = ''

        if self.upload_type == 'url' and os.path.exists( os.path.join(DATA_DIR,'raw_docs') ):
            self.DATA_DIR= os.path.join(DATA_DIR,'raw_docs')
            logger.info(f"Data directory: {self.DATA_DIR}")

        elif self.upload_type == 'url':
            self.DATA_DIR= os.path.join(DATA_DIR,'html_files')
            logger.info(f"Data directory: {self.DATA_DIR}")

        elif self.upload_type =='doc' or self.upload_type == 'pdf':
            self.DATA_DIR= os.path.join(DATA_DIR,'raw_docs')
            logger.info(f"Data directory: {self.DATA_DIR}")
            
        self.chat_engine = None


        logger.debug(f"embed model: {self.embed_model}")
        logger.debug(f"llm model: {self.llm},  {self.model_name}")

    def documents_loader(self, DIR=None):
        """
        Placeholder for a RAG-specific document loader method.

        Parameters:
            DIR (str, optional): Directory from which documents should be loaded.
        """
        pass

    async def create_index_async(self, documents = None):
        """
        Asynchronously creates an index from the provided documents using specified embedding models and parsers.

        Parameters:
            documents (list, optional): List of documents to be indexed. If None, uses pre-loaded documents.

        Returns:
            VectorStoreIndex: The created index object containing the document vectors.
        """

        persistent_path = os.path.join(self.PERSIST_DIR, self.index_name)
        if not documents:
            documents = self.documents
        
        # Index creation logic remains unchanged


        if self.upload_type == 'url':

            html_parser = HTMLNodeParser(tags=["p","h1", "h2", "h3", "h4", "h5", "h6", "li", "b", "i", "u", "section"])  # Keeping the default tags which llama-index provides
            pdf_files = [file for file in os.listdir(self.DATA_DIR) if file.lower().endswith(".pdf")] 
            html_files = [file for file in os.listdir(self.DATA_DIR) if file.lower().endswith(".html")]

            all_nodes=[]

            if pdf_files:

                logger.info("Using  Simple Node  parser to parse parent pdf page")            
                pdf_parser = SimpleNodeParser.from_defaults()
                pdf_nodes=pdf_parser.get_nodes_from_documents(documents, show_progress=True)           
                all_nodes.extend(pdf_nodes)

            if html_files:

                logger.info("Using  HTML nodes parser to parse htmls and index")            
                html_nodes = html_parser.get_nodes_from_documents(documents, show_progress=True)
                all_nodes.extend(html_nodes)

            index = VectorStoreIndex(all_nodes, embed_model=self.embed_model, show_progress=True)
        else:
            index = VectorStoreIndex.from_documents(documents, 
                                                    embed_model=self.embed_model,
                                                    show_progress=True, #use_async=True
                                                    )

        os.makedirs(os.path.dirname(self.PERSIST_DIR), exist_ok=True)  
        index.storage_context.persist(persist_dir=persistent_path)

        return index

    async def retrieve_index_async(self, documents = None, upload_index: bool = False):
        """
        Asynchronously retrieves an existing index either from local storage or Azure blob storage, based on configuration.

        Parameters:
            documents (list, optional): List of documents to validate against the index.
            upload_index (bool): Flag to denote whether to attempt downloading the index from Azure storage.

        Returns:
            VectorStoreIndex: The retrieved index object.

        Raises:
            Exception: If unable to download index from Azure and no local index is found.
        """

        persistent_path = os.path.join(self.PERSIST_DIR, self.index_name)

        if os.path.exists(persistent_path):
            logger.debug(f"Retrieveing index from the persistent path  : {persistent_path}")
            storage_context = StorageContext.from_defaults(persist_dir=persistent_path)
            self.index = load_index_from_storage(storage_context, embed_model = self.embed_model)
        
        return self.index

    async def get_chat_engine(self):
        """
        Configures and retrieves a chat engine based on the index and provided settings.

        Returns:
            ChatEngine: The configured chat engine ready for use.

        Kwargs:
            Additional parameters that might be required for chat engine configuration.
        """
        index = await self.retrieve_index_async()
        if not index:
            raise Exception("Index is not available or unable to fetch")

        self.chat_engine = self.index.as_chat_engine(
        chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT,
        system_prompt=self.system_prompt,
        llm=self.llm,
        context_prompt=self.context_prompt
        )

        return self.chat_engine

    
    async def achat(self, query, chat_history=None, is_evaluate=False):

        """
        Asynchronously handles a single chat query, performs evaluation if required, and logs the chat response.

        Note: This is used for introductory questions

        Parameters:
            query (str): The chat query to process.
            chat_history (list, optional): History of chat to provide context.
            is_evaluate (bool, optional): Flag to perform evaluation of the chat response for relevancy and accuracy.

        Returns:
            dict: A dictionary containing the chat response, evaluation score, and additional metadata.
        """
        await self.get_chat_engine()

        logger.debug(f"Chat engine: {self.chat_engine}")   
        start_time = time.time()
        response= await self.chat_engine.achat(query, chat_history=chat_history)
        logger.debug(f"Base response: {response.response}")
        interim_time = time.time()
        try:
            page_labels=[i.metadata['page_label'] for i in response.source_nodes]
            page_labels.sort()
        except Exception as e:
            logger.info(f"Could not retrieve page labels in response source nodes {e}")
            page_labels=[]

        evaluation_score = None
        if is_evaluate:

            contexts = []
            contexts.append([c.node.get_content() for c in response.source_nodes])
            
            dataset_dict = {
                    "question": [query],
                    "answer": [response.response],
                    "contexts": contexts,
                }

            ds_chat_engine = Dataset.from_dict(dataset_dict)
            evaluator1 = Evaluator(self.documents, 
                                   None, 
                                   self.llm, 
                                   self.embed_model, 
                                   rag_name=f"{'aggrag'}_{self.name}", 
                                   project_path=f"{os.getcwd()}",
                                   model_name=self.model_name)
            
            eval_result = await evaluator1.aevaluate_models(None, None, metrics, ds_chat_engine)
            
            evaluation_score = round(eval_result.answer_relevancy.values[0],2)

        final_time = time.time()
        return {"response":response.response, 
                "page_labels":page_labels, 
                "evaluation_score":evaluation_score,
                "time_taken": get_time_taken(start_time, interim_time, final_time),
                "rag_name":  f"{self.name}"
                }
    
    async def astream_chat(self, query, chat_history= None, is_evaluate=False):

        """
        Asynchronously handles a streaming chat session, providing real-time responses to a continuous query stream.

        Note: Streaming is only available for for open questions 
        
        Parameters:
            query (str): The continuous stream query to process.
            chat_history (list, optional): Existing chat history to maintain context.
            is_evaluate (bool, optional): Whether to evaluate responses in real-time for their relevancy and accuracy.

        Returns:
            ChatResponse: The chat response from the streaming interaction.
        """
        # logger.debug(f"Chat engine: {self.chat_engine}")
        # start_time = time.time() 
        # response = await self.chat_engine.astream_chat(query, chat_history=chat_history)
        
        # return response 
        pass
