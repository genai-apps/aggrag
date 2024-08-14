import logging
import os
import time


from typing import Optional


import nest_asyncio
from datasets import Dataset
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.packs.raptor import RaptorPack
from llama_index.packs.raptor import RaptorRetriever
from llama_index.packs.raptor.base import SummaryModule
from ragas.metrics import (
    answer_relevancy
)

from library.aggrag.core.config import settings, AzureOpenAIModelNames, AzureOpenAIModelEngines

metrics = [
        answer_relevancy,
        ]

from library.aggrag.evals.llm_evaluator import Evaluator
from library.aggrag.core.utils import get_time_taken

logger = logging.getLogger(__name__)


class Raptor:
    def __init__(self,
                 usecase_name: str, 
                 iteration: str,
                 DATA_DIR: Optional[str] = None,
                 upload_type: Optional[str] = None,
                 raptor_rag_setting= None,
                 llm:str = None,
                 embed_model:str = None):
        
        self.name = 'Raptor'


        # TODO: Ingest variables wherever required
        # This will be used in later phases of developemnt
        self.ai_service = raptor_rag_setting.ai_service 
        self.chunk_size = raptor_rag_setting.chunk_size

        self.llm = llm
        self.llm_summary = llm
        self.summary_module = SummaryModule(summary_prompt=raptor_rag_setting.summary_prompt, llm=self.llm_summary)

        self.embed_model = embed_model or AzureOpenAIEmbedding(
            model=raptor_rag_setting.embed_model,
            deployment_name=raptor_rag_setting.embed_deployment,
            api_key=settings.AZURE_OPENAI_KEY,
            azure_endpoint=settings.AZURE_API_BASE,
            api_version=settings.OPENAI_API_VERSION,
        )

        self.documents = None
        self.index_name = raptor_rag_setting.index_name
        self.index = None

        from library.aggrag.core.schema import UserConfig

        self.usecase_name=usecase_name or UserConfig.usecase_name
        self.iteration=iteration or UserConfig.iteration
        
        self.BASE_DIR=os.path.join("configurations",self.usecase_name, self.iteration)  

        self.DATA_DIR = os.path.join(self.BASE_DIR,'raw_docs')
        self.PERSIST_DIR = os.path.join(self.BASE_DIR,'index')
        self.upload_type = upload_type 
          
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
        self.raptor_pack = None

        logger.debug(f"azure embed: {self.embed_model}")
        logger.debug(f"azure llm: {self.llm}")

    def documents_loader(self, DIR=None):
        """ This is the rag specific documents loader"""
        pass

    async def create_index_async(self, documents = None):
        nest_asyncio.apply()
        try: 
            persistent_path = os.path.join(self.PERSIST_DIR, self.index_name)
            if not documents:
                documents = self.documents
            
            self.raptor_pack = RaptorPack(
                documents,
                embed_model=self.embed_model,  # used for embedding clusters
                summary_module=self.summary_module,# used for generating summaries
                # vector_store=vector_store,  # used for storage
                similarity_top_k=4,  # top k for each layer, or overall top-k for collapsed
                mode="collapsed",  # sets default mode
                transformations=[
                    SentenceSplitter(chunk_size=400, chunk_overlap=50)
                ],  # transformations applied for ingestion
                )

            os.makedirs(os.path.dirname(self.PERSIST_DIR), exist_ok=True)
            self.raptor_pack.retriever.persist(persistent_path)

            return True
        except Exception as e:
            logger.error(f"Error while creating Raptor index: {e}")
            return False
    
    async def retrieve_index_async(self):
        # - first check if index exists locally otherwise retrieve from blob storage
        logger.debug(f"Retrieving RAPTOR index")
        persistent_path = os.path.join(self.PERSIST_DIR, self.index_name)
        # if not documents:
        #     documents = self.documents
        
        if os.path.exists(persistent_path):
            pass

        
        retriever = RaptorRetriever.from_persist_dir(persistent_path, 
                                                     embed_model=self.embed_model,
                                                     llm=self.llm,
                                                     similarity_top_k=4,
                                                     summary_module = self.summary_module,
                                                     mode="tree_traversal", # collapsed might be faster
                                                     verbose=True)
        self.index = retriever
        logger.debug(f"RAPTOR index is: {self.index}")

        return self.index
    
    async def get_chat_engine(self):
        logger.debug("Retrieving RAPTOR chat engine")
        if not self.index:
            await self.retrieve_index_async()
        
        if self.index:
            query_engine = RetrieverQueryEngine.from_args(
            self.index, llm=self.llm, use_async=False, streaming=False, verbose=True
            )
            
            self.chat_engine = query_engine
        else:
            logger.debug(f"Raptor index not found yet. Not setting chat engine.")
            self.chat_engine = None
        
        logger.debug(f"RAPTOR chat engine is: {self.chat_engine}")
        return self.chat_engine
    
    async def achat(self, query, chat_history=None, is_evaluate=False):
             
        logger.debug(f"RAPTOR Chat engine: {self.chat_engine}")
        start_time = time.time()       
        response= await self.chat_engine.aquery(query)
        logger.debug(f"RAPTOR response: {response.response}")
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
                                   project_path=f"{os.getcwd()}")
            
            eval_result = await evaluator1.aevaluate_models(None, None, metrics, ds_chat_engine)

            evaluation_score = round(eval_result.answer_relevancy.values[0],2)
        final_time = time.time()
        return {"response":response.response, 
                "page_labels":page_labels, 
                "evaluation_score":evaluation_score,
                "time_taken": get_time_taken(start_time, interim_time, final_time),
                "rag_name": self.name
                }

