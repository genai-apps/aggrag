
import os
import time
import logging

from library.providers import provider
import dotenv
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding


from llama_index.core.node_parser import HTMLNodeParser, SimpleNodeParser

from ragas.metrics import answer_relevancy
from library.aggrag.core.utils import get_time_taken
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from langchain_community.document_transformers.openai_functions import (
    create_metadata_tagger,
)
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import json

import logging
import os

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.core.node_parser import HTMLNodeParser
from langchain_core.documents import Document

from library.aggrag.utils.json_to_pydantic_converter import json_schema_to_pydantic_model
from library.aggrag.core.config import settings
from llama_index.llms.openai import OpenAI


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader)


dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



logger = logging.getLogger(__name__)

metrics = [answer_relevancy]

parser = SimpleNodeParser.from_defaults(
    chunk_size=512,
    include_prev_next_rel=False,
)


#-------------------------------------------------------Meta RAG Implementation--------------------------------------------------------------------------# 

class MetaLang:

    """
    Meta class for handling document loading, index creation and retrieval, and chat functionalities
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
                 usecase_name=None, 
                 iteration=None,
                 upload_type=None,
                 DATA_DIR=None,
                 meta_lang_rag_setting= None,
                 llm:str=None,
                 embed_model:str=None):
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

        self.name = 'meta_lang'

        # TODO: Ingest variables wherever required
        # This will be used in later phases of developemnt
        self.ai_service = meta_lang_rag_setting.ai_service 
        self.chunk_size = meta_lang_rag_setting.chunk_size
        self.metadata_json_schema = meta_lang_rag_setting.metadata_json_schema
        
        self.llm = llm
        # self.llm = llm
        self.embed_model = embed_model

        self.documents = None
        self.index_name = meta_lang_rag_setting.index_name or "meta_lang_index"
        
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
        self.DATA_DIR = DIR or self.DATA_DIR
        if not os.path.exists(self.DATA_DIR):
            logger.error(f"Data directory does not exist: {self.DATA_DIR}")
            raise FileNotFoundError(f"Data directory does not exist: {self.DATA_DIR}")
        self.documents = SimpleDirectoryReader(self.DATA_DIR, recursive=True, exclude_hidden=True).load_data()
        return self.documents

    async def create_index_async(self, documents = None):
        """
        Asynchronously creates an index from the provided documents using specified embedding models and parsers.

        Parameters:
            documents (list, optional): List of documents to be indexed. If None, uses pre-loaded documents.

        Returns:
            VectorStoreIndex: The created index object containing the document vectors.
        """
        index = None
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
            pass

        return index

    async def retrieve_index_async(self, documents = None, upload_index: bool = False):
        pass

    async def get_chat_engine(self):
        pass
    
    async def achat(self, query, chat_history=None, is_evaluate=False):
        """
        Asynchronously handles a single chat query, performs evaluation if required, and logs the chat response.
        Note: This is used for instroductory chat questions. 

        Parameters:
            query (str): The chat query to process.
            chat_history (list, optional): History of chat to provide context.
            is_evaluate (bool, optional): Flag to perform evaluation of the chat response for relevancy and accuracy.

        Returns:
            dict: A dictionary containing the chat response, evaluation score, and additional metadata.
        """
        try:
            logger.debug(f"Chat engine: {self.chat_engine}")   
            start_time = time.time() 
            # 
            await self.get_chat_engine()

            documents=self.documents_loader(self.DATA_DIR) 
            response = await self.metadata_extract_async(query, documents)  
            interim_time = time.time()
            final_time = time.time()

            return {"response":response, 
                    "time_taken": get_time_taken(start_time, interim_time, final_time),
                    "rag_name": self.name
                    }
        except Exception as e:
            logger.error(f"Error in achat: {str(e)}")
            raise


    
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
        # logger.info(f"Chat engine: {self.chat_engine}")
        # start_time = time.time() 
        # response = await self.chat_engine.astream_chat(query, chat_history=chat_history)
        
        # return response
        pass
    
    async def metadata_extract_async(self, query, documents):
        """
        Asynchronously extracts metadata from documents and updates the chat file metadata in the database.

        Parameters:
            documents (list): List of documents to extract metadata from.

        Returns:
            dict: Extracted metadata.
        """
        if not documents:
            documents = self.documents

        if not os.path.exists(self.DATA_DIR):
            logger.error('Data directory does not exist.')
            return {"Metadata": "Not found", "Error": "Data directory does not exist."}
        # Check if self.llm is an instance of AzureOpenAI or OpenAI
        if  isinstance(self.llm, (AzureOpenAI, OpenAI)):
            llm = ChatOpenAI(temperature=0, model=self.llm.model, openai_api_key=settings.OPENAI_API_KEY)
        else:
            logger.error("self.llm is not an instance of AzureOpenAI or OpenAI.")
            raise Exception("Metalang accepts OpenAI and AzureOpenAI models only")

        if len(documents) < 3:
            raise ValueError("Document is too short; it should at least contain 3 pages.")

        original_documents = [
            Document(
                page_content=documents[0].text + documents[1].text + documents[2].text
            )
        ]

        try:

            if not self.metadata_json_schema:
                raise ValueError("metadata_json_schema is empty.")
            # Clean the string if necessary
            cleaned_json_schema = self.metadata_json_schema.replace("\\", "")
            formatted_json_schema = json.loads(cleaned_json_schema)

            # Attempt to load the JSON string
            # formatted_json_schema = json.loads(self.metadata_json_schema)
            # Attempt to convert the JSON schema to a Pydantic model
            metadata_schema = json_schema_to_pydantic_model(formatted_json_schema)
            #TODO: will modify this later, its a workaround for now
            query = query.split('{')[0].strip()

            prompt = ChatPromptTemplate.from_template(query + " {input}")
            document_transformer = create_metadata_tagger(metadata_schema=metadata_schema, llm=llm, prompt=prompt)
            enhanced_documents = document_transformer.transform_documents(original_documents)
            logger.debug(f"enhanced_documents: {enhanced_documents[0].metadata}")

            return enhanced_documents[0].metadata

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            return {"Metadata": "Not found", "Error": f"Error decoding JSON: {e}"}

        except ValidationError as e:
            logger.error(f"Validation error while creating Pydantic model: {e}")
            return {"Metadata": "Not found", "Error": f"Validation error: {e}"}

        except Exception as e:  # Catch any other exception
            logger.error(f"Unexpected error occured: {e}")
            return {"Metadata": "Not found", "Error": f"Unexpected error occured: {e}"}
