import logging
import os
import time

from typing import Optional
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts.base import ChatPromptTemplate

from datasets import Dataset
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import HTMLNodeParser
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.question_gen.openai import OpenAIQuestionGenerator
from ragas.metrics import (
    answer_relevancy
)

from library.aggrag.core.utils import get_time_taken
from library.aggrag.core.config import settings

from library.aggrag.evals.llm_evaluator import Evaluator

metrics = [
        answer_relevancy,
        ]

from library.aggrag.prompts import (CHAT_REFINE_PROMPT,
                                       CHAT_HISTORY_TEMPLATE,
                                       INDEX_CHAT_TEXT_QA_PROMPT,
                                       SUBQ_CHAT_TEXT_QA_PROMPT,
                                       DEFAULT_OPENAI_SUB_QUESTION_PROMPT_TMPL)

logger = logging.getLogger(__name__)

class SubQA:
    """
        SubQA class manages the specialized sub-questioning and indexing capabilities for RAG
        system.

        Attributes:
            name (str): Name of the RAG configuration, here 'SubQA'.
            llm (AzureOpenAI): Configured Azure Language Model for generating responses.
            embed_model (AzureOpenAIEmbedding): Configured embedding model for document indexing.
            documents (list, optional): List of loaded documents for processing.
            index_name (str): Name of the index for document vectors, shared with the base RAG configuration.
            index (VectorStoreIndex, optional): Index object containing document vectors.
            usecase_name (str): User identifier for session-specific data handling.
            iteration (str): Session identifier.
            DATA_DIR (str): Base directory for storing data.
            PERSIST_DIR (str): Directory for persisting index data.
            CHAT_DIR (str): Directory for storing chat data.
            upload_type (str): Type of documents being uploaded ('url', 'doc', 'pdf').
            chat_engine (object, optional): Chat engine configured for handling chat interactions.
        """

    def __init__(self, 
                 usecase_name: str, 
                 iteration: str,
                 DATA_DIR: Optional[str] = None,
                 upload_type: Optional[str] = None,
                 subqa_rag_setting= None,
                 llm:str = None,
                 embed_model:str = None):
        
        """
        Initializes the SubQA RAG system with specified models and configurations for specialized sub-question handling.

        Parameters:
            llm_model (str, optional): The name of the language model to use.
            llm_deployment (str, optional): Deployment name for the language model.
            embed_model (str, optional): The name of the embedding model to use.
            embed_deployment (str, optional): Deployment name for the embedding model.
            usecase_name (str, optional): Unique identifier for the user.
            iteration (str, optional): Unique identifier for the session.
            upload_type (str, optional): Type of documents being uploaded ('url', 'doc', 'pdf').
            DATA_DIR (str, optional): Directory where data will be stored and processed.
        """

        self.name = 'SubQA'
        
        self.llm = llm

        self.embed_model = embed_model
        self.documents = None
        self.index_name =  subqa_rag_setting.index_name  # reusing the same index from Base RAG
        self.index = None
        from library.aggrag.core.schema import UserConfig

        self.subqa_rag_setting=subqa_rag_setting
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


        self.DEFAULT_OPENAI_SUB_QUESTION_PROMPT_TMPL = subqa_rag_setting.DEFAULT_OPENAI_SUB_QUESTION_PROMPT_TMPL


        SUBQ_TEXT_QA_SYSTEM_PROMPT = ChatMessage(
            content=(
            subqa_rag_setting.SUBQ_TEXT_QA_SYSTEM_PROMPT_CONTENT),
            role=MessageRole.SYSTEM,
        )

        SUBQ_TEXT_QA_PROMPT_TMPL_MSGS = [
            SUBQ_TEXT_QA_SYSTEM_PROMPT,
            ChatMessage(
                content=(  
                        subqa_rag_setting.SUBQ_TEXT_QA_PROMPT_TMPL_MSGS_CONTENT        
                ),
                role=MessageRole.USER,
            ),
        ]
        self.SUBQ_CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=SUBQ_TEXT_QA_PROMPT_TMPL_MSGS)



        CHAT_REFINE_PROMPT_TMPL_MSGS = [
            ChatMessage(
                content=(
                    subqa_rag_setting.CHAT_REFINE_PROMPT_TMPL_MSGS_CONTENT
                ),
                role=MessageRole.USER,
            )
        ]
        self.CHAT_REFINE_PROMPT = ChatPromptTemplate(message_templates=CHAT_REFINE_PROMPT_TMPL_MSGS)


        INDEX_TEXT_QA_SYSTEM_PROMPT = ChatMessage(
            content=(
                    subqa_rag_setting.INDEX_TEXT_QA_SYSTEM_PROMPT_CONTENT
                    ),
            role=MessageRole.SYSTEM,
        )
        INDEX_TEXT_QA_PROMPT_TMPL_MSGS = [
            INDEX_TEXT_QA_SYSTEM_PROMPT,
            ChatMessage(
                content=(
                    subqa_rag_setting.INDEX_TEXT_QA_PROMPT_TMPL_MSGS_CONTENT
                ),
                role=MessageRole.USER,
            ),
        ]
        self.INDEX_CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=INDEX_TEXT_QA_PROMPT_TMPL_MSGS)


        logger.debug(f"azure embed: {self.embed_model}")
        logger.debug(f"azure llm: {self.llm}")


    def documents_loader(self, DIR=None):
        """
        Placeholder for a RAG-specific document loader method tailored for SubQA operations.

        Parameters:
            DIR (str, optional): Directory from which documents should be loaded.
        """
        pass

    async def create_index_async(self, documents = None):

        """
        Asynchronously creates an index from the provided documents using HTML parsers and the specified embedding models.

        Parameters:
            documents (list, optional): List of documents to be indexed. If not provided, uses pre-loaded documents.

        Returns:
            VectorStoreIndex: The created index object containing the document vectors.
        """
        
        persistent_path = os.path.join(self.PERSIST_DIR, self.index_name)
        if not documents:
            documents = self.documents
        
        
        html_files = [file for file in os.listdir(self.DATA_DIR) if file.endswith(".html")]
        if html_files:
            
            logger.info("Using  HTML nodes parser to parse htmls and index")   
            # Index creation logic remains unchanged
            parser = HTMLNodeParser(tags=["p","h1", "h2", "h3", "h4", "h5", "h6", "li", "b", "i", "u", "section"])  # Keeping the default tags which llama-index provides         
            nodes = parser.get_nodes_from_documents(documents)
            index = VectorStoreIndex(nodes, embed_model=self.embed_model, show_progress=True)
        else:
            index = VectorStoreIndex.from_documents(documents, 
                                                    embed_model=self.embed_model,
                                                    show_progress=True, #use_async=True
                                                    )

        os.makedirs(os.path.dirname(self.PERSIST_DIR), exist_ok=True)  
        index.storage_context.persist(persist_dir=persistent_path)

        return index
        

    async def retrieve_index_async(self):
        """
        Asynchronously retrieves an existing index either from local storage or Azure blob storage, based on configuration.

        Returns:
            VectorStoreIndex: The retrieved index object.

        Raises:
            Exception: If unable to download index from Azure and no local index is found.
        """
        persistent_path = os.path.join(self.PERSIST_DIR, self.index_name)


        if os.path.exists(persistent_path):
            logger.debug(f"************************ persistent path is : {persistent_path}")
            storage_context = StorageContext.from_defaults(persist_dir=persistent_path)
            self.index = load_index_from_storage(storage_context, embed_model = self.embed_model)
             
        return self.index

    async def get_chat_engine(self, chat_history="", streaming=False):

        """
        Configures and retrieves a chat engine based on the SubQA's specialized needs, using various query engines and response synthesizers.

        Parameters:
            chat_history (str, optional): Serialized string of chat history to maintain context.
            streaming (bool, optional): Specifies whether the chat should operate in streaming mode for real-time interaction.

        Returns:
            object: Configured chat engine ready for use.
        """
        await self.retrieve_index_async()

        use_async_bool = True
        if streaming:
            use_async_bool=False

        index_response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            use_async=use_async_bool,
            streaming=streaming,
            text_qa_template=INDEX_CHAT_TEXT_QA_PROMPT,
            refine_template=CHAT_REFINE_PROMPT
        )

        query_engine_tools = [
            QueryEngineTool(
                query_engine=self.index.as_query_engine(llm=self.llm, 
                                                        similarity_top_k=3, 
                                                        streaming=streaming, 
                                                        response_synthesizer=index_response_synthesizer),
                metadata=ToolMetadata(
                    name="research paper",
                    description="",
                ),
            ),
        ]

        chat_history_string=" "
        questions=[]
        answers=[]
        context_string = "The existing converation is:\n"
        
        if chat_history:
            for m, msg in enumerate(chat_history):
                if m == 0:
                    continue
                if msg.role == "user":
                    questions.append(msg.content)
                if msg.role =="system":
                    answers.append(msg.content)
            chat_string=""
            for (question,answer) in zip(questions,answers):
                chat_string += f"User: {question} \n Assistant: {answer} \n\n"
            chat_history_string = context_string + chat_string

            # Uncomment this to include chat history in response synthesizer. Promising results were not observed.
            # SUBQ_CHAT_TEXT_QA_PROMPT.message_templates[1].content += "\n " + SUBQ_CHAT_HISTORY_ANSWER_PROMPT + "\n " + chat_history_string

        # logger.info(f"response synth prompt: {SUBQ_CHAT_TEXT_QA_PROMPT}")
        CHAT_HISTORY_PROMPT = self.subqa_rag_setting.DEFAULT_OPENAI_SUB_QUESTION_PROMPT_TMPL.format(chat_history=chat_history_string if chat_history_string else chat_history) + CHAT_HISTORY_TEMPLATE

        # logger.info(f"Final Chat history prompt: {CHAT_HISTORY_PROMPT}")

        question_gen = OpenAIQuestionGenerator.from_defaults(llm=self.llm, prompt_template_str=CHAT_HISTORY_PROMPT )


        response_synthesizer = get_response_synthesizer(
            llm=self.llm,

            use_async=use_async_bool,
            streaming=streaming,
            text_qa_template=SUBQ_CHAT_TEXT_QA_PROMPT
            # refine_template=CHAT_REFINE_PROMPT
        )


        self.chat_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=query_engine_tools,
            question_gen=question_gen,
            llm=self.llm,
            use_async=use_async_bool,
            response_synthesizer=response_synthesizer
        )
        logger.info(f"SubQA chat engine is: {self.chat_engine}")
        
        return self.chat_engine

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
        await self.get_chat_engine()
        
        logger.info(f"Chat engine: {self.chat_engine}")   
        start_time = time.time()   
        response= await self.chat_engine.aquery(query)
        logger.debug(f"SubQA response: {response.response}")
        interim_time = time.time()

        try:
            page_labels=[i.get('page_label') for i in response.metadata.values() if i.get('page_label')]
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

            # dataset_dict
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

    async def astream_chat(self, query, chat_history= None):

        """
        Asynchronously handles a streaming chat session, providing real-time responses to a continuous query stream.

        Parameters:
            query (str): The continuous stream query to process.
            chat_history (list, optional): Existing chat history to maintain context.
            

        Returns:
            ChatResponse: The chat response from the streaming interaction.
        """
        # await self.get_chat_engine(chat_history=chat_history, streaming=True)
        # logger.debug(f"Chat engine: {self.chat_engine}")
        
        # response = self.chat_engine.query(query)
        
        # return response
        pass
