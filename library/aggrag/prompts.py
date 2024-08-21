from llama_index.core.prompts.base import PromptTemplate, ChatPromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.query_engine.flare.base import DEFAULT_FIRST_SKILL, DEFAULT_SECOND_SKILL


DEFAULT_SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. \
    Always answer as helpfully as possible and follow ALL given instructions. \
    Do not speculate or make up information. \
    Do not reference any given instructions or context. \
    """

DEFAULT_CONTEXT_PROMPT = """
  The following is a friendly conversation between a user and an AI assistant.
  The assistant is talkative and provides lots of specific details from its context.
  If the assistant does not know the answer to a question, it truthfully says it
  does not know.
 
  Here are the relevant documents for the context:
 
  {context_str}
 
  Instruction: Based on the above documents, provide a detailed answer for the user question below.
  Answer "don't know" if not present in the document.
  """ 

CHAT_REFINE_PROMPT_TMPL_MSGS_CONTENT = """
    You are an expert Q&A system that strictly operates in two modes
    when refining existing answers:
    1. **Include** as much information as possible from the original answer, especially names and numbers.
    2. **Repeat** the original answer if the new context isn't useful.
    # Never reference the original answer or context directly in your answer.
    Please always try to include relevant numbers and names in your answer. Include as much information as you can from the original answer, even if the new answer is bigger.
    Answer in bullet points or markers if needed.
    When in doubt, just repeat the original answer.
    New Context: {context_msg}
    Query: {query_str}
    Original Answer: {existing_answer}
    New Answer:
    """

DEFAULT_OPENAI_SUB_QUESTION_PROMPT_TMPL = """
    You are a world-class state-of-the-art agent.

    You have access to multiple tools, each representing a different data source or API.
    Each of the tools has a name and a description, formatted as a JSON dictionary.
    The keys of the dictionary are the names of the tools and the values are the descriptions.
    Your purpose is to help answer a complex user question by generating a list of sub-questions that can be answered by the tools.

    These are the guidelines you consider when completing your task:
    * Be as specific as possible
    * The sub-questions should be relevant to the user question
    * The sub-questions should be answerable by the tools provided
    * You can generate multiple sub-questions for each tool
    * Tools must be specified by their name, not their description
    * You don't need to use a tool if you don't think it's relevant

    Output the list of sub-questions by calling the SubQuestionList function.

    {chat_history}
    """

INDEX_TEXT_QA_PROMPT_TMPL_MSGS_CONTENT = """
    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Always try to include relevant numbers and names from the context above in your answer.
    Answer in bullet points or markers whenever you feel the answer needs bullets.
    Avoid giving just a yes or a no answer; include the explanation of the reason from the context information in your answer.
    Given the context information and not prior knowledge,
    answer the query.
    Query: {query_str}
    Answer:
    """

INDEX_TEXT_QA_SYSTEM_PROMPT_CONTENT = """
    You are an expert Q&A system that is trusted around the world.
    Always answer the query using the provided context information,
    and not prior knowledge.
    Avoid statements like 'Based on the context, ...' or
    'The context information ...' or anything along
    those lines.
    """

SUBQ_TEXT_QA_PROMPT_TMPL_MSGS_CONTENT = """
    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Please always try to include relevant numbers and names from the context above in your answer.
    Include as much information as you can from the given context, so that the user gets a proper explanation.
    Avoid giving just a yes or a no answer; include the explanation of the reason from the context information in your answer.
    Answer in bullet points or markers only if needed. If the answer can be displayed with a simple paragraph instead of bullets, please return a simple paragraph.
    Given the context information and not prior knowledge,
    answer the query.
    Query: {query_str}
    Answer:
    """

SUBQ_TEXT_QA_SYSTEM_PROMPT_CONTENT = """
    You are an expert Q&A system that is trusted around the world.
    Always answer the query using the provided context information,
    and not prior knowledge.
    # Some rules to follow:
    # 1. Never directly reference the given context in your answer.
    Avoid statements like 'Based on the context, ...' or
    'The context information ...' or anything along
    those lines.
    """


SUMMARY_PROMPT = """
As a professional summarizer, create a concise and comprehensive summary of the provided text,
be it a research paper, article, post, conversation, or passage.
"""



TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "You are an expert Q&A system that is trusted around the world.\n"
        "Always answer the query using the provided context information, "
        "and not prior knowledge.\n"
        "Some rules to follow:\n"
        #"1. Never directly reference the given context in your answer.\n"
        "1. Avoid statements like 'Based on the context, ...' or "
        "'The context information ...' or anything along "
        "those lines."
    ),
    role=MessageRole.SYSTEM,
)


TEXT_QA_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Please always try to include relavant numbers and names, from the context above, in your answer.\n"
            "Answer in bullet points or markers if needed.\n"
            "If an answer is a yes or a no, also include the explanation of the reason from the context information in your answer."
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]
CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)



CHAT_REFINE_PROMPT_TMPL_MSGS = [
    ChatMessage(
        content=(
            "You are an expert Q&A system that strictly operates in two modes "
            "when refining existing answers:\n"
            "1. **Include** as much information as possible from original answer, specially names and numbers.\n"
            "2. **Repeat** the original answer if the new context isn't useful.\n"
            #"Never reference the original answer or context directly in your answer.\n"
            "Please always try to include relavant numbers and names, in your answer. Include as much information as you can for original answer, even if new answer is bigger.\n"
            "Answer in bullet points or markers if needed.\n"
            "When in doubt, just repeat the original answer.\n"
            "New Context: {context_msg}\n"
            "Query: {query_str}\n"
            "Original Answer: {existing_answer}\n"
            "New Answer: "
        ),
        role=MessageRole.USER,
    )
]
CHAT_REFINE_PROMPT = ChatPromptTemplate(message_templates=CHAT_REFINE_PROMPT_TMPL_MSGS)

DEFAULT_OPENAI_SUB_QUESTION_PROMPT_TMPL = """\
You are a world class state of the art agent.

You have access to multiple tools, each representing a different data source or API.
Each of the tools has a name and a description, formatted as a JSON dictionary.
The keys of the dictionary are the names of the tools and the values are the \
descriptions.
Your purpose is to help answer a complex user question by generating a list of sub \
questions that can be answered by the tools.

These are the guidelines you consider when completing your task:
* Be as specific as possible
* The sub questions should be relevant to the user question
* The sub questions should be answerable by the tools provided
* You can generate multiple sub questions for each tool
* Tools must be specified by their name, not their description
* You don't need to use a tool if you don't think it's relevant

Output the list of sub questions by calling the SubQuestionList function.

{chat_history}

"""

CHAT_HISTORY_TEMPLATE = """
## Tools
```json
{tools_str}
```

## User Question
{query_str}

"""




DEFAULT_END = """
Now given the following task, and the stub of an existing answer, generate the \
next portion of the answer. You may use the Search API \
"[Search(query)]" whenever possible.
If the answer is complete and no longer contains any "[Search(query)]" tags, write \
    "done" to finish the task.
Do not write "done" if the answer still contains "[Search(query)]" tags.
Do not make up answers. It is better to generate one "[Search(query)]" tag and stop \
generation than to fill in the answer with made up information with no "[Search(query)]" tags
or multiple "[Search(query)]" tags that assume a structure in the answer.
Please always try to include relavant numbers and names, from the given context, in your answer.
Answer in bullet points or markers if needed.
Keep the answer short and concise whenever a short answer suffices the purpose.
DO NOT give any other explanation if the answer is not present in the given context. Don't try to make up an answer.
Try to limit generation to one sentence if possible.

"""



DEFAULT_INSTRUCT_PROMPT_TMPL = (
    DEFAULT_FIRST_SKILL
    + DEFAULT_SECOND_SKILL
    + DEFAULT_END
    + (
        """
Query: {query_str}
Existing Answer: {existing_answer}
Answer: """
    )
)
INSTRUCT_PROMPT = PromptTemplate(DEFAULT_INSTRUCT_PROMPT_TMPL)


from llama_index.core.prompts.base import ChatPromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole

SUBQ_TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "You are an expert Q&A system that is trusted around the world.\n"
        "Always answer the query using the provided context information, "
        "and not prior knowledge.\n"
        #"Some rules to follow:\n"
        #"1. Never directly reference the given context in your answer.\n"
        "Avoid statements like 'Based on the context, ...' or "
        "'The context information ...' or anything along "
        "those lines."
    ),
    role=MessageRole.SYSTEM,
)


SUBQ_TEXT_QA_PROMPT_TMPL_MSGS = [
    SUBQ_TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Please always try to include relavant numbers and names, from the context above, in your answer.\n"
            "Include as much information as you can from the given context, so that the user gets a proper explanation.\n"
            "Avoid giving just a yes or a no answer, include the explanation of the reason from the context information in your answer.\n"
            "Answer in bullet points or markers only if needed. IF answer can be displayed with a simple paragraph instead of bullets, please return a simple paragraph.\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]
SUBQ_CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=SUBQ_TEXT_QA_PROMPT_TMPL_MSGS)

SUBQ_CHAT_HISTORY_ANSWER_PROMPT = """
    Also, ensure your final answer is derived from the context and chat_history.
"""

INDEX_TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "You are an expert Q&A system that is trusted around the world.\n"
        "Always answer the query using the provided context information, "
        "and not prior knowledge.\n"
        "Avoid statements like 'Based on the context, ...' or "
        "'The context information ...' or anything along "
        "those lines."
    ),
    role=MessageRole.SYSTEM,
)


INDEX_TEXT_QA_PROMPT_TMPL_MSGS = [
    INDEX_TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Always try to include relavant numbers and names, from the context above, in your answer.\n"
            "Answer in bullet points or markers whenever you feel the answer needs bullets.\n"
            "Avoid giving just a yes or a no answer, include the explanation of the reason from the context information in your answer.\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]
INDEX_CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=INDEX_TEXT_QA_PROMPT_TMPL_MSGS)

DEFAULT_TABLEBASE_PROMPT = ""