from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send
from typing import Optional
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import pipeline, BitsAndBytesConfig
import torch
from utils_async import AsyncPostgresDataBaseUser
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import BaseTool
import asyncio
from pydantic import BaseModel, Field
from langchain_community.retrievers import WikipediaRetriever
from langchain_ollama import ChatOllama
from datetime import datetime, timezone
from utils_async import create_embedding

# --------------------------
# LLM
# --------------------------
load_dotenv(override=True)

llm = ChatOllama(model=os.environ.get('LLM_MODEL'))

# --------------------------
# Tools
# --------------------------
short_db_name = os.environ.get('SHORT_TERM_DB')
short_port = os.environ.get('SHORT_TERM_HOST_PORT')
long_db_name = os.environ.get('LONG_TERM_DB')
long_port = os.environ.get('LONG_TERM_HOST_PORT')
# bot access
bot_user = os.environ.get('POSTGRESS_BOT_USER')
bot_password = os.environ.get('POSTGRES_BOT_PASSWORD')
embedding_model = os.environ.get('EMBEDDING_MODEL')
program_session = os.environ.get('PROGRAM_SESSION')

# list of dbs
db_registry = {
    "short_term_db": AsyncPostgresDataBaseUser(password=bot_password,
                    user=bot_user,
                    db_name=short_db_name,
                    port=short_port,
                    hide_parameters=True),
    "long_term_db": AsyncPostgresDataBaseUser(password=bot_password,
                    user=bot_user,
                    db_name=long_db_name,
                    port=short_port,
                    hide_parameters=True)
}

@tool 
async def create_embedding_tool(input: str) -> list:
    """
    Gets vector embedding for user input.

    Args:
        input: user input string
    """
    embedding_vec = await create_embedding(model_name=embedding_model, input=input)
    embedding_vec = await asyncio.to_thread(embedding_vec.tolist)

    return embedding_vec

# @tool
# async def get_db_schemas(db:str) -> str:
#     """
#     Gets database schemas, which will be used to create SQL query.

#     Args:
#         - db: database
#     """
#     if db not in db_registry:
#         return [{"error": f"Unknown database key '{db}'. Expected one of {list(db_registry.keys())}"}]

#     db_client = db_registry[db]

#     # get schema
#     if not db_client.schemas:
#         db_client.schemas = await db_client.get_all_schemas()

#     return db_client.schemas

# @tool
# async def create_sql():
#     if db not in db_registry:
#         return [{"error": f"Unknown database key '{db}'. Expected one of {list(db_registry.keys())}"}]

#        # get schema
#     if not db_client.schemas:
#         db_client.schemas = await db_client.get_all_schemas()

#     top_k = 5
#     rdbms_type = "PostgreSQL"
#     # <=> is cosine DISTANCE (1 - cosine similarity); lower the distance, the better
#     # Note: pgvectorscale currently supports: cosine distance (<=>) queries, for indices created with vector_cosine_ops; L2 distance (<->) queries, for indices created with vector_l2_ops; and inner product (<#>) queries, for indices created with vector_ip_ops. This is the same syntax used by pgvector.
#     distance_search = "cosine distance, or <=>"
#     query_system_prompt = f"""
#         Given an input question, create a syntactically correct {rdbms_type} query to run based on this schema:

#         {db_client.schemas}

#         To start you should ALWAYS look at the schema to see what you can query. Do NOT skip this step.

#         It's a vector database, so make sure to use {distance_search}.
#         Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.

#         Never query for all the columns from a specific table, only ask for the relevant columns given the question.

#         DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

#         ex. 'What did user1 say about AI agents 2 days ago?'
#             WITH relaxed_results AS MATERIALIZED (
#                 SELECT 
#                     timestamp,
#                     speaker,
#                     text,
#                     embedding <=> '[...]' AS distance
#                 FROM transcriptionsvectors
#                 WHERE speaker = 'user1' AND timestamp = timezone('UTC', now()) - INTERVAL '2 days'
#                 ORDER BY distance
#                 LIMIT 5)
            
#             SELECT * 
#             FROM relaxed_results 
#             ORDER BY distance;
#     """

#     sql = llm.ainvoke([SystemMessage(content=query_system_prompt),
#                        HumanMessage(content=input + f"\n Embedding: {embedding}")])
#     print(sql)

#     # run query
#     try:
#         return await db_client.query(query=sql)
#     except Exception as e:
#         print("Query error: ", {e})
#         return []


import re
def remove_think_tags(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

@tool
async def query_db(db: str, input: str, embedding: list) -> list:
    """
    Runs SQL query to retrieve relevant results based on user input and embedding.
    
    Args:
        db: database name
        input: input string to use for SQL query
        embedding: input embedding for vector database
    """

    if db not in db_registry:
        return [{"error": f"Unknown database key '{db}'. Expected one of {list(db_registry.keys())}"}]

    db_client = db_registry[db]

    # get schema
    if not db_client.schemas:
        db_client.schemas = await db_client.get_all_schemas()

    top_k = 5
    rdbms_type = "PostgreSQL"
    # <=> is cosine DISTANCE (1 - cosine similarity); lower the distance, the better
    # Note: pgvectorscale currently supports: cosine distance (<=>) queries, for indices created with vector_cosine_ops; L2 distance (<->) queries, for indices created with vector_l2_ops; and inner product (<#>) queries, for indices created with vector_ip_ops. This is the same syntax used by pgvector.
    distance_search = "cosine distance, or <=>"
    query_system_prompt = f"""
        You are a helpful assistant that can output SQL statement.
        Given an input question, create a syntactically correct {rdbms_type} query to run based on this schema:

        {db_client.schemas}

        To start, you should ALWAYS look at the schema to see what you can query. Do NOT skip this step.

        It's a vector database, so make sure to use {distance_search}.
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.

        Never query for all the columns from a specific table, only ask for the relevant columns given the question.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

        ex. 'What did user1 say about AI agents 2 days ago?'
            WITH relaxed_results AS MATERIALIZED (
                SELECT 
                    timestamp,
                    speaker,
                    text,
                    embedding <=> '[...]' AS distance
                FROM transcriptionsvectors
                WHERE speaker = 'user1' AND timestamp = timezone('UTC', now()) - INTERVAL '2 days'
                ORDER BY distance
                LIMIT 5)
            
            SELECT * 
            FROM relaxed_results 
            ORDER BY distance;

        Output ONLY the sql statement.
    """

    sql = await llm.ainvoke([SystemMessage(content=query_system_prompt),
                       HumanMessage(content=input + f"\n Embedding: {embedding}")])
    print(remove_think_tags(sql))

    # run query
    try:
        return await db_client.query(query=sql)
    except Exception as e:
        print("Query error: ", {e})
        return []
    
class QueryDB(BaseTool):
    name: str = "query_db"
    description: str = "Run semantic SQL queries against Postgres vector database."

    db: AsyncPostgresDataBaseUser

    def _run(self, query: str, embedding: list) -> list:
        """
            Run asynchronously.

            Args:
                query: user input
                embedding: embedding of user input
        """
        return asyncio.run(self._arun(query, embedding))
    
    async def _arun(self, query: str, embedding: list) -> list:
        """
            Run asynchronously.

            Args:
                query: user input
                embedding: embedding of user input
        """
        return await query_db(db=self.db, input=query, embedding=embedding)




# wiki
@tool
def wiki_search(query: str) -> list:
    """
    Searches Wikipedia.

    Args:
        query: Search query
    """
    retriever = WikipediaRetriever(top_k_results=3, load_all_available_meta=False)
    results = retriever.invoke(query)

    return results

# Duck Duck Go search
@tool
def web_search(query: str) -> list:
    """
    Searches web.

    Args:
        query: Search query
    
    """
    search = DuckDuckGoSearchRun()
    
    return search.invoke(query)

# current time
@tool
def get_current_time(utc:bool = False) -> datetime:
    """
    Get current time, local or UTC
    
    Args:
        utc: Should datetime be in UTC or not
    """
    if utc:
        current_datetime = datetime.now(timezone.utc)
    else:
        current_datetime = datetime.now()

    return current_datetime

tools = [create_embedding_tool, query_db, web_search, wiki_search, get_current_time]
# tools = [web_search, wiki_search]

# --------------------------
# Graph
# --------------------------
llm_with_tools = llm.bind_tools(tools)
# llm_with_tools_strctured = llm.with_structured_output(ResponseFormatter, method="json_schema")


class State(TypedDict):
    messages: Annotated[list, add_messages]


agent_system_msg = f"""
    You are a helpful, conservational assistant who can respond to user by reasoning through your thoughts and actions using various tools. Use however many tools you need. Note that if the user input doesn't require any tools DO NOT use any tools. Also, unless explicitly specified, don't include your thoughts in the output.

    You have these databases available: {db_registry.keys()}.
    You don't need to use database if you think it's not necessary.

    Finally, you MUST format the response in markdown for Discord.
    Use stuff like these for formatting:
        Heading: # H1
        Bold: **bold text**
        Blockquote: > blockquote
        <br>: This is the first line <br> And this is another line
"""

async def assistant(state: State):
    """
    Agent node
    """

    messages = await llm_with_tools.ainvoke([SystemMessage(agent_system_msg)] + state["messages"])
    print(f"AI: {messages.content}")

    return {
        "messages": messages
    }


#TODO
formatter_system_msg = """
    You are a helpful agent that formats the message into markdown style. 
    Examples:
        Heading: # H1
        Bold: **bold text**
        Blockquote: > blockquote
        <br>: This is the first line <br> And this is another line
"""
# formatter
# async def markdown_formatter(state: State):
#     """
#     Formats the message into markdown.
#     """

#     messages = await llm.ainvoke([SystemMessage(formatter_system_msg)] + state["messages"])
#     print(f"Formatter: {messages[-1].content}")

#     return {
#         "messages": messages
#     }

#TODO: create formatter node and create conditional node
# def should_continue(state: State):
#     """
#     Determine if tool calling should end or not
#     """
#     if 

builder = StateGraph(State)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
# builder.add_node("markdown_formatter", markdown_formatter)

builder.add_edge(START, "assistant")
#TODO
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")


# short-term in-memory
checkpointer = InMemorySaver()

app = builder.compile(checkpointer=checkpointer)



# available_dbs = {
#     "bot_short": "Short-term memory database where only today's messages are saved.",
#     "bot_long": "Long-term database memory where all past messges are saved."
# }


# from utils_async import (
#     AsyncPostgresDataBaseUser,
#     get_detailed_instruct, 
#     create_embedding,
#     )
# # import asyncio

# # load_dotenv(override=True)
# # pg_username = os.environ.get('POSTGRESS_USER')
# # pg_password = os.environ.get('POSTGRES_PASSWORD')
# # short_db_name = os.environ.get('SHORT_TERM_DB')
# # short_port = os.environ.get('SHORT_TERM_HOST_PORT')
# # long_db_name = os.environ.get('LONG_TERM_DB')
# # long_port = os.environ.get('LONG_TERM_HOST_PORT')
# # # bot access
# # bot_user = os.environ.get('POSTGRESS_BOT_USER')
# # bot_password = os.environ.get('POSTGRES_BOT_PASSWORD')
# # embedding_model = os.environ.get('EMBEDDING_MODEL')
# # program_session = os.environ.get('PROGRAM_SESSION')

# # #! start docker first
# # bot
# bot_short = AsyncPostgresDataBaseUser(password=bot_password,
#                     user=bot_user,
#                     db_name=short_db_name,
#                     port=short_port,
#                     hide_parameters=True)
# bot_long = AsyncPostgresDataBaseUser(password=bot_password,
#                     user=bot_user,
#                     db_name=long_db_name,
#                     port=short_port,
#                     hide_parameters=True)

# available_dbs = {
#     "short_term_db": "For messages that was saved today only.",
#     "long_term_db": "For messages that was saved in the past."
# }
# system_prompt = f"""You are a helpful assistant that can use tools to respond to user. Use however many tools needed to respond to user's input.
# """

# # text = "What is the most popular color?"
# task = "Given user's message query, retrieve relevant messages that answer the query."


# async def main():
#     while True:
#         text = input("User: ")

#         instruct_query = get_detailed_instruct(query=text,
#                                             task_description=task)

#         instruct_embedding = await create_embedding(model_name=embedding_model, input=instruct_query)

#         initial_state = {
#                 "embedding": instruct_embedding.tolist(),
#                 # "available_dbs": {
#                 #     "short_term_db": bot_short,
#                 #     "long_term_db": bot_long
#                 # },
#                 "messages": [SystemMessage(content=system_prompt),
#                             HumanMessage(content=text)]
#         }

#         config = {"configurable": {"thread_id": program_session}}
#         agent_task = await app.ainvoke(initial_state, config)
#         print(agent_task["messages"])

# asyncio.run(main())