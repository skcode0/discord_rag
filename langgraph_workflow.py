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


# --------------------------
# LLM
# --------------------------
load_dotenv(override=True)

# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
# )

llm_model = pipeline("text-generation", 
                     model=os.environ.get('LLM_MODEL'),
                     token=os.environ.get('HF_TOKEN'),
                     max_new_tokens=5000,
                    #  model_kwargs={
                    #     "quantization_config": quant_config
                    #  }
                    ) 
hf_llm = HuggingFacePipeline(pipeline=llm_model)
llm = ChatHuggingFace(llm=hf_llm)

# --------------------------
# Tools
# --------------------------
async def query_db(db: type[AsyncPostgresDataBaseUser], input: str, embedding: list) -> list:
    """
    Runs SQL query to retrieve relevant results based on user input and embedding.
    
    Args:
        db: database class instance
        input: input string to use for SQL query
        embedding: input embedding for vector database
    
    Returns query results
    """
    # get schema
    if not db.schemas:
        db.schemas = await db.get_all_schemas()

    top_k = 5
    rdbms_type = "PostgreSQL"
    # <=> is cosine DISTANCE (1 - cosine similarity); lower the distance, the better
    # Note: pgvectorscale currently supports: cosine distance (<=>) queries, for indices created with vector_cosine_ops; L2 distance (<->) queries, for indices created with vector_l2_ops; and inner product (<#>) queries, for indices created with vector_ip_ops. This is the same syntax used by pgvector.
    distance_search = "cosine distance, or <=>"
    query_system_prompt = f"""
        Given an input question, create a syntactically correct {rdbms_type} query to run based on this schema:"

        {db.schemas}

        To start you should ALWAYS look at the schema to see what you can query. Do NOT skip this step.

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
    """

    sql = llm.ainvoke([SystemMessage(content=query_system_prompt),
                       HumanMessage(content=input + f"\n Embedding: {embedding}")])

    # run query
    try:
        return await db.query(query=sql)
    except Exception as e:
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


short_db_name = os.environ.get('SHORT_TERM_DB')
short_port = os.environ.get('SHORT_TERM_HOST_PORT')
long_db_name = os.environ.get('LONG_TERM_DB')
long_port = os.environ.get('LONG_TERM_HOST_PORT')
# bot access
bot_user = os.environ.get('POSTGRESS_BOT_USER')
bot_password = os.environ.get('POSTGRES_BOT_PASSWORD')
embedding_model = os.environ.get('EMBEDDING_MODEL')
program_session = os.environ.get('PROGRAM_SESSION')

# bot
bot_short = AsyncPostgresDataBaseUser(password=bot_password,
                    user=bot_user,
                    db_name=short_db_name,
                    port=short_port,
                    hide_parameters=True)
bot_long = AsyncPostgresDataBaseUser(password=bot_password,
                    user=bot_user,
                    db_name=long_db_name,
                    port=short_port,
                    hide_parameters=True)

short_db_tool = QueryDB(db=bot_short)
short_db_tool.name = "short_term_memory_db"
short_db_tool.description = "Run semantic SQL queries against Postgres vector database that only stores today's discord messages."

long_db_tool = QueryDB(db=bot_long)
long_db_tool.name = "long_term_memory_db"
long_db_tool.description = "Run semantic SQL queries against Postgres vector database that only stores past discord messages."

search_tool = DuckDuckGoSearchRun()
tools = [short_db_tool, long_db_tool, search_tool]

# --------------------------
# Graph
# --------------------------
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    embedding: list
    available_dbs: dict
    messages: Annotated[list, add_messages]


async def assistant(state: State):
    messages = await llm_with_tools.ainvoke(state["messages"])
    return {
        "messages": messages
    }

builder = StateGraph(State)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
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
# import asyncio

# load_dotenv(override=True)
# pg_username = os.environ.get('POSTGRESS_USER')
# pg_password = os.environ.get('POSTGRES_PASSWORD')
# short_db_name = os.environ.get('SHORT_TERM_DB')
# short_port = os.environ.get('SHORT_TERM_HOST_PORT')
# long_db_name = os.environ.get('LONG_TERM_DB')
# long_port = os.environ.get('LONG_TERM_HOST_PORT')
# # bot access
# bot_user = os.environ.get('POSTGRESS_BOT_USER')
# bot_password = os.environ.get('POSTGRES_BOT_PASSWORD')
# embedding_model = os.environ.get('EMBEDDING_MODEL')
# program_session = os.environ.get('PROGRAM_SESSION')

# #! start docker first
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

# text = "What is the most popular color?"
# task = "Given user's message query, retrieve relevant messages that answer the query."
# instruct_query = get_detailed_instruct(query=text,
#                                             task_description=task)

# async def main():
#     instruct_embedding = await create_embedding(model_name=embedding_model, input=instruct_query)

#     initial_state = {
#             "embedding": instruct_embedding.tolist(),
#             # "available_dbs": {
#             #     "short_term_db": bot_short,
#             #     "long_term_db": bot_long
#             # },
#             "messages": [SystemMessage(content=system_prompt),
#                         HumanMessage(content=text)]
#     }

#     config = {"configurable": {"thread_id": program_session}}
#     agent_task = await app.ainvoke(initial_state, config)
#     print(agent_task["messages"])

# asyncio.run(main())