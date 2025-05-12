from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send
from typing import Optional 
import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline, BitsAndBytesConfig
import torch
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from utils_async import AsyncPostgresDataBaseUser
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLDatabaseTool,
)
from langgraph.prebuilt import create_react_agent

load_dotenv(override=True)

#* Note: prebuilt

long_db_name = os.environ.get('LONG_TERM_DB')
long_port = os.environ.get('LONG_TERM_HOST_PORT')
short_db_name = os.environ.get('SHORT_TERM_DB')
short_port = os.environ.get('SHORT_TERM_HOST_PORT')
bot_user = os.environ.get('POSTGRESS_BOT_USER')
bot_password = os.environ.get('POSTGRES_BOT_PASSWORD')


# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
# )

llm_model = pipeline("text-generation", 
                     model=os.environ.get('LLM_MODEL'),
                     token=os.environ.get('HF_TOKEN'),
                    #  model_kwargs={
                    #     "quantization_config": quant_config
                    #  }
                    ) 
hf_llm = HuggingFacePipeline(pipeline=llm_model)
llm = ChatHuggingFace(llm=hf_llm)

#! Program gets stuck...
db_short = SQLDatabase.from_uri(f'postgresql+psycopg://{bot_user}:{bot_password}@localhost:{short_port}/{short_db_name}')
db_long = SQLDatabase.from_uri(f'postgresql+psycopg://{bot_user}:{bot_password}@localhost:{long_port}/{long_db_name}')


toolkit_short = SQLDatabaseToolkit(db=db_short, llm=llm)
toolkit_long = SQLDatabaseToolkit(db=db_long, llm=llm)

def make_sql_tools(db: SQLDatabase, prefix: str):
    return [
        InfoSQLDatabaseTool(db=db, name=f"{prefix}_sql_db_schema"),
        ListSQLDatabaseTool(db=db, name=f"{prefix}_sql_db_list_tables"),
        QuerySQLDatabaseTool(db=db, name=f"{prefix}_sql_db_query"),
    ]

print("tools")

tools_short = make_sql_tools(db_short, "short")
tools_long  = make_sql_tools(db_long, "long")

rdbms_type = "PostGreSQL"
distance_search = "cosine distance, or <=>"
top_k = 5
prompt = f"""Given an input question, create a syntactically correct {rdbms_type} query to run based on this schema:
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

agent = create_react_agent(
    llm,
    tools_short + tools_long,
    prompt=prompt,
)
