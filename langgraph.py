from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# https://langchain-ai.github.io/langgraph/tutorials/introduction/
# https://langchain-ai.github.io/langgraph/how-tos/branching/

#TODO: add child state for max db tries and retry cnts

#TODO: add state to tools 
class State(TypedDict):
    db_access = []
    table_schema = []
    query_results: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# user input/query --> tools agent (get table schemas, run queries, keep running until no error) --> evaluate (answer good/not good) --> format/synthesize (clean up results) --> end



# --------------------------
# For DB querying
# --------------------------

def tool_decider():
    pass



# --------------------------
# For DB querying
# --------------------------

def fetch_table_schema(db: Type(DeclarativeBase), table_name) -> str:
    """
    Gets table schema from database.

    Args:
        db: class that contains database info
        table_name: table name to get schema from

    Returns string of table schema
    """
    return db.get_table_schema(tablename = table_name)



top_k = 5
rdbms_type = "PostgreSQL"
# <=> is cosine DISTANCE (1 - cosine similarity); lower the distance, the better
# Note: pgvectorscale currently supports: cosine distance (<=>) queries, for indices created with vector_cosine_ops; L2 distance (<->) queries, for indices created with vector_l2_ops; and inner product (<#>) queries, for indices created with vector_ip_ops. This is the same syntax used by pgvector.
distance_search = "cosine distance, or <=>"
sql_system_prompt_1 = f"""Given an input question, create a syntactically correct {rdbms_type} query to run based on this schema:

"""
sql_system_prompt_2 = f"""To start you should ALWAYS look at the schema to see what you can query. Do NOT skip this step.

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
def text2sql(text: str, schema: str) -> str:
    """
    Converts text to SQL.

    Args:
        text: input text that needs to be converted to SQL
        db: class that contains database info
        table_name: table schema to use for reference when creating SQL

    Returns sql in string format
    """
    sys_prompt = sql_system_prompt_1 + schema + "\n" + sql_system_prompt_1




# TODO: system prompt
validate_system_prompt = """You are a SQL expert with a strong attention to detail.
    Double check the query for common mistakes, including:
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should have been used
    - Using BETWEEN for exclusive ranges
    - Data type mismatch in predicates
    - Properly quoting identifiers
    - Using the correct number of arguments for functions
    - Casting to the correct data type
    - Using the proper columns for joins

    If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

    You will call the appropriate tool to execute the query after running this check.
    """
def validate_query():
    """
    Double check to see 
    
    """



async def run_query(db: Type(DeclarativeBase), sql: str) -> list[dict]:
    """
    Runs sql and fetches results.

    Args:
        db: class that contains database info
        sql: sql code to query

    Returns list of ditionary of results
    """
    return await db.query(sql)


# --------------------------
# Review
# --------------------------

#TODO: if not good, loop back and rerun
def evaluate_results():
    """
    Evaluates sql query results and see if the results answer the user input text.
    
    """
    pass

#TODO: format result (list messages, generate response)
def format_result():
    """
    Formats sql query results and generates response.
    
    """
    pass



    

