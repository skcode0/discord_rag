from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# https://langchain-ai.github.io/langgraph/tutorials/introduction/
# https://langchain-ai.github.io/langgraph/how-tos/branching/

llm = ""

#TODO:
def merge_dict(old: dict, new: dict) -> dict:
    """
    Merge old and new dict

    - old: old dict
    - new: new dict

    Returns merged dict
    """
    return {**old, **new}


#TODO: add state to tools 
class State(TypedDict):
    user_input: str
    embedding: list
    tools_needed: set
    dbs: Annotated[dict, merge_dict]
    query_results: Annotated[list, add_messages]

# user input/query --> tools agent (get table schemas, run queries, keep running until no error) --> evaluate (answer good/not good) --> format/synthesize (clean up results) --> end



# --------------------------
# For DB querying
# --------------------------
#! Change tool list as needed
available_tools = {"short_term_db": "Database that stores anything said on the same day as today.", 
                   "long_term_db": "Database that stores anything said in the past."}
def tool_decider(state: State):
    f"""
    Decides which tool(s) to use based on user input.

    """
    system_prompt = {
        "role": "system",
        "content": f"""Choose the best tool(s) for asnwer user input.
        Available tools are:
        {available_tools}
        "Do not response with anything else other than the tools needed. List them as a python set."""
    }
    user_prompt = {
        "role": "user",
        "content": state.user_input
    }

    response = llm.invoke([system_prompt, user_prompt])
    
    return {"tools_needed": set(response)}


# --------------------------
# For DB querying
# --------------------------

def fetch_schema(state: State) -> Optional[dict]:
    """
    Gets schema from database.

    Returns string of table schema
    """
    for tool in state.tools_needed:
        if tool in state.dbs:
            db = state.dbs[tool]
            if not db.schemas:
                db.schemas = db.get_table_schema()


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

def fan_out_sql_generation(state: State) -> list[Send]:
    return [
        Send(
            "generate_sql_and_query", 
            {"db": db, "user_input": state.user_input, "embedding": state.embedding}
    ) for db in state["tools_needed"] if state["tools_needed"] in state["dbs"]]



def generate_sql_and_query(params) -> Union[dict, str]:
    """
    Converts text to SQL.

    Args:


    Returns sql in string format
    """
    db = params["db"]
    user_input = params["user_input"]
    embedding = params["embedding"]

    system_prompt = {
        "role": "system",
        "content": sql_system_prompt_1 + db.schemas + "\n" + sql_system_prompt_1
    }

    user_prompt = {
        "role": "user",
        "content": user_input
    }

    # text2query
    sql = llm.invoke([system_prompt, user_prompt])

    # validate
    isValid = validate_query(sql)

    if isValid:
        results = run_query(db, sql)
    
    return {"query_results": results}


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

    Return True if everything is good, else False.
    """
def validate_query(query) -> bool:
    """
    Double check to see the SQL query is valid.
    
    Args:
        query: SQL query to check.

    Returns True if valid, False otherwise.
    """
    system_prompt = {
        "role": "system",
        "content": validate_system_prompt 
    }

    user_prompt = {
        "role": "user",
        "content": query
    }

    return bool(llm.invoke([system_prompt, user_prompt]))


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
def evaluate_results(state: State):
    """
    Evaluates sql query results and see if the results answer the user input text.
    
    """
    system_prompt = {
        "role": "system",
        #TODO
        "content": "Your job is to evaluate if the query results answer user's input. If good, return True, else, return False."
    }
    user_prompt = {
        "role": "user",
        "content": 
    }

    

#TODO: format result (list messages, generate response)
def format_result():
    """
    Formats sql query results and generates response.
    
    """
    pass


# --------------------------
# Build Graph
# --------------------------
graph = StateGraph(State)

graph.add_node("decide_tools", tool_decider)
graph.add_node("fetch_schema", fetch_schema)
graph.add_node("fan_out_sql_generation", fan_out_sql_generation)
graph.add_node("evaluate_results", evaluate_results)
graph.add_node("format_results", format_result)

graph.add_edge(START, "decide_tools")
graph.add_edge("decide_tools", "fetch_schema")
#TODO: fix conditional edges
graph.add_conditional_edges("fetch_schema", fan_out_sql_generation,  "fan_out_sql_generation")
graph.add_edge("fan_out_sql_generation", "evaluate_results")
#TODO:
graph.add_conditional_edges("evaluate_results", "format_results")
graph.addedge("format_results", END)

app = graph.compile()

    

