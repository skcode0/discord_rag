from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# https://langchain-ai.github.io/langgraph/tutorials/introduction/
# https://langchain-ai.github.io/langgraph/how-tos/branching/

#TODO: fix 
class State(TypedDict):
    input_text: str
    input_embedding: list
    query_results: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# user input/query --> tools agent (get table schemas, run queries, keep running until no error) --> evaluate (answer good/not good) --> format/synthesize (clean up results) --> end


    



    

