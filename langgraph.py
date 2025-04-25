from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# https://langchain-ai.github.io/langgraph/tutorials/introduction/
# https://langchain-ai.github.io/langgraph/how-tos/branching/

class State(TypedDict):
    short_term: Annotated[list, add_messages]
    long_term: Annotated[list, add_messages]

graph_builder = StateGraph(State)

