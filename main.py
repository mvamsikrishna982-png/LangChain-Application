from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict,Annotated
import operator


class Mystate(TypedDict):
    message: str
    result: Annotated[list,add_messages]


def node(state: Mystate)->dict:
    return {'result':f"node 1 says {state['message']}"}

def node2(state: Mystate)->dict:
    return {'result':f"node 2 says{state['message']}"}


graph = StateGraph(Mystate)

graph.add_node("node1",node)
graph.set_entry_point("node1")
graph.add_node("node2",node2)
graph.add_edge("node1","node2")
graph.add_edge("node2",END)


app = graph.compile()

# output = app.invoke({'message':'hello','result':''})
output = app.invoke({'message':'hello'})

print(output)