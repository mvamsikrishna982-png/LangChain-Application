from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph,END
from langgraph.graph.message import MessagesState
from nodes import tool_node,run_agent_reasoning
load_dotenv()

ACT = "act"
LAST = -1
AGENt_REASONING = "agent_reasoning"

def should_continue(state:MessagesState):
    if state["messages"][-1].tool_calls:
        return ACT
    return END

workflow= StateGraph(MessagesState)
workflow.add_node(AGENt_REASONING,run_agent_reasoning)
workflow.set_entry_point(AGENt_REASONING)
workflow.add_node(ACT,tool_node)

workflow.add_conditional_edges(
    AGENt_REASONING,
    should_continue,
    {
        END:END,
        ACT:ACT
    }
)

workflow.add_edge(ACT,AGENt_REASONING)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")
    res = app.invoke({"messages": [HumanMessage(content="What is the temperature in Mysore? List it and then triple it")]})
    print(res["messages"][LAST].content)
