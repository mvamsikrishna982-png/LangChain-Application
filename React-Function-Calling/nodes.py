from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from react import llm,tools
load_dotenv()

SYSTEM_MESSAGE="""
You are a helpful assistant that can use tools to answer questions.
"""

def run_agent_reasoning(state: MessagesState)->dict:
    """
    Run the agent reasoning node.
    """
    result = llm.invoke([{'role':'system','content':SYSTEM_MESSAGE},*state['messages']])
    return {'messages':result}

tool_node = ToolNode(tools)
