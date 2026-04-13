from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
load_dotenv()



@tool
def triple(temp:float)->float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    return float(temp)*3

tools = [TavilySearch(),triple]

llm = ChatGroq(model="openai/gpt-oss-120b").bind_tools(tools)