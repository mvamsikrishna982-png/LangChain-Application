from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from tavily import TavilyClient
load_dotenv()

tavily = TavilyClient()

@tool
def search(query:str)->str:
    """
    Returns the current weather for a chicago city.
    Use this tool when the user asks about weather.
    Agrs:
        query: The query to search for.
    Returns:
        The search result
    """
    print("This is the input query:",query)
    return tavily.search(query)



llm = ChatGroq(   
    model="llama-3.3-70b-versatile",
    api_key = os.getenv('GROQ_API_KEY')
    )

tools = [search]
agent = create_agent(model=llm, tools=tools)

result= agent.invoke({'messages':[HumanMessage(content="How is the weather in chicago today")]})
print(result)
print(type(result))
