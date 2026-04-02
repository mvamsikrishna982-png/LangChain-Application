from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage,SystemMessage,ToolMessage
from langchain.tools import tool
from langchain.agents import create_agent
from langsmith import traceable
from langchain_groq import ChatGroq
import os

load_dotenv()
MAX_ITERATIONS = 4
#------------- defining tools ---------------------
@tool
def get_price_of_product(product:str) ->float:
    """ 
    This tool return the price of the given product.
    Args:
        product: The name of the product.

    """
    print(f"<<<<<<<<<<<< looking for pice of :{product} >>>>>>>>>>>>>>>>>>>>")
    stuff = {'laptop':47500,'mobile':25000,'earbuds':3000}
    return stuff.get(product,0)


@tool
def apply_discount(price:float, level:str) -> float:
    """
    This tool applys discount to the price based on level
    Args:
        price: Price of the product for which we need discount.
        level: This is the discount level, discount applied based on the level and possible values are silver,gold,diamond.
    """
    print(f"<<<<<<<<<<<< claculating the discount for teh price:{price}, and level:{level} >>>>>>>>>>>>>>>>>>>>")

    discount = {'gold':15,'silver':'10','diamond':20}

    return price*((100-discount.get(level))/100)


# ------------- defining agents--------------------
@traceable
def run_agent(query:str):
    tools = [get_price_of_product,apply_discount]
    tools_dict = {t.name:t for t in tools}
    llm = init_chat_model(model="llama-3.3-70b-versatile",model_provider = "groq",temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    messages =[
        SystemMessage(
            content="You helpful shopping assistant, You have acess to product catalog tool and discount tool." \
            "Use the appropriate tool for the given query." \
            "STRICT RULES: 1. Never assume the discount,2.Never assume price of a product,3.Never caluclate price after discount use tools for it." \
            "4.If user don't specify which level of deiscount he belongs, ask him don't assume."
        ),
        HumanMessage(content=query)
    ]

    for i in range(1,MAX_ITERATIONS+1):
        print(f'================================ Iteration:{i} =================================')
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print("Final answer:",ai_message.content)
            return ai_message.content
        


        # if tools are not empty process the forst tool
        tool_call = tool_calls[0]
        tool_name = tool_call.get('name')
        tool_args = tool_call.get('args')
        tool_call_id = tool_call.get('id')

        print(f"tools using:{tool_name}")
        tool_to_use = tools_dict.get(tool_name)

        observation = tool_to_use.invoke(tool_args)

        print("tool observarion:",observation)

        messages.append(ai_message)

        messages.append(
            ToolMessage(content=str(observation),tool_call_id=tool_call_id)
            )
        

    print("ERROR: Not able to resolve the query after {MAX_ITERATIONS} retries")
    return None



if __name__=="__main__":
    response = run_agent("what is the price of mobile after discount to diamond level customer.")
  


    