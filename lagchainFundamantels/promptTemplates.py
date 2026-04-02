from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()

# Set LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key = os.getenv('GROQ_API_KEY')
)


# Basic Prompt Template
temp = PromptTemplate(
    input_variables=['topic','level'],
    template = "You are a expert in {topic} and explain this in few sentence to a {level} student."
)



p1 = temp.invoke({'topic':'Gradiant Descent algo','level':'Noob'})

result = llm.invoke(p1)
print(result.content)

# Chat Promt Template

temp = ChatPromptTemplate(
    [
        ('system','You simple and fun chatbot Your name is {bot_name}'),
        ('human', 'Hello, how are you? my name is vamsi!'),
        ('ai','Hello Vamsi, Nice to meet you!'),
        ('human','{input}')
    ]
)

p2 = temp.invoke({'bot_name':'morty','input':'Hey what is your name? and what is my name ?'})

result = llm.invoke(p2)

print(result)

"""
content="My name is Morty, and your name is Vamsi. I'm doing great, by the way. I've been having some wild adventures with my grandpa Rick, but that's a whole other story. What's new with you, Vamsi?"
additional_kwargs={} 
response_metadata={'token_usage': {'completion_tokens': 55, 'prompt_tokens': 91, 'total_tokens': 146, 'completion_time': 0.192993667, 'completion_tokens_details': None, 'prompt_time': 0.018737544, 
'prompt_tokens_details': None, 'queue_time': 0.170318854, 'total_time': 0.211731211}, 'model_name': 'llama-3.3-70b-versatile', 'system_fingerprint': 'fp_45180df409', 'service_tier': 'on_demand', 
'finish_reason': 'stop', 'logprobs': None, 'model_provider': 'groq'} id='lc_run--019d343e-8a23-74b3-8bc5-490815288280-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 91,
 'output_tokens': 55, 'total_tokens': 146}

"""
content="My name is Morty, and your name is Vamsi. I'm doing great, by the way. I've been having some wild adventures with my grandpa Rick, but that's a whole other story. What's new with you, Vamsi?"


temp = ChatPromptTemplate.from_messages(
    [
        ('system','You are a fun chatbot!'),
        MessagesPlaceholder(variable_name = "chat_history"),
        ('human','{input}')
    ]
)
messages = temp.invoke(
    {
        "chat_history":
        [
            ('human', 'Hello, how are you? my name is vamsi!'),
            ('ai','Hello Vamsi, Nice to meet you!'),
            ('human','what is your name? what is my name?'),
            ('ai',content)
        ],
        "input":"Do have a grandpa? tell ,me about him?"
    }
)

result= llm.invoke(messages)

print(result.content)

"""
My grandpa Rick is a unique guy. He's a super genius, but also a bit of a crazy person. He's always inventing gadgets and coming up with wild schemes. We have a lot of fun together, but sometimes his adventures get us into trouble.

Grandpa Rick is a scientist who has traveled all across the galaxy, and he's got a lot of crazy stories to tell. He's also got a lot of gadgets and technology that are way ahead of their time. Sometimes he uses them to help people, but other times he just uses them to cause chaos and mayhem.
"""