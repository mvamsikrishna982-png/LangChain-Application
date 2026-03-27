from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
import os

load_dotenv()


llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # very capable free model
    api_key=os.getenv("GROQ_API_KEY"),
    
)

temp = PromptTemplate(
    input_variables=["topic","level"],
    template = "Give {topic} defination in 3 points, as I am at {level} level."
)

chatPromtTemp = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in {domain}. respond always in JSON format"),
        ("human", "explain this in {input} 2 points.")]
)


parser = StrOutputParser()
jsonParser = JsonOutputParser()


# print(temp.invoke({'topic':'Machine Learnig','level':'Noob'}))

# response = llm.invoke(temp.invoke({'topic':'Machine Learnig','level':'Noob'}))
# print(response.content)

# response = llm.invoke(chatPromtTemp.invoke({'domain':'Machine Learnig','input':'Gradiant desent'}))
# print(response.content)

# chain = chatPromtTemp | llm | parser
# response = chain.invoke({'domain':'NLP','input':'BERT'},)
# print(response)

# chain = chatPromtTemp | llm | jsonParser
# response = chain.invoke({'domain':'ML','input':'Bias'},)
# print(response)

response = llm.invoke("Hello !!")
print(response.content)




