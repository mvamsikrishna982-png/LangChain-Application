from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser,PydanticOutputParser
from pydantic import BaseModel, Field
import os
load_dotenv()

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    api_key = os.getenv("GROQ_API_KEY")
)



################## StrOutputParser ###############
# Return the .content form AiMessage Calss.

prompt = ChatPromptTemplate.from_messages(
    [
        ('system','You are a expert in ML.'),
        ('human','explin this ML topic in very few sentenses:{topic}')
    ]
)

chain = prompt | llm | StrOutputParser()
print(chain.invoke({'topic':'dropout'}))


################# JsonOutputParseer ###############
# When LLM respoonds in JSON this parserr converts that into a python dict.

prompt = ChatPromptTemplate.from_messages(
    [
        ('system','You are a expert in ML and responds in JSON format strictly and no extra text.'),
        ('human','explin this ML topic:{topic}, with keys name,defination,example')
    ]
)

chain = prompt | llm | JsonOutputParser()
print(chain.invoke({'topic':'dropout'}))

"""
{
    "name": "Dropout",
    "definition": "A regularization technique in deep learning where a fraction of neurons in a layer are randomly dropped
                     during training, helping to prevent overfitting",
    "example": {
        "description": "In a neural network with 2 hidden layers, each with 100 neurons, a dropout rate of 0.2 means that 20 neurons in each
         layer will be randomly dropped during training, reducing the capacity of the network and preventing overfitting",
        "code": "model.add(Dense(100, activation='relu'))\nmodel.add(Dropout(0.2))\nmodel.add(Dense(100, activation='relu'))\nmodel.add(Dropout(0.2))",
    },
}
"""

################## PydanticOutputParser ###############

class output_format(BaseModel):
    name: str = Field(description="Name of the example")
    defination:str = Field(description = "Therotical defination in very few terms.")
    example:str = Field(description="example in code.")

parser = PydanticOutputParser(pydantic_object = output_format)


prompt = ChatPromptTemplate.from_messages(
    [
        ('system','You are a expert in ML.{format}'),
        ('human','explin this ML topic:{topic}, with keys name,defination,example')
    ]
).partial(format=parser.get_format_instructions())

chain = prompt | llm | parser
result = chain.invoke({'topic':'dropout'})
print("result type",type(result))
print(result)

"""
result type <class '__main__.output_format'>
name='Dropout' 
defination='A regularization technique to prevent overfitting'
example='In Python using Keras: model.add(Dropout(0.2)) to randomly drop 20% of neurons during training'
"""