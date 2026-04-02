from dotenv import load_dotenv
import os
from typing import List
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()


class JobPosting(BaseModel):
    name: str = Field(description="The job title or role name")
    company: str = Field(description="The name of the company hiring")
    src: str = Field(description="The URL or source link of the job posting")

class JobResults(BaseModel):
    jobs: List[JobPosting] = Field(default_factory=list,description="A list of all the job postings found")


llm = ChatGroq(   
    model="openai/gpt-oss-120b",
    api_key = os.getenv('GROQ_API_KEY')
    )

formatter_llm = llm.with_structured_output(JobResults)




tavily_search = TavilySearch(max_results=3, topic="general")
tools = [tavily_search]
agent = create_agent(model=llm, tools=tools)

result= agent.invoke({'messages':[HumanMessage(content="Search for 3 job openings for role Gen AI role in Hyderabad city in linkedin")]})

raw_search_result = result['messages'][-1].content

structured_json = formatter_llm.invoke(
    f"Extract the job details from this text into a structured list: {raw_search_result}"
)

print("\nFinal Structured Results:")
print(structured_json.model_dump_json(indent=2))

"""
Final Structured Results:
{
  "jobs": [
    {
      "name": "Associate Director – Gen AI",
      "company": "KPMG India",
      "src": "https://in.linkedin.com/jobs/view/associate-director-gen-ai-at-kpmg-india-4389473216"
    },
    {
      "name": "Gen AI Developer",
      "company": "Coforge Ltd.",
      "src": "https://in.linkedin.com/jobs/view/gen-ai-developer-at-coforge-4395024664"
    },
    {
      "name": "Gen AI Developer (5+ yrs)",
      "company": "Zorba AI",
      "src": "https://in.linkedin.com/jobs/view/gen-ai-developer-5%2B-years-at-zorba-ai-4388278075"
    }
  ]
}

"""