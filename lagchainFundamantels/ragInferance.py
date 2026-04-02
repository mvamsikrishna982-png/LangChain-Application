from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.runnables import RunnableLambda,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(model='gpt-5.2')

embeddings = OpenAIEmbeddings()

vectorObj = PineconeVectorStore( embedding=embeddings, index_name='sample-rag-using-langchain')

doc_retriver = vectorObj.as_retriever(search_kwargs={'k':3})

prompt = ChatPromptTemplate.from_template(
    """
    Answer the given question only based on the this context:{context}
    Question:{question}
    """ 
)

def form_document(documents):
    return "\n\n".join(doc.page_content for doc in documents)


if __name__ == "__main__":
    question = "What trade does Rick make to retrieve Morty's borrowed week of confidence?and what does that reveal about Rick's character?"


        
    
    # print("<<<<<<<<<<<<<<<<<<<<<<LLM Output without context >>>>>>>>>>>>>>>>>>>>>>>>")
    # result = llm.invoke([HumanMessage(content=question)])
    # print("LLM Output without context")
    # print(result.content)
    # print("<<<<<<<<<<<<<<<<<<<<<< USING RAG WITHOUT LCEL >>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # docs = doc_retriver.invoke(question)
    # context = form_document(docs)
    # new_promt = prompt.format_messages(context=context,question=question)
    # result = llm.invoke(new_promt)
    # print("LLM ouput with out LCEL:")
    # print(result.content)

    print("<<<<<<<<<<<<<<<<<<<<<< USING RAG WITH LCEL >>>>>>>>>>>>>>>>>>>>>>>>>>>")

    runnable_form_document = RunnableLambda(form_document)

    rag_chain = (
        {
            'context':doc_retriver|runnable_form_document,
            'question': RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    result = rag_chain.invoke(question)

    print(result)





"""
Actual answer:
---------------
Rick trades a deeply personal "rooftop memory" an authentic, 
emotionally weighted moment—to recover the memory they had sold;
this reveals Rick's willingness to sacrifice private anchors for a greater good,
and it exposes his complex mix of pragmatism, guilt, and occasional vulnerability.

output:
-------
<<<<<<<<<<<<<<<<<<<<<< LLM Output without context >>>>>>>>>>>>>>>>>>>>>>>>

Rick trades away **Morty’s “borrowed week of confidence” by giving the remaining supply of the *confidence serum* 
(from the alien seeds/mega-fruit) to the galactic authorities in exchange for getting out of trouble**—effectively cashing in
Morty’s temporary boost to clean up Rick’s own mess.

What that reveals about Rick:

- **Instrumental, utilitarian morality:** Rick treats Morty’s wellbeing (even something Morty earned/needed) as a bargaining chip when it’s convenient.
- **Self-preservation first:** When pressured, Rick prioritizes escaping consequences over protecting Morty’s emotional gains.
- **Control and cynicism:** He’s uncomfortable with Morty becoming more self-assured/independent, and he’s quick to reassert the “Rick is in charge” dynamic.
- **Not purely heartless—but deeply dysfunctional:** Rick does care in his own way, but his default mode is transactional and evasive rather than openly nurturing.


<<<<<<<<<<<<<<<<<<<<<< USING RAG WITHOUT LCEL >>>>>>>>>>>>>>>>>>>>>>>>

LLM ouput with out LCEL:
-----------------------
Rick trades away his own “rooftop memory” in order to get back Morty’s borrowed week of confidence. 
That reveals Rick is willing to sacrifice something personally meaningful—his own cherished experiences—for Morty’s
well-being, showing protective love and regret beneath his tough, calculating exterior.

<<<<<<<<<<<<<<<<<<<<<< USING RAG WITH LCEL >>>>>>>>>>>>>>>>>>>>>>>>>>>
Rick trades away a “rooftop memory” to get back Morty’s borrowed week of confidence.

That trade reveals Rick will sacrifice even something personally meaningful and emotionally 
grounding to protect or help Morty—showing care and regret beneath his tough, cynical posture.
"""


