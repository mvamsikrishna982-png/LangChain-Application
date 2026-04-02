from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()




if __name__ == "__main__":
    print("<<<<<<<<<<<<<<<<<<< Loading documents >>>>>>>>>>>>>>>>>>")
    loader  = TextLoader(file_path="C:/Users/mvams/OneDrive/Desktop/GenAI/langchain/langchain_1/rickAndMorty.txt",encoding="utf-8")

    documents = loader.load()

    print("<<<<<<<<<<<<<<<<<<< Spliting documents >>>>>>>>>>>>>>>>>>")

    chunker = RecursiveCharacterTextSplitter(
        chunk_size=250, chunk_overlap=50
    )
    chunks = chunker.split_documents(documents=documents)

    print(f"--------------Total chuncks:{len(chunks)}-----------")


    print("<<<<<<<<<<<<<<<<<<< Embedding documents >>>>>>>>>>>>>>>>>>")
    embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))

    # PineconeVectorStore.from_documents(documents=chunks,
    #                                    embedding=embeddings,
    #                                    index_name='sample-rag-using-langchain')
