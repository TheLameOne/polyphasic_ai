import os
from fastapi import FastAPI
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
import google.generativeai as genai
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
load_dotenv()

GEMINI_API = os.getenv('GEMINI_API')
MONGODB_URL = os.getenv('MONGODB_URL')
OPENAI_API = os.getenv('OPENAI_API')

api = GEMINI_API

def get_stored_vectorstore():
    uri= MONGODB_URL
    client = MongoClient(uri)
    DB_NAME = "langchain_db"
    COLLECTION_NAME = "test"
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
    MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]
    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    uri,
    DB_NAME + "." + COLLECTION_NAME,
    OpenAIEmbeddings(api_key= OPENAI_API,disallowed_special=()),
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)
    return vector_search

app=FastAPI()

@app.get("/")
def root(question):
    vector=get_stored_vectorstore()
    genai.configure(api_key=api)
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash',google_api_key=api)
    retriever=vector.as_retriever(search_type="similarity",search_kwargs={"k": 25},)
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context and remove all * from answer aswell:

    <context>
    {context}
    </context>

    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": question})
    return {"message":response['answer']}
