
import openai
import os
import langchain
import shutil
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader,PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

f=open("C:\\Users\\ALEKHYA AZMEERA\\OneDrive\\Desktop\\API\\open_api_key.txt")
api_key=f.read()

os.environ['OPENAI_API_KEY']=api_key

llms=ChatOpenAI(model='gpt-4o-mini',temperature=0.5)

loader=DirectoryLoader('C:\\Users\\ALEKHYA AZMEERA\\OneDrive\\Desktop\\rag_aa\\sample_2',glob='*.pdf',show_progress=True,loader_cls=PyMuPDFLoader)
doc=loader.load()

chunk=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=50)
text_split=chunk.split_documents(doc)
#print(text_split)
#print(len(text_split))

embedding_models=OpenAIEmbeddings(api_key=api_key)
#print(embedding_models)


db=Chroma(collection_name="vectordb",embedding_function=embedding_models,
         persist_directory='./alekhya_db')
db.add_documents(text_split)


# print(os.getcwd())
#source='C:\\Users\\ALEKHYA AZMEERA\\Rag\\alekhya_db'
#distnation='C:\\Users\\ALEKHYA AZMEERA\\OneDrive\\Desktop\\rag_aa'
#shutil.move(source,distnation)

retrive=db.as_retriever(search_kwargs={"k":5})
qa_chain=RetrievalQA.from_chain_type(llm=llms, retriever=retrive)


#streamlit

st.title("question from pdf")
question=st.text_input("asking question& answer based on pdf")

if question:
    with st.spinner("searching....."):
        answer=qa_chain.run(question)
        st.success("answer:  ")
        st.write(answer)
