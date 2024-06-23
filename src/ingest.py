import os
import sys

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle
from langchain_community.vectorstores import FAISS

print (sys.argv[1])

rawDocs = UnstructuredFileLoader(sys.argv[1]).load()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(rawDocs)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

vectorstore.save_local("vectorstore.pkl")

