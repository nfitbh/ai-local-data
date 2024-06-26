import os
import glob

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.tsv import UnstructuredTSVLoader


class AiData:

    def __init__(self, llmName: str, openAiApiKey: str, dataLocation: str):
        #set up some config
        #use OpenAI (you'll need an OPENAI_API_KEY)
        self.embeddings = OpenAIEmbeddings()

        #store the init data
        self.dataLocation = dataLocation
        self.openAiApiKey = openAiApiKey
        self.llm_name = llmName
        
        #init the openai things
        self.llm = ChatOpenAI(
            model_name=self.llm_name,
            temperature=0,
            openai_api_key=self.openAiApiKey
        )

        #init the converstaion memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        #we dont have an index yet (but this is an important thingy)
        self.index = None

    #gets a list of documents from raw documents
    def getDocuments(self, rawDocs: list) -> list:
        text_splitter = RecursiveCharacterTextSplitter()
        return  text_splitter.split_documents(rawDocs)

    #loads data from the data dir
    def loadData(self):
        documents = []
        #get text files at the location and load using UnstructuredFileLoader
        for file in glob.glob(self.dataLocation+"/*.txt"):
            print("Loading file %s", file)
            documents.extend(self.getDocuments(UnstructuredFileLoader(file).load()))
        #get json files at the location and load using JSONLoader
        for file in glob.glob(self.dataLocation+"/*.json"):
            print("Loading file %s", file)
            documents.extend(self.getDocuments(JSONLoader(file_path=file, jq_schema='.', text_content=False).load()))
        for file in glob.glob(self.dataLocation+"/*.csv"):
            print("Loading file: ", file)
            documents.extend(self.getDocuments(CSVLoader(file_path=file).load()))
        for file in glob.glob(self.dataLocation+"/*.tsv"):
            print("Loading file: ", file)
            docs = UnstructuredTSVLoader(file_path=file, mode="elements").load()
            print(docs[1].metadata["text_as_html"])
            documents.extend(self.getDocuments(docs))
            
        #create the index - here be data
        self.index = FAISS.from_documents(documents, self.embeddings)

        #create the question & answer chain - a specific llm, index and memory
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.index.as_retriever(),
            memory=self.memory
        )

    #take a question, invoke the q&a chain and return the answer
    def query(self, question: str) -> str:
        result = self.qa_chain.invoke({"question": question})
        return result['answer']

if __name__ == '__main__':

    #init all the things
    aiData = AiData("gpt-3.5-turbo", os.environ['OPENAI_API_KEY'], "../data")

    #load the things
    aiData.loadData()

    #use stdin to ask away to your dreams content
    while True:
        # Prompt the user to introduce a question
        question = input("Ask a question or type 'exit': ")
        
        if question.lower() == "exit":
            break

        #if this doesn't make sense, go to a doctor now
        answer = aiData.query(question)
        print("Answer: ", answer)