import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "./chroma_db"
DATA_PATH = "./data"

def build_or_load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # Ingest data
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH
    )
    return vectorstore

def query_rag(query: str, k: int = 3):
    vectorstore = build_or_load_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])
