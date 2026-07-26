import os
import shutil
from langchain_core.documents import Document

os.environ["PYTHONIOENCODING"] = "utf-8"

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain_community.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "./chroma_db"
DATA_PATH = "./data"

def load_documents_utf8():
    documents = []
    if os.path.exists(DATA_PATH):
        for filename in sorted(os.listdir(DATA_PATH)):
            if filename.endswith(".txt"):
                filepath = os.path.join(DATA_PATH, filename)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read()
                        documents.append(Document(page_content=text, metadata={"source": filename}))
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
    return documents

def build_or_load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        try:
            return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        except Exception:
            try:
                shutil.rmtree(CHROMA_PATH)
            except Exception:
                pass
    
    # Ingest data safely using custom UTF-8 reader with fallback
    documents = load_documents_utf8()
    
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
