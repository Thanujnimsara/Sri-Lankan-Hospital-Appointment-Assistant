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

_vectorstore_instance = None

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
    global _vectorstore_instance
    if _vectorstore_instance is not None:
        return _vectorstore_instance

    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
            try:
                _vectorstore_instance = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
                return _vectorstore_instance
            except Exception:
                try:
                    shutil.rmtree(CHROMA_PATH)
                except Exception:
                    pass
        
        documents = load_documents_utf8()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        
        _vectorstore_instance = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=CHROMA_PATH
        )
        return _vectorstore_instance
    except Exception as e:
        print(f"ChromaDB initialization notice: {e}")
        return None

def query_rag(query: str, k: int = 3):
    try:
        vectorstore = build_or_load_vectorstore()
        if vectorstore is not None:
            results = vectorstore.similarity_search(query, k=k)
            if results:
                return "\n\n".join([doc.page_content for doc in results])
    except Exception as err:
        print(f"Chroma similarity search fallback: {err}")
    
    # Fail-safe in-memory retrieval over 20 UTF-8 healthcare documents
    documents = load_documents_utf8()
    query_words = set(query.lower().split())
    scored_docs = []
    for doc in documents:
        score = sum(1 for word in query_words if word in doc.page_content.lower())
        scored_docs.append((score, doc.page_content))
    
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [content for score, content in scored_docs[:k]]
    return "\n\n".join(top_chunks)
