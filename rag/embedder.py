import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

EMBED_MODEL = "all-MiniLM-L6-v2"
FAISS_DIR = "/tmp/faiss_db"

def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks

def get_embedding_model():
    print("⏳ Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    print("✅ Embedding model ready!")
    return embeddings

def store_documents(documents: list[Document]):
    chunks = chunk_documents(documents)
    embeddings = get_embedding_model()
    vectordb = FAISS.from_documents(chunks, embeddings)
    os.makedirs(FAISS_DIR, exist_ok=True)
    vectordb.save_local(FAISS_DIR)
    print(f"✅ Stored {len(chunks)} chunks in FAISS")
    return vectordb

def load_vectorstore():
    embeddings = get_embedding_model()
    vectordb = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ FAISS loaded!")
    return vectordb