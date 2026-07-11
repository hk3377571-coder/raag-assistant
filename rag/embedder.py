import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_DIR = "/tmp/chroma_db"      # ✅ works on Streamlit Cloud
EMBED_MODEL = "all-MiniLM-L6-v2"

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
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB")
    return vectordb

def load_vectorstore():
    embeddings = get_embedding_model()
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    print("✅ ChromaDB loaded!")
    return vectordb