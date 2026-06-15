import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ─────────────────────────────────────────
# ⚙️ Settings
# ─────────────────────────────────────────
CHROMA_DIR = "chroma_db"        # where vectors are saved
EMBED_MODEL = "all-MiniLM-L6-v2"  # free, fast, good quality

# ─────────────────────────────────────────
# ✂️ Split documents into small chunks
# ─────────────────────────────────────────
def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # ~500 characters per chunk
        chunk_overlap=50,     # overlap so nothing is lost at edges
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────
# 🧠 Load the embedding model
# ─────────────────────────────────────────
def get_embedding_model():
    print("⏳ Loading embedding model (first time may take 1-2 min)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    print("✅ Embedding model ready!")
    return embeddings


# ─────────────────────────────────────────
# 🗃️ Store chunks into ChromaDB
# ─────────────────────────────────────────
def store_documents(documents: list[Document]):
    chunks = chunk_documents(documents)
    embeddings = get_embedding_model()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB at '{CHROMA_DIR}'")
    return vectordb


# ─────────────────────────────────────────
# 🔍 Load existing ChromaDB
# ─────────────────────────────────────────
def load_vectorstore():
    embeddings = get_embedding_model()
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    print("✅ ChromaDB loaded!")
    return vectordb