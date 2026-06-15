from rag.loader import load_document
from rag.embedder import store_documents, load_vectorstore
import shutil, os

# Clear old ChromaDB
if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")
    print("🗑️ Cleared old ChromaDB")

# Load our local text file
print("📄 Loading document...")
docs = load_document("uploaded_docs/ai_info.txt")
print(f"✅ Loaded {len(docs)} doc, {len(docs[0].page_content)} characters")

# Store in ChromaDB
print("\n🗃️ Storing in ChromaDB...")
store_documents(docs)

# Test search
print("\n🔍 Testing search...")
db = load_vectorstore()
results = db.similarity_search("What is machine learning?", k=2)
for i, r in enumerate(results):
    print(f"\n--- Chunk {i+1} ---")
    print(r.page_content)