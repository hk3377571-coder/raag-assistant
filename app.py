import streamlit as st
import shutil
import os
from rag.loader import load_document
from rag.embedder import store_documents
from rag.retriever import build_rag_chain, ask

# ─────────────────────────────────────────
# 🎨 Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="RAAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# ─────────────────────────────────────────
# 💅 Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stChatMessage { border-radius: 12px; margin: 8px 0; }
    .stSidebar { background-color: #1a1a2e; }
    h1 { color: #00d4ff; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 🔧 Session State
# ─────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False

# ─────────────────────────────────────────
# 📌 Sidebar — Upload Documents
# ─────────────────────────────────────────
with st.sidebar:
    st.title("📁 Document Manager")
    st.markdown("---")

    # File Upload
    st.subheader("📤 Upload Files")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    # URL Input
    st.subheader("🌐 Or Add a URL")
    url_input = st.text_input("Paste a website URL")

    # Process Button
    if st.button("⚡ Process Documents", type="primary"):
        all_docs = []

        # Save and load uploaded files
        if uploaded_files:
            os.makedirs("uploaded_docs", exist_ok=True)
            for file in uploaded_files:
                path = f"uploaded_docs/{file.name}"
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
                with st.spinner(f"Reading {file.name}..."):
                    docs = load_document(path)
                    all_docs.extend(docs)
                st.success(f"✅ {file.name}")

        # Load URL
        if url_input:
            with st.spinner(f"Reading URL..."):
                docs = load_document(url_input)
                all_docs.extend(docs)
            st.success(f"✅ URL loaded!")

        # Store in ChromaDB
        if all_docs:
            with st.spinner("🧠 Building knowledge base..."):
                if os.path.exists("chroma_db"):
                    shutil.rmtree("chroma_db")
                store_documents(all_docs)
                st.session_state.chain = build_rag_chain()
                st.session_state.docs_loaded = True
            st.success("🎉 Ready to answer questions!")
        else:
            st.warning("Please upload a file or enter a URL!")

    st.markdown("---")

    # Clear Chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    # Status
    st.markdown("---")
    if st.session_state.docs_loaded:
        st.success("📚 Documents loaded!")
    else:
        st.info("⬆️ Upload documents to start")

# ─────────────────────────────────────────
# 💬 Main Chat Area
# ─────────────────────────────────────────
st.title("🤖 RAAG Assistant")
st.caption("Chat with your documents using AI")
st.markdown("---")

# Show chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Show user message
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt
    })

    # Get AI answer
    if st.session_state.docs_loaded and st.session_state.chain:
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                answer = ask(st.session_state.chain, prompt)
                st.write(answer)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })
    else:
        with st.chat_message("assistant"):
            st.warning("⬆️ Please upload documents first using the sidebar!")
            