from audio_recorder_streamlit import audio_recorder
from groq import Groq
import tempfile
from gtts import gTTS
import base64
import streamlit as st
import shutil
import os
from rag.loader import load_document
from rag.embedder import store_documents
from rag.retriever import build_rag_chain, ask

# ─────────────────────────────────────────
# 🔧 Helper Functions
# ─────────────────────────────────────────
def speech_to_text(audio_bytes):
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    with open(tmp_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
        )
    return transcription.text

def text_to_audio(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tts.save(f.name)
        tmp_path = f.name
    with open(tmp_path, "rb") as audio_file:
        b64 = base64.b64encode(audio_file.read()).decode()
    return f'<audio autoplay controls><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'

# ─────────────────────────────────────────
# 🎨 Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
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
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

# ─────────────────────────────────────────
# 📌 Sidebar — Upload Documents
# ─────────────────────────────────────────
with st.sidebar:
    st.title("📁 Document Manager")
    st.markdown("---")

    st.subheader("📤 Upload Files")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    st.subheader("🌐 Or Add a URL")
    url_input = st.text_input("Paste a website URL")

    if st.button("⚡ Process Documents", type="primary"):
        all_docs = []

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

        if url_input:
            with st.spinner("Reading URL..."):
                docs = load_document(url_input)
                all_docs.extend(docs)
            st.success("✅ URL loaded!")

        if all_docs:
            with st.spinner("🧠 Building knowledge base..."):
                if os.path.exists("faiss_db"):
                    shutil.rmtree("faiss_db")
                store_documents(all_docs)
                st.session_state.chain = build_rag_chain()
                st.session_state.docs_loaded = True
            st.success("🎉 Ready to answer questions!")
        else:
            st.warning("Please upload a file or enter a URL!")

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    if st.session_state.docs_loaded:
        st.success("📚 Documents loaded!")
    else:
        st.info("⬆️ Upload documents to start")

# ─────────────────────────────────────────
# 💬 Main Chat Area
# ─────────────────────────────────────────
st.title("🤖 RAG Assistant")
st.caption("Chat with your documents using AI")
st.markdown("---")

# Chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ── Mic Input ──
st.markdown("🎤 **Or speak your question:**")
audio_bytes = audio_recorder(text="", icon_size="2x", pause_threshold=2.0)

prompt = None

# Handle mic
if audio_bytes and audio_bytes != st.session_state.last_audio and len(audio_bytes) > 1000:
    st.session_state.last_audio = audio_bytes
    with st.spinner("🎤 Transcribing..."):
        prompt = speech_to_text(audio_bytes)
    st.info(f"🎤 You said: *{prompt}*")

# Handle text input
text_prompt = st.chat_input("Ask a question about your documents...")
if text_prompt:
    prompt = text_prompt

# ── Process & Answer ──
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    if st.session_state.docs_loaded and st.session_state.chain:
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                answer = ask(st.session_state.chain, prompt)
                st.write(answer)
                st.markdown(text_to_audio(answer), unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
    else:
        with st.chat_message("assistant"):
            st.warning("⬆️ Please upload documents first!")