import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rag.embedder import load_vectorstore

load_dotenv()

# ─────────────────────────────────────────
# 🧠 Load Groq AI (Llama 3)
# ─────────────────────────────────────────
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",   # free & very powerful!
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
        max_tokens=1024
    )

# ─────────────────────────────────────────
# 📋 Prompt Template
# ─────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in the documents."

Context:
{context}

Question: {question}

Answer:"""

# ─────────────────────────────────────────
# 🔗 Build the RAG Chain
# ─────────────────────────────────────────
def build_rag_chain():
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = get_llm()
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✅ RAG chain ready!")
    return chain

# ─────────────────────────────────────────
# 💬 Ask a Question
# ─────────────────────────────────────────
def ask(chain, question: str):
    print(f"\n❓ Question: {question}")
    answer = chain.invoke(question)
    print(f"💬 Answer: {answer}")
    return answer