from rag.retriever import build_rag_chain, ask

chain = build_rag_chain()

ask(chain, "What is artificial intelligence?")
ask(chain, "What is machine learning?")