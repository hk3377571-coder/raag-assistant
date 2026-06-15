from rag.loader import load_document

# Test with a URL
docs = load_document("https://en.wikipedia.org/wiki/Artificial_intelligence")
print(docs[0].page_content[:300])  # Print first 300 characters