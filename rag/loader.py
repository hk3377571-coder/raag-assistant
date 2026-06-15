import os
import requests
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)


# ─────────────────────────────────────────
# 📄 Load a PDF file
# ─────────────────────────────────────────
def load_pdf(file_path: str) -> list[Document]:
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"✅ PDF loaded: {len(pages)} pages from '{file_path}'")
    return pages


# ─────────────────────────────────────────
# 📝 Load a TXT file
# ─────────────────────────────────────────
def load_txt(file_path: str) -> list[Document]:
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    print(f"✅ TXT loaded: '{file_path}'")
    return docs


# ─────────────────────────────────────────
# 📘 Load a DOCX file
# ─────────────────────────────────────────
def load_docx(file_path: str) -> list[Document]:
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    print(f"✅ DOCX loaded: '{file_path}'")
    return docs


# ─────────────────────────────────────────
# 🌐 Load a Web URL
# ─────────────────────────────────────────
def load_url(url: str) -> list[Document]:
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script/style noise
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    doc = Document(
        page_content=text,
        metadata={"source": url}
    )
    print(f"✅ URL loaded: '{url}'")
    return [doc]


# ─────────────────────────────────────────
# 🚀 Master Loader — detects type automatically
# ─────────────────────────────────────────
def load_document(source: str) -> list[Document]:
    """
    Pass any file path or URL.
    It will automatically detect the type and load it.
    """
    if source.startswith("http://") or source.startswith("https://"):
        return load_url(source)

    ext = os.path.splitext(source)[-1].lower()

    if ext == ".pdf":
        return load_pdf(source)
    elif ext == ".txt":
        return load_txt(source)
    elif ext == ".docx":
        return load_docx(source)
    else:
        raise ValueError(f"❌ Unsupported file type: '{ext}'")


# ─────────────────────────────────────────
# 📂 Load all files from a folder
# ─────────────────────────────────────────
def load_all_from_folder(folder_path: str) -> list[Document]:
    all_docs = []
    supported = (".pdf", ".txt", ".docx")

    for filename in os.listdir(folder_path):
        if filename.endswith(supported):
            full_path = os.path.join(folder_path, filename)
            docs = load_document(full_path)
            all_docs.extend(docs)

    print(f"\n📚 Total documents loaded: {len(all_docs)}")
    return all_docs