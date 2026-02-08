import os
import re
import pickle
import faiss
import numpy as np

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

load_dotenv()

VECTOR_PATH = "vector_store"
CONFIDENCE_THRESHOLD = 0.25

# ✅ LOAD MODEL ONCE (IMPORTANT)
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# TEXT CLEANING
# =========================
def clean_text(text):
    text = re.sub(r'(?<=\w)\s(?=\w)', '', text)
    text = text.replace("U S D", "USD")
    text = re.sub(r'\$ (\d)', r'$\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# =========================
# LOAD DOCUMENT
# =========================
def load_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise Exception("Unsupported file type")

    docs = loader.load()

    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    return docs


# =========================
# CHUNKING
# =========================
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


# =========================
# VECTOR STORE CREATION
# =========================
def create_vector_store(chunks):

    texts = [c.page_content for c in chunks]
    embeddings = EMBED_MODEL.encode(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    os.makedirs(VECTOR_PATH, exist_ok=True)

    faiss.write_index(index, f"{VECTOR_PATH}/index.faiss")

    with open(f"{VECTOR_PATH}/docs.pkl", "wb") as f:
        pickle.dump(chunks, f)
