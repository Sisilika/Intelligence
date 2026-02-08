import os
import pickle
import faiss
import numpy as np
import re

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

VECTOR_PATH = "vector_store"

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text

def load_document(file_path):

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise Exception("Unsupported file")

    docs = loader.load()

    for d in docs:
        d.page_content = clean_text(d.page_content)

    return docs

def chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(documents)

def create_vector_store(chunks):

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [c.page_content for c in chunks]
    embeddings = model.encode(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    os.makedirs(VECTOR_PATH, exist_ok=True)

    faiss.write_index(index, f"{VECTOR_PATH}/index.faiss")

    with open(f"{VECTOR_PATH}/docs.pkl", "wb") as f:
        pickle.dump(chunks, f)

def retrieve_context(question, k=3):

    model = SentenceTransformer("all-MiniLM-L6-v2")

    index = faiss.read_index(f"{VECTOR_PATH}/index.faiss")

    with open(f"{VECTOR_PATH}/docs.pkl", "rb") as f:
        docs = pickle.load(f)

    query_vector = model.encode([question])

    distances, indices = index.search(query_vector, k)

    retrieved_docs = [docs[i] for i in indices[0] if i < len(docs)]

    if not retrieved_docs:
        return "", [], 0.0

    context = "\n\n".join([d.page_content for d in retrieved_docs])
    sources = [d.metadata for d in retrieved_docs]

    dists = distances[0]
    confidence = float(np.mean(1/(1+dists)))

    return context, sources, confidence
