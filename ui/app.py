import streamlit as st
import os
import re
import pickle
import json
import faiss
import numpy as np
import requests

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

# =========================
# CONFIG
# =========================

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

VECTOR_PATH = "vector_store"
UPLOAD_DIR = "uploads"

CONFIDENCE_THRESHOLD = 0.35

os.makedirs(VECTOR_PATH, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================
# MODEL (LOAD ONCE)
# =========================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

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
# DOCUMENT LOAD
# =========================

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
# VECTOR STORE
# =========================

def create_vector_store(chunks):

    texts = [c.page_content for c in chunks]
    embeddings = model.encode(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    faiss.write_index(index, f"{VECTOR_PATH}/index.faiss")

    with open(f"{VECTOR_PATH}/docs.pkl", "wb") as f:
        pickle.dump(chunks, f)

# =========================
# RETRIEVE + CONFIDENCE (YOUR EXACT LOGIC)
# =========================

def retrieve_context(question, k=3):

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

    # ⭐ YOUR CONFIDENCE MATH
    dists = distances[0]

    max_d = max(dists)
    min_d = min(dists)

    if max_d == min_d:
        confidence = 0.5
    else:
        norm_scores = [(max_d - d) / (max_d - min_d) for d in dists]
        confidence = float(np.mean(norm_scores) * (1 - np.std(norm_scores)))

    confidence = max(0.0, min(confidence, 1.0))

    return context, sources, confidence

# =========================
# LLM CALL
# =========================

def ask_llm(prompt):

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "mistralai/mixtral-8x7b-instruct",
        "messages": [
            {"role": "system", "content": "Answer ONLY using document context."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    resp_json = response.json()

    if "choices" not in resp_json:
        return str(resp_json)

    return resp_json["choices"][0]["message"]["content"]

# =========================
# EXTRACTION
# =========================

FIELDS = [
    "shipment_id",
    "shipper",
    "consignee",
    "pickup_datetime",
    "delivery_datetime",
    "equipment_type",
    "mode",
    "rate",
    "currency",
    "weight",
    "carrier_name"
]

def extract_shipment_data():

    context, sources, confidence = retrieve_context(
        "bill of lading shipment rate confirmation logistics shipment details carrier shipper consignee pickup delivery freight rate USD equipment truck mode",
        k=5
    )

    if not context or len(context.strip()) < 50 or confidence < 0.3:
        return {f: None for f in FIELDS}

    prompt = f"""
Extract shipment data.

Return ONLY JSON.
Missing fields must be null.

Fields:
{FIELDS}

Document:
{context}
"""

    response = ask_llm(prompt)

    try:
        return json.loads(response)
    except:
        return {f: None for f in FIELDS}

# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Ultra Doc Intelligence", layout="wide")

st.title("📄 Ultra Doc Intelligence")

# Upload
st.header("Upload Document")

uploaded = st.file_uploader("Upload PDF/DOCX/TXT")

if uploaded:
    path = os.path.join(UPLOAD_DIR, uploaded.name)

    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())

    docs = load_document(path)
    chunks = chunk_documents(docs)
    create_vector_store(chunks)

    st.success("Document processed + indexed")

# Ask
st.header("Ask Questions")

q = st.text_input("Ask something about document")

if st.button("Ask"):

    context, sources, confidence = retrieve_context(q)

    if not context:
        st.error("Not found in document")
    else:
        prompt = f"""
Answer ONLY using document context.

Context:
{context}

Question:
{q}
"""

        ans = ask_llm(prompt)

        if confidence < CONFIDENCE_THRESHOLD:
            ans = "(Low confidence — verify manually)\n\n" + ans

        st.subheader("Answer")
        st.write(ans)

        st.subheader("Confidence")
        st.write(confidence)

        st.subheader("Sources")
        st.json(sources)

# Extraction
st.header("Structured Extraction")

if st.button("Extract Shipment Data"):
    data = extract_shipment_data()
    st.json(data)
