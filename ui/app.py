import streamlit as st
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.rag_pipeline import load_document, chunk_documents, create_vector_store, retrieve_context
from backend.extractor import extract_shipment_data
from backend.llm import ask_llm

st.set_page_config(page_title="Doc Intelligence", layout="wide")

st.title("📄 Ultra Doc Intelligence")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Upload
st.header("Upload Document")

file = st.file_uploader("Upload PDF/DOCX/TXT", type=["pdf","docx","txt"])

if file and st.button("Process Document"):

    path = os.path.join(UPLOAD_DIR, file.name)

    with open(path,"wb") as f:
        f.write(file.getbuffer())

    docs = load_document(path)
    chunks = chunk_documents(docs)
    create_vector_store(chunks)

    st.success("Document processed")

# Ask
st.header("Ask Questions")

question = st.text_input("Enter question")

if st.button("Ask") and question:

    context, sources, confidence = retrieve_context(question)

    if not context:
        st.error("Not found in document")
    else:
        prompt = f"""
Answer ONLY using document.

Context:
{context}

Question:
{question}
"""

        answer = ask_llm(prompt)

        st.write("Answer:", answer)
        st.write("Confidence:", confidence)
        st.json(sources)

# Extract
st.header("Structured Extraction")

if st.button("Run Extraction"):
    result = extract_shipment_data()
    st.json(result)
