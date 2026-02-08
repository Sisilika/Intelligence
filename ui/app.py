import streamlit as st
import requests
import json

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Doc Intelligence", layout="wide")

st.title("📄 Doc Intelligence")
st.write("Upload logistics documents, ask questions, or extract structured shipment data.")

# -------------------
# Upload Section
# -------------------
st.header("📤 Upload Document")

uploaded_file = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    if st.button("Upload Document"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}

        response = requests.post(f"{API_BASE}/upload", files=files)

        if response.status_code == 200:
            st.success("✅ Document uploaded and indexed!")
        else:
            st.error("❌ Upload failed")


# -------------------
# Ask Question Section
# -------------------
st.header("❓ Ask Questions About Document")

question = st.text_input("Enter your question")

if st.button("Ask Question"):
    if question.strip() != "":
        payload = {"question": question}

        response = requests.post(f"{API_BASE}/ask", json=payload)

        if response.status_code == 200:
            result = response.json()

            st.subheader("Answer")
            st.write(result.get("answer"))

            st.subheader("Confidence")
            st.write(result.get("confidence"))

            st.subheader("Sources")
            st.json(result.get("sources"))

        else:
            st.error("Failed to get answer")


# -------------------
# Extraction Section
# -------------------
st.header("📦 Structured Shipment Extraction")

if st.button("Run Extraction"):
    response = requests.post(f"{API_BASE}/extract")

    if response.status_code == 200:
        result = response.json()

        st.subheader("Extracted Shipment Data")
        st.json(result)

    else:
        st.error("Extraction failed")
