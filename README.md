# Intelligence --- Logistics AI Assistant (POC)

## 🚀 Overview

Ultra Doc Intelligence is a Proof-of-Concept AI system that allows users
to upload logistics documents and interact with them using natural
language questions.

The system simulates an AI assistant inside a Transportation Management
System (TMS) by enabling: - Document upload and processing -
Retrieval-based question answering (RAG) - Hallucination guardrails -
Confidence scoring - Structured shipment data extraction - Lightweight
review UI

------------------------------------------------------------------------

## 🎯 Project Goal

Build an end-to-end AI pipeline that: ✅ Understands logistics
documents\
✅ Answers grounded questions only from document context\
✅ Prevents hallucinated answers\
✅ Returns confidence score with every response\
✅ Extracts structured shipment data in JSON

------------------------------------------------------------------------

## 🧠 Tech Stack

-   UI: Streamlit
-   Embeddings: Sentence Transformers (MiniLM)
-   Vector DB: FAISS
-   LLM: OpenRouter (Mixtral 8x7B Instruct)
-   Document Parsing: PyPDF, DOCX2TXT
-   RAG Utilities: LangChain helpers
-   Hosting: Streamlit Cloud

------------------------------------------------------------------------

## 🏗 Architecture

User → Streamlit UI → RAG Pipeline → FAISS → LLM → Answer + Sources +
Confidence

------------------------------------------------------------------------

## 📂 Project Structure

ultra-doc-intelligence-demo/ backend/ rag_pipeline.py extractor.py
llm.py ui/ app.py requirements.txt README.md

------------------------------------------------------------------------

## 📥 Supported Document Types

-   PDF
-   DOCX
-   TXT

------------------------------------------------------------------------

## ⚙️ Core Features

### Document Upload & Processing

System performs: - Text Parsing\
- Intelligent Chunking\
- Embedding Generation\
- Vector Storage (FAISS)

------------------------------------------------------------------------

### Ask Questions (RAG)

System returns: - Answer (Grounded) - Source Text Metadata - Confidence
Score

------------------------------------------------------------------------

### Guardrails

-   Retrieval presence guard → "Not found in document"
-   Similarity-based confidence threshold
-   Context-only LLM prompting

------------------------------------------------------------------------

### Confidence Scoring

confidence = mean( 1 / (1 + distance) )

Higher similarity → Higher confidence

------------------------------------------------------------------------

### Structured Shipment Extraction

Fields: shipment_id, shipper, consignee, pickup_datetime,
delivery_datetime, equipment_type, mode, rate, currency, weight,
carrier_name

Returns JSON with null if missing.

------------------------------------------------------------------------

## 🧩 Chunking Strategy

chunk_size = 1000\
chunk_overlap = 200

------------------------------------------------------------------------

## 🔍 Retrieval Method

Embedding Model: all-MiniLM-L6-v2\
Vector Search: FAISS IndexFlatL2\
Top-K Retrieval: 3

------------------------------------------------------------------------

## ⚠ Known Failure Cases

-   Poor scan quality PDFs
-   Extremely large documents
-   Heavy tables/images
-   Ambiguous shipment references

------------------------------------------------------------------------

## 🚀 Future Improvements

-   Hybrid Search (BM25 + Embeddings)
-   Table-aware parsing
-   Multi-document querying
-   Async processing
-   Streaming responses
-   Advanced confidence scoring

------------------------------------------------------------------------

## 🛠 Run Locally

pip install -r requirements.txt\
streamlit run ui/app.py

------------------------------------------------------------------------

## ☁ Deployment

Hosted on Streamlit Cloud\
Requires GitHub repo + Streamlit secrets setup

------------------------------------------------------------------------

## ❤️ Engineering Philosophy

Focus on grounded answers, reliability, guardrails, and modular design
for real-world AI deployment.
