# 🚚 Doc-Intelligence

> **End-to-end RAG-based AI system for logistics document understanding
> with guardrails, confidence scoring, and structured data extraction.**

------------------------------------------------------------------------

## 🌟 Project Overview

**Doc-Intelligence** is a Proof-of-Concept AI assistant designed for
**Transportation Management Systems (TMS)**.

It enables users to:

-   Upload logistics documents\
-   Ask natural language questions\
-   Retrieve grounded answers using RAG\
-   Prevent hallucinations using guardrails\
-   Generate response confidence scores\
-   Extract structured shipment data

This project demonstrates **real-world AI engineering practices**
including retrieval grounding, production-style guardrails, and
confidence modeling.

------------------------------------------------------------------------

## 🎯 System Objective

### Primary Goal

Build an AI system that allows users to upload logistics documents and
interact with them using natural language queries.

### Core Capabilities

✅ Grounded document retrieval using RAG\
✅ Hallucination prevention using guardrails\
✅ Confidence scoring per response\
✅ Structured shipment data extraction

------------------------------------------------------------------------

## 🏗️ Architecture

### Design Pattern

**Retrieval Augmented Generation (RAG)**

### End-to-End Pipeline

    Document Upload
       ↓
    Text Parsing & Cleaning
       ↓
    Intelligent Chunking
       ↓
    Embedding Generation
       ↓
    Vector Index Storage (FAISS)
       ↓
    Query Embedding
       ↓
    Top-K Semantic Retrieval
       ↓
    Context Assembly
       ↓
    LLM Grounded Answer Generation
       ↓
    Guardrail Enforcement
       ↓
    Confidence Score Calculation
       ↓
    Structured / Natural Response Delivery

------------------------------------------------------------------------

## 📂 Project Structure

    Doc-Intelligence/
    │
    ├ backend/
    │   ├ rag_pipeline.py
    │   ├ extractor.py
    │   ├ llm.py
    │
    ├ ui/
    │   ├ app.py
    │
    ├ main.py
    ├ requirements.txt
    ├ .gitignore
    ├ .env.example
    ├ README.md
    │
    ├ vector_store/   (Ignored - FAISS index storage)
    ├ uploads/        (Ignored - Uploaded docs temp storage)

------------------------------------------------------------------------

## 📄 Document Processing

### Supported Formats

-   PDF\
-   DOCX\
-   TXT

### Supported Logistics Documents

-   Rate Confirmation\
-   Bill of Lading (BOL)\
-   Shipment Instructions\
-   Invoice

### Parsing Loaders

  Format   Loader
  -------- ----------------
  PDF      PyPDFLoader
  DOCX     Docx2txtLoader
  TXT      TextLoader

------------------------------------------------------------------------

## 🧹 Text Preprocessing

-   Whitespace normalization\
-   Broken token merging\
-   Currency normalization (USD formatting)\
-   Spacing artifact cleanup

------------------------------------------------------------------------

## ✂️ Chunking Strategy

  Parameter       Value
  --------------- ------------------------------------
  Algorithm       Recursive Character Text Splitting
  Chunk Size      1000
  Chunk Overlap   200

**Why?**\
Logistics data is distributed across headers, tables, and paragraphs.\
Larger chunks preserve shipment context while overlap maintains
continuity.

------------------------------------------------------------------------

## 🧠 Embedding Layer

  Component   Value
  ----------- -----------------------
  Model       all-MiniLM-L6-v2
  Framework   Sentence Transformers

**Reason for Selection** - Fast inference\
- Strong semantic retrieval quality\
- Production-friendly latency

------------------------------------------------------------------------

## 🗄️ Vector Index

  Component    Value
  ------------ -------------
  Engine       FAISS
  Index Type   IndexFlatL2

### Stored Artifacts

    vector_store/index.faiss
    vector_store/docs.pkl

------------------------------------------------------------------------

## 🔍 Retrieval Layer

### Retrieval Strategy

Dense Vector Semantic Retrieval

### Retrieval Modes

  Mode              Top-K
  ----------------- -------
  QA Mode           3
  Extraction Mode   5

------------------------------------------------------------------------

## 🤖 Question Answering

  Component    Value
  ------------ ------------------
  LLM Access   OpenRouter
  Model        Mixtral Instruct

### Grounding Rule

LLM answers **only using retrieved document context**.

### Response Format

-   Answer Text\
-   Source Metadata\
-   Confidence Score

------------------------------------------------------------------------

## 🛡️ Guardrails

### Hallucination Prevention Methods

-   Confidence threshold enforcement\
-   Context presence validation\
-   Retrieval similarity grounding\
-   JSON schema enforcement

### Guardrail Behaviors

  Scenario          Action
  ----------------- --------------------------------
  Low Confidence    Return answer + warning
  Missing Context   Return "Not found in document"
  JSON Failure      Return null-safe schema

------------------------------------------------------------------------

## 📊 Confidence Scoring

### Base Signal

Retrieval Similarity

### Computation Pipeline

1.  Compute FAISS L2 distances\
2.  Normalize → similarity scores\
3.  Compute mean similarity\
4.  Apply stability penalty using standard deviation

### Formula

    confidence = mean(normalized_similarity_scores) *
                 (1 - std(normalized_similarity_scores))

### Output Range

`0.0 → 1.0`

------------------------------------------------------------------------

## 📦 Structured Shipment Extraction

### Method

Retrieval-Grounded LLM Extraction

### Output Format

Strict JSON Only

### Required Fields

-   shipment_id\
-   shipper\
-   consignee\
-   pickup_datetime\
-   delivery_datetime\
-   equipment_type\
-   mode\
-   rate\
-   currency\
-   weight\
-   carrier_name

### Fallback Rules

-   Missing Field → `null`\
-   Invalid JSON → Return null-safe schema

------------------------------------------------------------------------

## 🔌 API Specification

### Endpoints

#### 📤 Upload Document

    POST /upload

#### ❓ Ask Question

    POST /ask

#### 📦 Extract Shipment Data

    POST /extract

------------------------------------------------------------------------

## 🖥️ UI Layer

**Framework:** Streamlit

### Reviewer Capabilities

-   Upload document\
-   Ask natural language questions\
-   View answers + sources + confidence\
-   Trigger structured extraction

------------------------------------------------------------------------

## ⚙️ Local Deployment

### 1️⃣ Install Dependencies

``` bash
pip install -r requirements.txt
```

### 2️⃣ Configure Environment

Create `.env`

    OPENROUTER_API_KEY=your_key_here

### 3️⃣ Start Backend

``` bash
uvicorn main:app --reload
```

### 4️⃣ Start UI

``` bash
streamlit run ui/app.py
```

------------------------------------------------------------------------

## ⚠️ Known Limitations

-   Scanned PDFs without OCR layer\
-   Table-heavy documents\
-   Very short documents\
-   Multi-document ambiguity\
-   Ambiguous shipment naming

------------------------------------------------------------------------

## 🚀 Future Improvements

### Short Term

-   Hybrid retrieval (BM25 + Vector)\
-   Reranking models\
-   Table-aware extraction\
-   Metadata weighted retrieval

### Long Term

-   Multi-document session memory\
-   Layout-aware embeddings\
-   Automated evaluation benchmarks\
-   Hallucination detection scoring

------------------------------------------------------------------------

## 📈 Evaluation Alignment

  Area                  Implementation
  --------------------- ---------------------------------------------------
  Retrieval Grounding   FAISS + Context-only answering
  Extraction Accuracy   Retrieval-grounded JSON prompting
  Guardrails            Threshold + Context validation + JSON enforcement
  Confidence Scoring    Similarity + Stability penalty
  Code Structure        Modular backend architecture
  Usability             End-to-end Streamlit workflow

------------------------------------------------------------------------

## 👩‍💻 Author Statement

This project demonstrates practical AI engineering capability including
real-world RAG system architecture, production-grade guardrails,
confidence modeling, and applied document intelligence pipeline design.
