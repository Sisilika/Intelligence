from backend.rag_pipeline import load_document, chunk_documents, create_vector_store
from backend.llm import ask_llm
from backend.rag_pipeline import retrieve_context
from backend.extractor import extract_shipment_data

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import os

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.35


@app.get("/")
def root():
    return {"message": "Ultra Doc Intelligence API Running"}


# =========================
# UPLOAD ENDPOINT
# =========================
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        documents = load_document(file_path)
        chunks = chunk_documents(documents)
        create_vector_store(chunks)

        return {
            "status": "uploaded + indexed",
            "file": file.filename
        }

    except Exception as e:
        return {"error": str(e)}


# =========================
# ASK ENDPOINT
# =========================
class AskRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(req: AskRequest):
    try:
        question = req.question

        context, sources, confidence = retrieve_context(question)

        if not context:
            return {
                "answer": "Not found in document",
                "sources": [],
                "confidence": 0.0
            }

        prompt = f"""
Answer ONLY using this document context.

Context:
{context}

Question:
{question}
"""

        answer = ask_llm(prompt)

        if confidence < CONFIDENCE_THRESHOLD:
            answer = f"(Low confidence — verify manually)\n\n{answer}"

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}


# =========================
# EXTRACTION ENDPOINT
# =========================
@app.post("/extract")
async def extract_data():
    try:
        data = extract_shipment_data()
        return data
    except Exception as e:
        return {"error": str(e)}
