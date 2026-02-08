import json
from backend.rag_pipeline import retrieve_context
from backend.llm import ask_llm


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

    # 🔥 Better retrieval query (KEY FIX)
    context, sources, confidence = retrieve_context(
        "shipment id shipper consignee pickup delivery carrier rate currency weight equipment mode logistics bill of lading rate confirmation shipment details",
        k=5
    )

    # 🔥 Context Guardrail
    if not context or len(context.strip()) < 50:
        return {field: None for field in FIELDS}

    prompt = f"""
You are a logistics document data extractor.

STRICT RULES:
- Output ONLY valid JSON
- No explanation
- No markdown
- No text before or after JSON
- If field missing → null

JSON Schema:
{FIELDS}

DOCUMENT:
{context}
"""

    response = ask_llm(prompt)

    try:
        data = json.loads(response)
    except:
        data = {field: None for field in FIELDS}

    return data
