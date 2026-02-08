import json
from backend.rag_pipeline import retrieve_context
from backend.llm import ask_llm

FIELDS = [
    "shipment_id","shipper","consignee",
    "pickup_datetime","delivery_datetime",
    "equipment_type","mode","rate",
    "currency","weight","carrier_name"
]

def extract_shipment_data():

    context, sources, confidence = retrieve_context(
        "shipment logistics bill of lading rate carrier shipper consignee weight pickup delivery"
    )

    if not context:
        return {f: None for f in FIELDS}

    prompt = f"""
Extract shipment data.

Return ONLY JSON with keys:
{FIELDS}

Document:
{context}
"""

    response = ask_llm(prompt)

    try:
        return json.loads(response)
    except:
        return {f: None for f in FIELDS}
