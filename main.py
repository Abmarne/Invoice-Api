from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from pydantic import BaseModel
from supabase import create_client, Client
import os
import ollama
import json
import re
from pdf2image import convert_from_path
import tempfile

# --- Environment Variables ---
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ajcjqjkrpgdkvpkppjyj.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_DEffvKR_hgpRYRMl4Dj5sg_eV4NmQvL")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI()

# --- Canonical Schema ---
class Item(BaseModel):
    name: str
    qty: int

class Invoice(BaseModel):
    Invoice_id: str
    Invoice_date: str
    Vendor_name: str
    List_of_items: list[Item]

# --- Helper: Deduplicate & Merge Quantities ---
def deduplicate_items(items):
    merged = {}
    for item in items:
        name = item.get("name") if isinstance(item, dict) else str(item)
        qty = item.get("qty", 1) if isinstance(item, dict) else 1
        if name in merged:
            merged[name] += qty
        else:
            merged[name] = qty
    return [{"name": n, "qty": q} for n, q in merged.items()]

# --- LLM Extraction from Image ---
def extract_invoice_from_image(image_path: str) -> dict:
    prompt = """
    Extract the following details from this invoice image and return them as a JSON object:
    - Invoice_id
    - Invoice_date
    - Vendor_name
    - List_of_items (each item should have 'name' and 'qty')

    Return only one valid JSON object with these exact keys.
    Ignore any extra fields not listed above.
    """

    response = ollama.chat(
        model="llama3.2-vision",
        format="json",
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_path]
        }]
    )

    raw = response["message"]["content"]

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise HTTPException(status_code=500, detail="No JSON found in LLM response")

    data = json.loads(match.group(0))

    # Ensure Invoice_id is always a string
    if "Invoice_id" in data:
        data["Invoice_id"] = str(data["Invoice_id"])

    if "List_of_items" in data and isinstance(data["List_of_items"], list):
        data["List_of_items"] = deduplicate_items(data["List_of_items"])

    allowed_keys = {"Invoice_id", "Invoice_date", "Vendor_name", "List_of_items"}
    return {k: v for k, v in data.items() if k in allowed_keys}

# --- LLM Normalization for JSON ---
def normalize_with_llm(body: dict) -> dict:
    prompt = f"""
    Map the following JSON keys to these canonical fields:
    - Invoice_id
    - Invoice_date
    - Vendor_name
    - List_of_items (each item should have 'name' and 'qty')

    Input JSON: {body}

    Return only one valid JSON object with these exact keys.
    Ignore any extra fields not listed above.
    """

    response = ollama.chat(
        model="llama3.2",
        format="json",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response["message"]["content"]

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise HTTPException(status_code=500, detail="No JSON found in LLM response")

    data = json.loads(match.group(0))

    # Ensure Invoice_id is always a string
    if "Invoice_id" in data:
        data["Invoice_id"] = str(data["Invoice_id"])

    if "List_of_items" in data and isinstance(data["List_of_items"], list):
        data["List_of_items"] = deduplicate_items(data["List_of_items"])

    allowed_keys = {"Invoice_id", "Invoice_date", "Vendor_name", "List_of_items"}
    return {k: v for k, v in data.items() if k in allowed_keys}

# --- Unified Endpoint ---
@app.post("/invoices")
async def create_invoice(request: Request, file: UploadFile = File(None)):
    try:
        # Case 1: File upload (PDF or image)
        if file:
            suffix = os.path.splitext(file.filename)[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(await file.read())
                tmp_path = tmp_file.name

            if suffix == ".pdf":
                images = convert_from_path(tmp_path)
                image_path = tmp_path.replace(".pdf", ".png")
                images[0].save(image_path, "PNG")
            else:
                image_path = tmp_path

            normalized = extract_invoice_from_image(image_path)

        # Case 2: JSON body
        else:
            body = await request.json()
            normalized = normalize_with_llm(body)

        # Validate against Pydantic model
        invoice = Invoice(**normalized)

        # Check if Invoice_id already exists
        existing = supabase.table("invoices").select("invoice_id").eq("invoice_id", invoice.Invoice_id).execute()
        if existing.data:
            raise HTTPException(status_code=400, detail=f"Invoice_id {invoice.Invoice_id} already exists")

        # Insert into Supabase
        data = {
            "invoice_id": invoice.Invoice_id,
            "invoice_date": invoice.Invoice_date,
            "vendor_name": invoice.Vendor_name,
            "list_of_items": [item.model_dump() for item in invoice.List_of_items]
        }
        response = supabase.table("invoices").insert(data).execute()

        if not response.data:
            raise HTTPException(status_code=400, detail="Insert failed")

        return {"message": "Invoice inserted successfully", "data": response.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))