from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from pydantic import BaseModel
from supabase import create_client, Client
import os
import ollama
import json
import re
import tempfile
import fitz  # PyMuPDF
import logging
from datetime import datetime
import chromadb
from chromadb.config import Settings

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    Invoice_id: str | None = None
    Invoice_date: str | None = None
    Vendor_name: str | None = None
    List_of_items: list[Item] = []

# --- ChromaDB Setup ---
chroma_client = chromadb.PersistentClient(path="./chroma_storage")
invoice_collection = chroma_client.get_or_create_collection(name="invoices")

# --- Helper: Deduplicate & Merge Quantities ---
def deduplicate_items(items):
    merged = {}
    for item in items:
        name = item.get("name") if isinstance(item, dict) else str(item)
        try:
            qty = int(item.get("qty", 1)) if isinstance(item, dict) else 1
        except Exception:
            qty = 1
        merged[name] = merged.get(name, 0) + qty
    return [{"name": n, "qty": q} for n, q in merged.items()]

# --- Helper: Normalize Date ---
def normalize_date(date_str: str) -> str:
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d %b %Y", "%d %B %Y"):
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str

# --- LLM Extraction from Image ---
def extract_invoice_from_image(image_path: str) -> dict:
    prompt = """
    Extract the following details from this invoice image and return them as a JSON object:
    - Invoice_id
    - Invoice_date
    - Vendor_name
    - List_of_items (each item should have 'name' and 'qty')

    Return ONLY one valid JSON object with these exact keys and casing:
    {
      "Invoice_id": "...",
      "Invoice_date": "...",
      "Vendor_name": "...",
      "List_of_items": [{"name": "...", "qty": ...}]
    }
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
    logger.info(f"Raw LLM output (image): {raw}")

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise HTTPException(status_code=500, detail="No JSON found in LLM response")

    data = json.loads(match.group(0))

    if "Invoice_date" in data:
        data["Invoice_date"] = normalize_date(str(data["Invoice_date"]))
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

    Return ONLY one valid JSON object with these exact keys and casing:
    {{
      "Invoice_id": "...",
      "Invoice_date": "...",
      "Vendor_name": "...",
      "List_of_items": [{{"name": "...", "qty": ...}}]
    }}

    Do not invent or rename keys. Do not output explanations.
    """

    response = ollama.chat(
        model="llama3.2",
        format="json",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response["message"]["content"]
    logger.info(f"Raw LLM output (JSON): {raw}")

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise HTTPException(status_code=500, detail="No JSON found in LLM response")

    data = json.loads(match.group(0))

    if "Invoice_date" in data:
        data["Invoice_date"] = normalize_date(str(data["Invoice_date"]))
    if "List_of_items" in data and isinstance(data["List_of_items"], list):
        data["List_of_items"] = deduplicate_items(data["List_of_items"])

    allowed_keys = {"Invoice_id", "Invoice_date", "Vendor_name", "List_of_items"}
    return {k: v for k, v in data.items() if k in allowed_keys}

# --- File Processing ---
def process_file(file: UploadFile) -> dict:
    suffix = os.path.splitext(file.filename)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file.file.read())
        tmp_path = tmp_file.name

    normalized = {}
    all_items = []

    if suffix == ".pdf":
        doc = fitz.open(tmp_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap()
            image_path = tmp_path.replace(".pdf", f"_{page_num+1}.png")
            pix.save(image_path)
            extracted = extract_invoice_from_image(image_path)
            if not normalized:
                normalized.update({
                    "Invoice_id": extracted.get("Invoice_id"),
                    "Invoice_date": extracted.get("Invoice_date"),
                    "Vendor_name": extracted.get("Vendor_name")
                })
            if "List_of_items" in extracted:
                all_items.extend(extracted["List_of_items"])
        doc.close()
    else:
        extracted = extract_invoice_from_image(tmp_path)
        normalized.update(extracted)
        if "List_of_items" in extracted:
            all_items.extend(extracted["List_of_items"])

    normalized["List_of_items"] = deduplicate_items(all_items)
    return normalized

# --- Helper: Store in ChromaDB ---
def store_in_chromadb(invoice: Invoice):
    text_repr = f"Invoice {invoice.Invoice_id} from {invoice.Vendor_name} on {invoice.Invoice_date}. Items: " + \
                ", ".join([f"{item.name} (qty {item.qty})" for item in invoice.List_of_items])

    # Generate embedding
    embedding_response = ollama.embeddings(model="llama3.2", prompt=text_repr)
    embedding = embedding_response["embedding"]

    # Flatten items into strings for metadata
    item_names = [item.name for item in invoice.List_of_items]
    item_qtys = [item.qty for item in invoice.List_of_items]

    invoice_collection.add(
        ids=[invoice.Invoice_id],
        embeddings=[embedding],
        metadatas=[{
            "vendor_name": invoice.Vendor_name,
            "invoice_date": invoice.Invoice_date,
            "items": item_names,   # list of strings ✅ valid
            "quantities": item_qtys  # list of ints ✅ valid
        }],
        documents=[text_repr]
    )
    logger.info(f"Stored invoice {invoice.Invoice_id} in ChromaDB")
    
# --- Unified Endpoint ---

@app.get("/invoices/debug")
def debug_invoices():
    # Retrieve everything stored in the collection
    all_invoices = invoice_collection.get()
    return {
        "ids": all_invoices.get("ids", []),
        # "metadatas": all_invoices.get("metadatas", []),
        # "documents": all_invoices.get("documents", []),
        # "embeddings": all_invoices.get("embeddings", [])
    }
@app.post("/invoices")
async def create_invoice(request: Request, file: UploadFile = File(None), merge: bool = False):
    try:
        normalized = {}
        all_items = []

        # --- File upload handling ---
        if file:
            file_data = process_file(file)
            normalized.update({k: v for k, v in file_data.items() if k != "List_of_items"})
            all_items.extend(file_data.get("List_of_items", []))

        # --- JSON body handling ---
        try:
            body = await request.json()
            json_normalized = normalize_with_llm(body)
            for key in ["Invoice_id", "Invoice_date", "Vendor_name"]:
                if json_normalized.get(key):
                    normalized[key] = json_normalized[key]
            if "List_of_items" in json_normalized:
                all_items.extend(json_normalized["List_of_items"])
        except Exception:
            pass

        normalized["List_of_items"] = deduplicate_items(all_items)

        # --- Validation ---
        required_fields = ["Invoice_id", "Invoice_date", "Vendor_name"]
        for field in required_fields:
            if not normalized.get(field):
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}. Cannot insert into database."
                )

        invoice = Invoice(**normalized)

        # --- Duplicate handling ---
        existing = supabase.table("invoices").select("*").eq("invoice_id", invoice.Invoice_id).execute()
        if existing.data:
            existing_vendor = existing.data[0]["vendor_name"]
            if existing_vendor != invoice.Vendor_name:
                raise HTTPException(status_code=400, detail="Invoice_id already exists for a different vendor. Strict mode: no duplicates allowed.")

            if not merge:
                return {"message": f"Invoice_id {invoice.Invoice_id} already exists for vendor {invoice.Vendor_name}. No action taken."}

            existing_items = existing.data[0]["list_of_items"]
            merged_items = deduplicate_items(existing_items + [item.model_dump() for item in invoice.List_of_items])
            update_data = {"list_of_items": merged_items}
            response = supabase.table("invoices").update(update_data).eq("invoice_id", invoice.Invoice_id).execute()

            # --- Store merged invoice in ChromaDB ---
            store_in_chromadb(invoice)

            return {"message": "Invoice merged successfully", "data": response.data}

        # --- Insert new invoice ---
        data = {
            "invoice_id": invoice.Invoice_id,
            "invoice_date": invoice.Invoice_date,
            "vendor_name": invoice.Vendor_name,
            "list_of_items": [item.model_dump() for item in invoice.List_of_items]
        }
        response = supabase.table("invoices").insert(data).execute()

        if not response.data:
            raise HTTPException(status_code=400, detail="Insert failed")

        # --- Store new invoice in ChromaDB ---
        store_in_chromadb(invoice)

        return {"message": "Invoice inserted successfully", "data": response.data}

    except Exception as e:
        logger.error(f"Error processing invoice: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    