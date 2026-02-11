from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from pydantic import BaseModel
from supabase import create_client, Client
import os
import ollama
import json
import re
import tempfile
import fitz  # PyMuPDF
from datetime import datetime
import chromadb 
from chromadb.config import Settings

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

    Return only one valid JSON object with these exact keys.
    Ignore any extra fields not listed above.
    """

    response = ollama.chat(
        model="llama3.2-vision",  # vision-capable model
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

    if "Invoice_id" in data:
        data["Invoice_id"] = str(data["Invoice_id"])

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

    Return only one valid JSON object with these exact keys.
    Ignore any extra fields not listed above.
    """

    response = ollama.chat(
        model="llama3.2",  # text-only model
        format="json",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response["message"]["content"]

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise HTTPException(status_code=500, detail="No JSON found in LLM response")

    data = json.loads(match.group(0))

    if "Invoice_id" in data:
        data["Invoice_id"] = str(data["Invoice_id"])

    if "Invoice_date" in data:
        data["Invoice_date"] = normalize_date(str(data["Invoice_date"]))

    if "List_of_items" in data and isinstance(data["List_of_items"], list):
        data["List_of_items"] = deduplicate_items(data["List_of_items"])

    allowed_keys = {"Invoice_id", "Invoice_date", "Vendor_name", "List_of_items"}
    return {k: v for k, v in data.items() if k in allowed_keys}

# --- ChromaDB Setup ---

chroma_client = chromadb.Client(Settings(persist_directory="./chroma_db"))
collection = chroma_client.get_or_create_collection(name="invoices")

def embed_text(text: str):
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]

# --- Create embeddings ---
def add_invoice_to_chroma(invoice: Invoice):
    # Convert invoice into a text string
    text = f"Invoice {invoice.Invoice_id} from {invoice.Vendor_name} on {invoice.Invoice_date}. Items: " + \
           ", ".join([f"{item.qty} x {item.name}" for item in invoice.List_of_items])

    # Generate embedding
    embedding = embed_text(text)

    # Store in ChromaDB
    collection.add(
        ids=[invoice.Invoice_id],
        embeddings=[embedding],
        metadatas=[{
            "vendor_name": invoice.Vendor_name,
            "invoice_date": invoice.Invoice_date,
            "items": [item.model_dump() for item in invoice.List_of_items]
        }],
        documents=[json.dumps(invoice.model_dump())]
    )

# --- Unified Endpoint ---
@app.post("/invoices")
async def create_invoice(request: Request, file: UploadFile = File(None), merge: bool = False):
    try:
        normalized = {}
        all_items = []

        # --- File upload handling ---
        if file:
            suffix = os.path.splitext(file.filename)[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(await file.read())
                tmp_path = tmp_file.name

            if suffix == ".pdf":
                doc = fitz.open(tmp_path)
                image_paths = []
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    pix = page.get_pixmap()
                    image_path = tmp_path.replace(".pdf", f"_{page_num+1}.png")
                    pix.save(image_path)
                    image_paths.append(image_path)

                for img_path in image_paths:
                    extracted = extract_invoice_from_image(img_path)
                    if not normalized:
                        normalized.update({
                            "Invoice_id": extracted.get("Invoice_id"),
                            "Invoice_date": extracted.get("Invoice_date"),
                            "Vendor_name": extracted.get("Vendor_name")
                        })
                    if "List_of_items" in extracted:
                        all_items.extend(extracted["List_of_items"])
            else:
                image_path = tmp_path
                extracted = extract_invoice_from_image(image_path)
                normalized.update(extracted)
                if "List_of_items" in extracted:
                    all_items.extend(extracted["List_of_items"])

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
            add_invoice_to_chroma(invoice)
            return {"message": "Invoice merged successfully", "data": response.data}
            

        # --- Insert new invoice ---
        data = {
            "invoice_id": invoice.Invoice_id,
            "invoice_date": invoice.Invoice_date,
            "vendor_name": invoice.Vendor_name,
            "list_of_items": [item.model_dump() for item in invoice.List_of_items]
        }
        add_invoice_to_chroma(invoice)
        response = supabase.table("invoices").insert(data).execute()
        

        if not response.data:
            raise HTTPException(status_code=400, detail="Insert failed")

        return {"message": "Invoice inserted successfully", "data": response.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# --- New Endpoint: Read from ChromaDB ---
@app.post("/invoices/vector")
async def search_invoices(request: Request):
    try:
        body = await request.json()
        query = body.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Missing 'query' in request body")

        # Generate embedding for query
        embedding = embed_text(query)

        # Query ChromaDB
        results = collection.query(query_embeddings=[embedding], n_results=5)

        return {
            "message": "Search completed",
            "query": query,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))