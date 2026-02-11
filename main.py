from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from pydantic import BaseModel
from supabase import create_client, Client
import os
import ollama
import json
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

class QueryRequest(BaseModel):
    query: str

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

# --- ChromaDB Setup ---
chroma_client = chromadb.Client(Settings(persist_directory="./chroma_db"))
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="invoices")

def embed_text(text: str):
    try:
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response["embedding"]
    except Exception as e:
        print("Embedding failed:", e)
        return None

# --- Create embeddings ---
def add_invoice_to_chroma(invoice: Invoice):
    text = f"Invoice {invoice.Invoice_id} from {invoice.Vendor_name} on {invoice.Invoice_date}. Items: " + \
           ", ".join([f"{item.qty} x {item.name}" for item in invoice.List_of_items])

    embedding = embed_text(text)
    if embedding is None:
        print("Skipping ChromaDB insert because embedding failed.")
        return

    collection.add(
        ids=[invoice.Invoice_id],
        embeddings=[embedding],
        metadatas=[{
            "vendor_name": invoice.Vendor_name,
            "invoice_date": invoice.Invoice_date,
            "items": ", ".join([f"{item.qty} x {item.name}" for item in invoice.List_of_items])
        }],
        documents=[json.dumps(invoice.model_dump())]
    )
    print("Inserted into ChromaDB:", invoice.Invoice_id)

# --- Insert Endpoint ---
@app.post("/invoices")
async def create_invoice(request: Request, file: UploadFile = File(None), merge: bool = False):
    try:
        normalized = {}
        all_items = []

        # --- JSON body handling ---
        try:
            body = await request.json()
            normalized.update(body)
            if "List_of_items" in body:
                all_items.extend(body["List_of_items"])
        except Exception:
            pass

        # Ensure required fields exist
        required_keys = ["Invoice_id", "Invoice_date", "Vendor_name"]
        for key in required_keys:
            if key not in normalized:
                raise HTTPException(status_code=400, detail=f"Missing required field: {key}")

        normalized["Invoice_date"] = normalize_date(normalized["Invoice_date"])
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

            # Merge items
            existing_items = existing.data[0]["list_of_items"]
            merged_items = deduplicate_items(existing_items + [item.model_dump() for item in invoice.List_of_items])
            update_data = {"list_of_items": merged_items}
            response = supabase.table("invoices").update(update_data).eq("invoice_id", invoice.Invoice_id).execute()

            # Update ChromaDB with merged invoice
            invoice.List_of_items = [Item(**item) for item in merged_items]
            add_invoice_to_chroma(invoice)

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

        # Add to ChromaDB
        add_invoice_to_chroma(invoice)

        return {"message": "Invoice inserted successfully", "data": response.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
# --- New Endpoint: Read from ChromaDB ---
# --- Vector + Filter Search Endpoint ---
@app.post("/invoices/vector")
async def search_invoices(query_request: QueryRequest):
    try:
        # Build metadata filter dynamically
        filters = {}
        if query_request.vendor_name:
            filters["vendor_name"] = query_request.vendor_name
        if query_request.invoice_date:
            filters["invoice_date"] = query_request.invoice_date
        if query_request.item:
            filters["items"] = {"$contains": query_request.item}

        # Handle semantic query
        query_embedding = None
        if query_request.query:
            query_embedding = embed_text(query_request.query)
            if query_embedding is None:
                raise HTTPException(status_code=500, detail="Embedding failed for query")

        # Perform search
        if query_embedding:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                where=filters if filters else None,
                include=["metadatas", "documents", "distances"]
            )
        else:
            # If no semantic query, just filter by metadata
            results = collection.get(
                where=filters if filters else None,
                include=["metadatas", "documents"]
            )

        return {
            "message": "Search completed",
            "query": query_request.query,
            "filters": filters,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))