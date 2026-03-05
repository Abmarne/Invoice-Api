
# Invoice Processing API

A FastAPI-based application for extracting, normalizing, deduplicating, and storing invoice data from images, PDFs, or JSON payloads using LLMs (Ollama), Supabase, and ChromaDB for vector search.

**Author:** Abhiraj Marne  
**All Rights Reserved © 2026**

---

## Features

- **Multi-format Support:** Process invoices from PDF files, images, or JSON data
- **AI-Powered Extraction:** Uses Ollama's `llama3.2-vision` model to extract invoice details from images and PDFs
- **LLM-Powered Normalization:** All incoming data (file or JSON) is mapped to a canonical schema using Ollama LLM
- **Deduplication & Quantity Merging:** Automatically merges duplicate items and sums their quantities
- **Supabase Storage:** Stores processed invoices in Supabase (PostgreSQL)
- **ChromaDB Vector Storage:** Stores invoice embeddings in ChromaDB for semantic search and retrieval
- **Validation:** Pydantic-based validation ensures data integrity
- **Duplicate Handling & Merge:** Prevents duplicate invoice IDs for different vendors; supports merging items for the same invoice/vendor

---

## Requirements

- Python 3.8+
- FastAPI
- Pydantic
- Supabase client
- Ollama (with `llama3.2-vision` and `llama3.2` models)
- pymupdf
- python-multipart
- chromadb
- uvicorn

---

## Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   cd /Invoice-Api
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables (recommended):
   ```bash
   set SUPABASE_URL=your_supabase_url
   set SUPABASE_KEY=your_supabase_api_key
   ```

---

## Configuration

If not using environment variables, update the following in `main.py`:
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase API key

---

## API Endpoints

### POST `/invoices`

Process and store an invoice. Handles both file uploads (PDF/image) and JSON payloads. Deduplicates items and merges quantities. Prevents duplicate invoice IDs for different vendors. Supports merging items for the same invoice/vendor if `merge=true` is set.

**Option 1: File Upload (PDF or Image)**
```bash
curl -X POST http://localhost:8000/invoices \
  -F "file=@invoice.pdf"
```

**Option 2: JSON Payload**
```bash
curl -X POST http://localhost:8000/invoices \
  -H "Content-Type: application/json" \
  -d '{
    "Invoice_id": "INV001",
    "Invoice_date": "2024-01-15",
    "Vendor_name": "ACME Corp",
    "List_of_items": [
      {"name": "Widget", "qty": 5},
      {"name": "Gadget", "qty": 3}
    ]
  }'
```

**Merge Items for Existing Invoice:**
```bash
curl -X POST "http://localhost:8000/invoices?merge=true" \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

**Response:**
```json
{
  "message": "Invoice inserted successfully",
  "data": [
    {
      "invoice_id": "INV001",
      "invoice_date": "2024-01-15",
      "vendor_name": "ACME Corp",
      "list_of_items": [...]
    }
  ]
}
```

### GET `/invoices/debug`

Returns all invoice IDs currently stored in ChromaDB (for debugging vector storage).

---

## Data Schema

### Canonical Invoice Structure
```json
{
  "Invoice_id": "string (unique)",
  "Invoice_date": "string (YYYY-MM-DD)",
  "Vendor_name": "string",
  "List_of_items": [
    {
      "name": "string",
      "qty": "integer"
    }
  ]
}
```

---

## Running the Application

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

---

## Error Handling

- **400**: Invoice_id already exists for a different vendor, or invalid input
- **500**: LLM extraction failure, ChromaDB/embedding error, or database insert error

---

## Database Setup

Ensure your Supabase `invoices` table has the following schema:
- `invoice_id` (text, primary key)
- `invoice_date` (text)
- `vendor_name` (text)
- `list_of_items` (jsonb)

ChromaDB will automatically create its own vector storage in the `chroma_storage/` directory.

---

## ChromaDB Vector Search

Each invoice is embedded using Ollama and stored in ChromaDB for semantic search and retrieval. You can extend the API to add search endpoints using ChromaDB's similarity search features.

---

## License

This project is provided as-is for invoice processing automation.
