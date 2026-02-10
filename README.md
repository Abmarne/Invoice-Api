# Invoice Processing API

A FastAPI-based application that extracts and processes invoice data from images, PDFs, or JSON payloads using LLM (Ollama) and stores the data in Supabase.

**Author:** Abhiraj Marne  
**All Rights Reserved © 2026**

## Features

- **Multi-format Support**: Process invoices from PDF files, images, or JSON data
- **AI-Powered Extraction**: Uses Ollama's `llama3.2-vision` model to extract invoice details from images
- **Automatic Normalization**: Normalizes incoming data to a canonical schema using LLM
- **Deduplication**: Automatically merges duplicate items and sums quantities
- **Database Storage**: Stores processed invoices in Supabase
- **Validation**: Pydantic-based validation ensures data integrity

## Requirements

- Python 3.8+
- FastAPI
- Pydantic
- Supabase client
- Ollama (with `llama3.2-vision` and `llama3.2` models)
- pymupdf
- python-multipart

## Installation

1. Clone the repository and navigate to the project directory:
```bash
cd /Invoice-Api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables:
```bash
set SUPABASE_URL=your_supabase_url
set SUPABASE_KEY=your_supabase_api_key
```

## Configuration

Update the following in `main.py` if not using environment variables:
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase API key

## API Endpoints

### POST `/invoices`

Process and store an invoice.

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
    "invoice_id": "INV001",
    "invoice_date": "2024-01-15",
    "vendor_name": "ACME Corp",
    "list_of_items": [
      {"name": "Widget", "qty": 5},
      {"name": "Gadget", "qty": 3}
    ]
  }'
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

## Data Schema

### Invoice Structure
```json
{
  "Invoice_id": "string (unique)",
  "Invoice_date": "string (date format)",
  "Vendor_name": "string",
  "List_of_items": [
    {
      "name": "string",
      "qty": "integer"
    }
  ]
}
```

## Running the Application

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`


## Error Handling

- **400**: Invoice_id already exists or invalid input
- **500**: LLM extraction failure or database insert error

## Database Setup

Ensure your Supabase `invoices` table has the following schema:
- `invoice_id` (text, primary key)
- `invoice_date` (text)
- `vendor_name` (text)
- `list_of_items` (jsonb)

## License

This project is provided as-is for invoice processing automation.
