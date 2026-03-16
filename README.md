
# Universal Document Extractor

A powerful, fully customizable AI-powered document extraction system with an intuitive web UI. Extract any data from any document (PDFs, images, etc.) using user-defined schemas and custom extraction prompts.

**Author:** Abhiraj Marne  
**All Rights Reserved © 2026**

---

## Features

- **🎯 Complete User Control**: Define your own schema - no hardcoded fields or formats
- **🤖 AI-Powered Extraction**: Uses Ollama's `llama3.2-vision` for intelligent data extraction
- **📄 Multi-Format Support**: Process PDFs, images (PNG, JPG, JPEG, WEBP), and more
- **🎨 Beautiful Web UI**: Intuitive interface with schema builder and real-time results
- **🔄 Dynamic Schema**: Transform any input format to any output structure
- **💾 Flexible Storage**: Supabase for structured storage + ChromaDB for vector search
- **🔍 Semantic Search**: Find documents using natural language queries
- **⚡ Template System**: Pre-built templates for common document types
- **🚀 Real-time Processing**: Live feedback and detailed metadata
- **📥 Multi-Format Downloads**: Export to JSON, CSV, Excel, or PDF
- **⚙️ Advanced Extraction Controls**:
  - ⚠️ **Strict Mode** - Only extract explicitly visible data
  - 🔍 **Double Check** - Verify extraction twice for accuracy
  - 💡 **Intelligent Inference** - Fill in missing data using context
  - 🤖 **Auto-Detect Fields** - Map fields regardless of naming convention

---

## Requirements

- Python 3.8+
- FastAPI
- Pydantic
- Supabase client
- Ollama (with `llama3.2-vision` and `llama3.2` models)
- pymupdf (PyMuPDF)
- python-multipart
- chromadb
- uvicorn

### Supported File Formats

- **Documents**: PDF
- **Images**: PNG, JPG, JPEG, WEBP
- **More formats**: Any format that can be converted to images

---

## Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   cd c:\Users\Abhiraj\Desktop\codes\Python
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

4. Make sure Ollama is running with the required models:
   ```bash
   ollama pull llama3.2-vision
   ollama pull llama3.2
   ollama serve
   ```

5. Start the application:
   ```bash
   uvicorn main:app --reload
   ```

6. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

---

## Configuration

If not using environment variables, update the following in `main.py`:
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase API key

---

## API Endpoints

### POST `/api/extract`

Extract data from any document with user-defined schema and prompt.

**Parameters:**
- `file` (multipart/form-data): The document file (PDF, image, etc.)
- `schema` (string): JSON object defining desired output structure
- `extraction_prompt` (string): Custom instructions for the AI
- `document_type` (string): Type classification (invoice, receipt, etc.)
- `table_name` (string): Supabase table name for storage

**Example:**
```bash
curl -X POST http://localhost:8000/api/extract \
  -F "file=@document.pdf" \
  -F 'schema={"invoice_num":"string","date":"string","total":"number","items":[]}' \
  -F 'extraction_prompt=Extract invoice number, date, total amount, and line items' \
  -F 'document_type=invoice' \
  -F 'table_name=invoices'
```

**Response:**
```json
{
  "success": true,
  "message": "Document processed successfully",
  "document_id": "uuid-here",
  "vector_id": "doc_timestamp_filename",
  "extracted_data": {
    "invoice_num": "INV-2024-001",
    "date": "2024-01-15",
    "total": 1250.00,
    "items": [...]
  },
  "metadata": {
    "filename": "document.pdf",
    "pages_processed": 3,
    "page_details": [...]
  }
}
```

---

### POST `/api/extract/json`

Transform JSON data using user-defined schema (no file upload).

**Request:**
```json
{
  "data": {
    "raw_field_1": "value1",
    "raw_field_2": "value2"
  },
  "schema": {
    "transformedField1": "string",
    "transformedField2": "number"
  }
}
```

**Response:**
```json
{
  "success": true,
  "transformed_data": {
    "transformedField1": "value1",
    "transformedField2": 123
  }
}
```

---

### POST `/api/search`

Search documents using semantic similarity.

**Parameters:**
- `query` (string): Natural language search query
- `collection_name` (string): ChromaDB collection (default: "documents")
- `limit` (int): Number of results (default: 5)

**Example:**
```bash
curl -X POST "http://localhost:8000/api/search?query=invoices%20from%20ACME%20Corp&limit=10"
```

---

### GET `/api/documents`

Retrieve all stored documents.

**Parameters:**
- `table_name` (string): Table to query (default: "documents")
- `limit` (int): Maximum results (default: 100)

---

## Schema Definition

### Field Types

- **string**: Text data (names, addresses, descriptions)
- **number**: Numeric values (amounts, quantities)
- **date**: Date values (automatically normalized to YYYY-MM-DD)
- **boolean**: True/false values
- **array**: List of items (can contain objects or strings)

### Schema Examples

#### Simple Invoice Schema
```json
{
  "invoice_number": "string",
  "date": "string",
  "vendor": "string",
  "total": "number"
}
```

#### Complex Schema with Nested Objects
```json
{
  "contract_title": "string",
  "effective_date": "string",
  "parties": {
    "party_a": "string",
    "party_b": "string"
  },
  "terms": [],
  "financial_value": "number"
}
```

#### Resume Schema
```json
{
  "name": "string",
  "email": "string",
  "phone": "string",
  "education": [],
  "experience": [],
  "skills": []
}
```

---

## Running the Application

### Development Mode

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Auto-reloads on code changes. Access at `http://localhost:8000`

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Multiple workers for better performance.

### Background Process (Optional)

On Windows PowerShell:
```powershell
Start-Process powershell -ArgumentList "-NoExit", "-Command", "uvicorn main:app --reload"
```

---

## Error Handling

### Common Errors

**Invalid Schema JSON**
```json
{
  "detail": "Invalid schema JSON format"
}
```
Solution: Ensure your schema is valid JSON with proper quotes and brackets.

**No JSON in LLM Response**
```json
{
  "detail": "No JSON found in LLM response"
}
```
Solution: Check that your extraction prompt clearly requests JSON output.

**Unsupported File Format**
```json
{
  "detail": "Unsupported file format: .xyz"
}
```
Solution: Convert to PDF or image format (PNG, JPG).

**Missing Required Fields**
The system requires at least one field in the schema. Add fields before extracting.

---

## Database Setup

### Supabase Configuration

1. Create a Supabase project at https://supabase.com

2. Get your credentials:
   - Project URL → `SUPABASE_URL`
   - API Key → `SUPABASE_KEY`

3. The system will automatically create tables based on your schema, or you can use a generic `documents` table with this structure:

```sql
CREATE TABLE documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  document_type TEXT,
  filename TEXT,
  extracted_data JSONB,
  metadata JSONB,
  processed_at TIMESTAMPTZ
);
```

### ChromaDB Configuration

ChromaDB automatically creates its storage in `./chroma_storage/`. No manual setup required.

Collections are created on-demand based on document type.

---

## ChromaDB Vector Search

Each invoice is embedded using Ollama and stored in ChromaDB for semantic search and retrieval. You can extend the API to add search endpoints using ChromaDB's similarity search features.

---

## Quick Start Guide

### Using the Web UI (Recommended)

1. **Open the Application**: Navigate to `http://localhost:8000`

2. **Choose a Template** (optional):
   - Click on Invoice, Receipt, Contract, Resume, or Custom
   - This pre-populates the schema with common fields

3. **Define Your Schema**:
   - Add fields you want to extract (e.g., `invoice_number`, `total_amount`, `vendor_name`)
   - Choose field types: Text, Number, Date, Boolean, Array/List
   - The AI will extract exactly these fields

4. **Customize Extraction Instructions**:
   - Tell the AI what to look for
   - Example: "Extract all line items with quantities and prices, calculate totals if not shown"

5. **Upload Your Document**:
   - Drag & drop or click to upload
   - Supports PDF, PNG, JPG, JPEG, WEBP

6. **Click Extract**:
   - Watch real-time processing
   - View extracted data in JSON format
   - See metadata (pages processed, storage status)

7. **Search Your Documents**:
   - Use semantic search to find documents by content
   - Query naturally: "Show me invoices from ACME Corp over $1000"

---

## Pre-built Templates

The web UI includes templates for common document types:

### Invoice Template
Fields: `invoice_number`, `invoice_date`, `vendor_name`, `vendor_address`, `total_amount`, `tax_amount`, `line_items`

### Receipt Template
Fields: `store_name`, `transaction_date`, `receipt_number`, `payment_method`, `subtotal`, `tax`, `total`, `items`

### Contract Template
Fields: `contract_title`, `effective_date`, `expiration_date`, `party_a`, `party_b`, `key_terms`, `value`

### Resume Template
Fields: `full_name`, `email`, `phone`, `education`, `work_experience`, `skills`, `certifications`

### Custom Template
Start from scratch and define your own fields

---

## Advanced Extraction Settings

The web UI provides three powerful toggle options to control extraction behavior:

### ⚠️ Strict Mode
**When to use:** Legal documents, compliance, auditing, financial records

**Behavior:**
- Only extracts information that is EXPLICITLY visible
- No inference, assumptions, or hallucinations
- Returns null/empty for unclear fields
- Prioritizes accuracy over completeness

**Example:** If invoice total is smudged, returns `null` instead of guessing

### 🔍 Double Check
**When to use:** Critical data, high-value transactions, medical records

**Behavior:**
- Reviews extraction twice before responding
- Verifies each value against the original image
- Cross-checks related fields (e.g., total = sum of items)
- Adds verification metadata to results

**Example:** Validates that line item totals match the grand total

### 💡 Infer Missing Data
**When to use:** Damaged documents, poor scans, incomplete forms

**Behavior:**
- Uses context clues to fill in missing information
- Infers dates from surrounding text
- Calculates totals from line items
- Recognizes patterns (phone numbers, emails, addresses)

**Default:** Enabled (toggle off for "No Inference" mode)

**Example:** If "Jan 15, 2024" is partially visible, infers full date

### 🤖 Auto-Detect Fields
**When to use:** Documents with non-standard field names, international documents, OCR results

**Behavior:**
- Automatically identifies required fields regardless of naming convention
- Handles synonyms, abbreviations, and regional variations
- Maps different field names to your standard schema
- Intelligently detects field types from context

**Default:** Enabled (recommended for most use cases)

**Handles These Variations:**
- **Synonyms:** `invoice_number` → `invoice_id` → `inv_no` → `bill_number`
- **Case differences:** `InvoiceNumber` → `invoice_number` → `INVOICE_NUMBER`
- **Abbreviations:** `qty` → `quantity`, `amt` → `amount`, `desc` → `description`
- **Regional:** `colour` → `color`, `organisation` → `organization`
- **Domain terms:** `vendor` → `supplier` → `merchant` → `seller`
- **Formatting:** `totalAmount` (camelCase) → `total_amount` (snake_case)
- **Typos/OCR errors:** Uses context to infer correct field

**Example:** 
- Your schema expects: `invoice_number`, `total_amount`
- Document has: `inv_no`, `grand_total`
- System automatically maps them correctly!

---

### Usage Combinations

#### Maximum Accuracy (Recommended for Legal/Medical)
✅ Strict Mode  
✅ Double Check  
❌ Infer Missing  

**Result:** Most accurate but may have missing fields

#### Balanced (Recommended for Business Documents)
❌ Strict Mode  
✅ Double Check  
✅ Infer Missing  

**Result:** Good balance of accuracy and completeness

#### Maximum Completion (Recommended for Poor Quality Scans)
❌ Strict Mode  
❌ Double Check  
✅ Infer Missing  

**Result:** Most fields filled, higher risk of errors

## Advanced Usage

### Multi-Page Document Processing

The system automatically handles multi-page PDFs:
- Each page is processed separately
- Data is merged intelligently
- Line items from all pages are combined
- Metadata includes page-by-page breakdown

### Custom Extraction Prompts

Best practices for writing extraction prompts:

1. **Be Specific**: "Extract the invoice number from the top right corner"
2. **Handle Missing Data**: "If a field is not found, set it to null"
3. **Specify Format**: "Return dates in YYYY-MM-DD format"
4. **Complex Logic**: "Calculate total by summing all line item amounts"

### Use Cases

#### 1. Accounts Payable Automation
- Extract invoice data from vendor PDFs
- Validate against purchase orders
- Store in accounting system
- Search by vendor, date, or amount

#### 2. Expense Management
- Process employee receipts
- Extract merchant, date, amount
- Categorize expenses
- Generate reports

#### 3. Legal Document Analysis
- Extract contract terms
- Identify parties and dates
- Track obligations
- Compare contracts

#### 4. HR Resume Screening
- Parse candidate CVs
- Extract skills, experience, education
- Match against job requirements
- Rank candidates

#### 5. Medical Records Processing
- Extract patient information
- Process lab results
- Track medications
- Maintain audit trail

---

## Troubleshooting

### Ollama Not Responding
```bash
# Check if Ollama is running
curl http://localhost:11434

# If not running, start it
ollama serve

# Pull required models
ollama pull llama3.2-vision
ollama pull llama3.2
```

### Supabase Connection Error
- Verify `SUPABASE_URL` and `SUPABASE_KEY` are set correctly
- Check network connectivity
- Ensure table permissions allow INSERT/SELECT

### Slow Processing
- Large PDFs take longer (normal)
- Vision models require GPU for best performance
- Consider reducing page count or image resolution

---

## Security Considerations

⚠️ **Important Notes:**

1. **Environment Variables**: Never commit `.env` files or hardcode credentials
2. **File Upload Limits**: Implement size limits in production
3. **Authentication**: Add auth middleware for production use
4. **Input Validation**: Sanitize all user inputs
5. **Rate Limiting**: Protect against abuse
6. **Data Privacy**: Ensure compliance with GDPR, HIPAA, etc.

---

## License

This project is provided as-is for universal document processing automation.

**Author:** Abhiraj Marne  
**All Rights Reserved © 2026**

---

## Changelog

### Version 2.0 - Universal Document Extractor
- 🎉 Complete redesign with user-controlled schemas
- 🎨 New web UI with schema builder
- 🔄 Support for any document type
- 📊 Dynamic database schema creation
- 🔍 Semantic search functionality
- 📱 Responsive design
- ⚡ Template system for common types

### Version 1.0 - Invoice Processing API
- Initial release
- Invoice-specific extraction
- Fixed schema
- Basic API endpoints

---

## Support

For issues, questions, or contributions:
1. Check the troubleshooting section
2. Review API docs at `/docs`
3. Test with sample documents first
4. Report bugs with reproduction steps

---

## Acknowledgments

- **Ollama** - Local LLM runtime
- **FastAPI** - Modern Python web framework
- **Supabase** - Open-source Firebase alternative
- **ChromaDB** - Vector database for embeddings
- **PyMuPDF** - PDF processing library

---

**Built with ❤️ using AI and modern web technologies**
