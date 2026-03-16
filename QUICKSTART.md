# 🚀 Quick Start Guide - Universal Document Extractor

## What's New? Version 2.0

Your project has been **completely transformed** from an invoice-specific API to a **universal document processing platform** with full user control!

### Key Changes

✅ **No Hardcoded Fields** - You define what to extract  
✅ **Beautiful Web UI** - Visual schema builder included  
✅ **Any Document Type** - Invoices, receipts, contracts, resumes, etc.  
✅ **Custom Schemas** - Complete control over output format  
✅ **Template System** - Pre-built templates for common documents  
✅ **Semantic Search** - Find documents using natural language  

---

## Getting Started in 3 Steps

### Step 1: Start the Application

```bash
cd c:\Users\Abhiraj\Desktop\codes\Python
uvicorn main:app --reload
```

The application will be available at: **http://localhost:8000**

### Step 2: Open the Web UI

Navigate to **http://localhost:8000** in your browser

You'll see:
- 📋 Schema Builder (left panel)
- ⚙️ Extraction Configuration (right panel)
- 🎯 Template buttons at the top

### Step 3: Extract Your First Document

#### Option A: Use a Template (Easiest)

1. Click **"Invoice"** template button
2. Upload a PDF or image of an invoice
3. Click **"🚀 Extract Data"**
4. View results instantly!

#### Option B: Custom Schema (Full Control)

1. Click **"Custom"** template
2. Add fields you want:
   - Field name: `your_field_name`
   - Field type: Text, Number, Date, Array
3. Write extraction instructions:
   ```
   Extract all relevant information from this document.
   Focus on: [what you care about]
   Return dates in YYYY-MM-DD format.
   ```
4. Upload any document
5. Click **"🚀 Extract Data"**

---

## Feature Walkthrough

### 1. Schema Builder

**What it does:** Lets you define exactly what data to extract

**How to use:**
- Click **"+ Add Field"** to add fields
- Choose field types:
  - **Text** - for names, addresses, descriptions
  - **Number** - for amounts, quantities
  - **Date** - for dates (auto-formatted to YYYY-MM-DD)
  - **Boolean** - for true/false values
  - **Array/List** - for lists of items

**Example:**
```
Fields:
- invoice_number (Text)
- total_amount (Number)
- invoice_date (Date)
- line_items (Array)
```

### 2. Templates

**Available Templates:**

#### Invoice
- Pre-defined fields for invoices
- Extracts: number, date, vendor, totals, items
- Best for: Vendor invoices, bills, purchase orders

#### Receipt
- Retail receipt fields
- Extracts: store, date, items, payment method, totals
- Best for: Expense reports, purchases

#### Contract
- Legal document fields
- Extracts: parties, dates, terms, value
- Best for: Agreements, contracts, leases

#### Resume
- CV/Resume fields
- Extracts: contact info, education, experience, skills
- Best for: HR screening, candidate evaluation

#### Custom
- Start from scratch
- Define any fields you want
- Best for: Unique document types

### 3. Extraction Prompt

**What it does:** Tells the AI exactly how to extract data

**Tips for Better Results:**

✅ **Be Specific:**
```
Extract the invoice number from the top right corner.
Find the total amount including tax.
List all line items with quantity and price.
```

✅ **Handle Missing Data:**
```
If a field cannot be found, set it to null.
Do not invent information.
```

✅ **Specify Format:**
```
Return all dates in YYYY-MM-DD format.
Round monetary values to 2 decimal places.
```

❌ **Too Vague:**
```
Get the data
```

### 4. File Upload

**Supported Formats:**
- ✅ PDF (multi-page supported)
- ✅ PNG images
- ✅ JPG/JPEG images
- ✅ WEBP images

**How it works:**
1. Drag & drop or click to upload
2. Each page is processed separately
3. Data is merged intelligently
4. Results show page-by-page breakdown

### 5. Results Display

**You'll see:**

📊 **Statistics:**
- Number of fields extracted
- Pages processed
- Storage status (DB & Vector)

📄 **Extracted Data:**
- JSON format
- Matches your schema exactly
- Copy-paste ready

ℹ️ **Metadata:**
- Filename
- Pages processed
- Processing details

---

## API Usage (For Developers)

### Extract from File

```bash
curl -X POST http://localhost:8000/api/extract \
  -F "file=@document.pdf" \
  -F 'schema={"field1":"string","field2":"number"}' \
  -F 'extraction_prompt=Extract field1 and field2' \
  -F 'document_type=general' \
  -F 'table_name=documents'
```

### Transform JSON Data

```bash
curl -X POST http://localhost:8000/api/extract/json \
  -H "Content-Type: application/json" \
  -d '{
    "data": {"old_field": "value"},
    "schema": {"new_field": "string"}
  }'
```

### Search Documents

```bash
curl -X POST "http://localhost:8000/api/search?query=invoices from ACME Corp&limit=10"
```

### Get All Documents

```bash
curl http://localhost:8000/api/documents
```

---

## Common Use Cases

### 1. Accounts Payable

**Goal:** Automate invoice processing

**Steps:**
1. Select **Invoice** template
2. Upload vendor invoices
3. Extract: invoice #, date, vendor, amount, items
4. Store in `invoices` table
5. Search: "Show me all invoices from Vendor X"

**Schema:**
```json
{
  "invoice_number": "string",
  "vendor_name": "string",
  "invoice_date": "string",
  "total_amount": "number",
  "line_items": []
}
```

### 2. Expense Management

**Goal:** Process employee receipts

**Steps:**
1. Select **Receipt** template
2. Upload receipts
3. Extract: merchant, date, amount, category
4. Export to accounting system

**Schema:**
```json
{
  "merchant": "string",
  "transaction_date": "string",
  "amount": "number",
  "category": "string",
  "payment_method": "string"
}
```

### 3. Resume Screening

**Goal:** Parse candidate CVs

**Steps:**
1. Select **Resume** template
2. Upload resumes
3. Extract: skills, experience, education
4. Match against job requirements

**Schema:**
```json
{
  "full_name": "string",
  "email": "string",
  "phone": "string",
  "skills": [],
  "work_experience": [],
  "education": []
}
```

### 4. Contract Analysis

**Goal:** Extract key terms from legal docs

**Steps:**
1. Select **Contract** template
2. Upload contracts
3. Extract: parties, dates, obligations, values
4. Track renewal dates

**Schema:**
```json
{
  "contract_title": "string",
  "effective_date": "string",
  "expiration_date": "string",
  "party_a": "string",
  "party_b": "string",
  "contract_value": "number",
  "key_terms": []
}
```

---

## Configuration (Optional)

### Supabase Setup (For Database Storage)

The system works without Supabase (local-only mode), but for production use:

1. Create account at https://supabase.com
2. Create new project
3. Get credentials from Settings → API
4. Set environment variables:

```bash
set SUPABASE_URL=https://your-project.supabase.co
set SUPABASE_KEY=your-api-key-here
```

5. Restart the application

### Ollama Setup

The system requires Ollama with these models:

```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2-vision
ollama pull llama3.2
ollama serve
```

---

## Troubleshooting

### Server Won't Start

**Error:** `SupabaseException: Invalid URL`

**Solution:** This is normal if you haven't configured Supabase. The app runs in local-only mode. Look for:
```
WARNING: Supabase not configured. Database storage will be disabled.
```
This is fine! The app still extracts data, just doesn't save to database.

### Slow Processing

**Cause:** Large files or CPU-only inference

**Solutions:**
- Use GPU for Ollama (faster)
- Process one document at a time
- Reduce image resolution
- Split large PDFs

### Poor Extraction Quality

**Solutions:**
- Be more specific in extraction prompt
- Ensure document is clear/readable
- Add examples to prompt
- Check field names are descriptive

### Port Already in Use

**Error:** `Address already in use`

**Solution:** Use different port:
```bash
uvicorn main:app --reload --port 8001
```

---

## Tips for Best Results

### 1. Document Quality
✅ High-resolution scans  
✅ Good contrast  
✅ No shadows or glare  
✅ Straight orientation  

❌ Blurry photos  
❌ Handwritten text (limited support)  
❌ Very old/faded documents  

### 2. Schema Design
✅ Use descriptive field names  
✅ Keep it simple (start with 5-10 fields)  
✅ Match field types to data  
✅ Use arrays for repeating items  

❌ Too many fields (>20)  
❌ Vague field names  
❌ Wrong field types  

### 3. Extraction Prompts
✅ Be conversational  
✅ Give examples  
✅ Specify edge cases  
✅ Request JSON format  

❌ One-word instructions  
❌ Unclear requirements  

### 4. Performance
✅ Batch similar documents  
✅ Use templates when possible  
✅ Cache frequent queries  
✅ Index database fields  

---

## Next Steps

### Beginner
1. ✅ Try the Invoice template with a sample invoice
2. ✅ Create a custom schema with 3-5 fields
3. ✅ Test with different document types
4. ✅ Experiment with extraction prompts

### Intermediate
1. Set up Supabase for storage
2. Create custom templates for your use case
3. Use semantic search to find documents
4. Export data to CSV/Excel

### Advanced
1. Customize the UI for your team
2. Add authentication middleware
3. Implement batch processing
4. Integrate with other systems via API

---

## What Changed from v1.0?

### Old System (Invoice API v1.0)
❌ Only worked with invoices  
❌ Fixed schema (couldn't change fields)  
❌ Hardcoded field names  
❌ No UI (API only)  
❌ Required specific input format  

### New System (Universal Extractor v2.0)
✅ Works with ANY document type  
✅ User-defined schemas  
✅ You choose field names  
✅ Beautiful web UI included  
✅ Accepts any format  
✅ Template system  
✅ Semantic search  
✅ Multi-page support  
✅ Local or cloud storage  

---

## Support & Resources

### Documentation
- Full README: See `README.md`
- API Docs: http://localhost:8000/docs
- Source Code: See `main.py` and `index.html`

### Testing
1. Sample invoices in test folder (if available)
2. Download sample PDFs online
3. Scan your own documents

### Community
- Report bugs with reproduction steps
- Suggest features via issues
- Share your custom templates

---

## Summary

You now have a **fully user-controlled document extraction system** that:

1. 🎯 **You Control** - Define what to extract
2. 🤖 **AI-Powered** - Uses advanced vision models
3. 📱 **Beautiful UI** - Easy to use interface
4. 🔍 **Smart Search** - Find anything instantly
5. 💾 **Flexible Storage** - Local or cloud database
6. ⚡ **Fast Processing** - Multi-page support
7. 🎨 **Templates** - Pre-built for common types
8. 🔧 **Customizable** - Adapt to any need

**Start extracting:** http://localhost:8000

Happy extracting! 🚀
