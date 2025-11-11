# HR Policy Assistant (RAG) — Streamlit with PDF Uploads

This version supports **PDF uploads**. PDFs are converted to `.txt` on upload using `pypdf`, then embedded and indexed with FAISS.

## Quickstart
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env  # add your OPENAI_API_KEY
streamlit run app.py
```
- Upload `.pdf` or `.txt`
- Click **Rebuild Index**
- Ask questions

> Note: `pypdf` extracts text from digital PDFs. Scanned-image PDFs need OCR (e.g., Tesseract) before ingestion.
