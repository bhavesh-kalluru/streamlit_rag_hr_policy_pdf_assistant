📚 HR Policy Assistant — RAG + Streamlit (PDF Uploads)

A lightweight Retrieval-Augmented Generation (RAG) app that lets you upload PDFs or .txt files (e.g., HR policies), builds a FAISS index, and answers questions grounded in your documents using the OpenAI API.

Built with Python + Streamlit + FAISS + OpenAI.

Live demo flow: Upload files → Rebuild index → Ask questions → Get answers with retrieved context & sources.

✨ Features

🧩 RAG pipeline: Embed docs → build FAISS index → retrieve → LLM answers grounded in context

📄 PDF support: Converts PDFs to clean .txt (with de-noising)

⚡ Fast QA: Top-K chunk retrieval with scores + source attribution

🔐 Secure key handling: Use .env or set key in the sidebar

🛠️ Robust UX: Clear errors, spinners, and helpful messages

🧱 Architecture (High Level)
          ┌───────────────┐
          │   Upload UI   │  .pdf/.txt
          └──────┬────────┘
                 │
       ┌─────────▼───────────┐
       │  PDF → Text (clean) │  -> ./data/*.txt
       └─────────┬───────────┘
                 │
       ┌─────────▼───────────┐
       │   Index Builder     │  -> FAISS @ ./index
       └─────────┬───────────┘
                 │
       ┌─────────▼───────────┐
       │   Retriever (Top-K) │  -> contexts + scores
       └─────────┬───────────┘
                 │
       ┌─────────▼───────────┐
       │  LLM (OpenAI Chat)  │  -> grounded answer
       └──────────────────────┘

📁 Project Structure
project_root/
│
├─ app.py
├─ .env                      # contains OPENAI_API_KEY
│
├─ data/                     # uploaded + converted texts
├─ index/                    # FAISS index (chunks.index)
│
├─ scripts/
│   ├─ index_builder.py      # build_index(data_dir, index_dir)
│   └─ retrieve.py           # retrieve(query, k, index_dir)
│
└─ utils/
    ├─ __init__.py
    ├─ pdf_utils.py          # pdf_bytes_to_text(...)
    └─ embedding.py          # chat_complete(messages, stream=False)

🚀 Quickstart
1) Requirements

Python 3.10+

Packages:

pip install -r requirements.txt


Minimal set (if you don’t have a requirements file):

pip install streamlit python-dotenv openai faiss-cpu pypdf


Optional OCR for scanned PDFs (if you add it later):
pip install pytesseract pillow and install Tesseract on your OS.

2) Environment

Create a .env in the project root:

OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx


You can also paste the key in the app sidebar at runtime.

3) Run
streamlit run app.py


Then in the browser:

Upload PDFs or .txt files

Click “🔨 Rebuild Index”

Ask your question in the chat input

⚙️ Configuration

Data directory (default: ./data) – where cleaned .txt files are stored

Index directory (default: ./index) – FAISS artifacts (e.g., chunks.index)

Both are editable in the sidebar.

🧪 Example Prompts

“What day does health insurance start?”

“Summarize paid leave accrual and carryover limits.”

“What’s the 401(k) match and when does it begin?”

🩹 Troubleshooting
1) 'str' object has no attribute 'choices'

Your utils/embedding.chat_complete may return either a raw string or an OpenAI-style response object.
This repo’s app.py is resilient and accepts both via extract_answer(resp).

If you want to standardize: make chat_complete return the OpenAI response object:

# utils/embedding.py (example using OpenAI SDK >= 1.0)
from openai import OpenAI
import os

def chat_complete(messages, stream=False, model="gpt-4o-mini"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client.chat.completions.create(model=model, messages=messages, stream=stream)

2) Garbled PDF text (broken line breaks / letters spaced out)

This app runs a cleaner on extracted text to remove hyphenation and fix single line breaks. If your PDFs are scanned images, add OCR.

3) ModuleNotFoundError: utils.embedding

Ensure:

utils/__init__.py exists

You run streamlit run app.py from the project root

You didn’t name any file openai.py

🧩 Key Files
utils/embedding.py (example skeleton)
import os
from openai import OpenAI

def chat_complete(messages, stream=False, model="gpt-4o-mini"):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client.chat.completions.create(model=model, messages=messages, stream=stream)

scripts/index_builder.py (contract)
def run(data_dir: str = "data", index_dir: str = "index"):
    """
    Reads .txt files from data_dir, chunks & embeds them, and writes a FAISS index to index_dir.
    Creates an index marker file 'chunks.index' (or equivalent).
    """
    ...

scripts/retrieve.py (contract)
def retrieve(query: str, k: int, index_dir: str = "index"):
    """
    Returns a list of dicts: [{"score": float, "text": str, "metadata": {"source": str, "chunk_id": int}}, ...]
    """
    ...

🗺️ Roadmap

 OCR toggle for scanned PDFs (Tesseract)

 Source highlighting in answers

 Streaming token output for answers

 “Rebuild only changed files” fast path

 Export Q&A to Markdown/PDF

🤝 Contributing

PRs welcome! Please open an issue for feature requests or bugs.

👤 Author

Bhavesh Kalluru — GenAI Engineer (5+ years)

Portfolio: add your link

GitHub: add your link

LinkedIn: add your link

I’m open to full-time roles in Generative AI / LLM Engineering / Applied ML. Let’s connect!

📄 License

MIT — see LICENSE for details.# streamlit_rag_hr_policy_pdf_assistant
The streamlit rag_hr_policy_pdf assistant is an AI-powered interactive web application that uses Retrieval-Augmented Generation (RAG) to allow users to ask questions and get answers based on the content of uploaded Human Resources (HR) policy PDF documents. 
