import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

# --- Project-local modules (expected to exist in your repo) ---
from scripts.index_builder import run as build_index
from scripts.retrieve import retrieve
from utils.pdf_utils import pdf_bytes_to_text
from utils.embedding import chat_complete  # ensure this function exists

# =========================
# App Setup
# =========================
load_dotenv()

st.set_page_config(
    page_title="HR Policy Assistant (RAG, PDF-ready)",
    page_icon="📚",
    layout="wide",
)

# =========================
# Helpers
# =========================
@dataclass
class RetrievedChunk:
    score: float
    text: str
    metadata: Dict[str, Any]


def index_exists(index_dir: str = "index") -> bool:
    """Best-effort check for a built index."""
    expected = os.path.join(index_dir, "chunks.index")
    return os.path.isdir(index_dir) and os.path.exists(expected)


def clean_extracted_text(t: str) -> str:
    """Make PDF-extracted text readable before saving."""
    t = t.replace("\r", "")
    # remove hyphenation across line breaks
    t = re.sub(r"-\n", "", t)
    # convert single newlines to spaces but keep paragraph breaks
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    # collapse multiple spaces/tabs
    t = re.sub(r"[ \t]+", " ", t)
    # collapse excessive blank lines
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def save_uploaded_files(files, data_dir: str = "data") -> Dict[str, int]:
    """Persist uploaded PDFs as extracted .txt and .txt files as-is."""
    os.makedirs(data_dir, exist_ok=True)
    saved_txt = 0
    converted_pdf = 0
    for uf in files:
        raw = uf.read()
        name_lower = uf.name.lower()
        if uf.type == "application/pdf" or name_lower.endswith(".pdf"):
            text = pdf_bytes_to_text(raw)
            if not text or not text.strip():
                st.warning(
                    f"No extractable text found in PDF: {uf.name}. It may be a scanned image without OCR."
                )
                continue
            text = clean_extracted_text(text)
            out_name = os.path.splitext(uf.name)[0] + ".txt"
            with open(os.path.join(data_dir, out_name), "w", encoding="utf-8") as f:
                f.write(text)
            converted_pdf += 1
        elif name_lower.endswith(".txt"):
            with open(os.path.join(data_dir, uf.name), "wb") as f:
                f.write(raw)
            saved_txt += 1
        else:
            st.info(f"Skipping unsupported file: {uf.name}")
    return {"saved_txt": saved_txt, "converted_pdf": converted_pdf}


def render_retrieved_context(contexts: List[Dict[str, Any]]):
    st.write("### Retrieved Context")
    for i, c in enumerate(contexts, start=1):
        score = c.get("score")
        md = c.get("metadata", {})
        src = md.get("source", "unknown")
        chunk_id = md.get("chunk_id", "?")
        text = c.get("text", "")
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
        st.markdown(
            f"""**{i}.** _score={score_str}_  
**Source:** `{src}` (chunk {chunk_id})  
{text}"""
        )


def extract_answer(resp) -> str:
    """
    Accepts either:
      - a plain string (already the final answer), or
      - an OpenAI-like response object/dict with .choices[0].message.content
    Returns final answer text.
    """
    if isinstance(resp, str):
        return resp

    # Object path (OpenAI SDK v1+)
    try:
        return resp.choices[0].message.content
    except Exception:
        pass

    # Dict path
    if isinstance(resp, dict):
        try:
            return resp["choices"][0]["message"]["content"]
        except Exception:
            pass

    # Fallback representation
    return str(resp)


# =========================
# Sidebar: Keys & Settings
# =========================
with st.sidebar:
    st.header("⚙️ Settings")
    openai_env = os.getenv("OPENAI_API_KEY")
    key_input = st.text_input("OpenAI API Key (overrides .env)", type="password", value="" if openai_env else "")
    effective_key_present = bool(key_input or openai_env)
    if key_input:
        os.environ["OPENAI_API_KEY"] = key_input

    st.markdown(f"**OpenAI API Key detected:** {'✅' if effective_key_present else '❌'}")

    st.divider()

    st.caption("Index locations")
    data_dir = st.text_input("Data directory", value="data")
    index_dir = st.text_input("Index directory", value="index")

# =========================
# Main
# =========================
st.title("📚 HR Policy Assistant — RAG + Streamlit (PDF Uploads)")
st.write(
    """
Upload **PDFs** or **.txt** files containing HR policies (or any internal knowledge). Build the index,
then ask questions grounded in those documents.
"""
)

# =========================
# Upload Area
# =========================
with st.expander("➕ Upload policy files (.pdf or .txt)"):
    uploaded = st.file_uploader(
        "Upload files",
        accept_multiple_files=True,
        type=["txt", "pdf"],
        key="uploader",
    )
    if uploaded:
        counts = save_uploaded_files(uploaded, data_dir=data_dir)
        st.success(
            f"Saved {counts['saved_txt']} text files and converted {counts['converted_pdf']} PDFs to text in ./{data_dir}"
        )

# =========================
# Index Controls
# =========================
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("🔨 Rebuild Index"):
        try:
            build_index(data_dir=data_dir, index_dir=index_dir)
            st.success("Index rebuilt successfully!")
        except Exception as e:
            st.error(f"Indexing failed: {e}")
with col2:
    exists = index_exists(index_dir)
    st.markdown(f"**Index available:** {'✅' if exists else '❌'}")
with col3:
    st.caption("Tip: Rebuild after uploading or changing files.")

st.divider()

# =========================
# Ask / Chat
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []  # chat transcript

query = st.chat_input("Ask a question about your documents…")

# place slider above results for clarity
top_k = st.slider("Top-K context chunks", min_value=2, max_value=8, value=4)

if query:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": query})

    if not (os.getenv("OPENAI_API_KEY")):
        st.error(
            "Missing OPENAI_API_KEY. Provide one in the sidebar or set it in your environment, then rerun."
        )
    elif not index_exists(index_dir):
        st.error("Index not found. Click 'Rebuild Index' first.")
    else:
        with st.spinner("Retrieving relevant chunks…"):
            try:
                contexts = retrieve(query, k=top_k, index_dir=index_dir)
            except Exception as e:
                contexts = []
                st.error(f"Retrieval failed: {e}")

        if not contexts:
            st.warning(
                "No results retrieved. Try rebuilding the index, widening Top-K, or rephrasing your question."
            )
        else:
            render_retrieved_context(contexts)

            system_msg = {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question using only the provided context. "
                    "If the answer is not in the context, say you don't know and suggest which document might contain it."
                ),
            }
            context_str = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(contexts)])
            user_msg = {
                "role": "user",
                "content": f"Question: {query}\n\nContext:\n{context_str}",
            }

            with st.spinner("Generating grounded answer…"):
                try:
                    resp = chat_complete([system_msg, user_msg], stream=False)
                    answer = extract_answer(resp)
                except Exception as e:
                    answer = f"Generation failed: {e}"

            st.session_state.messages.append({"role": "assistant", "content": answer})

# Render transcript
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
