# utils/embedding.py
from typing import List, Dict, Any
import os

# If you use OpenAI >= 1.0
try:
    from openai import OpenAI
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:  # fall back if older SDK
    _client = None

# ---- Embeddings ----
def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Return a single embedding vector for one piece of text.
    """
    if _client is None:
        raise RuntimeError("OpenAI client not available. Check openai package and API key.")
    resp = _client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding

def get_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Batch embeddings for a list of texts.
    """
    if _client is None:
        raise RuntimeError("OpenAI client not available. Check openai package and API key.")
    resp = _client.embeddings.create(model=model, input=texts)
    # Preserve input ordering
    return [d.embedding for d in resp.data]

# ---- Chat completion wrapper used by app.py ----
def chat_complete(messages: List[Dict[str, Any]], model: str = "gpt-4o-mini", **kwargs) -> str:
    """
    Minimal wrapper returning the assistant's message content as a string.
    messages = [{"role":"system","content":"..."},{"role":"user","content":"..."}]
    """
    if _client is None:
        raise RuntimeError("OpenAI client not available. Check openai package and API key.")
    resp = _client.chat.completions.create(model=model, messages=messages, **kwargs)
    return resp.choices[0].message.content
