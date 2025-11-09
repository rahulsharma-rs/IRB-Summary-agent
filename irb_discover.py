# irb_discover.py
# Lightweight agentic IRB Protocol Explorer powered by OpenAI

import os
import io
import re
import json
import time
import base64
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import streamlit as st

from dotenv import load_dotenv
from openai import OpenAI

from pypdf import PdfReader
import fitz  # PyMuPDF

try:
    import docx  # python-docx
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

# -----------------------------------------------------------------------------
# Environment & client
# -----------------------------------------------------------------------------

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_OK = bool(OPENAI_API_KEY)
client = OpenAI() if OPENAI_OK else None

AGENT_MODEL_CHOICES = ["gpt-4o-mini", "gpt-5-nano"]
DEFAULT_AGENT_MODEL = os.getenv("OPENAI_MODEL") or AGENT_MODEL_CHOICES[0]
DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# -----------------------------------------------------------------------------
# Regex helpers
# -----------------------------------------------------------------------------

CHECKMAP = {
    "☑": "[CHECKED]",
    "☒": "[CHECKED]",
    "✓": "[CHECKED]",
    "✔": "[CHECKED]",
    "☐": "[UNCHECKED]",
    "✗": "[UNCHECKED]",
    "✘": "[UNCHECKED]",
}

HEADING_RE = re.compile(
    "|".join(
        [
            r"^\s*(title|study title)\b",
            r"^\s*(principal investigator|pi)\b",
            r"^\s*(objectives?)\b",
            r"^\s*(background|rationale)\b",
            r"^\s*(study design|methods?)\b",
            r"^\s*(population|subjects?)\b",
            r"^\s*(inclusion criteria)\b",
            r"^\s*(exclusion criteria)\b",
            r"^\s*(recruitment)\b",
            r"^\s*(consent|assent|waiver)\b",
            r"^\s*(phi|hipaa|privacy|confidentiality)\b",
            r"^\s*(data (security|management)|security)\b",
            r"^\s*(data sharing|dua|baa|agreements?)\b",
            r"^\s*(risks?|benefits?)\b",
            r"^\s*(compensation|payments?)\b",
            r"^\s*(adverse events?|monitoring|safety)\b",
            r"^\s*(multi[- ]?site|central irb)\b",
            r"^\s*(retention|destruction|data retention)\b",
        ]
    ),
    re.IGNORECASE | re.MULTILINE,
)


def normalize_check_chars(text: str) -> str:
    return "".join(CHECKMAP.get(ch, ch) for ch in text)


YN_RE = re.compile(r"(Yes|No)\s*(\[CHECKED\]|\[UNCHECKED\])", re.I)


def annotate_yes_no(text: str) -> str:
    def repl(match):
        return f"{match.group(1)} {match.group(2)}"

    return YN_RE.sub(repl, text)


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


@dataclass
class DocChunk:
    page: int
    text: str


# -----------------------------------------------------------------------------
# PDF helpers (text, forms, optional GPT vision OCR)
# -----------------------------------------------------------------------------


def extract_acroform_values(pdf_bytes: bytes) -> Dict[str, str]:
    vals: Dict[str, str] = {}
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        fields = reader.get_fields() or {}
        for name, field in fields.items():
            raw = field.get("V") or field.get("/V") or ""
            if isinstance(raw, bytes):
                try:
                    raw = raw.decode("utf-8", "ignore")
                except Exception:
                    raw = str(raw)
            vals[str(name)] = str(raw)
    except Exception:
        pass
    return vals


def vision_transcribe_page(doc: fitz.Document, page_index: int, model: str) -> str:
    if not (OPENAI_OK and client and model):
        return ""
    try:
        page = doc[page_index]
        pix = page.get_pixmap(dpi=220)
        image_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
        resp = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": "You are a meticulous OCR agent. Return only the textual transcription, preserving layout when possible.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe this IRB form page verbatim."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                },
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


def _read_pdf_bytes(upload: bytes, use_vision: bool, vision_model: str) -> List[Tuple[int, str]]:
    reader = PdfReader(io.BytesIO(upload))
    pages: List[Tuple[int, str]] = []
    doc = None
    if use_vision and OPENAI_OK:
        try:
            doc = fitz.open(stream=upload, filetype="pdf")
        except Exception:
            doc = None
    for idx, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = txt.strip()
        if use_vision and doc:
            if len(txt) < 40:  # likely scanned or image only
                vision_txt = vision_transcribe_page(doc, idx, vision_model)
                if vision_txt:
                    txt = vision_txt
        txt = normalize_check_chars(txt)
        txt = annotate_yes_no(txt)
        pages.append((idx + 1, txt))
    if doc:
        doc.close()
    return pages


# -----------------------------------------------------------------------------
# DOCX/TXT helpers
# -----------------------------------------------------------------------------


def _read_docx_bytes(upload: bytes) -> List[Tuple[int, str]]:
    if not HAS_DOCX:
        raise RuntimeError("Install python-docx to read DOCX files.")
    document = docx.Document(io.BytesIO(upload))
    text = "\n".join(par.text for par in document.paragraphs)
    text = normalize_check_chars(text)
    text = annotate_yes_no(text)
    return [(1, text)]


def _read_txt_bytes(upload: bytes) -> List[Tuple[int, str]]:
    try:
        txt = upload.decode("utf-8", errors="ignore")
    except Exception:
        txt = upload.decode("latin-1", errors="ignore")
    txt = normalize_check_chars(txt)
    txt = annotate_yes_no(txt)
    return [(1, txt)]


def load_pages(uploaded_file, use_vision: bool, vision_model: str) -> Tuple[List[Tuple[int, str]], bytes]:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        pages = _read_pdf_bytes(data, use_vision=use_vision, vision_model=vision_model)
    elif name.endswith((".docx", ".doc")):
        pages = _read_docx_bytes(data)
    else:
        pages = _read_txt_bytes(data)
    return pages, data


def augment_pdf_with_forms(pdf_bytes: bytes, pages: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    form_vals = extract_acroform_values(pdf_bytes)
    if not form_vals:
        return pages
    prefix = "\n[FORM_VALUES] " + "; ".join(f"{k}={v}" for k, v in form_vals.items()) + "\n"
    updated = list(pages)
    first_page_no, first_text = updated[0]
    updated[0] = (first_page_no, first_text + prefix)
    return updated


# -----------------------------------------------------------------------------
# Chunking & embeddings
# -----------------------------------------------------------------------------


def split_chunks(pages: List[Tuple[int, str]], max_chars=1500, overlap=200) -> List[DocChunk]:
    chunks: List[DocChunk] = []
    for page_no, text in pages:
        if not text.strip():
            continue
        segments: List[str] = []
        last = 0
        for match in HEADING_RE.finditer(text):
            start = match.start()
            if start > last:
                segments.append(text[last:start])
            segments.append(text[start : start + 400])
            last = start + 400
        if last < len(text):
            segments.append(text[last:])
        if not segments:
            segments = [text]
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            i = 0
            while i < len(seg):
                end = min(i + max_chars, len(seg))
                chunk_text = seg[i:end]
                chunks.append(DocChunk(page=page_no, text=chunk_text))
                if end == len(seg):
                    break
                i = max(0, end - overlap)
    return chunks


def embed_texts(texts: List[str], model: str) -> np.ndarray:
    if not (OPENAI_OK and client):
        raise RuntimeError("OpenAI API key missing; cannot embed text.")
    embeddings: List[List[float]] = []
    batch = 64
    for start in range(0, len(texts), batch):
        batch_texts = texts[start : start + batch]
        resp = client.embeddings.create(model=model, input=batch_texts)
        embeddings.extend([item.embedding for item in resp.data])
    arr = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


class SimpleIndex:
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings

    def search(self, query_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.embeddings.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        sims = self.embeddings @ query_vec
        idx = np.argsort(-sims)[:k]
        return idx, sims[idx]


def embed_query(query: str, model: str) -> np.ndarray:
    vec = embed_texts([query], model=model)[0]
    return vec.astype(np.float32)


def retrieve(index: SimpleIndex, chunks: List[DocChunk], query: str, model: str, k: int) -> List[Dict[str, Any]]:
    if not index:
        return []
    q_vec = embed_query(query, model=model)
    ids, sims = index.search(q_vec, k)
    hits: List[Dict[str, Any]] = []
    for rank, (idx, score) in enumerate(zip(ids, sims), start=1):
        chunk = chunks[int(idx)]
        hits.append(
            {"rank": rank, "score": float(score), "page": chunk.page, "text": chunk.text}
        )
    return hits


# -----------------------------------------------------------------------------
# Agentic answer
# -----------------------------------------------------------------------------


def agentic_answer(question: str, context_blocks: List[Dict[str, Any]], model: str) -> str:
    if not (OPENAI_OK and client):
        return ""
    context = "\n\n".join([f"(Page {b['page']}) {b['text']}" for b in context_blocks])
    prompt = (
        "You are an IRB protocol analysis agent. Use the retrieved context only. "
        "Follow this workflow:\n"
        "1. Quickly outline the plan (2 short steps max).\n"
        "2. Produce the final answer with citations like (p. X).\n"
        "If information is missing, say so explicitly. Be concise."
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext:\n{context}",
            },
        ],
    )
    return resp.choices[0].message.content.strip()


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------


st.set_page_config(page_title="IRB Agentic Explorer", page_icon="🛰️", layout="wide")
st.title("🛰️ IRB Protocol Agent")

if not OPENAI_OK:
    st.warning("Add an OpenAI API key to a .env file (`OPENAI_API_KEY=...`) before using the agent.")

with st.sidebar:
    st.markdown("### Upload Protocol")
    uploaded = st.file_uploader("PDF, DOCX, or TXT", type=["pdf", "docx", "doc", "txt"])

    st.markdown("---")
    st.markdown("### Agent Settings")
    agent_model = st.selectbox(
        "Agent model",
        options=AGENT_MODEL_CHOICES,
        index=AGENT_MODEL_CHOICES.index(DEFAULT_AGENT_MODEL)
        if DEFAULT_AGENT_MODEL in AGENT_MODEL_CHOICES
        else 0,
    )
    embed_model = st.selectbox(
        "Embedding model",
        options=["text-embedding-3-small", "text-embedding-3-large"],
        index=0 if DEFAULT_EMBED_MODEL == "text-embedding-3-small" else 1,
        help="Use the small model to minimize cost.",
    )
    vision_rescue = st.checkbox(
        "Use GPT vision for scanned PDFs (costs more)",
        value=False,
        disabled=not OPENAI_OK,
    )
    top_k = st.slider("Top-k passages", min_value=3, max_value=10, value=6, step=1)

if "chunks" not in st.session_state:
    st.session_state.chunks = None
    st.session_state.index = None
    st.session_state.meta = {}

if uploaded and OPENAI_OK:
    with st.spinner("Running agentic ingestion…"):
        pages, raw_bytes = load_pages(uploaded, use_vision=vision_rescue, vision_model=agent_model)
        if uploaded.name.lower().endswith(".pdf"):
            pages = augment_pdf_with_forms(raw_bytes, pages)

        chunks = split_chunks(pages)
        if not chunks:
            st.error("No text detected in the document.")
        else:
            embeddings = embed_texts([c.text for c in chunks], model=embed_model)
            index = SimpleIndex(embeddings)
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.meta = {
                "filename": uploaded.name,
                "n_pages": len(pages),
                "n_chunks": len(chunks),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "agent_model": agent_model,
                "embed_model": embed_model,
                "vision_enabled": vision_rescue,
            }

if st.session_state.chunks:
    meta = st.session_state.meta
    note = f" • Vision rescue: {'on' if meta.get('vision_enabled') else 'off'}"
    st.caption(
        f"**Loaded:** {meta['filename']} • Pages: {meta['n_pages']} • Chunks: {meta['n_chunks']} • Agent: {meta['agent_model']} • Embed: {meta['embed_model']}{note}"
    )

    st.markdown("#### Ask the agent")
    default_q = "List every Yes/No prompt and whether it was marked."
    question = st.text_input("Question", value=default_q)
    if st.button("Run agent"):
        if not question.strip():
            st.error("Enter a question.")
        else:
            with st.spinner("Retrieving supporting context…"):
                hits = retrieve(
                    st.session_state.index,
                    st.session_state.chunks,
                    question.strip(),
                    model=meta["embed_model"],
                    k=top_k,
                )
            if not hits:
                st.error("No relevant context found.")
            else:
                answer = agentic_answer(question.strip(), hits, model=meta["agent_model"])
                st.subheader("Agent Answer")
                st.write(answer or "Agent reply unavailable.")

                st.subheader("Context used")
                for hit in hits:
                    badge = "🟢" if hit["score"] >= 0.6 else ("🟡" if hit["score"] >= 0.4 else "🔴")
                    st.markdown(
                        f"**#{hit['rank']}** {badge} score `{hit['score']:.3f}` • **(p. {hit['page']})**"
                    )
                    st.write(hit["text"])

                with st.expander("JSON payload"):
                    st.code(json.dumps({"question": question, "hits": hits}, indent=2))
elif uploaded and not OPENAI_OK:
    st.error("OpenAI key missing. Set OPENAI_API_KEY in .env and restart.")
else:
    st.info("Upload a protocol to begin.")

st.markdown("---")
st.caption("Agentic workflow: ingest → chunk → embed → retrieve → reason with GPT.")
