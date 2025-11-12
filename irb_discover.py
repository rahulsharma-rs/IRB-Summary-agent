# irb_discover.py
# Lightweight agentic IRB Protocol Explorer powered by OpenAI

import os
import io
import re
import json
import time
import base64
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from pathlib import Path

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

CACHE_DIR = Path(".rag_cache")
CACHE_VERSION = 1
CHUNKER_VERSION = 1

SUMMARY_ITEMS = [
    {
        "id": "irb_number_pi_title",
        "label": "IRB number, PI, title",
        "query": "IRB number protocol number principal investigator study title",
        "instruction": "Report the IRB protocol number, the principal investigator(s), and the full study title.",
    },
    {
        "id": "study_purpose",
        "label": "Study purpose or description",
        "query": "study purpose summary background description objective",
        "instruction": "Summarize the study purpose or high-level description in 1-2 sentences.",
    },
    {
        "id": "cohort_criteria",
        "label": "Cohort criteria (inclusion/exclusion)",
        "query": "inclusion criteria exclusion criteria eligibility subjects",
        "instruction": "List key inclusion and exclusion criteria; mention if not explicitly stated.",
    },
    {
        "id": "data_elements",
        "label": "Approved data elements (date range, identifiers)",
        "query": "data elements identifiers date range data requested approved data",
        "instruction": "Describe which data elements or identifiers are approved for release, including any date ranges.",
    },
    {
        "id": "funding",
        "label": "Funding / sponsor info",
        "query": "funding sponsor grant support",
        "instruction": "Identify funding sources or sponsors; if none, state that it is not specified.",
    },
    {
        "id": "protocol_status",
        "label": "Protocol status (approved, expired, exempt)",
        "query": "protocol status approval exempt withdrawn",
        "instruction": "State the current protocol status (approved, pending, expired, exempt, etc.).",
    },
    {
        "id": "expiration_date",
        "label": "Expiration date",
        "query": "expiration date approval expiration continuing review",
        "instruction": "Provide the protocol approval or expiration date(s).",
    },
    {
        "id": "privacy_confidentiality",
        "label": "Privacy & confidentiality items approved",
        "query": "privacy confidentiality HIPAA PHI data security checkboxes",
        "instruction": "List which privacy or confidentiality items/checkboxes were approved or checked.",
    },
]


def _stringify_value(val) -> str:
    if val is None:
        return ""
    if isinstance(val, (list, tuple)):
        return "; ".join(_stringify_value(v) for v in val if v is not None)
    if isinstance(val, dict):
        return "; ".join(f"{k}: {_stringify_value(v)}" for k, v in val.items())
    return str(val)


def log_event(message: str):
    print(f"[IRB] {message}", flush=True)

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


def load_pages(name: str, data: bytes, use_vision: bool, vision_model: str) -> List[Tuple[int, str]]:
    lname = name.lower()
    if lname.endswith(".pdf"):
        pages = _read_pdf_bytes(data, use_vision=use_vision, vision_model=vision_model)
    elif lname.endswith((".docx", ".doc")):
        pages = _read_docx_bytes(data)
    else:
        pages = _read_txt_bytes(data)
    return pages


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


def compute_file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def chunks_checksum(chunks: List[DocChunk]) -> str:
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(str(chunk.page).encode("utf-8"))
        h.update(b"\x00")
        h.update(chunk.text.encode("utf-8", errors="ignore"))
        h.update(b"\x01")
    return h.hexdigest()


def _model_slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name)


def _cache_bucket(file_hash: str, embed_model: str, vision_flag: bool) -> Path:
    model_slug = _model_slug(embed_model)
    return CACHE_DIR / file_hash[:16] / f"{model_slug}_{int(bool(vision_flag))}"


def _chunks_to_json(chunks: List[DocChunk]) -> List[Dict[str, Any]]:
    return [{"page": c.page, "text": c.text} for c in chunks]


def _chunks_from_json(payload: List[Dict[str, Any]]) -> List[DocChunk]:
    return [DocChunk(page=int(item["page"]), text=item["text"]) for item in payload]


def load_cached_index(file_hash: str, embed_model: str, vision_flag: bool):
    bucket = _cache_bucket(file_hash, embed_model, vision_flag)
    meta_path = bucket / "meta.json"
    chunks_path = bucket / "chunks.json"
    emb_path = bucket / "embeddings.npy"
    if not (meta_path.exists() and chunks_path.exists() and emb_path.exists()):
        log_event(f"[cache] miss – files not found for {bucket}")
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        log_event(f"[cache] miss – could not read meta for {bucket}")
        return None
    if meta.get("cache_version") != CACHE_VERSION:
        log_event(f"[cache] miss – cache version mismatch ({meta.get('cache_version')} != {CACHE_VERSION})")
        return None
    if meta.get("chunker_version") != CHUNKER_VERSION:
        log_event(f"[cache] miss – chunker version mismatch ({meta.get('chunker_version')} != {CHUNKER_VERSION})")
        return None
    if meta.get("file_hash") != file_hash:
        log_event("[cache] miss – file hash mismatch")
        return None
    if meta.get("embed_model") != embed_model:
        log_event("[cache] miss – embedding model mismatch")
        return None
    if bool(meta.get("vision_enabled")) != bool(vision_flag):
        log_event("[cache] miss – vision flag mismatch")
        return None
    try:
        embeddings = np.load(emb_path)
        chunk_data = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = _chunks_from_json(chunk_data)
    except Exception:
        log_event(f"[cache] miss – failed to load embeddings/chunks for {bucket}")
        return None
    if meta.get("chunk_checksum") and meta["chunk_checksum"] != chunks_checksum(chunks):
        log_event("[cache] miss – checksum mismatch")
        return None
    log_event(f"[cache] hit for bucket {bucket}")
    return {"chunks": chunks, "embeddings": embeddings, "meta": meta}


def save_cached_index(
    file_hash: str,
    embed_model: str,
    vision_flag: bool,
    chunks: List[DocChunk],
    embeddings: np.ndarray,
    meta: Dict[str, Any],
):
    bucket = _cache_bucket(file_hash, embed_model, vision_flag)
    bucket.mkdir(parents=True, exist_ok=True)
    np.save(bucket / "embeddings.npy", embeddings)
    (bucket / "chunks.json").write_text(
        json.dumps(_chunks_to_json(chunks), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    meta_out = dict(meta)
    meta_out.update(
        {
            "file_hash": file_hash,
            "embed_model": embed_model,
            "vision_enabled": bool(vision_flag),
            "chunk_checksum": chunks_checksum(chunks),
            "cache_version": CACHE_VERSION,
            "chunker_version": CHUNKER_VERSION,
            "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    (bucket / "meta.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    log_event(f"[cache] saved embeddings and chunks to {bucket}")


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
# Structured metadata summary
# -----------------------------------------------------------------------------

SUMMARY_PROMPT = (
    "You are an expert IRB protocol summarizer. Only use the provided context. "
    "Return a JSON object with keys: field, value, status, pages, evidence. "
    "status must be one of FOUND, PARTIAL, NOT_FOUND. "
    "pages is an array of integers. evidence is an array of short quotes (<=120 chars). "
    "If the information is missing, set value to 'Not specified' and status to NOT_FOUND."
)


def summarize_field(item: Dict[str, str], hits: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    if not hits:
        return {
            "id": item["id"],
            "label": item["label"],
            "value": "Not specified",
            "status": "NOT_FOUND",
            "pages": [],
            "evidence": [],
        }
    context = "\n\n".join([f"(Page {h['page']}) {h['text']}" for h in hits])
    user_content = (
        f"Field: {item['label']}\n"
        f"Instruction: {item['instruction']}\n"
        "Respond with JSON using the schema described earlier.\n\n"
        f"Context:\n{context}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", raw)
            raw = raw.rstrip("`").strip()
        parsed = json.loads(raw)
    except Exception:
        parsed = {
            "field": item["label"],
            "value": "Not specified",
            "status": "NOT_FOUND",
            "pages": [],
            "evidence": [],
        }
    return {
        "id": item["id"],
        "label": item["label"],
        "value": _stringify_value(parsed.get("value")) or "Not specified",
        "status": (parsed.get("status") or "NOT_FOUND").upper(),
        "pages": parsed.get("pages") or [],
        "evidence": parsed.get("evidence") or [],
    }


def summarize_metadata(chunks: List[DocChunk], index: SimpleIndex, embed_model: str, agent_model: str) -> List[Dict[str, Any]]:
    if not (OPENAI_OK and client):
        return []
    results: List[Dict[str, Any]] = []
    for item in SUMMARY_ITEMS:
        hits = retrieve(index, chunks, item["query"], model=embed_model, k=4)
        summary = summarize_field(item, hits, model=agent_model)
        summary["hits"] = hits
        results.append(summary)
    return results


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
    summarize_choice = st.radio(
        "Summarize “IRB/Study Administrative Data”?",
        options=["No", "Yes"],
        horizontal=True,
        help="Automatically extract key protocol metadata after upload.",
    )
    summary_requested = summarize_choice == "Yes"
    st.session_state["summary_pref"] = summary_requested

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
if "summary" not in st.session_state:
    st.session_state.summary = None
    st.session_state.summary_meta = {}
if "summary_modal" not in st.session_state:
    st.session_state.summary_modal = None

if uploaded and OPENAI_OK:
    raw_bytes = uploaded.read()
    if not raw_bytes:
        st.error("Uploaded file is empty.")
    else:
        file_hash = compute_file_hash(raw_bytes)
        log_event(
            f"Received '{uploaded.name}' ({len(raw_bytes)/1024:.1f} KB) hash={file_hash[:12]}… embed_model={embed_model} vision={vision_rescue}"
        )
        log_event("Checking local cache…")
        cache_payload = load_cached_index(file_hash, embed_model, vision_rescue)
        if cache_payload:
            log_event("Cache hit – skipping re-embedding.")
            chunks = cache_payload["chunks"]
            embeddings = cache_payload["embeddings"]
            index = SimpleIndex(embeddings)
            cached_meta = cache_payload["meta"]
            meta = {
                "filename": uploaded.name,
                "n_pages": cached_meta.get("n_pages", 0),
                "n_chunks": cached_meta.get("n_chunks", len(chunks)),
                "time": cached_meta.get("cached_at", time.strftime("%Y-%m-%d %H:%M:%S")),
                "agent_model": agent_model,
                "embed_model": embed_model,
                "vision_enabled": vision_rescue,
                "file_hash": file_hash,
                "cache_hit": True,
            }
        else:
            log_event("Cache miss – parsing document and computing embeddings.")
            with st.spinner("Running agentic ingestion…"):
                log_event("Reading pages…")
                pages = load_pages(uploaded.name, raw_bytes, use_vision=vision_rescue, vision_model=agent_model)
                if uploaded.name.lower().endswith(".pdf"):
                    log_event("Augmenting PDF with AcroForm data.")
                    pages = augment_pdf_with_forms(raw_bytes, pages)

                log_event(f"Chunking {len(pages)} pages…")
                chunks = split_chunks(pages)
                if not chunks:
                    st.error("No text detected in the document.")
                    chunks = None
                else:
                    log_event(f"Generated {len(chunks)} chunks. Requesting embeddings from OpenAI…")
                    embeddings = embed_texts([c.text for c in chunks], model=embed_model)
                    log_event("Embeddings received. Saving to cache.")
                    index = SimpleIndex(embeddings)
                    base_meta = {
                        "filename": uploaded.name,
                        "n_pages": len(pages),
                        "n_chunks": len(chunks),
                    }
                    save_cached_index(file_hash, embed_model, vision_rescue, chunks, embeddings, base_meta)
                    meta = {
                        **base_meta,
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "agent_model": agent_model,
                        "embed_model": embed_model,
                        "vision_enabled": vision_rescue,
                        "file_hash": file_hash,
                        "cache_hit": False,
                    }
        if chunks:
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.meta = meta
            if st.session_state.summary_meta.get("file_hash") != meta.get("file_hash"):
                st.session_state.summary = None
                st.session_state.summary_meta = {}

if st.session_state.chunks:
    meta = st.session_state.meta
    note = f" • Vision rescue: {'on' if meta.get('vision_enabled') else 'off'}"
    cache_note = f" • Cache: {'hit' if meta.get('cache_hit') else 'miss'}"
    st.caption(
        f"**Loaded:** {meta['filename']} • Pages: {meta['n_pages']} • Chunks: {meta['n_chunks']} • Agent: {meta['agent_model']} • Embed: {meta['embed_model']}{note}{cache_note}"
    )

    if st.session_state.get("summary_pref") and st.session_state.index:
        needs_summary = (
            st.session_state.summary is None
            or st.session_state.summary_meta.get("file_hash") != meta.get("file_hash")
            or st.session_state.summary_meta.get("agent_model") != meta.get("agent_model")
            or st.session_state.summary_meta.get("embed_model") != meta.get("embed_model")
        )
        if needs_summary:
            with st.spinner("Summarizing IRB/Study Administrative Data…"):
                log_event("Running metadata summarization…")
                summary_data = summarize_metadata(
                    st.session_state.chunks,
                    st.session_state.index,
                    embed_model=meta["embed_model"],
                    agent_model=meta["agent_model"],
                )
                st.session_state.summary = summary_data
                st.session_state.summary_meta = {
                    "file_hash": meta.get("file_hash"),
                    "agent_model": meta.get("agent_model"),
                    "embed_model": meta.get("embed_model"),
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

    if st.session_state.get("summary_pref") and st.session_state.summary:
        display_items = []
        for item in st.session_state.summary:
            status = (item.get("status") or "").upper()
            value_str = _stringify_value(item.get("value")).strip().lower()
            if status != "NOT_FOUND" and value_str != "not specified" and value_str != "":
                display_items.append(item)
        if display_items:
            with st.expander("IRB/Study Administrative Data Summary", expanded=True):
                for idx, item in enumerate(display_items, start=1):
                    exp = st.expander(f"{idx}. {item['label']}", expanded=False)
                    with exp:
                        cols = st.columns([10, 1])
                        with cols[0]:
                            st.markdown(f"**Value:** {item['value']}")
                            pages = item.get("pages") or []
                            page_str = ", ".join(f"p. {p}" for p in pages) if pages else "N/A"
                            st.caption(f"Status: {item['status']} • References: {page_str}")
                        with cols[1]:
                            if st.button(
                                "👁️",
                                key=f"refs_btn_{meta.get('file_hash')}_{item['id']}",
                                help="View references and supporting evidence",
                            ):
                                st.session_state.summary_modal = item
                                st.rerun()
        else:
            st.info("No structured metadata fields were confidently extracted from this document.")

    modal_item = st.session_state.get("summary_modal")
    if modal_item:
        ref_pages = modal_item.get("pages") or []
        ref_text = ", ".join(f"p. {p}" for p in ref_pages) if ref_pages else "N/A"
        evidence = modal_item.get("evidence") or []
        st.markdown("---")
        st.markdown(f"### References – {modal_item['label']}")
        st.markdown(f"**References:** {ref_text}")
        st.markdown("**Evidence**")
        if evidence:
            for ev in evidence:
                st.write(f"• {ev}")
        else:
            st.write("No supporting quotes captured.")
        if st.button("Close references panel", key="close_summary_modal"):
            st.session_state.summary_modal = None
            st.rerun()

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
