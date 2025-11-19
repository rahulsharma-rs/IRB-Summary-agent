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

AGENT_MODEL_CHOICES = ["gpt-5-nano"]
DEFAULT_AGENT_MODEL = os.getenv("OPENAI_MODEL") or AGENT_MODEL_CHOICES[0]
DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

CACHE_DIR = Path(".rag_cache")
CACHE_VERSION = 1
CHUNKER_VERSION = 1

MAX_FILE_SIZE_MB = 200


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
            #temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a meticulous OCR and form-understanding agent for IRB protocol documents (PDF scans and Word exports).\n\nYour job:\n- Extract EVERY visible piece of information from the page image.\n- Preserve the logical structure (sections, questions, tables, checkboxes, fill-in fields) in a way that is easy for a program to parse.\n- Do NOT summarize, skip, or rephrase. Transcribe verbatim as much as possible.\n\nGENERAL RULES\n1. Do not invent content. If you truly cannot read something, write '[[ILLEGIBLE]]' in its place.\n2. Preserve original spelling, capitalization, and punctuation, even if they look wrong.\n3. Keep the original question order and section order.\n4. Show page breaks as: '--- PAGE BREAK ---'.\n\nSECTIONS & HEADERS\n- Use 'SECTION: <exact title>' for main section headings.\n- Use 'SUBSECTION: <exact title>' for subheadings.\n\nCHECKBOXES AND RADIO BUTTONS\n- Represent all checkboxes and radio buttons using this exact syntax:\n  - Checked:   '[x]'\n  - Unchecked: '[ ]'\n- If a label has multiple options (e.g., YES/NO), keep them on the same line in order. Example:\n  'Is the project funded?  [x] YES   [ ] NO'\n- Treat any mark (X, x, ✓, checkmark, filled box) as checked.\n\nFILL-IN-THE-BLANK FIELDS\n- When a field label is followed by a blank line, transcribe as:\n  'Field Label: {{VALUE}}'\n- If the blank is filled, put the transcribed text inside {{ }}.\n- If the blank is empty, write '{{EMPTY}}'.\n- Example:\n  'Contact Name: {{Julie Kanter}}'\n  'Degree: {{EMPTY}}'\n\nTABLES\n- Represent tables using pipe '|' delimited rows, with one header row if visible.\n- Example:\n  '| Column 1 | Column 2 |\\n| value11 | value12 |\\n| value21 | value22 |'\n- If a cell spans multiple lines, keep the line breaks inside the cell as '\\n' (literal backslash-n).\n\nMULTI-LINE ANSWERS\n- If an answer (like \"Purpose\" or \"Background\") continues across lines or pages, concatenate lines into a single block under the same label.\n- If it continues on the next page, add '[[CONTINUES_ON_NEXT_PAGE]]' at end of the page and '[[CONTINUED_FROM_PREVIOUS_PAGE]]' at the beginning of the continuation.\n\nSIGNATURES & DATES\n- For handwritten signatures, do NOT try to guess the name. Use:\n  'Signature: {{HANDWRITTEN, ILLEGIBLE}}' or 'Signature: {{Printed Name if readable}}'.\n- For dates, transcribe exactly as printed.\n\nMARGINAL NOTES / COMMENTS / STAMPS\n- If any handwritten notes, stamps, or comments are visible, append a 'NOTES:' section at the end of the page listing each note on its own line.\n\nOUTPUT FORMAT\n- Return plain UTF-8 text only. No Markdown, no JSON, no bullet points beyond what is needed for tables or checkboxes.\n- Start the page with 'PAGE: <number if visible or UNKNOWN>'."
                        }
                    ]
                }
                ,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe this IRB form page following the IRB-OCR rules. Do not summarize or omit anything. Return only the transcription text."},
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
        #temperature=0.2,
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
    st.info(f"Maximum file size: {MAX_FILE_SIZE_MB}MB")
    uploaded = st.file_uploader(
        "PDF, DOCX, or TXT",
        type=["pdf", "docx", "doc", "txt"],
        help=f"Upload IRB protocol documents (max {MAX_FILE_SIZE_MB}MB)"
    )

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
        options=["text-embedding-3-large"],
        index=0 if DEFAULT_EMBED_MODEL == "text-embedding-3-large" else 1,
        #help="Use the small model to minimize cost.",
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
    raw_bytes = uploaded.read()

    # Validate file size
    file_size_mb = len(raw_bytes) / (1024 * 1024)

    if not raw_bytes:
        st.error("Uploaded file is empty.")
    elif file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large ({file_size_mb:.1f}MB). Maximum size is {MAX_FILE_SIZE_MB}MB.")
    else:
        file_hash = compute_file_hash(raw_bytes)
        log_event(
            f"Received '{uploaded.name}' ({file_size_mb:.1f} MB) hash={file_hash[:12]}… embed_model={embed_model} vision={vision_rescue}"
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

if st.session_state.chunks:
    meta = st.session_state.meta
    note = f" • Vision rescue: {'on' if meta.get('vision_enabled') else 'off'}"
    cache_note = f" • Cache: {'hit' if meta.get('cache_hit') else 'miss'}"
    st.caption(
        f"**Loaded:** {meta['filename']} • Pages: {meta['n_pages']} • Chunks: {meta['n_chunks']} • Agent: {meta['agent_model']} • Embed: {meta['embed_model']}{note}{cache_note}"
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
