# irb_discover.py
# IRB Protocol Explorer with checkbox handling for PDF & DOCX
# - .env support, privacy toggle, model selector
# - AcroForm extraction for fillable PDFs
# - OCR fallback for graphical checkboxes/X’s on PDFs
# - Checkbox glyph normalization -> [CHECKED]/[UNCHECKED] tokens

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence HF tokenizer fork warning

import re
import io
import json
import time
import faiss
import numpy as np
import streamlit as st
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

# --- .env support ---
from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY, OPENAI_MODEL, EMBEDDING_MODEL if present

# --------- Optional LLM (OpenAI) ----------
OPENAI_OK = False
try:
    from openai import OpenAI
    if os.getenv("OPENAI_API_KEY"):
        OPENAI_OK = True
except Exception:
    OPENAI_OK = False

# --------- Parsing utilities ----------
from pypdf import PdfReader
try:
    import docx  # python-docx
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

# Optional OCR/rendering deps (gracefully degrade if missing)
try:
    import pypdfium2 as pdfium
    from PIL import Image
    import pytesseract
    HAS_OCR = True
except Exception:
    HAS_OCR = False

# --------- Embeddings ----------
from sentence_transformers import SentenceTransformer

# ----------------------------- Helpers -----------------------------
@dataclass
class DocChunk:
    page: int
    text: str

HEADING_RE = re.compile("|".join([
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
]), re.IGNORECASE | re.MULTILINE)

CHECKMAP = {
    "☑":"[CHECKED]", "☒":"[CHECKED]", "✓":"[CHECKED]", "✔":"[CHECKED]",
    "☐":"[UNCHECKED]", "✗":"[UNCHECKED]", "✘":"[UNCHECKED]"
}

def normalize_check_chars(text: str) -> str:
    return "".join(CHECKMAP.get(ch, ch) for ch in text)

# Annotate explicit “Yes/No [CHECKED]” near each other
YN_RE = re.compile(r"(Yes|No)\s*(\[CHECKED\]|\[UNCHECKED\])", re.I)
def annotate_yes_no(text: str) -> str:
    def repl(m):
        return f"{m.group(1)} {m.group(2)}"
    return YN_RE.sub(repl, text)

# ---------- PDF helpers ----------
def _read_pdf_bytes(upload: bytes) -> List[Tuple[int, str]]:
    reader = PdfReader(io.BytesIO(upload))
    pages = []
    for i, p in enumerate(reader.pages):
        try:
            txt = p.extract_text() or ""
        except Exception:
            txt = ""
        pages.append((i+1, txt))
    return pages

def extract_acroform_values_from_bytes(pdf_bytes: bytes) -> Dict[str, str]:
    """Return dict of {field_name: value} for fillable PDFs; empty if none."""
    vals = {}
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        fields = reader.get_fields() or {}
        for name, f in fields.items():
            v = f.get("V") or f.get("/V") or ""
            if isinstance(v, bytes):
                try:
                    v = v.decode("utf-8", "ignore")
                except Exception:
                    v = str(v)
            vals[str(name)] = str(v)
    except Exception:
        pass
    return vals

def ocr_pdf_pages(pdf_bytes: bytes, dpi=200) -> List[str]:
    """OCR each page to text. Requires pypdfium2 + pytesseract."""
    if not HAS_OCR:
        return []
    try:
        pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
        out = []
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=dpi/72, rotation=0).to_pil()
            txt = pytesseract.image_to_string(bitmap, config="--psm 6")
            out.append(txt)
        return out
    except Exception:
        return []

def augment_pdf_with_forms_and_ocr(pdf_bytes: bytes, pages: List[Tuple[int,str]], use_ocr: bool) -> List[Tuple[int,str]]:
    """Merge normal text + AcroForm values + optional OCR and normalize checkboxes."""
    # 1) form values (once, page 1)
    form_vals = extract_acroform_values_from_bytes(pdf_bytes)
    # 2) OCR (optional / heuristic)
    need_ocr = use_ocr or any(len(t.strip()) < 40 for _, t in pages)
    ocr_texts = ocr_pdf_pages(pdf_bytes) if need_ocr else []

    merged = []
    for i, (pno, txt) in enumerate(pages):
        extra_parts = []

        if i == 0 and form_vals:
            kv = "; ".join(f"{k}={v}" for k, v in form_vals.items())
            extra_parts.append(f"\n[FORM_VALUES] {kv}\n")

        if ocr_texts and i < len(ocr_texts):
            # If page text is sparse, prefer OCR supplement
            if len(txt.strip()) < max(40, 0.3 * len(ocr_texts[i].strip())):
                extra_parts.append("\n[OCR]\n" + ocr_texts[i])

        merged_text = txt + "".join(extra_parts)
        merged_text = normalize_check_chars(merged_text)
        merged_text = annotate_yes_no(merged_text)
        merged.append((pno, merged_text))
    return merged

# ---------- DOCX helpers ----------
def _read_docx_bytes(upload: bytes) -> List[Tuple[int, str]]:
    if not HAS_DOCX:
        raise RuntimeError("Install python-docx to read .docx files.")
    f = io.BytesIO(upload)
    d = docx.Document(f)
    text = "\n".join(par.text for par in d.paragraphs)
    # Normalize checkbox glyphs if present in DOCX
    text = normalize_check_chars(text)
    text = annotate_yes_no(text)
    return [(1, text)]

# ---------- TXT helper ----------
def _read_txt_bytes(upload: bytes) -> List[Tuple[int, str]]:
    try:
        txt = upload.decode("utf-8", errors="ignore")
    except Exception:
        txt = upload.decode("latin-1", errors="ignore")
    txt = normalize_check_chars(txt)
    txt = annotate_yes_no(txt)
    return [(1, txt)]

# Unified loader
def load_pages(file) -> Tuple[List[Tuple[int, str]], bytes]:
    name = file.name.lower()
    data = file.read()
    if name.endswith(".pdf"):
        pages = _read_pdf_bytes(data)
    elif name.endswith(".docx") or name.endswith(".doc"):
        pages = _read_docx_bytes(data)
    else:
        pages = _read_txt_bytes(data)
    return pages, data

def split_chunks(pages: List[Tuple[int,str]], max_chars=1500, overlap=200) -> List[DocChunk]:
    chunks: List[DocChunk] = []
    for page, text in pages:
        if not text.strip():
            continue
        # break on headings first
        segments, last = [], 0
        for m in HEADING_RE.finditer(text):
            s = m.start()
            if s > last: segments.append(text[last:s])
            segments.append(text[s:s+400]); last = s+400
        if last < len(text): segments.append(text[last:])
        if not segments: segments = [text]
        # then size-based with overlap
        for seg in segments:
            seg = seg.strip()
            if not seg: continue
            i = 0
            while i < len(seg):
                end = min(i + max_chars, len(seg))
                chunks.append(DocChunk(page=page, text=seg[i:end]))
                if end == len(seg): break
                i = max(0, end - overlap)
    return chunks

# ----------------- Embedding & Search -----------------
@st.cache_resource(show_spinner=False)
def load_embedder(name=None):
    name = name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    return SentenceTransformer(name)

def build_index(embedder, chunks: List[DocChunk]):
    texts = [c.text for c in chunks]
    embs = embedder.encode(texts, normalize_embeddings=False, show_progress_bar=False)
    embs = np.asarray(embs, dtype=np.float32)
    faiss.normalize_L2(embs)  # cosine
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index, embs

def retrieve(embedder, index, chunks, query: str, k=6):
    q = embedder.encode([query], normalize_embeddings=False)
    q = np.asarray(q, dtype=np.float32)
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    out = []
    for rank, idx in enumerate(I[0]):
        if idx < 0: continue
        c = chunks[idx]
        out.append({"rank": rank+1, "score": float(D[0][rank]), "page": c.page, "text": c.text})
    return out

# --------------- Optional LLM answer ------------------
def llm_answer(question: str, context_blocks: List[Dict[str, Any]]) -> str:
    if not OPENAI_OK:
        return ""
    client = OpenAI()
    context = "\n\n".join([f"[p.{b['page']}] {b['text']}" for b in context_blocks])
    sys = ("You are an IRB analyst. Use ONLY the provided context. "
           "Ground every claim with page refs like (p. X). Be concise.")
    msg = [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}
    ]
    model_name = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    params = {"model": model_name, "messages": msg}
    # Many 'o1*' / reasoning models disallow temperature
    # if not model_name.startswith("o1"):
    #     params["temperature"] = 0.2
    resp = client.chat.completions.create(**params)
    return resp.choices[0].message.content.strip()

# -------------------- UI --------------------
st.set_page_config(page_title="IRB Protocol Explorer", page_icon="📄", layout="wide")
st.title("📄 IRB Protocol Explorer")

with st.sidebar:
    st.markdown("### Upload Protocol")
    file = st.file_uploader("PDF, DOCX, or TXT", type=["pdf","docx","doc","txt"])

    st.markdown("---")
    st.markdown("### Retrieval Settings")
    model_choice = st.selectbox(
        "Embedding model",
        ["BAAI/bge-base-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"],
        index=0
    )
    k = st.slider("Top-k passages", 3, 12, 6)
    use_ocr = st.checkbox("Use OCR to detect graphical checkboxes (PDF) — slower", value=False)

    st.markdown("---")
    st.markdown("### Privacy")
    key_loaded = bool(os.getenv("OPENAI_API_KEY"))
    use_llm = st.checkbox("Use OpenAI for answers", value=key_loaded)
    if key_loaded:
        st.success("✅ OpenAI key loaded from .env")
        st.caption(f"Model: {os.getenv('OPENAI_MODEL', 'gpt-5')}")
    else:
        st.warning("⚠️ No OpenAI key detected (.env missing or invalid)")
        st.caption("You can still search and view supporting passages.")

if "chunks" not in st.session_state:
    st.session_state.chunks = None
    st.session_state.index = None
    st.session_state.embedder = None
    st.session_state.meta = {}

if file is not None and st.session_state.chunks is None:
    with st.spinner("Parsing & indexing…"):
        pages, raw_bytes = load_pages(file)

        # If PDF: augment with AcroForm + optional OCR + checkbox normalization
        if file.name.lower().endswith(".pdf"):
            pages = augment_pdf_with_forms_and_ocr(raw_bytes, pages, use_ocr)

        chunks = split_chunks(pages, max_chars=1500, overlap=200)
        embedder = load_embedder(model_choice)
        index, _ = build_index(embedder, chunks)
        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.embedder = embedder
        st.session_state.meta = {
            "filename": file.name,
            "n_pages": len(pages),
            "n_chunks": len(chunks),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ocr_enabled": use_ocr and file.name.lower().endswith(".pdf") and HAS_OCR
        }

if st.session_state.chunks:
    meta = st.session_state.meta
    ocr_note = " • OCR: on" if meta.get("ocr_enabled") else ""
    st.caption(f"**Loaded:** {meta['filename']} • Pages: {meta['n_pages']} • Chunks: {meta['n_chunks']}{ocr_note}")

    st.markdown("#### Quick Prompts")
    cols = st.columns(4)
    prompts = {
        "What is this protocol about?": cols[0].button("Overview"),
        "List all mentions of PHI or HIPAA and where they occur.": cols[1].button("PHI & HIPAA"),
        "Describe the consent pathway (consent/assent/waiver) with page citations.": cols[2].button("Consent Pathway"),
        "Summarize data security, sharing, and retention/destruction details with page refs.": cols[3].button("Data & Security"),
    }

    st.markdown("#### Ask a question")
    q = st.text_input("Try: “What are the inclusion/exclusion criteria?” or “Did they select Yes for compensation?”")
    ask = st.button("Search")

    chosen_q = None
    for text, pressed in prompts.items():
        if pressed: chosen_q = text
    if ask and q.strip():
        chosen_q = q.strip()

    if chosen_q:
        with st.spinner("Retrieving…"):
            hits = retrieve(st.session_state.embedder, st.session_state.index, st.session_state.chunks, chosen_q, k=k)

        # Optional LLM answer (privacy toggle enforced)
        answer = ""
        if OPENAI_OK and use_llm:
            with st.spinner("Generating answer…"):
                answer = llm_answer(chosen_q, hits)

        st.subheader("Answer")
        if answer:
            st.write(answer)
        else:
            st.info("LLM disabled — showing top supporting passages instead.")

        st.subheader("Top supporting passages")
        for h in hits:
            # simple relevance tint
            label = "🟢" if h["score"] >= 0.6 else ("🟡" if h["score"] >= 0.4 else "🔴")
            st.markdown(f"**#{h['rank']}** • {label} score: `{h['score']:.3f}` • **(p. {h['page']})**")
            st.write(h["text"])

        with st.expander("JSON (for programmatic use)"):
            st.code(json.dumps({"question": chosen_q, "hits": hits}, indent=2))

else:
    st.info("Upload an IRB protocol on the left to begin.")

st.markdown("---")
st.markdown("Prototype")