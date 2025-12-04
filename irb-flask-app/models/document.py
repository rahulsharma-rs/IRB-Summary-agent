import io
import re
import hashlib
import base64
from typing import List, Tuple, Dict, Optional

from pypdf import PdfReader
import fitz  # PyMuPDF
from openai import OpenAI
from config import Config
from services.embedder import get_openai_client

try:
    import docx

    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

# Character normalization map
CHECKMAP = {
    "☑": "[CHECKED]", "☒": "[CHECKED]", "✓": "[CHECKED]",
    "✔": "[CHECKED]", "☐": "[UNCHECKED]", "✗": "[UNCHECKED]",
    "✘": "[UNCHECKED]",
}

YN_RE = re.compile(r"(Yes|No)\s*(\[CHECKED\]|\[UNCHECKED\])", re.I)


def normalize_check_chars(text: str) -> str:
    """Normalize checkbox characters"""
    return "".join(CHECKMAP.get(ch, ch) for ch in text)


def annotate_yes_no(text: str) -> str:
    """Append [CHECKED]/[UNCHECKED] labels to Yes/No pairs if present"""
    def repl(match):
        return f"{match.group(1)} {match.group(2)}"

    return YN_RE.sub(repl, text)


def extract_acroform_values(file_bytes: bytes) -> Dict[str, str]:
    """Extract PDF form field values"""
    vals: Dict[str, str] = {}
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
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
    """OCR a PDF page using OpenAI vision when normal text extraction is empty."""
    if not Config.OPENAI_API_KEY:
        return ""
    try:
        client = get_openai_client()
    except Exception:
        try:
            client = OpenAI(api_key=Config.OPENAI_API_KEY)
        except Exception:
            return ""

    try:
        page = doc[page_index]
        # Lower DPI to speed up processing
        pix = page.get_pixmap(dpi=170)
        image_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
        params = {
            "model": model or Config.OPENAI_VISION_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an OCR agent. Transcribe all visible text from this page. Do not summarize. Preserve section headers, checkboxes, and tables. Output plain text only."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe this page accurately. Return only text."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]
                }
            ]
        }
        # Add timeout to avoid hanging
        resp = client.chat.completions.create(timeout=Config.OPENAI_VISION_TIMEOUT, **params)
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


def compute_file_hash(data: bytes) -> str:
    """Compute SHA-256 hash of file"""
    return hashlib.sha256(data).hexdigest()


def extract_text_from_pdf(file_bytes: bytes) -> List[Tuple[int, str]]:
    """Extract text from PDF, returns list of (page_number, text) tuples with OCR fallback."""
    pages = []
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        # Optional vision doc for OCR
        try:
            vision_doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception:
            vision_doc = None

        for idx, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""

            # If little/no text, try OCR via vision model
            if len(text.strip()) < 40 and vision_doc:
                ocr_text = vision_transcribe_page(vision_doc, idx, Config.OPENAI_VISION_MODEL)
                if ocr_text:
                    text = ocr_text

            text = normalize_check_chars(text.strip())
            text = annotate_yes_no(text)
            pages.append((idx + 1, text))

        if vision_doc:
            vision_doc.close()
    except Exception as e:
        raise ValueError(f"Failed to read PDF: {str(e)}")
    return pages


def extract_text_from_docx(file_bytes: bytes) -> List[Tuple[int, str]]:
    """Extract text from DOCX"""
    if not HAS_DOCX:
        raise RuntimeError("python-docx not installed")
    try:
        document = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join(par.text for par in document.paragraphs)
        text = annotate_yes_no(normalize_check_chars(text))
        return [(1, text)]
    except Exception as e:
        raise ValueError(f"Failed to read DOCX: {str(e)}")


def extract_text_from_txt(file_bytes: bytes) -> List[Tuple[int, str]]:
    """Extract text from plain text file"""
    try:
        text = file_bytes.decode('utf-8', errors='ignore')
    except Exception:
        text = file_bytes.decode('latin-1', errors='ignore')
    text = annotate_yes_no(normalize_check_chars(text))
    return [(1, text)]


def extract_pages(filename: str, file_bytes: bytes) -> List[Tuple[int, str]]:
    """Main function to extract pages based on file type"""
    filename_lower = filename.lower()

    if filename_lower.endswith('.pdf'):
        pages = extract_text_from_pdf(file_bytes)
        # Append form fields if present
        form_vals = extract_acroform_values(file_bytes)
        if form_vals and pages:
            prefix = "\n[FORM_VALUES] " + "; ".join(f"{k}={v}" for k, v in form_vals.items()) + "\n"
            first_page_no, first_text = pages[0]
            pages[0] = (first_page_no, first_text + prefix)
        return pages
    elif filename_lower.endswith(('.docx', '.doc')):
        return extract_text_from_docx(file_bytes)
    elif filename_lower.endswith('.txt'):
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {filename}")


def chunk_text(pages: List[Tuple[int, str]], max_chars=1500, overlap=200) -> List[dict]:
    """Split pages into overlapping chunks"""
    chunks = []
    chunk_index = 0

    for page_no, text in pages:
        if not text.strip():
            continue

        # Simple sliding window chunking
        i = 0
        while i < len(text):
            end = min(i + max_chars, len(text))
            chunk_text = text[i:end].strip()

            if chunk_text:
                chunks.append({
                    'page_number': page_no,
                    'chunk_index': chunk_index,
                    'text': chunk_text
                })
                chunk_index += 1

            if end == len(text):
                break
            i = max(0, end - overlap)

    return chunks


def get_full_text(pages: List[Tuple[int, str]]) -> str:
    """Combine all pages into single text for extraction"""
    return "\n\n".join(text for _, text in pages)
