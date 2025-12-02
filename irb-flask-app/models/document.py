import io
import re
import hashlib
from typing import List, Tuple

from pypdf import PdfReader
import fitz  # PyMuPDF

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


def normalize_check_chars(text: str) -> str:
    """Normalize checkbox characters"""
    return "".join(CHECKMAP.get(ch, ch) for ch in text)


def compute_file_hash(data: bytes) -> str:
    """Compute SHA-256 hash of file"""
    return hashlib.sha256(data).hexdigest()


def extract_text_from_pdf(file_bytes: bytes) -> List[Tuple[int, str]]:
    """Extract text from PDF, returns list of (page_number, text) tuples"""
    pages = []
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        for idx, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            text = normalize_check_chars(text.strip())
            pages.append((idx + 1, text))
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
        text = normalize_check_chars(text)
        return [(1, text)]
    except Exception as e:
        raise ValueError(f"Failed to read DOCX: {str(e)}")


def extract_text_from_txt(file_bytes: bytes) -> List[Tuple[int, str]]:
    """Extract text from plain text file"""
    try:
        text = file_bytes.decode('utf-8', errors='ignore')
    except Exception:
        text = file_bytes.decode('latin-1', errors='ignore')
    text = normalize_check_chars(text)
    return [(1, text)]


def extract_pages(filename: str, file_bytes: bytes) -> List[Tuple[int, str]]:
    """Main function to extract pages based on file type"""
    filename_lower = filename.lower()

    if filename_lower.endswith('.pdf'):
        return extract_text_from_pdf(file_bytes)
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