#!/usr/bin/env python3
import argparse, os, sys, pathlib, tempfile, subprocess, shutil, re, itertools
from typing import List, Tuple

# PDF engines
import fitz  # PyMuPDF
try:
    import pymupdf_layout as pml  # preferred for layout → MD/HTML
except ImportError:
    pml = None

try:
    import pymupdf4llm  # fallback MD converter
except ImportError:
    pymupdf4llm = None

# DOCX pipeline
import mammoth, markdownify
from PIL import Image


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


# ------------------------ PDF utils ------------------------

def pdf_has_text(pdf_path: str, max_pages_check: int = 3) -> bool:
    doc = fitz.open(pdf_path)
    try:
        n = min(len(doc), max_pages_check)
        for i in range(n):
            page = doc[i]
            if page.get_text("text").strip():
                return True
        return False
    finally:
        doc.close()


def run_ocrmypdf(input_pdf: str, output_pdf: str, lang: str = "eng", force: bool = True):
    """
    Uses OCRmyPDF to add a text layer to scanned PDFs.
    """
    cmd = [
        "ocrmypdf",
        "--language", lang,
        "--optimize", "3",
        "--output-type", "pdf",
        "--skip-text" if not force else "--force-ocr",
        input_pdf,
        output_pdf,
    ]
    # flatten flags
    cmd = [c for c in cmd if c]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"OCRmyPDF failed: {result.stderr.decode(errors='ignore')}")


def export_pdf_images(pdf_path: str, assets_dir: str, prefix: str = "page") -> List[Tuple[int, str]]:
    """
    Extract embedded images from PDF via PyMuPDF.
    Returns [(page_index, relative_path), ...]
    """
    ensure_dir(assets_dir)
    rel_assets = os.path.basename(assets_dir)
    results = []
    doc = fitz.open(pdf_path)
    try:
        for pi, page in enumerate(doc):
            for im_index, im in enumerate(page.get_images(full=True)):
                xref = im[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                fname = f"{prefix}_{pi+1}_{im_index+1}.png"
                out_path = os.path.join(assets_dir, fname)
                pix.save(out_path)
                results.append((pi, os.path.join(rel_assets, fname)))
    finally:
        doc.close()
    return results


def normalize_checkboxes(md_text: str) -> str:
    """
    Convert common checkbox glyphs to ASCII markdown:
    ☑ ☒ → [x], ☐ → [ ]
    """
    md_text = md_text.replace("☑", "[x]").replace("☒", "[x]").replace("✅", "[x]")
    md_text = md_text.replace("☐", "[ ]").replace("⬜", "[ ]")
    # Some OCRs output [ ] as separate chars; lightly fix orphaned box chars
    md_text = re.sub(r"\[\s*\]", "[ ]", md_text)
    return md_text


def inject_page_images(md_text: str, page_imgs: List[Tuple[int, str]]) -> str:
    """
    If section headers like '## Page N' exist (pymupdf_layout often does),
    inject images under each page section; else append a gallery at end.
    """
    if not page_imgs:
        return md_text

    injected = md_text
    any_injected = False
    for pi, rel in page_imgs:
        pattern = rf"(^|\n)#+\s*Page\s*{pi+1}\b.*\n"
        m = re.search(pattern, injected, flags=re.IGNORECASE)
        if m:
            pos = m.end()
            injected = injected[:pos] + f"\n![]({rel})\n" + injected[pos:]
            any_injected = True

    if any_injected:
        return injected

    # fallback gallery
    out = [injected, "\n\n---\n\n### Extracted Images\n"]
    last = -1
    for pi, rel in page_imgs:
        if pi != last:
            out.append(f"\n**Page {pi+1}**\n\n")
            last = pi
        out.append(f"![]({rel})\n\n")
    return "".join(out)


def pdf_to_markdown_layout(pdf_path: str) -> str:
    """
    Prefer pymupdf_layout → Markdown, else fall back to pymupdf4llm.
    """
    if pml is not None:
        doc = fitz.open(pdf_path)
        try:
            pages = [pml.Page(pg) for pg in doc]
            return "\n\n".join(pg.to_markdown() for pg in pages)
        finally:
            doc.close()
    if pymupdf4llm is not None:
        return pymupdf4llm.to_markdown(pdf_path)
    raise RuntimeError("Neither 'pymupdf_layout' nor 'pymupdf4llm' is installed.")


def convert_pdf_to_markdown(input_path: str, out_md: str, assets_dir: str,
                            ocr_lang: str = "eng", force_ocr: bool = False) -> str:
    """
    - Detects scanned PDFs; runs OCR (or forces OCR).
    - Extracts images.
    - Converts to Markdown with layout-aware engine.
    """
    ensure_dir(assets_dir)
    work_pdf = input_path

    is_scanned = not pdf_has_text(input_path)
    if is_scanned or force_ocr:
        with tempfile.TemporaryDirectory() as td:
            ocred = os.path.join(td, "searchable.pdf")
            run_ocrmypdf(input_path, ocred, lang=ocr_lang, force=True)
            # copy to a stable temp path in same folder (so relative assets are nice)
            work_pdf = os.path.join(os.path.dirname(out_md), f"{pathlib.Path(input_path).stem}.ocr.pdf")
            shutil.copyfile(ocred, work_pdf)
    # extract images (from original is fine; from OCRed also ok)
    imgs = export_pdf_images(input_path, assets_dir)

    md = pdf_to_markdown_layout(work_pdf)
    md = inject_page_images(md, imgs)
    md = normalize_checkboxes(md)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    return out_md


# ------------------------ DOCX utils ------------------------

def _docx_image_saver(assets_dir: str):
    ensure_dir(assets_dir)
    counter = itertools.count(1)
    def convert_image(image):
        ext = (image.content_type or "image/png").split("/")[-1].lower()
        if ext == "jpeg":
            ext = "jpg"
        fname = f"image_{next(counter)}.{ext}"
        rel = os.path.join(os.path.basename(assets_dir), fname)
        abs_ = os.path.join(assets_dir, fname)
        with image.open() as img_bytes:
            with open(abs_, "wb") as out:
                out.write(img_bytes.read())
        return {"src": rel}
    return convert_image


def convert_docx_to_markdown(input_path: str, out_md: str, assets_dir: str) -> str:
    ensure_dir(assets_dir)
    with open(input_path, "rb") as f:
        result = mammoth.convert_to_html(
            f,
            convert_image=mammoth.images.img_element(_docx_image_saver(assets_dir)),
        )
    html = result.value
    md = markdownify.markdownify(html, heading_style="ATX", strip=["span"])
    md = normalize_checkboxes(md)
    with open(out_md, "w", encoding="utf-8") as out:
        out.write(md)
    return out_md


# ------------------------ CLI ------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert PDF (native/scanned) or DOCX to Markdown with images.")
    ap.add_argument("input", help="Path to input .pdf or .docx")
    ap.add_argument("-o", "--output", help="Output .md path (default: <inputname>.md)")
    ap.add_argument("--assets-dir", help="Folder for extracted images (default: <inputname>_assets)")
    ap.add_argument("--ocr-lang", default="eng", help="Tesseract language code(s), e.g., 'eng', 'eng+spa'")
    ap.add_argument("--force-ocr", action="store_true", help="Run OCR even if PDF already has text")
    args = ap.parse_args()

    in_path = os.path.abspath(args.input)
    if not os.path.exists(in_path):
        print(f"ERROR: File not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    ext = pathlib.Path(in_path).suffix.lower()
    stem = pathlib.Path(in_path).stem
    base_dir = os.path.dirname(in_path)

    out_md = os.path.abspath(args.output or os.path.join(base_dir, f"{stem}.md"))
    assets_dir = os.path.abspath(args.assets_dir or os.path.join(base_dir, f"{stem}_assets"))

    if ext == ".pdf":
        print("Converting PDF -> Markdown …")
        out = convert_pdf_to_markdown(in_path, out_md, assets_dir, ocr_lang=args.ocr_lang, force_ocr=args.force_ocr)
    elif ext == ".docx":
        print("Converting DOCX -> Markdown …")
        out = convert_docx_to_markdown(in_path, out_md, assets_dir)
    else:
        print("ERROR: Only .pdf and .docx are supported.", file=sys.stderr)
        sys.exit(2)

    print(f"✅ Done.\nMarkdown: {out}\nAssets:   {assets_dir}")


if __name__ == "__main__":
    main()
