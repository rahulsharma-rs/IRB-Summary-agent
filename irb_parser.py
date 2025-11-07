#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IRB PDF Parser (text + selections preserved)

Outputs per input PDF:
  - <name>.json : structured fields & page texts
  - <name>.txt  : flattened text with [x]/[ ] / (•) lines

Usage:
  python irb_parser.py /path/to/IRB.pdf
  python irb_parser.py --vision /path/to/IRB.pdf
  python irb_parser.py --no-ocr /path/to/IRB.pdf
  python irb_parser.py /folder/with/pdfs
"""

import os, sys, json, math, argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import fitz  # PyMuPDF
from pypdf import PdfReader

# Optional vision/OCR deps
try:
    import numpy as np
    import cv2
    import pytesseract
    HAVE_VISION = True
except Exception:
    HAVE_VISION = False


# ---------------- Logging ----------------
def dbg(msg: str):
    print(msg, flush=True)


# ---------------- Geometry/Text utils ----------------
def bbox_to_list(r: fitz.Rect) -> List[float]:
    return [float(r.x0), float(r.y0), float(r.x1), float(r.y1)]

def get_page_words(page: fitz.Page):
    words = page.get_text("words") or []
    items = []
    for w in words:
        if len(w) >= 5:
            x0,y0,x1,y1,text = w[0],w[1],w[2],w[3],w[4]
            if str(text).strip():
                items.append((text, [x0,y0,x1,y1]))
    return items

def nearest_text_right(page: fitz.Page, rect: fitz.Rect, max_dx=200, max_dy=40):
    words = get_page_words(page)
    cx = rect.x1
    cy = (rect.y0 + rect.y1) / 2.0
    candidates = []
    for text, bb in words:
        wx = (bb[0] + bb[2]) / 2.0
        wy = (bb[1] + bb[3]) / 2.0
        dx = wx - cx
        dy = abs(wy - cy)
        if dx >= 0 and dx < max_dx and dy < max_dy:
            candidates.append((dx + 0.25*dy, text))
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1] if candidates else None

def gather_phrase(page: fitz.Page, start_rect: fitz.Rect, max_len=8):
    words = page.get_text("words") or []
    row = []
    for w in words:
        if len(w) < 5:
            continue
        x0,y0,x1,y1,txt = w[0],w[1],w[2],w[3],w[4]
        if x0 >= start_rect.x1 - 2 and abs(((y0+y1)/2) - ((start_rect.y0+start_rect.y1)/2)) < 20:
            if str(txt).strip():
                row.append((x0, str(txt)))
    row.sort(key=lambda t: t[0])
    phrase = " ".join([t for _,t in row[:max_len]]).strip()
    return phrase or None


# ---------------- Text extraction (robust) ----------------
def extract_text_strategies(page: fitz.Page) -> str:
    # 1) Simple text
    t1 = page.get_text("text") or ""
    if len(t1.strip()) > 20: return t1

    # 2) Blocks
    blocks = page.get_text("blocks") or []
    if blocks:
        t2 = "\n".join([b[4] for b in blocks if isinstance(b, (list,tuple)) and len(b) >= 5 and str(b[4]).strip()])
        if len(t2.strip()) > 20:
            return t2

    # 3) Raw
    t3 = page.get_text("raw") or ""
    if len(t3.strip()) > 20: return t3

    # 4) JSON spans
    try:
        j = page.get_text("json")
        import json as _json
        jd = _json.loads(j)
        spans = []
        for b in jd.get("blocks", []):
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    tx = s.get("text", "")
                    if str(tx).strip():
                        spans.append(tx)
        t4 = "\n".join(spans)
        if len(t4.strip()) > 20:
            return t4
    except Exception:
        pass

    return ""  # let OCR try


def ocr_page(page: fitz.Page, dpi=300, psm=6) -> str:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # light denoise + adaptive threshold
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 35, 11)

    config = f"--oem 1 --psm {psm}"
    try:
        txt = pytesseract.image_to_string(thr, config=config)
    except Exception:
        txt = ""
    return txt or ""


def page_text_robust(page: fitz.Page, enable_ocr=True) -> str:
    txt = extract_text_strategies(page)
    if len(txt.strip()) > 20:
        return txt
    if enable_ocr and HAVE_VISION:
        txt2 = ocr_page(page, dpi=300, psm=6)
        if len((txt2 or "").strip()) > 0:
            return txt2
    return ""


# ---------------- Form fields (AcroForm) ----------------
def extract_widgets(page: fitz.Page) -> List[Dict[str, Any]]:
    """
    Return a list of widget dicts (safe across PyMuPDF versions where page.widgets() may be a generator).
    """
    out = []
    try:
        wlist = page.widgets() or []
        # Ensure we have a concrete iterable (generator-safe)
        wlist = list(wlist)
    except Exception:
        wlist = []

    for w in wlist:
        try:
            ftype = w.field_type
            item = {
                "name": w.field_name,
                "rect": bbox_to_list(w.rect),
                "value_raw": w.field_value,
                "label": (w.field_label or "").strip() if hasattr(w, "field_label") else "",
            }
            if ftype == fitz.PDF_WIDGET_TYPE_CHECKBOX:
                item["type"] = "checkbox"
                item["checked"] = bool(w.field_value)
            elif ftype == fitz.PDF_WIDGET_TYPE_RADIOBUTTON:
                item["type"] = "radio"
                item["selected"] = bool(w.field_value)
                item["export"] = w.field_value
            elif ftype == fitz.PDF_WIDGET_TYPE_TEXT:
                item["type"] = "text"
                item["text"] = str(w.field_value) if w.field_value is not None else ""
            elif ftype == fitz.PDF_WIDGET_TYPE_COMBOLIST:
                item["type"] = "combo"
                item["text"] = str(w.field_value) if w.field_value is not None else ""
            else:
                item["type"] = "other"
            out.append(item)
        except Exception:
            continue
    return out


# ---------------- Visual checkbox/X detection (optional) ----------------
def detect_visual_checkboxes(page: fitz.Page, dpi=200) -> List[Dict[str, Any]]:
    if not HAVE_VISION:
        return []
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 35, 9)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_w, page_h = page.rect.width, page.rect.height
    scale_x = page_w / img.shape[1]
    scale_y = page_h / img.shape[0]

    results = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        if area < 80 or area > 3500:
            continue
        ar = w/float(h)
        if 0.85 < ar < 1.2:
            roi = bw[y:y+h, x:x+w]
            fill_ratio = (roi > 0).mean()
            d1 = np.mean(np.diag(roi)) / 255.0
            d2 = np.mean(np.diag(np.fliplr(roi))) / 255.0
            has_x = (d1 > 0.5 and d2 > 0.5)
            checked = (fill_ratio > 0.25) or has_x

            # map to PDF coords
            px0, py0 = x * scale_x, y * scale_y
            px1, py1 = (x + w) * scale_x, (y + h) * scale_y
            rect = fitz.Rect(px0, py0, px1, py1)

            label = nearest_text_right(page, rect) or gather_phrase(page, rect) or ""
            results.append({
                "type": "checkbox_visual",
                "rect": [px0, py0, px1, py1],
                "checked": bool(checked),
                "label": label
            })
    return results


# ---------------- Build flattened text ----------------
def build_flat_text(doc_name: str,
                    page_texts: Dict[int, str],
                    fields_by_page: Dict[int, List[Dict[str, Any]]]) -> str:
    lines = [f"=== Document: {doc_name} ==="]
    for pno in sorted(page_texts.keys()):
        lines.append(f"\n--- Page {pno} ---")
        # selections & key fields first
        for f in fields_by_page.get(pno, []):
            if f["type"] in ("checkbox", "checkbox_visual"):
                mark = "[x]" if f.get("checked") else "[ ]"
                lbl = f.get("label") or f.get("name") or "checkbox"
                lines.append(f"{mark} {lbl}")
            elif f["type"] == "radio" and f.get("selected"):
                lbl = f.get("label") or f.get("name") or f.get("export") or "radio"
                lines.append(f"(•) {lbl}")
            elif f["type"] == "text":
                val = (f.get("text") or "").strip()
                if val:
                    nm = f.get("label") or f.get("name") or "text"
                    lines.append(f"{nm}: {val}")
        # then raw page text
        if page_texts[pno].strip():
            lines.append(page_texts[pno].rstrip())
    return "\n".join(lines)


# ---------------- Core parse ----------------
def parse_pdf(pdf_path: Path, enable_ocr=True, enable_vision=False):
    doc = fitz.open(pdf_path)
    all_pages_text: Dict[int, str] = {}
    widgets_by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    visual_by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for pno in range(len(doc)):
        page = doc[pno]

        # widgets FIRST (ensure list, avoids generator/len issues)
        ws = extract_widgets(page)

        # text (robust)
        txt = page_text_robust(page, enable_ocr=enable_ocr)
        all_pages_text[pno+1] = txt

        # safe logging using the concrete list
        dbg(f"[p{pno+1}] text_len={len((txt or '').strip())} widgets={len(ws)}")

        # infer labels if missing
        for w in ws:
            if w["type"] in ("checkbox", "radio") and not w.get("label"):
                try:
                    rect = fitz.Rect(*w["rect"])
                    lbl = nearest_text_right(page, rect) or gather_phrase(page, rect)
                    if lbl:
                        w["label"] = lbl
                except Exception:
                    pass
        widgets_by_page[pno+1].extend(ws)

        # vision fallback (drawn boxes)
        if enable_vision:
            vis = detect_visual_checkboxes(page)
            visual_by_page[pno+1].extend(vis)

    # assemble
    fields_by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for p in widgets_by_page:
        fields_by_page[p].extend(widgets_by_page[p])
    for p in visual_by_page:
        fields_by_page[p].extend(visual_by_page[p])

    out = {
        "document": pdf_path.name,
        "pages": len(doc),
        "fields": [],
        "fields_by_page": {},
        "text_by_page": {}
    }
    for pno in sorted(fields_by_page.keys()):
        out["fields_by_page"][str(pno)] = fields_by_page[pno]
        for f in fields_by_page[pno]:
            f2 = dict(f)
            f2["page"] = pno
            out["fields"].append(f2)
    for pno, txt in all_pages_text.items():
        out["text_by_page"][str(pno)] = txt

    flat = build_flat_text(pdf_path.name, all_pages_text, fields_by_page)
    doc.close()
    return out, flat


# ---------------- IO helpers ----------------
def process_one(pdf_path: Path, enable_ocr=True, enable_vision=False):
    dbg(f"[INFO] Parsing {pdf_path} (OCR={enable_ocr}, vision={enable_vision})")
    data, flat = parse_pdf(pdf_path, enable_ocr=enable_ocr, enable_vision=enable_vision)
    out_json = pdf_path.with_suffix(".json")
    out_txt  = pdf_path.with_suffix(".txt")
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    out_txt.write_text(flat)
    dbg(f"[OK] Wrote {out_json.name} and {out_txt.name}")

def process_path(path: Path, enable_ocr=True, enable_vision=False):
    if path.is_dir():
        pdfs = sorted(list(path.rglob("*.pdf")))
        if not pdfs:
            dbg(f"[WARN] No PDFs under {path}")
            return
        for p in pdfs:
            process_one(p, enable_ocr=enable_ocr, enable_vision=enable_vision)
    else:
        process_one(path, enable_ocr=enable_ocr, enable_vision=enable_vision)


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser("IRB PDF Parser")
    ap.add_argument("path", help="PDF file or folder")
    ap.add_argument("--no-ocr", action="store_true", help="disable OCR fallback")
    ap.add_argument("--vision", action="store_true", help="detect drawn checkboxes/X (needs OpenCV+pytesseract)")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        dbg(f("[ERR] Not found: {p}"))
        sys.exit(1)

    if args.vision and not HAVE_VISION:
        dbg("[WARN] --vision requested but OpenCV/Tesseract not available; continuing without vision.")

    enable_ocr = not args.no_ocr
    process_path(p, enable_ocr=enable_ocr, enable_vision=args.vision)

if __name__ == "__main__":
    main()
