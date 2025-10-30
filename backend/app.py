#!/usr/bin/env python3
"""
Flask backend: Upload a PDF → extract text (PyMuPDF + Tesseract OCR fallback) → Gemini summary + study materials.

Endpoints
---------
POST /api/extract_summary
  form-data:
    file=<PDF>                         (required if 'text' not provided)
    text=<string>                      (optional alternative to file)
    ocr_lang=eng                       (optional; e.g. 'eng', 'eng+hin')
    dpi=300                            (optional)
    force_ocr=false                    (optional; true/false)
    min_char_threshold=60              (optional)
    prefer_blocks=true                 (optional; true/false)
    psm=6                              (optional; Tesseract PSM)
    page_breaks=true                   (optional; true/false)
    tesseract_cmd=<path>               (optional; Windows path to tesseract.exe)
    prompt="Summarize..."              (optional; summary instruction)
    model="gemini-2.5-flash"           (optional)

Response JSON:
  { text, ocr_pages, page_count, summary, used_model, notes? }

POST /api/study_material
  form-data: same extractor params as above + either file=<PDF> or text=<string>
  plus:
    topic=<string>                     (optional hint for context)
    num_cards=12                       (optional; default 12)
    num_questions=10                   (optional; default 10)
    difficulty=easy|medium|hard       (optional; default 'medium')
    model="gemini-2.5-flash"           (optional)

Response JSON:
  {
    text, ocr_pages, page_count, used_model, notes?,
    summary: "...",
    flashcards: [{front, back}, ...],
    quiz: [{question, options, answer, explanation}, ...]
  }
"""

import io
import os
import re
import json
import tempfile
from typing import List, Tuple, Optional

from flask import Flask, request, jsonify, abort
from flask_cors import CORS

from dotenv import load_dotenv  # pip install python-dotenv
load_dotenv()  # Load .env if present

import fitz  # PyMuPDF  (pip install pymupdf)
from PIL import Image          # (pip install Pillow)
import pytesseract             # (pip install pytesseract)
from google import genai       # (pip install google-genai)

# ---------------- Flask app config ----------------
app = Flask(__name__)
CORS(app)

# Optional: limit uploads (change as you like)
app.config['MAX_CONTENT_LENGTH'] = 80 * 1024 * 1024  # 80 MB

# Defaults tuned for speed/quality
DEFAULT_DPI = 300
DEFAULT_MIN_CHAR_THRESHOLD = 60
DEFAULT_PSM = "6"


# ------------- Hybrid extractor helpers -------------

def _extract_blocks_text(page: fitz.Page) -> str:
    """Extract text using block order for better paragraph grouping."""
    blocks = page.get_text("blocks") or []
    # sort top→bottom, then left→right
    blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
    parts = []
    for b in blocks:
        txt = (b[4] or "").strip()
        if txt:
            parts.append(txt)
    return "\n\n".join(parts).strip()


def _count_embedded_chars(page: fitz.Page) -> int:
    """Heuristic to decide if a page is digital vs scanned."""
    try:
        raw = page.get_text("rawdict")
    except Exception:
        return 0
    total = 0
    for blk in (raw or {}).get("blocks", []):
        for line in blk.get("lines", []):
            for span in line.get("spans", []):
                total += len((span.get("text") or "").strip())
    return total


def _ocr_page(page: fitz.Page, dpi: int, ocr_lang: str, tesseract_psm: str) -> str:
    """Render a page to an image and OCR with Tesseract."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    config = f"--psm {tesseract_psm}"
    return pytesseract.image_to_string(img, lang=ocr_lang, config=config).strip()


def extract_text(
    pdf_path: str,
    *,
    ocr_lang: str = "eng",
    dpi: int = DEFAULT_DPI,
    force_ocr: bool = False,
    min_char_threshold: int = DEFAULT_MIN_CHAR_THRESHOLD,
    prefer_blocks: bool = True,
    tesseract_psm: str = DEFAULT_PSM,
    include_page_breaks: bool = True,
) -> Tuple[str, List[int], int]:
    """Extract text from a PDF using PyMuPDF for digital pages, OCR for scanned pages."""
    doc = fitz.open(pdf_path)
    pieces: List[str] = []
    ocr_pages: List[int] = []

    for i, page in enumerate(doc, start=1):
        if force_ocr:
            text = _ocr_page(page, dpi, ocr_lang, tesseract_psm)
            ocr_pages.append(i)
        else:
            char_count = _count_embedded_chars(page)
            if char_count >= min_char_threshold:
                text = _extract_blocks_text(page) if prefer_blocks else page.get_text("text").strip()
                # fallback if embedded text is junk/empty
                if len(text) < 10:
                    text = _ocr_page(page, dpi, ocr_lang, tesseract_psm)
                    ocr_pages.append(i)
            else:
                text = _ocr_page(page, dpi, ocr_lang, tesseract_psm)
                ocr_pages.append(i)

        pieces.append(text or "")
        if include_page_breaks:
            pieces.append(f"\n\n----- PAGE BREAK ({i}/{len(doc)}) -----\n\n")

    full_text = "".join(pieces).rstrip()
    page_count = len(doc)
    doc.close()
    return full_text, ocr_pages, page_count


# ------------- Gemini helpers -------------

def _trim_for_llm(text: str, limit: int = 120_000) -> str:
    """Keep the payload to a sane size for latency and cost."""
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return head + "\n\n...[truncated]...\n\n" + tail


def _gen_summary(client: genai.Client, model: str, text: str, prompt: str) -> str:
    """Get a concise summary from Gemini."""
    trimmed = _trim_for_llm(text)
    resp = client.models.generate_content(
        model=model,
        contents=f"{prompt}\n\n--- DOCUMENT START ---\n{trimmed}\n--- DOCUMENT END ---",
    )
    return getattr(resp, "text", "") or ""


def _gen_study_json(
    client: genai.Client,
    model: str,
    text: str,
    *,
    topic: Optional[str],
    num_cards: int,
    num_questions: int,
    difficulty: str,
) -> dict:
    """
    Ask Gemini to return STRICT JSON with summary, flashcards, and quiz.
    We do a robust JSON parse (strip code fences, find first {...} block).
    """
    trimmed = _trim_for_llm(text)
    topic_hint = f"Topic hint: {topic}\n" if topic else ""

    # Keep this as a plain triple-quoted string (NOT an f-string) so braces {} are safe.
    schema_block = """
You are a teaching assistant. Produce concise, accurate study materials.
Return STRICT JSON only. Do not include commentary.
Schema:
{
  "summary": "string",
  "key_topics": ["string", ...],
  "key_points": ["string", ...],
  "flashcards": [ {"front": "string", "back": "string"} ],
  "quiz": [ {"question": "string", "options": ["string", "..."], "answer": "string", "explanation": "string"} ]
}
""".strip()


    # Append variable lines after the static block
    system = (
    schema_block
    + f"\nNumber of flashcards: {num_cards}\n"
    + f"Number of quiz questions: {num_questions}\n"
    + f"Difficulty: {difficulty}\n"
    + "\nRules:\n"
      "- key_points MUST be 5–8 short, concrete bullets (strings in an array).\n"
      "- key_topics MUST be 3–8 short tags.\n"
      "- Avoid duplicating the same idea; no empty strings.\n"
    )

    contents = (
        f"{system}\n"
        f"{topic_hint}"
        "Base the materials ONLY on the document below. Keep answers unambiguous. "
        "Ensure all quiz items have 3–5 options, exactly one correct answer, and a brief explanation.\n\n"
        f"--- DOCUMENT START ---\n{trimmed}\n--- DOCUMENT END ---"
    )

    resp = client.models.generate_content(model=model, contents=contents)
    raw = getattr(resp, "text", "") or ""

    data = _robust_json_parse(raw)
    if isinstance(data, dict):
        return data
    return {"summary": "", "flashcards": [], "quiz": []}



def _robust_json_parse(s: str):
    """Parse JSON even if wrapped in code fences or has preceding text."""
    # Strip markdown ```json fences
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    # Find first { ... } block (greedy to last })
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        # Sometimes models escape newlines etc.; try a mild cleanup
        candidate = candidate.replace("\t", " ")
        try:
            return json.loads(candidate)
        except Exception:
            return None


# --------------------- API ---------------------

@app.post("/api/extract_summary")
def api_extract_summary():
    """
    Extract text and (optionally) summarize with Gemini.
    Accepts either PDF upload (file) or 'text' field.
    """
    incoming_text = request.form.get("text")
    pdf_file = request.files.get("file")

    if not incoming_text and not pdf_file:
        return abort(400, "Provide either 'file' (PDF) or 'text'.")

    # extractor params (all optional)
    ocr_lang = request.form.get("ocr_lang", "eng")
    dpi = int(request.form.get("dpi", DEFAULT_DPI))
    force_ocr = request.form.get("force_ocr", "false").lower() in {"1", "true", "yes", "y", "on"}
    min_char_threshold = int(request.form.get("min_char_threshold", DEFAULT_MIN_CHAR_THRESHOLD))
    prefer_blocks = request.form.get("prefer_blocks", "true").lower() in {"1", "true", "yes", "y", "on"}
    psm = request.form.get("psm", DEFAULT_PSM)
    page_breaks = request.form.get("page_breaks", "true").lower() in {"1", "true", "yes", "y", "on"}

    # Tesseract path (optional; handy on Windows)
    tesseract_cmd = request.form.get("tesseract_cmd")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    # Summarization
    prompt = request.form.get("prompt", "Summarize the document clearly and list 5 key insights.")
    model = request.form.get("model", "gemini-2.5-flash")

    # Text: either provided directly or extracted from PDF
    if incoming_text:
        text = incoming_text
        ocr_pages, page_count = [], 0
    else:
        if not pdf_file.filename.lower().endswith(".pdf"):
            return abort(400, "File must be a .pdf")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            pdf_file.save(tmp.name)
            text, ocr_pages, page_count = extract_text(
                tmp.name,
                ocr_lang=ocr_lang,
                dpi=dpi,
                force_ocr=force_ocr,
                min_char_threshold=min_char_threshold,
                prefer_blocks=prefer_blocks,
                tesseract_psm=psm,
                include_page_breaks=page_breaks,
            )

    # Summarize with Gemini if key present; otherwise return empty summary + note
    api_key = os.getenv("GEMINI_API_KEY")
    summary = ""
    notes = None
    if api_key:
        client = genai.Client(api_key=api_key)
        summary = _gen_summary(client, model, text, prompt)
    else:
        notes = "summary skipped; GEMINI_API_KEY not set"

    payload = {
        "text": text,
        "ocr_pages": ocr_pages,
        "page_count": page_count,
        "summary": summary,
        "used_model": model,
    }
    if notes:
        payload["notes"] = notes

    return jsonify(payload)


@app.post("/api/study_material")
def api_study_material():
    """
    Build study materials (summary + flashcards + quiz).
    Accepts either PDF upload (file) or 'text' field.

    Extra params:
      topic: optional context hint
      num_cards: default 12
      num_questions: default 10
      difficulty: 'easy'|'medium'|'hard' (default 'medium')
    """
    incoming_text = request.form.get("text")
    pdf_file = request.files.get("file")

    if not incoming_text and not pdf_file:
        return abort(400, "Provide either 'file' (PDF) or 'text'.")

    # extractor params (all optional)
    ocr_lang = request.form.get("ocr_lang", "eng")
    dpi = int(request.form.get("dpi", DEFAULT_DPI))
    force_ocr = request.form.get("force_ocr", "false").lower() in {"1", "true", "yes", "y", "on"}
    min_char_threshold = int(request.form.get("min_char_threshold", DEFAULT_MIN_CHAR_THRESHOLD))
    prefer_blocks = request.form.get("prefer_blocks", "true").lower() in {"1", "true", "yes", "y", "on"}
    psm = request.form.get("psm", DEFAULT_PSM)
    page_breaks = request.form.get("page_breaks", "true").lower() in {"1", "true", "yes", "y", "on"}

    # Tesseract path (optional; handy on Windows)
    tesseract_cmd = request.form.get("tesseract_cmd")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    # Study params
    topic = request.form.get("topic") or None
    num_cards = int(request.form.get("num_cards", 12))
    num_questions = int(request.form.get("num_questions", 10))
    difficulty = (request.form.get("difficulty", "medium") or "medium").lower()
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "medium"

    model = request.form.get("model", "gemini-2.5-flash")

    # Text: either provided directly or extracted from PDF
    if incoming_text:
        text = incoming_text
        ocr_pages, page_count = [], 0
    else:
        if not pdf_file.filename.lower().endswith(".pdf"):
            return abort(400, "File must be a .pdf")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            pdf_file.save(tmp.name)
            text, ocr_pages, page_count = extract_text(
                tmp.name,
                ocr_lang=ocr_lang,
                dpi=dpi,
                force_ocr=force_ocr,
                min_char_threshold=min_char_threshold,
                prefer_blocks=prefer_blocks,
                tesseract_psm=psm,
                include_page_breaks=page_breaks,
            )

    # Build study materials via Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    summary = ""
    flashcards: List[dict] = []
    quiz: List[dict] = []
    notes = None

    if api_key:
        client = genai.Client(api_key=api_key)
        # Summary
        summary = _gen_summary(
            client, model, text,
            prompt="Summarize clearly for quick revision. Use bullets where appropriate."
        )
        # Flashcards + quiz
        data = _gen_study_json(
            client, model, text,
            topic=topic,
            num_cards=num_cards,
            num_questions=num_questions,
            difficulty=difficulty,
        )
        flashcards = data.get("flashcards") or []
        quiz = data.get("quiz") or []
        # If summary field returned too, prefer that
        if data.get("summary"):
            summary = data["summary"]
    else:
        notes = "study material skipped; GEMINI_API_KEY not set"

    payload = {
    "text": text,
    "ocr_pages": ocr_pages,
    "page_count": page_count,
    "used_model": model,
    "summary": summary,
    "key_topics": data.get("key_topics") or [],
    "key_points": data.get("key_points") or [],
    "flashcards": flashcards,
    "quiz": quiz,
    }

    if notes:
        payload["notes"] = notes

    return jsonify(payload)


# ----------------- Entrypoint -----------------

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "0").lower() in {"1", "true", "yes", "y", "on"}
    app.run(host=host, port=port, debug=debug)
