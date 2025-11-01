#!/usr/bin/env python3
"""
Full Flask backend (single file).

Features:
- POST /api/extract_summary : upload PDF or send text -> hybrid extract (PyMuPDF + Tesseract) -> optional Gemini summary
- POST /api/study_material  : same input -> returns summary + flashcards + quiz (Gemini JSON)
- POST /api/recommend_videos: returns YouTube search results (server-side) or curated fallback

Notes:
- Install requirements in your venv:
    pip install flask flask-cors python-dotenv pymupdf Pillow pytesseract google-genai requests
- Install Tesseract OCR engine on the host (apt / brew / Windows installer).
- Set GEMINI_API_KEY and (optionally) YOUTUBE_API_KEY in the environment (or use a .env file).
"""

from __future__ import annotations
import io
import os
import re
import json
import random
import tempfile
from typing import List, Tuple, Optional, Dict, Any, Set

from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()  # loads .env into environment if present

import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from google import genai
import requests

# ---------------- Flask config ----------------
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 120 * 1024 * 1024  # 120 MB

# Defaults
DEFAULT_DPI = 300
DEFAULT_MIN_CHAR_THRESHOLD = 60
DEFAULT_PSM = "6"
LLM_TRIM_LIMIT = 120_000  # characters


# ------------------ Hybrid PDF extractor ------------------

def _extract_blocks_text(page: fitz.Page) -> str:
    blocks = page.get_text("blocks") or []
    blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
    parts = []
    for b in blocks:
        txt = (b[4] or "").strip()
        if txt:
            parts.append(txt)
    return "\n\n".join(parts).strip()


def _count_embedded_chars(page: fitz.Page) -> int:
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


# ------------------ Gemini helpers ------------------

def _trim_for_llm(text: str, limit: int = LLM_TRIM_LIMIT) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return head + "\n\n...[truncated]...\n\n" + tail


def _gen_summary(client: genai.Client, model: str, text: str, prompt: str) -> str:
    trimmed = _trim_for_llm(text)
    resp = client.models.generate_content(
        model=model,
        contents=f"{prompt}\n\n--- DOCUMENT START ---\n{trimmed}\n--- DOCUMENT END ---",
    )
    return getattr(resp, "text", "") or ""


def _robust_json_parse(s: str):
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        candidate = candidate.replace("\t", " ")
        try:
            return json.loads(candidate)
        except Exception:
            return None


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
    trimmed = _trim_for_llm(text)
    topic_hint = f"Topic hint: {topic}\n" if topic else ""

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
    return {"summary": "", "flashcards": [], "quiz": [], "key_topics": [], "key_points": []}


# ------------------ YouTube helpers ------------------

YOUTUBE_API_URL_SEARCH = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_API_URL_VIDEOS = "https://www.googleapis.com/youtube/v3/videos"


def fetch_youtube_videos(query: str, max_results: int = 6, api_key: Optional[str] = None) -> List[dict]:
    if not api_key:
        return []
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "key": api_key,
        "relevanceLanguage": "en",
    }
    r = requests.get(YOUTUBE_API_URL_SEARCH, params=params, timeout=15)
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        return []

    video_ids = ",".join([it["id"]["videoId"] for it in items if it.get("id", {}).get("videoId")])
    if not video_ids:
        return []

    vparams = {
        "part": "contentDetails,statistics",
        "id": video_ids,
        "key": api_key,
    }
    vr = requests.get(YOUTUBE_API_URL_VIDEOS, params=vparams, timeout=15)
    vr.raise_for_status()
    vmap = {v["id"]: v for v in vr.json().get("items", [])}

    videos = []
    for it in items:
        vid = it.get("id", {}).get("videoId")
        snippet = it.get("snippet", {})
        video_obj = {
            "videoId": vid,
            "title": snippet.get("title"),
            "channelTitle": snippet.get("channelTitle"),
            "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url")
                         or snippet.get("thumbnails", {}).get("default", {}).get("url"),
            "publishedAt": snippet.get("publishedAt"),
            "duration": None,
            "viewCount": None,
        }
        details = vmap.get(vid)
        if details:
            video_obj["duration"] = details.get("contentDetails", {}).get("duration")
            video_obj["viewCount"] = details.get("statistics", {}).get("viewCount")
        videos.append(video_obj)
    return videos


# curated fallback
CURATED_VIDEOS = {
    "machine learning": [
        {
            "videoId": "GwIo3gDZCVQ",
            "title": "Machine Learning by Andrew Ng (full course)",
            "channelTitle": "Stanford",
            "thumbnail": "https://i.ytimg.com/vi/GwIo3gDZCVQ/hqdefault.jpg",
            "duration": "PT2H30M",
            "viewCount": "1200000"
        },
        {
            "videoId": "Gv9_4yMHFhI",
            "title": "Intro to Machine Learning - Crash Course",
            "channelTitle": "CrashCourse",
            "thumbnail": "https://i.ytimg.com/vi/Gv9_4yMHFhI/hqdefault.jpg",
            "duration": "PT12M",
            "viewCount": "350000"
        }
    ],
    "neural networks": [
        {
            "videoId": "aircAruvnKk",
            "title": "Neural Networks Explained in 20 Minutes",
            "channelTitle": "AI Simplified",
            "thumbnail": "https://i.ytimg.com/vi/aircAruvnKk/hqdefault.jpg",
            "duration": "PT20M15S",
            "viewCount": "850000"
        }
    ],
}


def curated_for_query(query: str, max_results: int = 6) -> List[dict]:
    if not query:
        out = []
        for arr in CURATED_VIDEOS.values():
            out.extend(arr)
            if len(out) >= max_results:
                break
        random.shuffle(out)
        return out[:max_results]
    q = query.lower()
    for topic, vids in CURATED_VIDEOS.items():
        if topic in q:
            return vids[:max_results]
    out = []
    for arr in CURATED_VIDEOS.values():
        out.extend(arr)
        if len(out) >= max_results:
            break
    random.shuffle(out)
    return out[:max_results]


def _expand_queries(base: str) -> List[str]:
    base = (base or "").strip()
    if not base:
        return ["introduction to machine learning", "machine learning tutorial"]
    variants = [
        base,
        f"{base} tutorial",
        f"{base} course",
        f"{base} explained",
        f"{base} overview",
        f"{base} lecture",
        f"{base} for beginners",
        f"{base} advanced",
    ]
    seen: Set[str] = set()
    out: List[str] = []
    for v in variants:
        lv = v.lower()
        if lv not in seen:
            seen.add(lv)
            out.append(v)
    return out


def _collect_youtube_videos_for_queries(queries: List[str], max_results: int, api_key: str) -> List[dict]:
    seen_ids: Set[str] = set()
    collected: List[dict] = []
    for q in queries:
        try:
            items = fetch_youtube_videos(q, max_results=max_results, api_key=api_key)
        except Exception as e:
            print("YouTube fetch error for query", q, e)
            continue
        for v in items:
            vid = v.get("videoId")
            if not vid or vid in seen_ids:
                continue
            seen_ids.add(vid)
            collected.append(v)
            if len(collected) >= max_results:
                return collected
    return collected


def _shuffle_and_limit(videos: List[dict], max_results: int) -> List[dict]:
    if not videos:
        return []
    subset = videos[: max_results * 3] if len(videos) > max_results * 3 else videos[:]
    random.shuffle(subset)
    return subset[:max_results]


# ------------------ API endpoints ------------------

@app.post("/api/extract_summary")
def api_extract_summary():
    incoming_text = request.form.get("text")
    pdf_file = request.files.get("file")

    if not incoming_text and not pdf_file:
        return abort(400, "Provide either 'file' (PDF) or 'text'.")

    ocr_lang = request.form.get("ocr_lang", "eng")
    dpi = int(request.form.get("dpi", DEFAULT_DPI))
    force_ocr = request.form.get("force_ocr", "false").lower() in {"1", "true", "yes", "y", "on"}
    min_char_threshold = int(request.form.get("min_char_threshold", DEFAULT_MIN_CHAR_THRESHOLD))
    prefer_blocks = request.form.get("prefer_blocks", "true").lower() in {"1", "true", "yes", "y", "on"}
    psm = request.form.get("psm", DEFAULT_PSM)
    page_breaks = request.form.get("page_breaks", "true").lower() in {"1", "true", "yes", "y", "on"}

    prompt = request.form.get("prompt", "Summarize the document clearly and list 5 key insights.")
    model = request.form.get("model", "gemini-2.5-flash")

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

    api_key = os.getenv("GEMINI_API_KEY")
    summary = ""
    notes = None
    if api_key:
        client = genai.Client(api_key=api_key)
        try:
            summary = _gen_summary(client, model, text, prompt)
        except Exception as e:
            notes = f"summary failed: {e}"
            summary = ""
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
    incoming_text = request.form.get("text")
    pdf_file = request.files.get("file")

    if not incoming_text and not pdf_file:
        return abort(400, "Provide either 'file' (PDF) or 'text'.")

    ocr_lang = request.form.get("ocr_lang", "eng")
    dpi = int(request.form.get("dpi", DEFAULT_DPI))
    force_ocr = request.form.get("force_ocr", "false").lower() in {"1", "true", "yes", "y", "on"}
    min_char_threshold = int(request.form.get("min_char_threshold", DEFAULT_MIN_CHAR_THRESHOLD))
    prefer_blocks = request.form.get("prefer_blocks", "true").lower() in {"1", "true", "yes", "y", "on"}
    psm = request.form.get("psm", DEFAULT_PSM)
    page_breaks = request.form.get("page_breaks", "true").lower() in {"1", "true", "yes", "y", "on"}

    topic = request.form.get("topic") or None
    num_cards = int(request.form.get("num_cards", 12))
    num_questions = int(request.form.get("num_questions", 10))
    difficulty = (request.form.get("difficulty", "medium") or "medium").lower()
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "medium"
    model = request.form.get("model", "gemini-2.5-flash")

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

    api_key = os.getenv("GEMINI_API_KEY")
    summary = ""
    flashcards: List[dict] = []
    quiz: List[dict] = []
    data: Dict[str, Any] = {"key_topics": [], "key_points": []}
    notes = None

    if api_key:
        client = genai.Client(api_key=api_key)
        try:
            summary = _gen_summary(client, model, text, "Summarize clearly for quick revision. Use bullets where appropriate.")
            data = _gen_study_json(client, model, text, topic=topic, num_cards=num_cards, num_questions=num_questions, difficulty=difficulty)
            flashcards = data.get("flashcards") or []
            quiz = data.get("quiz") or []
            if data.get("summary"):
                summary = data["summary"]
        except Exception as e:
            notes = f"study material generation failed: {e}"
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


@app.post("/api/recommend_videos")
def api_recommend_videos():
    """
    Improved recommend_videos:
    - Accepts JSON/form with key_points (list) OR query/text
    - Expands each short key-point into multiple educational search terms
    - Collects results across variants, dedupes and filters by title/description
    - Ranks by viewCount (if available) and returns top `max_results`
    - Falls back to curated list when YouTube key missing or API fails
    """
    data = request.get_json(silent=True) or request.form or {}
    max_results = int(data.get("max_results") or 6)

    # Derive a compact query string if key_points provided (limit to first 8)
    key_points = data.get("key_points")
    if isinstance(key_points, list) and len(key_points) > 0:
        # Keep the raw list for multi-query expansion below
        kp_list = [str(k).strip() for k in key_points if str(k).strip()]
    else:
        kp_list = []

    # If no key_points, fall back to query/text string
    query_str = ""
    if not kp_list:
        query_str = (data.get("query") or data.get("text") or "").strip()

    youtube_key = os.getenv("YOUTUBE_API_KEY")

    # Small helper: generate educational query variants for a short phrase
    def _variants_for_phrase(phrase: str):
        p = phrase.strip()
        if not p:
            return []
        base = p.lower()
        return [
            f"{base} tutorial",
            f"{base} explained",
            f"{base} lecture",
            f"{base} crash course",
            f"{base} for beginners",
            f"{base} overview",
            f"{base} full course",
        ]

    # Relevance filter for titles/descriptions (allow if any keep keyword present and none of the ban words)
    KEEP = ["tutorial", "course", "lesson", "lecture", "explained", "learn", "introduction", "guide", "how to", "overview", "crash course"]
    BAN = ["funny", "shorts", "reaction", "music", "meme", "song", "asmr", "review (unboxing)"]

    def _is_relevant(video_obj: dict) -> bool:
        title = (video_obj.get("title") or "").lower()
        desc = (video_obj.get("description") or "").lower()
        text = title + " " + desc
        if any(b in text for b in BAN):
            return False
        return any(k in text for k in KEEP)

    # Collect videos by iterating variants for each key point (or single query)
    collected = []
    seen_ids = set()
    try:
        if youtube_key:
            # If we have multiple key points, query each separately (gives more focused results)
            queries = []
            if kp_list:
                for kp in kp_list[:8]:
                    queries.extend(_variants_for_phrase(kp))
            else:
                # if no kp_list but a query string exists, expand it a little
                base_q = query_str or ""
                if base_q:
                    queries = _variants_for_phrase(base_q) or [base_q]
            # Limit repeated long loops
            for q in queries:
                if not q:
                    continue
                try:
                    # per-term fetch a few candidates (we'll dedupe later)
                    items = fetch_youtube_videos(q, max_results=6, api_key=youtube_key) or []
                except Exception as e:
                    # keep going on partial failures, but log
                    print("YouTube fetch error for query:", q, str(e))
                    items = []
                for it in items:
                    vid = it.get("videoId")
                    if not vid or vid in seen_ids:
                        continue
                    # allow items even if filter rejects — we still keep a fallback
                    if _is_relevant(it):
                        seen_ids.add(vid)
                        collected.append(it)
                    else:
                        # keep less-relevant ones in a separate bucket (if we need to fill)
                        collected.append({**it, "_less_relevant": True})
                # stop early if we already have plenty
                if len(collected) >= max_results * 6:
                    break
        else:
            # No API key — leave collected empty so curated fallback is used
            collected = []
    except Exception as e:
        # unexpected error — print and continue to fallback
        print("Error during video collection:", e)
        collected = []

    # Deduplicate while preferring relevant items (those without _less_relevant)
    ordered = []
    seen = set()
    for v in collected:
        vid = v.get("videoId")
        if not vid or vid in seen:
            continue
        # prefer relevant items first
        if v.get("_less_relevant"):
            ordered.append(v)  # appended later in ordering
        else:
            ordered.insert(0, v)  # push relevant ones to front
        seen.add(vid)

    # If still empty, use curated fallback
    if not ordered:
        ordered = curated_for_query(query_str or (", ".join(kp_list) if kp_list else ""), max_results=max_results * 3)

    # Normalize url, ensure fields exist
    for v in ordered:
        if "videoId" in v and not v.get("url"):
            v["url"] = f"https://www.youtube.com/watch?v={v.get('videoId')}"

    # Sort by viewCount where present (desc), else keep existing order
    def _viewcount_key(v):
        vc = v.get("viewCount") or v.get("view_count") or v.get("statistics", {}).get("viewCount")
        try:
            return int(vc)
        except Exception:
            return 0

    ordered.sort(key=_viewcount_key, reverse=True)

    # Final take top N, but keep deterministic helpfulness: keep top (max_results*2) then shuffle lightly to add variety
    top_pool = ordered[: max_results * 3]
    # if pool larger than desired, shuffle only to vary but keep the highest-ranked near top
    if len(top_pool) > max_results:
        random.shuffle(top_pool)
    result = top_pool[:max_results]

    return jsonify({"videos": result})

@app.post("/api/ask_question")
def api_ask_question():
    """
    Ask a question about the uploaded PDF text.
    Requires 'text' (full extracted text) and 'question' (user input).
    Returns: { answer: "..." }
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    question = (data.get("question") or "").strip()
    model = data.get("model", "gemini-2.5-flash")

    if not text or not question:
        return jsonify({"error": "Missing 'text' or 'question'"}), 400

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY not set"}), 500

    client = genai.Client(api_key=api_key)
    prompt = f"Answer the following question based only on the provided document.\n\nQuestion: {question}\n\nDocument:\n{text[:40000]}"

    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        answer = getattr(resp, "text", "").strip() or "No relevant answer found."
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"answer": answer})


# ----------------- Entrypoint -----------------
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "0").lower() in {"1", "true", "yes", "y", "on"}
    app.run(host=host, port=port, debug=debug)
