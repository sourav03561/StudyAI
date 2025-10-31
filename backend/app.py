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
    Smarter YouTube recommendations for study topics.
    Now optimized for key_points input: expects short topic phrases joined into one query.
    """
    incoming = request.get_json(silent=True) or request.form or {}
    # Combine key points list into a compact query string if provided
    key_points = incoming.get("key_points")
    if isinstance(key_points, list):
        query = ", ".join(key_points[:5])  # limit to 5 key points
    else:
        query = (incoming.get("query") or incoming.get("text") or "").strip()

    max_results = int(incoming.get("max_results") or 6)
    youtube_key = os.getenv("YOUTUBE_API_KEY")

    def build_search_terms(q: str) -> list[str]:
        q = q.strip()
        if not q:
            return ["learning basics tutorial", "introduction to science"]
        base = q.lower()
        variants = [
            f"{base} tutorial",
            f"{base} explained",
            f"{base} full course",
            f"{base} lecture",
            f"{base} for beginners",
            f"{base} complete guide",
            f"{base} class",
            f"{base} crash course",
        ]
        seen, out = set(), []
        for v in variants:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def is_relevant_title(title: str) -> bool:
        if not title:
            return False
        t = title.lower()
        keep = ["tutorial", "course", "lesson", "lecture", "explained", "learn", "introduction", "training"]
        ban = ["funny", "shorts", "reaction", "music", "meme", "song", "asmr"]
        return any(k in t for k in keep) and not any(b in t for b in ban)

    videos = []
    if youtube_key and query:
        try:
            collected = []
            for term in build_search_terms(query):
                items = fetch_youtube_videos(term, max_results=max_results, api_key=youtube_key)
                collected.extend(items)
            filtered = [v for v in collected if is_relevant_title(v.get("title", ""))]
            seen_ids, unique = set(), []
            for v in filtered:
                vid = v.get("videoId")
                if vid and vid not in seen_ids:
                    seen_ids.add(vid)
                    unique.append(v)
            videos = unique[:max_results * 2]
        except Exception as e:
            print("YouTube error:", e)
            videos = []

    if not videos:
        videos = curated_for_query(query, max_results=max_results * 3)

    for v in videos:
        if "videoId" in v and not v.get("url"):
            v["url"] = f"https://www.youtube.com/watch?v={v['videoId']}"

    def safe_int(x): return int(x) if str(x).isdigit() else 0
    videos.sort(key=lambda v: safe_int(v.get("viewCount", 0)), reverse=True)
    videos = videos[:max_results]
    random.shuffle(videos)

    return jsonify({"videos": videos})


# ----------------- Entrypoint -----------------
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "0").lower() in {"1", "true", "yes", "y", "on"}
    app.run(host=host, port=port, debug=debug)
