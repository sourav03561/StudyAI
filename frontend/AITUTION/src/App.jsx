import React, { useMemo, useState, useEffect } from "react";
import Tabs from "./components/Tabs.jsx";
import UploadCard from "./components/UploadCard.jsx";
import SummaryView from "./components/SummaryView.jsx";
import FlashcardsView from "./components/FlashcardsView.jsx";
import Quiz from "./components/Quiz.jsx";
import RecommendedVideos from "./components/RecommendedVideos.jsx";
import { Brain } from "lucide-react";
import AskQuestions from "./components/AskQuestions.jsx";

export default function App() {
  const [activeTab, setActiveTab] = useState("Summary");

  // upload + status
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  // results
  const [fileName, setFileName] = useState("");
  const [summary, setSummary] = useState("");
  const [keyTopics, setKeyTopics] = useState([]);
  const [keyPoints, setKeyPoints] = useState([]);
  const [flashcards, setFlashcards] = useState([]);
  const [quiz, setQuiz] = useState([]);
  const [videos, setVideos] = useState([]); // <-- videos state
  const [rawText, setRawText] = useState("");
  const [ocrPages, setOcrPages] = useState([]);
  const [pageCount, setPageCount] = useState(0);

  // show tabs only after we have results
  const hasResults = useMemo(
    () =>
      Boolean(
        summary ||
          (keyTopics && keyTopics.length) ||
          (keyPoints && keyPoints.length) ||
          (flashcards && flashcards.length) ||
          (quiz && quiz.length)
      ),
    [summary, keyTopics, keyPoints, flashcards, quiz]
  );

  function resetOutputs() {
    setErr("");
    setSummary("");
    setKeyTopics([]);
    setKeyPoints([]);
    setFlashcards([]);
    setQuiz([]);
    setRawText("");
    setOcrPages([]);
    setPageCount(0);
    setVideos([]);
  }

  async function callForm(path, form) {
    const res = await fetch(path, { method: "POST", body: form });
    const text = await res.text();
    if (!res.ok) throw new Error(`HTTP ${res.status} – ${text}`);
    return JSON.parse(text);
  }

  // generate study pack (summary, key points, flashcards, quiz)
  async function onGeneratePack() {
    if (!file) {
      setErr("Please choose a PDF.");
      return;
    }
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setErr("File must be a .pdf");
      return;
    }

    resetOutputs();
    setLoading(true);
    setFileName(file?.name || "");

    try {
      const form = new FormData();
      form.append("file", file);
      form.append("ocr_lang", "eng");
      form.append("dpi", "200");
      form.append("min_char_threshold", "120");
      form.append("psm", "6");
      form.append("prefer_blocks", "true");
      form.append("num_cards", "10");
      form.append("num_questions", "6");
      form.append("difficulty", "medium");

      const json = await callForm("/api/study_material", form);

      setSummary(json.summary || "");
      setKeyTopics(Array.isArray(json.key_topics) ? json.key_topics : []);
      setKeyPoints(Array.isArray(json.key_points) ? json.key_points : []);
      setFlashcards(json.flashcards || []);
      setQuiz(json.quiz || []);
      setRawText(json.text || "");
      setOcrPages(json.ocr_pages || []);
      setPageCount(json.page_count || 0);

      setActiveTab("Summary");
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  // fetch recommended videos when user opens the "Recommended Videos" tab (lazy)
  useEffect(() => {
    if (activeTab !== "Recommended Videos") return;
    if (!keyPoints || keyPoints.length === 0) return;
    if (videos && videos.length > 0) return; // already loaded

    (async () => {
      try {
        const payload = { key_points: keyPoints.slice(0, 8), max_results: 8 };
        const res = await fetch("/api/recommend_videos", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          const txt = await res.text();
          console.error("Video API error:", res.status, txt);
          return;
        }
        const json = await res.json();
        setVideos(json.videos || []);
      } catch (err) {
        console.error("Failed to fetch videos:", err);
      }
    })();
  }, [activeTab, keyPoints]); // eslint-disable-line react-hooks/exhaustive-deps

  function startOver() {
    setFile(null);
    resetOutputs();
    setActiveTab("Summary");
  }

  return (
    <div style={{ maxWidth: 1000, margin: "30px auto", padding: 12, fontFamily: "Inter, system-ui, Arial" }}>
      <header style={{ textAlign: "center", marginBottom: 18 }}>
        <h1 style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
          <Brain /> StudyAI
        </h1>
        <p style={{ marginTop: 6, color: "#555" }}>Create a study pack (summary, key points, flashcards, quiz) from a PDF</p>
      </header>

      {hasResults && (
        <div style={{ display: "flex", justifyContent: "center", marginBottom: 12 }}>
          <Tabs value={activeTab} onChange={setActiveTab} />
        </div>
      )}

      {hasResults && (
        <div style={{ textAlign: "center", marginBottom: 12 }}>
          <button onClick={startOver} style={{ padding: "8px 12px", borderRadius: 8 }}>New PDF</button>
        </div>
      )}

      {!hasResults && (
        <UploadCard file={file} setFile={setFile} onGenerate={onGeneratePack} loading={loading} />
      )}

      {err && <div style={{ background: "#fff1f0", border: "1px solid #fecaca", padding: 10, borderRadius: 8 }}>{err}</div>}

      {/* Tab content */}
      {hasResults && activeTab === "Summary" && (
        <SummaryView fileName={fileName} keyTopics={keyTopics} keyPoints={keyPoints} summary={summary} />
      )}

      {hasResults && activeTab === "Flashcards" && <FlashcardsView cards={flashcards} />}

      {hasResults && activeTab === "Quiz" && <Quiz quiz={quiz} />}

      {hasResults && activeTab === "Recommended Videos" && (
        <RecommendedVideos keyPoints={keyPoints} videos={videos} />
      )}
      {hasResults && activeTab === "Ask" && (
        <AskQuestions pdfText={rawText} />
        )}

    </div>
  );
}
