import React, { useMemo, useState } from "react";
import Tabs from "./components/Tabs.jsx";
import UploadCard from "./components/UploadCard.jsx";
import SummaryView from "./components/SummaryView.jsx";
import FlashcardsView from "./components/FlashcardsView.jsx";
import Quiz from "./components/Quiz.jsx";
import { BookOpen, Brain, ListChecks } from "lucide-react";

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
  }

  async function call(path, form) {
    const res = await fetch(path, { method: "POST", body: form });
    const text = await res.text();
    if (!res.ok) throw new Error(`HTTP ${res.status} – ${text}`);
    return JSON.parse(text);
  }

  // ONE action: generate everything together
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
      // faster defaults
      form.append("ocr_lang", "eng");
      form.append("dpi", "200");
      form.append("min_char_threshold", "120");
      form.append("psm", "6");
      form.append("prefer_blocks", "true");
      // study pack knobs
      form.append("num_cards", "10");
      form.append("num_questions", "6");
      form.append("difficulty", "medium");

      const json = await call("/api/study_material", form);

      setSummary(json.summary || "");
      setKeyTopics(Array.isArray(json.key_topics) ? json.key_topics : []);
      setKeyPoints(Array.isArray(json.key_points) ? json.key_points : []);
      setFlashcards(json.flashcards || []);
      setQuiz(json.quiz || []);
      setRawText(json.text || "");
      setOcrPages(json.ocr_pages || []);
      setPageCount(json.page_count || 0);

      setActiveTab("Summary"); // show Summary first
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  function startOver() {
    setFile(null);
    resetOutputs();
    setActiveTab("Summary");
  }

  return (
    <div className="container">
      {/* Header */}
      <div className="header" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10, textAlign: "center" }}>
  <h1 className="h1" style={{ display: "flex", alignItems: "center", gap: "8px" }}>
    <Brain className="w-8 h-8 text-blue-600" />
    StudyAI
  </h1>        <div className="subtle">Create a study pack (key topics, key points, flashcards, quiz) from your PDF</div>
      </div>

      {/* Tabs row – only after results; centered */}
      {hasResults && (
        <div className="row" style={{ justifyContent: "center", marginBottom: 12 }}>
          <Tabs value={activeTab} onChange={setActiveTab} />
        </div>
      )}

      {/* Stats + reset */}
      {hasResults && (
        <div className="row" style={{ justifyContent: "center", marginBottom: 12 }}>
          <button className="btn secondary" onClick={startOver}>New PDF</button>
        </div>
      )}

      {/* Upload screen (no tabs shown) */}
      {!hasResults && (
        <UploadCard
          file={file}
          setFile={setFile}
          onGenerate={onGeneratePack}
          loading={loading}
        />
      )}

      {/* Error */}
      {err && (
        <div className="card" style={{ borderColor: "#fecaca", background: "#fff1f2" }}>
          {err}
        </div>
      )}

      {/* Results screens */}
      {hasResults && activeTab === "Summary" && (
        <SummaryView
          fileName={fileName}
          keyTopics={keyTopics}
          keyPoints={keyPoints}
          summary={summary}
        />
      )}

      {hasResults && activeTab === "Flashcards" && (
        <FlashcardsView cards={flashcards} />
      )}

      {hasResults && activeTab === "Quiz" && <Quiz quiz={quiz} />}
    </div>
  );
}
