import React, { useState } from "react";

export default function AskQuestions({ pdfText }) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  const suggested = [
    "What are the main topics in this PDF?",
    "Summarize this document in simple terms.",
    "Explain the key differences mentioned in the text.",
    "What are the most important points I should remember?",
  ];

  async function askQuestion(q) {
    if (!q || !pdfText) return;
    setLoading(true);
    setAnswer("");

    try {
      const res = await fetch("/api/ask_question", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: pdfText, question: q }),
      });
      const data = await res.json();
      const ans = data.answer || "No answer found.";
      setHistory([...history, { question: q, answer: ans }]);
      setAnswer(ans);
    } catch (e) {
      setAnswer("Error: " + e.message);
    } finally {
      setLoading(false);
      setQuestion("");
    }
  }

  return (
    <div className="card">
      <h2>Ask Questions</h2>
      <p>Ask any question about your PDF and get instant answers.</p>

      <div style={{ marginTop: 16 }}>
        <input
          type="text"
          value={question}
          placeholder="Ask a question about your PDF..."
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && askQuestion(question)}
          style={{
            width: "100%",
            padding: 8,
            border: "1px solid #ccc",
            borderRadius: 6,
            marginBottom: 10,
          }}
        />
        <button
          onClick={() => askQuestion(question)}
          disabled={!question || loading}
          className="btn primary"
        >
          {loading ? "Thinking..." : "Ask"}
        </button>
      </div>

      {answer && (
        <div style={{ marginTop: 16, background: "#f9fafb", padding: 10, borderRadius: 8 }}>
          <b>Answer:</b>
          <p>{answer}</p>
        </div>
      )}

      {history.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <h4>Previous Questions</h4>
          {history.map((h, i) => (
            <div key={i} style={{ marginBottom: 10 }}>
              <div><b>Q:</b> {h.question}</div>
              <div><b>A:</b> {h.answer}</div>
            </div>
          ))}
        </div>
      )}

      <div style={{ marginTop: 20 }}>
        <h4>Suggested Questions:</h4>
        {suggested.map((s, i) => (
          <button
            key={i}
            onClick={() => askQuestion(s)}
            className="btn secondary"
            style={{ margin: 4 }}
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
