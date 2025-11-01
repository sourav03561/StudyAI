import React from "react";
import { BookOpen, MessageCircle, Brain, ListChecks, Video } from "lucide-react";

const tabs = [
  { label: "Summary", icon: <BookOpen className="w-4 h-4" /> },
  { label: "Ask", icon: <MessageCircle className="w-4 h-4" /> },
  { label: "Flashcards", icon: <Brain className="w-4 h-4" /> },
  { label: "Quiz", icon: <ListChecks className="w-4 h-4" /> },
  { label: "Recommended Videos", icon: <Video className="w-4 h-4" /> },
];

export default function Tabs({ value, onChange }) {
  return (
    <div
      style={{
        display: "flex",
        gap: "12px",
        background: "#f4f5f7",
        borderRadius: "9999px", // full pill shape
        padding: "6px 10px",
        justifyContent: "center",
        alignItems: "center",
        boxShadow: "inset 0 0 3px rgba(0,0,0,0.05)",
      }}
    >
      {tabs.map((t) => (
        <button
          key={t.label}
          onClick={() => onChange(t.label)}
          className="tab-btn"
          style={{
            display: "flex",
            alignItems: "center",
            gap: "6px",
            padding: "6px 16px",
            borderRadius: "9999px",
            border: value === t.label ? "1px solid #ccc" : "1px solid transparent",
            background: value === t.label ? "#fff" : "transparent",
            fontWeight: value === t.label ? "600" : "500",
            cursor: "pointer",
            color: value === t.label ? "#000" : "#444",
            transition: "all 0.2s ease",
          }}
        >
          {t.icon}
          <span>{t.label}</span>
        </button>
      ))}
    </div>
  );
}
