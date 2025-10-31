import React from "react";
import { BookOpen, Brain, ListChecks, Video } from "lucide-react";

const tabs = [
  { label: "Summary", icon: <BookOpen /> },
  { label: "Flashcards", icon: <Brain /> },
  { label: "Quiz", icon: <ListChecks /> },
  { label: "Recommended Videos", icon: <Video /> },
];

export default function Tabs({ value, onChange }) {
  return (
    <div style={{ display: "flex", gap: 8 }}>
      {tabs.map((t) => {
        const active = t.label === value;
        return (
          <button
            key={t.label}
            onClick={() => onChange(t.label)}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 8,
              padding: "8px 14px",
              borderRadius: 8,
              border: "1px solid rgba(0,0,0,0.06)",
              background: active ? "#0b74ff" : "#f6f7fb",
              color: active ? "#fff" : "#111",
              cursor: "pointer",
              fontWeight: 600,
            }}
            aria-pressed={active}
          >
            {t.icon}
            <span style={{ marginLeft: 4 }}>{t.label}</span>
          </button>
        );
      })}
    </div>
  );
}
