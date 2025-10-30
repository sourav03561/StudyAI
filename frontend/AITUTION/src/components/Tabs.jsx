import React from "react";
import { BookOpen, Brain, ListChecks } from "lucide-react";

const tabs = [
  { label: "Summary", icon: <BookOpen className="w-4 h-4" /> },
  { label: "Flashcards", icon: <Brain className="w-4 h-4" /> },
  { label: "Quiz", icon: <ListChecks className="w-4 h-4" /> },
];

export default function Tabs({ value, onChange }) {
  return (
    <div className="tabs">
      {tabs.map(t => (
        <div
          key={t.label}
          className={"tab" + (value === t.label ? " active" : "")}
          onClick={() => onChange(t.label)}
          role="button"
          aria-pressed={value === t.label}
        >
          <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            {t.icon}
            {t.label}
          </span>
        </div>
      ))}
    </div>
  );
}

