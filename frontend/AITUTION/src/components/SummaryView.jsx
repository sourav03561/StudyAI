import React from "react";
import { CheckCircle2 } from "lucide-react";

export default function SummaryView({
  fileName,
  keyTopics = [],
  keyPoints = [],
  // fallback if backend didn’t fill arrays yet:
  summary = ""
}) {
  return (
    <>
          <div className="subtle" style={{display:"flex",alignItems:"center",gap:8,marginBottom:6}}>
        <span style={{fontSize:18}}>📄</span> {fileName || "Your document"}
      </div>
    <div className="card" style={{ marginBottom: 20 }}>
      <div className="section-title">Key Topics</div>
      {keyTopics.length ? (
        <div style={{marginBottom:12}}>
          {keyTopics.map((t,i)=>(<span className="kpill" key={i}>{t}</span>))}
        </div>
      ) : (
        <div className="small">No topics detected.</div>
      )}
      </div>
      <div className="card" style={{ marginBottom: 20 }}>
      <div className="section-title">Key Points</div>
      {keyPoints.length ? (
        <ul style={{listStyle:"none", padding:0, margin:0}}>
          {keyPoints.map((p,i)=>(
            <li key={i} style={{display:"flex", gap:10, alignItems:"flex-start", margin:"10px 0"}}>
              <CheckCircle2 size={18} color="#16a34a" style={{flexShrink:0, marginTop:2}} />
              <div style={{lineHeight:1.5}}>{p}</div>
            </li>
          ))}
        </ul>
      ) : (
        <textarea className="textbox" readOnly value={summary || "No key points yet."} />
      )}
      </div>
    </>
  );
}
