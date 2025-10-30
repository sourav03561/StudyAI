import React, { useState } from "react";

export default function FlashcardsView({ cards = [] }) {
  const [idx, setIdx] = useState(0);
  const [showBack, setShowBack] = useState(false);
  const total = cards.length || 0;
  const card = cards[idx] || { front: "No cards", back: "" };

  function prev(){ setShowBack(false); setIdx(i => Math.max(0, i-1)); }
  function next(){ setShowBack(false); setIdx(i => Math.min(total-1, i+1)); }
  function toggle(){ setShowBack(s => !s); }

  return (
    <div className="card fc-wrap">
      <div className="section-title">Flashcards</div>
      <div className="small">Card {Math.min(idx+1,total)} of {total}</div>

      <div className="fc-card" onClick={toggle}>
        <div className="small" style={{position:"absolute",top:10,right:14,color:"#64748b"}}>{showBack ? "Answer" : "Question"}</div>
        <div style={{fontSize:18, lineHeight:"1.5"}}>
          {showBack ? (card.back || "—") : (card.front || "—")}
        </div>
        <div className="small" style={{position:"absolute",bottom:12,color:"#64748b"}}>Click to {showBack ? "see question" : "reveal answer"}</div>
      </div>

      <div className="fc-ctl">
        <button className="btn secondary" onClick={prev} disabled={idx===0}>Previous</button>
        <div className="fc-indicators">
          {Array.from({length: total}).map((_,i)=>(
            <span className={"dot"+(i===idx?" active":"")} key={i}></span>
          ))}
        </div>
        <button className="btn secondary" onClick={next} disabled={idx===total-1}>Next</button>
      </div>
    </div>
  );
}
