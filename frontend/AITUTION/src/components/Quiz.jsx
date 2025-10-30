import React, { useMemo, useState } from "react";

export default function Quiz({ quiz = [] }) {
  const [qIndex, setQIndex] = useState(0);
  const [selected, setSelected] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);

  // Freeze options order per mount
  const prepared = useMemo(() => {
    return quiz.map(q => {
      const options = [...(q.options || [])];
      for (let i = options.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [options[i], options[j]] = [options[j], options[i]];
      }
      return { ...q, options };
    });
  }, [quiz]);

  const total = prepared.length;
  const q = prepared[qIndex] || null;

  function normalize(s){ return String(s||"").trim().toLowerCase(); }

  function pickOption(i){
    if (submitted) return;
    setSelected(prev => ({ ...prev, [qIndex]: i }));
  }

  function submit(){
    let s = 0;
    prepared.forEach((item, i) => {
      const sel = selected[i];
      if (sel == null) return;
      if (normalize(item.options[sel]) === normalize(item.answer)) s++;
    });
    setScore(s);
    setSubmitted(true);
  }

  return (
    <div className="card">
      <div className="row" style={{justifyContent:"space-between",marginBottom:10}}>
        <div className="section-title" style={{margin:0}}>Quiz</div>
        <div className="small">Question {Math.min(qIndex+1,total)} of {total}</div>
      </div>

      {/* progress bar */}
      <div className="progress" style={{margin:"8px 0 14px"}}>
        <span style={{ width: total ? `${((qIndex+1)/total)*100}%` : "0%" }} />
      </div>

      {q ? (
        <div className="quiz-card">
          <div style={{fontWeight:700, marginBottom:10}}>{q.question}</div>
          {(q.options || []).map((opt, i) => {
            const chosen = selected[qIndex] === i;
            const isRight = submitted && normalize(opt) === normalize(q.answer);
            const isWrongChosen = submitted && chosen && !isRight;
            const cls = ["option", chosen ? "selected" : "", isRight ? "correct" : "", isWrongChosen ? "incorrect" : ""].join(" ");
            return (
              <div className={cls} key={i} onClick={() => pickOption(i)}>{opt}</div>
            );
          })}

          <div className="row" style={{justifyContent:"space-between",marginTop:14}}>
            <button className="btn secondary" onClick={()=>setQIndex(i=>Math.max(0,i-1))} disabled={qIndex===0}>Previous</button>
            {!submitted ? (
              <button className="btn" onClick={submit} disabled={Object.keys(selected).length < total}>Submit Answer</button>
            ) : (
              <div className="row">
                <div className="badge">Score: {score}/{total}</div>
                <button className="btn" onClick={()=>{ setSelected({}); setSubmitted(false); setScore(0); }}>Retry</button>
              </div>
            )}
            <button className="btn secondary" onClick={()=>setQIndex(i=>Math.min(total-1,i+1))} disabled={qIndex===total-1}>Next</button>
          </div>

          {submitted && (
            <div className="small" style={{marginTop:10}}>
              <strong>Answer:</strong> {q.answer}
              {q.explanation && <div><strong>Why:</strong> {q.explanation}</div>}
            </div>
          )}
        </div>
      ) : (
        <div className="small">No questions yet.</div>
      )}
    </div>
  );
}
