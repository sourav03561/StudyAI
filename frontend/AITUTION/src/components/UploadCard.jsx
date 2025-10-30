import React from "react";

export default function UploadCard({ file, setFile, onGenerate, loading }) {
  return (
    <div className="upload-wrap card">
      <div className="upload-inner">
        <div className="upload-icon">📄</div>
        <div style={{fontWeight:700,fontSize:18}}>Upload Your PDF</div>
        <div className="subtle">We’ll create a study pack: summary, flashcards, and quiz</div>
        <label className="btn">
          <input
            type="file"
            accept="application/pdf"
            onChange={e => setFile(e.target.files?.[0] || null)}
            hidden
          />
          Choose PDF File
        </label>
        {file && <div className="small">Selected: {file.name}</div>}

        {/* Single action: build everything */}
        <button
          className="btn"
          style={{marginTop:8}}
          onClick={onGenerate}
          disabled={!file || loading}
        >
          {loading ? "Generating…" : "Generate Study Pack"}
        </button>
      </div>
    </div>
  );
}
