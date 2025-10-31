import React from "react";

export default function RecommendedVideos({ keyPoints = [], videos = [] }) {
  return (
    <div style={{ marginTop: 10 }}>
      <h2 style={{ marginBottom: 8, textAlign: "center" }}>Recommended Videos</h2>
      {(!videos || videos.length === 0) && (
        <div style={{ padding: 18, background: "#f8fafc", borderRadius: 8 }}>
          No videos found yet. Open the "Recommended Videos" tab after generating study materials.
        </div>
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
          gap: 14,
          marginTop: 12,
        }}
      >
        {videos?.map((v, i) => (
          <a
            key={v.videoId || i}
            href={v.url || `https://www.youtube.com/watch?v=${v.videoId}`}
            target="_blank"
            rel="noreferrer"
            style={{
              textDecoration: "none",
              color: "inherit",
              border: "1px solid #e6e9ef",
              borderRadius: 10,
              overflow: "hidden",
              background: "#fff",
              boxShadow: "0 1px 2px rgba(16,24,40,0.04)",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <div style={{ height: 150, background: `url(${v.thumbnail}) center/cover no-repeat` }} />
            <div style={{ padding: 10 }}>
              <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 6 }}>{v.title}</div>
              <div style={{ fontSize: 12, color: "#666" }}>{v.channelTitle}</div>
              <div style={{ fontSize: 12, color: "#999", marginTop: 8 }}>
                {v.viewCount ? `${Number(v.viewCount).toLocaleString()} views` : ""}
              </div>
            </div>
          </a>
        ))}
      </div>
    </div>
  );
}
