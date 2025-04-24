import React, { useState } from "react";
import "./MatchJobs.css";

function MatchJobs() {
  const [cvFile, setCvFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setCvFile(file);
      setError("");
    }
  };

  const handleMatch = () => {
    if (!cvFile) {
      setError("⚠️ Please upload a CV file first.");
      return;
    }

    setError("");
    setLoading(true);
    setTimeout(() => {
      setResult({
        title: "Backend Developer – Banking Project",
        score: 94,
        industry: 100,
        skills: 89,
        semantic: 91,
        explanation:
          "Matched perfectly due to strong experience in backend tech and financial projects.",
      });
      setLoading(false);
    }, 2000);
  };

  return (
    <div className="match-jobs glass-container">
      <h2>📄 Match Job for Uploaded CV</h2>

      <input type="file" accept=".txt,.docx" onChange={handleUpload} />
      {cvFile && (
        <p className="file-info">Uploaded: <strong>{cvFile.name}</strong></p>
      )}
      {error && <p className="error">{error}</p>}

      <button className="glow-button" onClick={handleMatch}>
        Find Best Matching Job
      </button>

      {loading && <div className="glow-spinner"></div>}

      {result && (
        <div className="result-card glass-card">
          <div className="badge">🏆 Best Match</div>
          <h3>{result.title}</h3>
          <div className="score-circle">{result.score}%</div>
          <ScoreBar
            label="Industry"
            value={result.industry}
            gradient="linear-gradient(to right, #00c9ff, #92fe9d)"
          />
          <ScoreBar
            label="Skills"
            value={result.skills}
            gradient="linear-gradient(to right, #fcb045, #fd1d1d, #833ab4)"
          />
          <ScoreBar
            label="Semantic"
            value={result.semantic}
            gradient="linear-gradient(to right, #43e97b, #38f9d7)"
          />
          <p className="explanation">{result.explanation}</p>
        </div>
      )}
    </div>
  );
}

function ScoreBar({ label, value, gradient }) {
  return (
    <div className="score-bar">
      <span>{label}: {value}%</span>
      <div className="bar-bg">
        <div
          className="bar-fill"
          style={{ width: `${value}%`, background: gradient }}
        />
      </div>
    </div>
  );
}

export default MatchJobs;
