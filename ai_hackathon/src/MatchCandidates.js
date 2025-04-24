import React, { useState } from "react";
import "./MatchCandidates.css";

function MatchCandidates() {
  const [jobDescription, setJobDescription] = useState("");
  const [results, setResults] = useState([]);
  const [weights, setWeights] = useState({
    python: 20,
    java: 20,
    react: 20,
    sql: 20,
    docker: 20
  });

  const handleWeightChange = (skill, value) => {
    setWeights({ ...weights, [skill]: parseInt(value) });
  };

  const handleMatch = () => {
    // Simulează răspunsul
    const simulatedResults = [
      { name: "CV 1", score: 87, industry: 100, skills: 85, semantic: 90 },
      { name: "CV 2", score: 81, industry: 50, skills: 80, semantic: 85 },
    ];
    setResults(simulatedResults);
  };

  return (
    <div className="match-candidates">
      <h2>Match Candidates</h2>
      <textarea
        placeholder="Paste Job Description Here"
        value={jobDescription}
        onChange={(e) => setJobDescription(e.target.value)}
      />
      <div className="sliders">
        {Object.entries(weights).map(([skill, value]) => (
          <div key={skill} className="slider-item">
            <label>{skill}: {value}%</label>
            <input
              type="range"
              min="0"
              max="100"
              value={value}
              onChange={(e) => handleWeightChange(skill, e.target.value)}
            />
          </div>
        ))}
      </div>
      <button onClick={handleMatch}>Find Top 5 Candidates</button>

      <div className="results">
        {results.map((cv, index) => (
          <div key={index} className="result-card">
            <h3>{cv.name} – Score: {cv.score}%</h3>
            <p>Industry: {cv.industry}%</p>
            <p>Skills: {cv.skills}%</p>
            <p>Semantic: {cv.semantic}%</p>
            <p className="explanation">Selected due to strong experience in the relevant industry and technologies required.</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default MatchCandidates;