import React from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import MatchCandidates from "./MatchCandidates";
import MatchJobs from "./MatchJobs";
import "./App.css";

function App() {
  return (
    <Router>
      <div className="app">
        <nav className="navbar">
          <h1>DEVMATCH</h1>
          <ul className="nav-links">
            <li><Link to="/">Match Candidates</Link></li>
            <li><Link to="/match-jobs">Match Jobs</Link></li>
          </ul>
        </nav>

        <Routes>
          <Route path="/" element={<MatchCandidates />} />
          <Route path="/match-jobs" element={<MatchJobs />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;