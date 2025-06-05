// App.js - Main App with Routing
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import "./App.css";

// Import your pages
import HomePage from "./pages/HomePage";
import GamesPage from "./pages/GamesPage";
import GameDetailsPage from "./pages/GameDetailsPage";
import AboutPage from "./pages/AboutPage";
import Navbar from "./components/Navbar";

function App() {
  return (
    <Router>
      <div style={{ fontFamily: "Arial" }}>
        <Navbar />
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/games" element={<GamesPage />} />
          <Route
            path="/game/:sport/:team1/:team2"
            element={<GameDetailsPage />}
          />
          <Route path="/about" element={<AboutPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
