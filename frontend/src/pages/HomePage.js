// pages/HomePage.js
import React from "react";
import { Link } from "react-router-dom";

function HomePage() {
  return (
    <div style={{ padding: "2rem", maxWidth: "1200px", margin: "0 auto" }}>
      {/* Hero Section */}
      <div
        style={{
          textAlign: "center",
          padding: "4rem 2rem",
          background: "linear-gradient(135deg, #007bff 0%, #6610f2 100%)",
          borderRadius: "16px",
          color: "white",
          marginBottom: "3rem",
        }}
      >
        <h1
          style={{ fontSize: "3rem", marginBottom: "1rem", fontWeight: "bold" }}
        >
          🤖 OddsGPT
        </h1>
        <p style={{ fontSize: "1.3rem", marginBottom: "2rem", opacity: 0.9 }}>
          AI-Powered Sports Betting Analysis with ESPN Machine Learning
        </p>
        <Link to="/games" style={{ textDecoration: "none" }}>
          <button
            style={{
              backgroundColor: "#28a745",
              color: "white",
              border: "none",
              padding: "1rem 2rem",
              borderRadius: "8px",
              fontSize: "1.2rem",
              fontWeight: "bold",
              cursor: "pointer",
              boxShadow: "0 4px 8px rgba(0,0,0,0.2)",
            }}
          >
            🎯 Start Analyzing Games
          </button>
        </Link>
      </div>

      {/* Features Grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
          gap: "2rem",
          marginBottom: "3rem",
        }}
      >
        <div
          style={{
            backgroundColor: "#f8f9fa",
            padding: "2rem",
            borderRadius: "12px",
            border: "1px solid #ddd",
            textAlign: "center",
          }}
        >
          <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>🏆</div>
          <h3 style={{ color: "#333", marginBottom: "1rem" }}>
            ESPN ML Predictions
          </h3>
          <p style={{ color: "#666", lineHeight: "1.6" }}>
            Advanced machine learning models trained on 6,717+ games from ESPN
            API. Our ensemble approach combines XGBoost, Random Forest, and
            Gradient Boosting for superior accuracy.
          </p>
        </div>

        <div
          style={{
            backgroundColor: "#f8f9fa",
            padding: "2rem",
            borderRadius: "12px",
            border: "1px solid #ddd",
            textAlign: "center",
          }}
        >
          <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>📊</div>
          <h3 style={{ color: "#333", marginBottom: "1rem" }}>
            Live Odds Integration
          </h3>
          <p style={{ color: "#666", lineHeight: "1.6" }}>
            Real-time odds and spreads from major sportsbooks. Compare our AI
            predictions with current betting lines to identify value
            opportunities and make informed decisions.
          </p>
        </div>

        <div
          style={{
            backgroundColor: "#f8f9fa",
            padding: "2rem",
            borderRadius: "12px",
            border: "1px solid #ddd",
            textAlign: "center",
          }}
        >
          <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>💬</div>
          <h3 style={{ color: "#333", marginBottom: "1rem" }}>
            ChatGPT Analysis
          </h3>
          <p style={{ color: "#666", lineHeight: "1.6" }}>
            Get detailed betting analysis powered by ChatGPT. Our AI considers
            team form, head-to-head records, and market conditions to provide
            comprehensive insights.
          </p>
        </div>
      </div>

      {/* Supported Sports */}
      <div
        style={{
          backgroundColor: "white",
          padding: "2rem",
          borderRadius: "12px",
          border: "1px solid #ddd",
          marginBottom: "3rem",
        }}
      >
        <h2
          style={{ textAlign: "center", color: "#333", marginBottom: "2rem" }}
        >
          Supported Sports
        </h2>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
            gap: "1.5rem",
          }}
        >
          <div style={{ textAlign: "center", padding: "1rem" }}>
            <div style={{ fontSize: "4rem", marginBottom: "0.5rem" }}>🏀</div>
            <h4 style={{ color: "#333", margin: 0 }}>NBA</h4>
            <p
              style={{
                color: "#666",
                fontSize: "0.9rem",
                margin: "0.5rem 0 0 0",
              }}
            >
              Basketball
            </p>
          </div>
          <div style={{ textAlign: "center", padding: "1rem" }}>
            <div style={{ fontSize: "4rem", marginBottom: "0.5rem" }}>🏈</div>
            <h4 style={{ color: "#333", margin: 0 }}>NFL</h4>
            <p
              style={{
                color: "#666",
                fontSize: "0.9rem",
                margin: "0.5rem 0 0 0",
              }}
            >
              American Football
            </p>
          </div>
          <div style={{ textAlign: "center", padding: "1rem" }}>
            <div style={{ fontSize: "4rem", marginBottom: "0.5rem" }}>⚾</div>
            <h4 style={{ color: "#333", margin: 0 }}>MLB</h4>
            <p
              style={{
                color: "#666",
                fontSize: "0.9rem",
                margin: "0.5rem 0 0 0",
              }}
            >
              Baseball
            </p>
          </div>
          <div style={{ textAlign: "center", padding: "1rem" }}>
            <div style={{ fontSize: "4rem", marginBottom: "0.5rem" }}>🏒</div>
            <h4 style={{ color: "#333", margin: 0 }}>NHL</h4>
            <p
              style={{
                color: "#666",
                fontSize: "0.9rem",
                margin: "0.5rem 0 0 0",
              }}
            >
              Hockey
            </p>
          </div>
        </div>
      </div>

      {/* Call to Action */}
      <div
        style={{
          textAlign: "center",
          padding: "3rem 2rem",
          backgroundColor: "#f8f9fa",
          borderRadius: "12px",
          border: "1px solid #ddd",
        }}
      >
        <h2 style={{ color: "#333", marginBottom: "1rem" }}>
          Ready to Start Winning?
        </h2>
        <p style={{ color: "#666", marginBottom: "2rem", fontSize: "1.1rem" }}>
          Our AI has analyzed thousands of games to give you the edge you need.
        </p>
        <div
          style={{
            display: "flex",
            gap: "1rem",
            justifyContent: "center",
            flexWrap: "wrap",
          }}
        >
          <Link to="/games" style={{ textDecoration: "none" }}>
            <button
              style={{
                backgroundColor: "#007bff",
                color: "white",
                border: "none",
                padding: "1rem 2rem",
                borderRadius: "8px",
                fontSize: "1.1rem",
                fontWeight: "bold",
                cursor: "pointer",
              }}
            >
              View Live Games
            </button>
          </Link>
          <Link to="/about" style={{ textDecoration: "none" }}>
            <button
              style={{
                backgroundColor: "white",
                color: "#007bff",
                border: "2px solid #007bff",
                padding: "1rem 2rem",
                borderRadius: "8px",
                fontSize: "1.1rem",
                fontWeight: "bold",
                cursor: "pointer",
              }}
            >
              Learn More
            </button>
          </Link>
        </div>
      </div>
    </div>
  );
}

export default HomePage;
