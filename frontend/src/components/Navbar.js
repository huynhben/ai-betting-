// components/Navbar.js
import React, { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";

function Navbar() {
  const [modelStatus, setModelStatus] = useState(null);
  const location = useLocation();

  useEffect(() => {
    // Check model status on load
    fetch("http://localhost:8000/model/status")
      .then((res) => res.json())
      .then((data) => setModelStatus(data))
      .catch((err) => console.error("Failed to fetch model status", err));
  }, []);

  const trainModel = () => {
    if (!window.confirm("Training will take several minutes. Continue?")) {
      return;
    }

    fetch("http://localhost:8000/model/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    })
      .then((res) => res.json())
      .then((data) => {
        alert(`Training ${data.status}: ${data.message}`);

        // Refresh model status
        fetch("http://localhost:8000/model/status")
          .then((res) => res.json())
          .then((statusData) => setModelStatus(statusData));
      })
      .catch((err) => {
        console.error("Training failed:", err);
        alert("Training failed: " + err.message);
      });
  };

  const getModelStatusColor = () => {
    if (!modelStatus) return "#6c757d";
    if (modelStatus.model_available && modelStatus.trained_sports?.length > 0)
      return "#28a745";
    if (modelStatus.model_available) return "#ffc107";
    return "#dc3545";
  };

  const getModelStatusText = () => {
    if (!modelStatus) return "Loading...";
    if (modelStatus.model_available && modelStatus.trained_sports?.length > 0) {
      return `Ready (${modelStatus.trained_sports.length} sports)`;
    }
    if (modelStatus.model_available) return "Needs Training";
    return "Not Available";
  };

  const isActive = (path) => location.pathname === path;

  return (
    <nav
      style={{
        backgroundColor: "#FFFFFF",
        padding: "1rem 2rem",
        display: "flex",
        alignItems: "center",
        gap: "2rem",
        position: "sticky",
        top: 0,
        zIndex: 1000,
        borderBottom: "2px solid #007bff",
        boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
      }}
    >
      {/* Logo */}
      <Link to="/" style={{ textDecoration: "none" }}>
        <button
          style={{
            backgroundColor: "white",
            border: "2px solid #007bff",
            padding: "0.5rem 1rem",
            color: "#007bff",
            borderRadius: "8px",
            cursor: "pointer",
            fontWeight: "bold",
            fontSize: "1.5rem",
          }}
        >
          OddsGPT
        </button>
      </Link>

      {/* Navigation Links */}
      <div style={{ display: "flex", gap: "1rem" }}>
        <Link to="/" style={{ textDecoration: "none" }}>
          <button
            style={{
              backgroundColor: isActive("/") ? "#28a745" : "white",
              border: "2px solid #28a745",
              padding: "0.5rem 1rem",
              color: isActive("/") ? "white" : "#28a745",
              fontSize: "1rem",
              borderRadius: "8px",
              cursor: "pointer",
              fontWeight: isActive("/") ? "bold" : "normal",
            }}
          >
            🏠 Home
          </button>
        </Link>

        <Link to="/games" style={{ textDecoration: "none" }}>
          <button
            style={{
              backgroundColor: isActive("/games") ? "#17a2b8" : "white",
              border: "2px solid #17a2b8",
              padding: "0.5rem 1rem",
              color: isActive("/games") ? "white" : "#17a2b8",
              fontSize: "1rem",
              borderRadius: "8px",
              cursor: "pointer",
              fontWeight: isActive("/games") ? "bold" : "normal",
            }}
          >
            🎯 Games
          </button>
        </Link>

        <Link to="/about" style={{ textDecoration: "none" }}>
          <button
            style={{
              backgroundColor: isActive("/about") ? "#6f42c1" : "white",
              border: "2px solid #6f42c1",
              padding: "0.5rem 1rem",
              color: isActive("/about") ? "white" : "#6f42c1",
              fontSize: "1rem",
              borderRadius: "8px",
              cursor: "pointer",
              fontWeight: isActive("/about") ? "bold" : "normal",
            }}
          >
            📊 About
          </button>
        </Link>
      </div>

      {/* Model Status & Training */}
      <div
        style={{
          marginLeft: "auto",
          display: "flex",
          alignItems: "center",
          gap: "1rem",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <div
            style={{
              width: "12px",
              height: "12px",
              borderRadius: "50%",
              backgroundColor: getModelStatusColor(),
            }}
          ></div>
          <span
            style={{ color: "#333", fontSize: "0.9rem", fontWeight: "bold" }}
          >
            ESPN ML: {getModelStatusText()}
          </span>
        </div>

        {modelStatus &&
          (!modelStatus.model_available ||
            modelStatus.trained_sports?.length === 0) && (
            <button
              onClick={trainModel}
              style={{
                backgroundColor: "#17a2b8",
                color: "white",
                border: "none",
                padding: "0.5rem 1rem",
                borderRadius: "6px",
                cursor: "pointer",
                fontSize: "0.9rem",
                fontWeight: "bold",
              }}
            >
              Train Model
            </button>
          )}
      </div>
    </nav>
  );
}

export default Navbar;
