// pages/GameDetailsPage.js
import React, { useState, useEffect } from "react";
import { useLocation, useParams, useNavigate } from "react-router-dom";

function GameDetailsPage() {
  const location = useLocation();
  const params = useParams();
  const navigate = useNavigate();

  // Get game data from navigation state or reconstruct from params
  const [game, setGame] = useState(location.state?.game || null);
  const [sportKey] = useState(
    location.state?.sportKey || params.sport || "basketball_nba"
  );

  const [prediction, setPrediction] = useState(null);
  const [chatGPTAnalysis, setChatGPTAnalysis] = useState(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [modelStatus, setModelStatus] = useState(null);

  // Check model status on load
  useEffect(() => {
    fetch("http://localhost:8000/model/status")
      .then((res) => res.json())
      .then((data) => {
        console.log("Model status:", data);
        setModelStatus(data);
      })
      .catch((err) => console.error("Failed to fetch model status", err));
  }, []);

  // If no game data from navigation state, try to fetch or reconstruct
  useEffect(() => {
    if (!game && params.team1 && params.team2) {
      // Try to reconstruct game data from URL params
      const reconstructedGame = {
        team_1: decodeURIComponent(params.team1.replace(/-/g, " ")),
        team_2: decodeURIComponent(params.team2.replace(/-/g, " ")),
        date: new Date().toISOString(), // Placeholder
        odds: null,
        spreads: null,
      };
      setGame(reconstructedGame);
    }
  }, [params, game]);

  const handlePredict = () => {
    if (!game) {
      console.warn("No game data available.");
      return;
    }

    setPredictionLoading(true);
    setPrediction(null);

    // Send correct data format for ESPN ML model
    const requestData = {
      sport: sportKey,
      team_1: game.team_1, // Home team
      team_2: game.team_2, // Away team
      // Keep backward compatibility
      team_id_home: game.team_1,
      team_id_away: game.team_2,
      fg_pct_home: 45.0,
      reb_home: 43,
      ast_home: 25,
      fg_pct_away: 45.0,
      reb_away: 43,
      ast_away: 25,
    };

    console.log("🎯 Sending ESPN ML prediction request:", requestData);

    fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestData),
    })
      .then((res) => res.json())
      .then((data) => {
        console.log("✅ ESPN ML prediction response:", data);
        setPrediction(data);
      })
      .catch((err) => {
        console.error("❌ Prediction failed:", err);
        setPrediction({
          error: "Prediction failed: " + err.message,
          confidence: 0.5,
          home_win_probability: 0.5,
          away_win_probability: 0.5,
        });
      })
      .finally(() => {
        setPredictionLoading(false);
      });
  };

  const handleChatGPTAnalysis = () => {
    if (!game) {
      console.warn("No game data available.");
      return;
    }

    setAnalysisLoading(true);

    // Enhanced prompt with ML prediction data
    let prompt = `Game Analysis:
${game.team_1} vs ${game.team_2} (${sportKey
      .replace("_", " ")
      .toUpperCase()}).`;

    if (game.odds) {
      prompt += `
Odds: ${game.team_1} ${game.odds[game.team_1]}, ${game.team_2} ${
        game.odds[game.team_2]
      }.`;
    }

    if (game.spreads) {
      prompt += `
Spread: ${game.team_1} ${game.spreads[game.team_1]?.point ?? "N/A"} (${
        game.spreads[game.team_1]?.price ?? "N/A"
      }), ${game.team_2} ${game.spreads[game.team_2]?.point ?? "N/A"} (${
        game.spreads[game.team_2]?.price ?? "N/A"
      }).`;
    }

    // Add ML prediction if available
    if (prediction && !prediction.error) {
      prompt += `

🤖 ESPN AI MODEL PREDICTION:
- Predicted Winner: ${prediction.prediction}
- Confidence: ${(prediction.confidence * 100).toFixed(1)}%
- Home Win Probability: ${(prediction.home_win_probability * 100).toFixed(1)}%
- Away Win Probability: ${(prediction.away_win_probability * 100).toFixed(1)}%
- Model: ${prediction.model_info?.type || "ESPN Ensemble"}
- Trained on: 6,717+ games from ESPN API`;
    }

    prompt += `

Should I bet on this game? Please analyze both the odds and the AI prediction to give betting advice.`;

    console.log("📝 Prompt sent to ChatGPT:", prompt);

    fetch("http://localhost:8000/analyze-bet/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    })
      .then((res) => res.json())
      .then((data) => {
        console.log("💬 ChatGPT response:", data);
        setChatGPTAnalysis(data.analysis || data.error || "No response");
      })
      .catch((err) => {
        console.error("Error contacting ChatGPT:", err);
        setChatGPTAnalysis("Error contacting ChatGPT.");
      })
      .finally(() => {
        setAnalysisLoading(false);
      });
  };

  const getSportDisplayName = (sport) => {
    const sportNames = {
      basketball_nba: "🏀 NBA",
      americanfootball_nfl: "🏈 NFL",
      baseball_mlb: "⚾ MLB",
      icehockey_nhl: "🏒 NHL",
      soccer_epl: "⚽ EPL",
    };
    return sportNames[sport] || sport;
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

  if (!game) {
    return (
      <div style={{ padding: "2rem", textAlign: "center" }}>
        <h2>Game not found</h2>
        <button
          onClick={() => navigate("/games")}
          style={{
            backgroundColor: "#007bff",
            color: "white",
            border: "none",
            padding: "0.75rem 1.5rem",
            borderRadius: "6px",
            cursor: "pointer",
            fontWeight: "bold",
          }}
        >
          ← Back to Games
        </button>
      </div>
    );
  }

  return (
    <div style={{ padding: "2rem", maxWidth: "1200px", margin: "0 auto" }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: "2rem",
        }}
      >
        <button
          onClick={() => navigate("/games")}
          style={{
            backgroundColor: "#6c757d",
            color: "white",
            border: "none",
            padding: "0.5rem 1rem",
            borderRadius: "6px",
            cursor: "pointer",
            fontWeight: "bold",
          }}
        >
          ← Back to Games
        </button>

        {/* Model Status Indicator */}
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
      </div>

      {/* Game Header */}
      <div style={{ textAlign: "center", marginBottom: "3rem" }}>
        <h1
          style={{ color: "#333", marginBottom: "0.5rem", fontSize: "2.5rem" }}
        >
          {game.team_2} @ {game.team_1}
        </h1>
        <p style={{ color: "#666", fontSize: "1.2rem", margin: "0 0 1rem 0" }}>
          {getSportDisplayName(sportKey)}
        </p>

        {/* Date and Time Section */}
        {game.date && (
          <div
            style={{
              backgroundColor: "#f8f9fa",
              border: "1px solid #dee2e6",
              borderRadius: "8px",
              padding: "1rem",
              display: "inline-block",
              marginTop: "0.5rem",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "1rem",
              }}
            >
              <div style={{ textAlign: "center" }}>
                <div
                  style={{
                    fontSize: "0.9rem",
                    color: "#666",
                    fontWeight: "bold",
                  }}
                >
                  📅 DATE
                </div>
                <div
                  style={{
                    fontSize: "1.1rem",
                    color: "#333",
                    fontWeight: "bold",
                  }}
                >
                  {new Date(game.date).toLocaleDateString("en-US", {
                    weekday: "long",
                    year: "numeric",
                    month: "long",
                    day: "numeric",
                  })}
                </div>
              </div>

              <div
                style={{
                  width: "1px",
                  height: "40px",
                  backgroundColor: "#dee2e6",
                }}
              ></div>

              <div style={{ textAlign: "center" }}>
                <div
                  style={{
                    fontSize: "0.9rem",
                    color: "#666",
                    fontWeight: "bold",
                  }}
                >
                  🕐 TIME
                </div>
                <div
                  style={{
                    fontSize: "1.1rem",
                    color: "#333",
                    fontWeight: "bold",
                  }}
                >
                  {new Date(game.date).toLocaleTimeString("en-US", {
                    hour: "numeric",
                    minute: "2-digit",
                    timeZoneName: "short",
                  })}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Game Details Grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
          gap: "2rem",
          marginBottom: "3rem",
        }}
      >
        {/* Home Team Section */}
        <div
          style={{
            backgroundColor: "#f8f9fa",
            border: "2px solid #28a745",
            borderRadius: "12px",
            padding: "2rem",
            textAlign: "center",
          }}
        >
          <h2
            style={{
              color: "#28a745",
              marginBottom: "1rem",
              fontSize: "1.5rem",
            }}
          >
            🏠 Home Team
          </h2>
          <h3 style={{ color: "#333", fontSize: "2rem", margin: "0.5rem 0" }}>
            {game.team_1}
          </h3>
          {game.odds && (
            <div style={{ marginTop: "1rem" }}>
              <p
                style={{
                  color: "#666",
                  fontSize: "0.9rem",
                  margin: "0.25rem 0",
                }}
              >
                Moneyline Odds
              </p>
              <div
                style={{
                  fontSize: "1.5rem",
                  fontWeight: "bold",
                  color: game.odds[game.team_1] > 0 ? "#28a745" : "#dc3545",
                }}
              >
                {game.odds[game.team_1] > 0 ? "+" : ""}
                {game.odds[game.team_1]}
              </div>
            </div>
          )}
        </div>

        {/* Away Team Section */}
        <div
          style={{
            backgroundColor: "#f8f9fa",
            border: "2px solid #dc3545",
            borderRadius: "12px",
            padding: "2rem",
            textAlign: "center",
          }}
        >
          <h2
            style={{
              color: "#dc3545",
              marginBottom: "1rem",
              fontSize: "1.5rem",
            }}
          >
            ✈️ Away Team
          </h2>
          <h3 style={{ color: "#333", fontSize: "2rem", margin: "0.5rem 0" }}>
            {game.team_2}
          </h3>
          {game.odds && (
            <div style={{ marginTop: "1rem" }}>
              <p
                style={{
                  color: "#666",
                  fontSize: "0.9rem",
                  margin: "0.25rem 0",
                }}
              >
                Moneyline Odds
              </p>
              <div
                style={{
                  fontSize: "1.5rem",
                  fontWeight: "bold",
                  color: game.odds[game.team_2] > 0 ? "#28a745" : "#dc3545",
                }}
              >
                {game.odds[game.team_2] > 0 ? "+" : ""}
                {game.odds[game.team_2]}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Betting Information */}
      {(game.odds || game.spreads) && (
        <div
          style={{
            backgroundColor: "#ffffff",
            border: "1px solid #ddd",
            borderRadius: "12px",
            padding: "2rem",
            marginBottom: "3rem",
          }}
        >
          <h2
            style={{ color: "#333", marginBottom: "2rem", textAlign: "center" }}
          >
            📊 Betting Information
          </h2>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
              gap: "2rem",
            }}
          >
            {/* Moneyline Section */}
            {game.odds && (
              <div
                style={{
                  backgroundColor: "#e8f5e8",
                  padding: "1.5rem",
                  borderRadius: "8px",
                  border: "1px solid #c3e6cb",
                }}
              >
                <h3
                  style={{
                    color: "#155724",
                    marginBottom: "1rem",
                    textAlign: "center",
                  }}
                >
                  💰 Moneyline
                </h3>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    marginBottom: "0.5rem",
                  }}
                >
                  <span style={{ fontWeight: "bold" }}>{game.team_1}:</span>
                  <span
                    style={{
                      fontWeight: "bold",
                      color: game.odds[game.team_1] > 0 ? "#28a745" : "#dc3545",
                    }}
                  >
                    {game.odds[game.team_1] > 0 ? "+" : ""}
                    {game.odds[game.team_1]}
                  </span>
                </div>
                <div
                  style={{ display: "flex", justifyContent: "space-between" }}
                >
                  <span style={{ fontWeight: "bold" }}>{game.team_2}:</span>
                  <span
                    style={{
                      fontWeight: "bold",
                      color: game.odds[game.team_2] > 0 ? "#28a745" : "#dc3545",
                    }}
                  >
                    {game.odds[game.team_2] > 0 ? "+" : ""}
                    {game.odds[game.team_2]}
                  </span>
                </div>
              </div>
            )}

            {/* Spread Section */}
            {game.spreads && (
              <div
                style={{
                  backgroundColor: "#e3f2fd",
                  padding: "1.5rem",
                  borderRadius: "8px",
                  border: "1px solid #bbdefb",
                }}
              >
                <h3
                  style={{
                    color: "#1565c0",
                    marginBottom: "1rem",
                    textAlign: "center",
                  }}
                >
                  📈 Point Spread
                </h3>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    marginBottom: "0.5rem",
                  }}
                >
                  <span style={{ fontWeight: "bold" }}>{game.team_1}:</span>
                  <span style={{ fontWeight: "bold" }}>
                    {game.spreads[game.team_1]?.point > 0 ? "+" : ""}
                    {game.spreads[game.team_1]?.point}
                    {game.spreads[game.team_1]?.price && (
                      <span style={{ color: "#666", marginLeft: "0.5rem" }}>
                        ({game.spreads[game.team_1].price > 0 ? "+" : ""}
                        {game.spreads[game.team_1].price})
                      </span>
                    )}
                  </span>
                </div>
                <div
                  style={{ display: "flex", justifyContent: "space-between" }}
                >
                  <span style={{ fontWeight: "bold" }}>{game.team_2}:</span>
                  <span style={{ fontWeight: "bold" }}>
                    {game.spreads[game.team_2]?.point > 0 ? "+" : ""}
                    {game.spreads[game.team_2]?.point}
                    {game.spreads[game.team_2]?.price && (
                      <span style={{ color: "#666", marginLeft: "0.5rem" }}>
                        ({game.spreads[game.team_2].price > 0 ? "+" : ""}
                        {game.spreads[game.team_2].price})
                      </span>
                    )}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div
        style={{
          display: "flex",
          gap: "1rem",
          justifyContent: "center",
          marginBottom: "3rem",
          flexWrap: "wrap",
        }}
      >
        <button
          onClick={handlePredict}
          disabled={predictionLoading}
          style={{
            backgroundColor: predictionLoading ? "#6c757d" : "#007bff",
            color: "white",
            border: "none",
            padding: "1rem 2rem",
            borderRadius: "8px",
            cursor: predictionLoading ? "not-allowed" : "pointer",
            fontWeight: "bold",
            fontSize: "1.1rem",
            minWidth: "200px",
            transition: "background-color 0.2s ease",
          }}
        >
          {predictionLoading ? "🔄 Analyzing..." : "🎯 AI Prediction"}
        </button>

        <button
          onClick={handleChatGPTAnalysis}
          disabled={analysisLoading}
          style={{
            backgroundColor: analysisLoading ? "#6c757d" : "#28a745",
            color: "white",
            border: "none",
            padding: "1rem 2rem",
            borderRadius: "8px",
            cursor: analysisLoading ? "not-allowed" : "pointer",
            fontWeight: "bold",
            fontSize: "1.1rem",
            minWidth: "200px",
            transition: "background-color 0.2s ease",
          }}
        >
          {analysisLoading ? "🔄 Analyzing..." : "💬 ChatGPT Analysis"}
        </button>
      </div>

      {/* ML Prediction Results */}
      {prediction && (
        <div
          style={{
            backgroundColor: prediction.error ? "#f8d7da" : "#d4edda",
            padding: "2rem",
            borderRadius: "12px",
            border: prediction.error
              ? "1px solid #f5c6cb"
              : "1px solid #c3e6cb",
            marginBottom: "2rem",
          }}
        >
          {prediction.error ? (
            <div style={{ color: "#721c24" }}>
              <h3>❌ Prediction Error</h3>
              <p>
                <strong>Error:</strong> {prediction.error}
              </p>
              {prediction.available_sports && (
                <p>
                  <strong>Available Sports:</strong>{" "}
                  {prediction.available_sports.join(", ")}
                </p>
              )}
            </div>
          ) : (
            <div style={{ color: "#155724" }}>
              <h3
                style={{
                  textAlign: "center",
                  marginBottom: "2rem",
                  fontSize: "1.8rem",
                }}
              >
                🤖 AI Prediction: {prediction.prediction}
              </h3>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
                  gap: "1rem",
                  marginBottom: "2rem",
                }}
              >
                <div
                  style={{
                    backgroundColor: "white",
                    padding: "1.5rem",
                    borderRadius: "8px",
                    textAlign: "center",
                  }}
                >
                  <strong style={{ display: "block", marginBottom: "0.5rem" }}>
                    Confidence
                  </strong>
                  <div
                    style={{
                      fontSize: "2rem",
                      fontWeight: "bold",
                      color: "#28a745",
                    }}
                  >
                    {((prediction.confidence || 0) * 100).toFixed(1)}%
                  </div>
                </div>

                <div
                  style={{
                    backgroundColor: "white",
                    padding: "1.5rem",
                    borderRadius: "8px",
                    textAlign: "center",
                  }}
                >
                  <strong style={{ display: "block", marginBottom: "0.5rem" }}>
                    Home Win Probability
                  </strong>
                  <div
                    style={{
                      fontSize: "2rem",
                      fontWeight: "bold",
                      color: "#007bff",
                    }}
                  >
                    {((prediction.home_win_probability || 0) * 100).toFixed(1)}%
                  </div>
                </div>

                <div
                  style={{
                    backgroundColor: "white",
                    padding: "1.5rem",
                    borderRadius: "8px",
                    textAlign: "center",
                  }}
                >
                  <strong style={{ display: "block", marginBottom: "0.5rem" }}>
                    Away Win Probability
                  </strong>
                  <div
                    style={{
                      fontSize: "2rem",
                      fontWeight: "bold",
                      color: "#dc3545",
                    }}
                  >
                    {((prediction.away_win_probability || 0) * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Betting Recommendation */}
              {game.odds && (
                <div
                  style={{
                    backgroundColor: "#fff3cd",
                    padding: "1.5rem",
                    borderRadius: "8px",
                    border: "1px solid #ffeaa7",
                  }}
                >
                  <h4
                    style={{
                      color: "#856404",
                      margin: "0 0 1rem 0",
                      textAlign: "center",
                    }}
                  >
                    🎯 Quick Betting Analysis
                  </h4>
                  <p
                    style={{
                      margin: 0,
                      color: "#856404",
                      textAlign: "center",
                      fontSize: "1.1rem",
                    }}
                  >
                    {prediction.prediction.includes("home") ? (
                      <>
                        <strong>AI recommends:</strong> {game.team_1} (Home) |
                        <strong> Current odds:</strong> {game.odds[game.team_1]}{" "}
                        |<strong> AI confidence:</strong>{" "}
                        {((prediction.confidence || 0) * 100).toFixed(1)}%
                      </>
                    ) : (
                      <>
                        <strong>AI recommends:</strong> {game.team_2} (Away) |
                        <strong> Current odds:</strong> {game.odds[game.team_2]}{" "}
                        |<strong> AI confidence:</strong>{" "}
                        {((prediction.confidence || 0) * 100).toFixed(1)}%
                      </>
                    )}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ChatGPT Analysis Results */}
      {chatGPTAnalysis && (
        <div
          style={{
            backgroundColor: "#e3f2fd",
            padding: "2rem",
            borderRadius: "12px",
            border: "1px solid #bbdefb",
          }}
        >
          <h3
            style={{
              color: "#1565c0",
              margin: "0 0 1.5rem 0",
              textAlign: "center",
            }}
          >
            💬 ChatGPT's Betting Analysis
          </h3>
          <div
            style={{
              whiteSpace: "pre-wrap",
              backgroundColor: "white",
              padding: "1.5rem",
              borderRadius: "8px",
              color: "#333",
              lineHeight: "1.6",
              fontSize: "1rem",
            }}
          >
            {chatGPTAnalysis}
          </div>
        </div>
      )}
    </div>
  );
}

export default GameDetailsPage;
