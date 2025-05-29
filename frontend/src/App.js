import React, { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [games, setGames] = useState([]);
  const [selectedGame, setSelectedGame] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [sportKey, setSportKey] = useState("basketball_nba");
  const [chatGPTAnalysis, setChatGPTAnalysis] = useState(null);

  useEffect(() => {
    setSelectedGame(null);
    fetch(`http://localhost:8000/games/${sportKey}`)
      .then((res) => res.json())
      .then((data) => setGames(data))
      .catch((err) => console.error("Failed to fetch games", err));
  }, [sportKey]);

  const handlePredict = () => {
    fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        team_id_home: selectedGame.team_id_1,
        pts_home: selectedGame.odds?.[selectedGame.team_1] ?? 100,
        team_id_away: selectedGame.team_id_2,
        pts_away: selectedGame.odds?.[selectedGame.team_2] ?? 100,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        setPrediction(data);
        handleChatGPTAnalysis(); // 🔁 auto-trigger ChatGPT analysis
      })
      .catch((err) => {
        console.error("Prediction failed:", err);
      });
  };

  const handleChatGPTAnalysis = () => {
    if (!selectedGame) {
      console.warn("No game selected.");
      return;
    }

    const prompt = `Game Analysis:
${selectedGame.team_1} vs ${selectedGame.team_2}.
Odds: ${selectedGame.team_1} ${selectedGame.odds?.[selectedGame.team_1]}, ${
      selectedGame.team_2
    } ${selectedGame.odds?.[selectedGame.team_2]}.
Spread: ${selectedGame.team_1} ${
      selectedGame.spreads?.[selectedGame.team_1]?.point ?? "N/A"
    } (${selectedGame.spreads?.[selectedGame.team_1]?.price ?? "N/A"}), 
        ${selectedGame.team_2} ${
      selectedGame.spreads?.[selectedGame.team_2]?.point ?? "N/A"
    } (${selectedGame.spreads?.[selectedGame.team_2]?.price ?? "N/A"}).
Should I bet on this game?`;

    console.log("Prompt sent to ChatGPT:", prompt);

    fetch("http://localhost:8000/analyze-bet/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    })
      .then((res) => res.json())
      .then((data) => {
        console.log("ChatGPT response:", data);
        setChatGPTAnalysis(data.analysis || data.error || "No response");
      })
      .catch((err) => {
        console.error("Error contacting ChatGPT:", err);
        setChatGPTAnalysis("Error contacting ChatGPT.");
      });
  };

  return (
    <div style={{ padding: "2rem", fontFamily: "Arial" }}>
      {/* Navigation Bar */}
      <nav
        style={{
          backgroundColor: "#FFFFFF",
          padding: "1rem",
          color: "white",
          display: "flex",
          alignItems: "center",
          gap: "1rem",
          position: "sticky",
          top: 0,
          zIndex: 1000,
        }}
      >
        <button
          style={{
            backgroundColor: "white",
            border: "2px solid white",
            padding: "0.5rem 1rem",
            color: "black",
            borderRadius: "8px",
            cursor: "pointer",
            fontWeight: "bold",
            fontSize: "1.5rem",
          }}
        >
          OddsGBT
        </button>
        <button
          style={{
            backgroundColor: "white",
            border: "2px solid white",
            padding: "0.5rem 1rem",
            color: "black",
            fontSize: "1rem",
            borderRadius: "8px",
            cursor: "pointer",
          }}
        >
          Home
        </button>
      </nav>
      <h1 style={{ textAlign: "center" }}>AI Sports Betting App</h1>

      {/* Sport Selector */}
      <label>Choose a sport: </label>
      <select onChange={(e) => setSportKey(e.target.value)} value={sportKey}>
        <option value="basketball_nba">NBA</option>
        <option value="americanfootball_nfl">NFL</option>
        <option value="baseball_mlb">MLB</option>
        <option value="soccer_epl">Soccer (EPL)</option>
        <option value="icehockey_nhl">NHL</option>
      </select>
      <h2 style={{ marginTop: "1rem" }}>Upcoming Games</h2>
      {/* Games List */}
      {games.length === 0 && (
        <div
          style={{
            marginTop: "2rem",
            textAlign: "center",
            padding: "3rem",
            backgroundColor: "#f8f9fa",
            borderRadius: "12px",
            border: "1px solid #ddd",
          }}
        >
          <h3 style={{ color: "#666", marginBottom: "0.5rem" }}>
            No Upcoming Games
          </h3>
          <p style={{ color: "#888", margin: 0 }}>
            Check back later for upcoming games and betting opportunities.
          </p>
        </div>
      )}
      <div
        style={{
          marginTop: "2rem",
          display: "grid",
          gap: "1.5rem",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
        }}
      >
        {games.map((game, idx) => (
          <div
            key={idx}
            style={{
              backgroundColor: "#ffffff",
              border: "1px solid #ddd",
              borderRadius: "12px",
              boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
              padding: "1.5rem",
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
            }}
          >
            <div>
              <h3 style={{ marginBottom: "0.5rem" }}>
                {game.team_1} vs {game.team_2}
              </h3>
              <small style={{ color: "#666" }}>
                {new Date(game.date).toLocaleString()}
              </small>

              {game.odds && (
                <p style={{ marginTop: "1rem" }}>
                  <strong>Moneyline:</strong>
                  <br />
                  {game.team_1}: {game.odds[game.team_1]} <br />
                  {game.team_2}: {game.odds[game.team_2]}
                </p>
              )}

              {game.spreads && (
                <p style={{ marginTop: "0.5rem" }}>
                  <strong>Spread:</strong>
                  <br />
                  {game.team_1}: {game.spreads[game.team_1].point} (
                  {game.spreads[game.team_1].price}) <br />
                  {game.team_2}: {game.spreads[game.team_2].point} (
                  {game.spreads[game.team_2].price})
                </p>
              )}
            </div>

            <button
              style={{
                marginTop: "1rem",
                padding: "0.5rem",
                borderRadius: "6px",
                border: "none",
                backgroundColor: "#007bff",
                color: "#fff",
                cursor: "pointer",
              }}
              onClick={() => setSelectedGame(game)}
            >
              Select
            </button>
          </div>
        ))}
      </div>

      {/* Prediction + ChatGPT */}
      {selectedGame && (
        <div
          style={{
            marginTop: "2rem",
            borderTop: "1px solid #ccc",
            paddingTop: "1rem",
          }}
        >
          <h2>Selected Game</h2>
          <p>
            <strong>{selectedGame.team_1}</strong> vs{" "}
            <strong>{selectedGame.team_2}</strong>
          </p>
          <button onClick={handlePredict}>Predict Winner (ML Model)</button>

          {/* ML Prediction */}
          {prediction && (
            <div style={{ marginTop: "1rem" }}>
              <h3>ML Prediction: {prediction.prediction}</h3>
              <p>Confidence: {(prediction.confidence * 100).toFixed(1)}%</p>
            </div>
          )}

          {/* ChatGPT Analysis */}
          {chatGPTAnalysis && (
            <div
              style={{
                marginTop: "1rem",
                background: "#f9f9f9",
                padding: "1rem",
              }}
            >
              <h3>ChatGPT's Advice:</h3>
              <p>{chatGPTAnalysis}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
