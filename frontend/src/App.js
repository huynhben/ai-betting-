import React, { useState, useEffect } from "react";

function App() {
  const [games, setGames] = useState([]);
  const [selectedGame, setSelectedGame] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [sportKey, setSportKey] = useState("basketball_nba");

  useEffect(() => {
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
        team_1: selectedGame.team_1,
        team_2: selectedGame.team_2,
        date: selectedGame.date,
        features: {}, // TODO: add real input
      }),
    })
      .then((res) => res.json())
      .then((data) => setPrediction(data));
  };

  return (
    <div style={{ padding: "2rem" }}>
      <h1>AI Sports Prediction</h1>

      {/* 🔽 Dropdown to select sport */}
      <label>Select a sport: </label>
      <select onChange={(e) => setSportKey(e.target.value)} value={sportKey}>
        <option value="basketball_nba">NBA</option>
        <option value="americanfootball_nfl">NFL</option>
        <option value="baseball_mlb">MLB</option>
        <option value="soccer_epl">Soccer (EPL)</option>
      </select>

      {/* 🏈 List of games */}
      {games.map((game, idx) => (
        <div key={idx}>
          <p>
            {game.team_1} vs {game.team_2} (
            {new Date(game.date).toLocaleString()})
          </p>
          {game.odds && (
            <p>
              Odds: {game.team_1} - {game.odds[game.team_1]} | {game.team_2} -{" "}
              {game.odds[game.team_2]}
            </p>
          )}
          <button onClick={() => setSelectedGame(game)}>Select</button>
        </div>
      ))}

      {/* 🔮 Prediction */}
      {selectedGame && (
        <div style={{ marginTop: "1rem" }}>
          <h2>
            Selected: {selectedGame.team_1} vs {selectedGame.team_2}
          </h2>
          <button onClick={handlePredict}>Predict Winner</button>
        </div>
      )}

      {prediction && (
        <div style={{ marginTop: "1rem" }}>
          <h3>Prediction: {prediction.prediction}</h3>
          <p>Confidence: {(prediction.confidence * 100).toFixed(1)}%</p>
        </div>
      )}
    </div>
  );
}

export default App;
