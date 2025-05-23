import React, { useState, useEffect } from "react";

function App() {
  const [games, setGames] = useState([]);
  const [selectedGame, setSelectedGame] = useState(null);
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    fetch("http://localhost:8000/games/upcoming")
      .then((res) => res.json())
      .then((data) => setGames(data));
  }, []);

  const handlePredict = () => {
    fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        team_1: selectedGame.team_1,
        team_2: selectedGame.team_2,
        date: selectedGame.date,
        features: {}, // later add real input
      }),
    })
      .then((res) => res.json())
      .then((data) => setPrediction(data));
  };

  return (
    <div style={{ padding: "2rem" }}>
      <h1>AI Sports Prediction</h1>
      {games.map((game, idx) => (
        <div key={idx}>
          <p>
            {game.team_1} vs {game.team_2} ({game.date})
          </p>
          <button onClick={() => setSelectedGame(game)}>Select</button>
        </div>
      ))}

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
