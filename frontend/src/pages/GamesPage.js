// pages/GamesPage.js
import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

function GamesPage() {
  const [games, setGames] = useState([]);
  const [sportKey, setSportKey] = useState("basketball_nba");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchGames = async () => {
      setLoading(true);
      try {
        const response = await fetch(`http://localhost:8000/games/${sportKey}`);
        const data = await response.json();
        setGames(data);
      } catch (err) {
        console.error("Failed to fetch games", err);
        setGames([]);
      } finally {
        setLoading(false);
      }
    };

    fetchGames();
  }, [sportKey]); // Now the dependency array is correct

  // Create a separate function for manual refresh
  const handleRefresh = async () => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/games/${sportKey}`);
      const data = await response.json();
      setGames(data);
    } catch (err) {
      console.error("Failed to fetch games", err);
      setGames([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectGame = (game) => {
    // Navigate to game details page with URL-safe team names
    const team1Safe = encodeURIComponent(game.team_1.replace(/\s+/g, "-"));
    const team2Safe = encodeURIComponent(game.team_2.replace(/\s+/g, "-"));

    navigate(`/game/${sportKey}/${team1Safe}/${team2Safe}`, {
      state: { game, sportKey },
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

  return (
    <div style={{ padding: "2rem", maxWidth: "1200px", margin: "0 auto" }}>
      <h1 style={{ textAlign: "center", color: "#333", marginBottom: "2rem" }}>
        🎯 Live Sports Games
      </h1>

      {/* Sport Selector */}
      <div style={{ marginBottom: "2rem", textAlign: "center" }}>
        <label
          style={{
            fontWeight: "bold",
            marginRight: "1rem",
            fontSize: "1.1rem",
          }}
        >
          Choose a sport:
        </label>
        <select
          onChange={(e) => setSportKey(e.target.value)}
          value={sportKey}
          style={{
            padding: "0.5rem",
            borderRadius: "6px",
            border: "2px solid #007bff",
            fontSize: "1rem",
            fontWeight: "bold",
            backgroundColor: "white",
            minWidth: "150px",
          }}
        >
          <option value="basketball_nba">🏀 NBA</option>
          <option value="americanfootball_nfl">🏈 NFL</option>
          <option value="baseball_mlb">⚾ MLB</option>
          <option value="icehockey_nhl">🏒 NHL</option>
          <option value="soccer_epl">⚽ Soccer (EPL)</option>
        </select>

        <button
          onClick={handleRefresh}
          style={{
            backgroundColor: "#28a745",
            color: "white",
            border: "none",
            padding: "0.5rem 1rem",
            borderRadius: "6px",
            marginLeft: "1rem",
            cursor: "pointer",
            fontWeight: "bold",
          }}
        >
          🔄 Refresh
        </button>
      </div>

      <h2 style={{ color: "#333", marginBottom: "1rem" }}>
        {getSportDisplayName(sportKey)} - Upcoming Games
      </h2>

      {/* Loading State */}
      {loading && (
        <div style={{ textAlign: "center", padding: "2rem" }}>
          <div style={{ fontSize: "2rem", marginBottom: "1rem" }}>⏳</div>
          <p>Loading games...</p>
        </div>
      )}

      {/* No Games State */}
      {!loading && games.length === 0 && (
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
          <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>📅</div>
          <h3 style={{ color: "#666", marginBottom: "0.5rem" }}>
            No Upcoming Games
          </h3>
          <p style={{ color: "#888", margin: 0 }}>
            Check back later for upcoming games and betting opportunities.
          </p>
          <button
            onClick={handleRefresh}
            style={{
              backgroundColor: "#007bff",
              color: "white",
              border: "none",
              padding: "0.75rem 1.5rem",
              borderRadius: "6px",
              marginTop: "1rem",
              cursor: "pointer",
              fontWeight: "bold",
            }}
          >
            Try Again
          </button>
        </div>
      )}

      {/* Games Grid */}
      {!loading && games.length > 0 && (
        <div
          style={{
            display: "grid",
            gap: "1.5rem",
            gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
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
                transition: "all 0.2s ease",
                cursor: "pointer",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = "translateY(-2px)";
                e.currentTarget.style.boxShadow = "0 4px 12px rgba(0,0,0,0.15)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = "translateY(0)";
                e.currentTarget.style.boxShadow = "0 2px 8px rgba(0,0,0,0.1)";
              }}
            >
              <div>
                <h3
                  style={{
                    marginBottom: "0.5rem",
                    color: "#333",
                    fontSize: "1.2rem",
                  }}
                >
                  {game.team_2} @ {game.team_1}
                </h3>
                <small style={{ color: "#666", fontSize: "0.9rem" }}>
                  📅 {new Date(game.date).toLocaleString()}
                </small>

                {game.odds && (
                  <div style={{ marginTop: "1rem" }}>
                    <strong style={{ color: "#333" }}>💰 Moneyline:</strong>
                    <div style={{ marginTop: "0.5rem" }}>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          marginBottom: "0.25rem",
                        }}
                      >
                        <span>{game.team_1}:</span>
                        <span
                          style={{
                            fontWeight: "bold",
                            color:
                              game.odds[game.team_1] > 0
                                ? "#28a745"
                                : "#dc3545",
                          }}
                        >
                          {game.odds[game.team_1] > 0 ? "+" : ""}
                          {game.odds[game.team_1]}
                        </span>
                      </div>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                        }}
                      >
                        <span>{game.team_2}:</span>
                        <span
                          style={{
                            fontWeight: "bold",
                            color:
                              game.odds[game.team_2] > 0
                                ? "#28a745"
                                : "#dc3545",
                          }}
                        >
                          {game.odds[game.team_2] > 0 ? "+" : ""}
                          {game.odds[game.team_2]}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {game.spreads && (
                  <div style={{ marginTop: "1rem" }}>
                    <strong style={{ color: "#333" }}>📊 Spread:</strong>
                    <div style={{ marginTop: "0.5rem", fontSize: "0.9rem" }}>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          marginBottom: "0.25rem",
                        }}
                      >
                        <span>{game.team_1}:</span>
                        <span>
                          {game.spreads[game.team_1]?.point > 0 ? "+" : ""}
                          {game.spreads[game.team_1]?.point}(
                          {game.spreads[game.team_1]?.price > 0 ? "+" : ""}
                          {game.spreads[game.team_1]?.price})
                        </span>
                      </div>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                        }}
                      >
                        <span>{game.team_2}:</span>
                        <span>
                          {game.spreads[game.team_2]?.point > 0 ? "+" : ""}
                          {game.spreads[game.team_2]?.point}(
                          {game.spreads[game.team_2]?.price > 0 ? "+" : ""}
                          {game.spreads[game.team_2]?.price})
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <button
                style={{
                  marginTop: "1.5rem",
                  padding: "0.75rem",
                  borderRadius: "6px",
                  border: "none",
                  backgroundColor: "#007bff",
                  color: "#fff",
                  cursor: "pointer",
                  fontWeight: "bold",
                  fontSize: "1rem",
                  transition: "background-color 0.2s ease",
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  handleSelectGame(game);
                }}
                onMouseEnter={(e) => {
                  e.target.style.backgroundColor = "#0056b3";
                }}
                onMouseLeave={(e) => {
                  e.target.style.backgroundColor = "#007bff";
                }}
              >
                🎯 Analyze Game
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default GamesPage;
