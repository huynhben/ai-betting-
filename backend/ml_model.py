import pandas as pd
import numpy as np
import joblib
import os
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_FILE = "model.pkl"
DB_PATH = "./data/game.db"
FEATURES = [
    "fg_pct_home", "reb_home", "ast_home",
    "fg_pct_away", "reb_away", "ast_away"
]

def train_model():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        fg_pct_home, reb_home, ast_home,
        fg_pct_away, reb_away, ast_away,
        pts_home, pts_away
    FROM games
    WHERE fg_pct_home IS NOT NULL AND fg_pct_away IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    df["winner"] = (df["pts_home"] > df["pts_away"]).astype(int)
    X = df[FEATURES]
    y = df["winner"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Trained with DB. Accuracy: {acc:.4f}")

    joblib.dump(model, MODEL_FILE)
    print(f"📦 Model saved to {MODEL_FILE}")

def predict_winner(features: dict):
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("Model not found. Train it first.")

    model = joblib.load(MODEL_FILE)
    X = np.array([features[f] for f in FEATURES]).reshape(1, -1)
    prob = model.predict_proba(X)[0][1]
    winner = "team_1" if prob > 0.5 else "team_2"
    return winner, prob

if __name__ == "__main__":
    train_model()