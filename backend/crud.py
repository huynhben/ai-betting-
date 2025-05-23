from models import PredictionInput

# Sample hardcoded games
def get_upcoming_games():
    return [
        {"team_1": "Lakers", "team_2": "Celtics", "date": "2025-06-01"},
        {"team_1": "Warriors", "team_2": "Heat", "date": "2025-06-03"}
    ]

# Fake prediction logic
def submit_prediction(input: PredictionInput):
    # Placeholder logic: random fake result
    winner = input.team_1  # Pretend team_1 always wins
    confidence = 0.75
    return {
        "prediction": f"{winner} win",
        "confidence": confidence
    }