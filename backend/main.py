from fastapi.middleware.cors import CORSMiddleware
from models import PredictionInput
from crud import get_upcoming_games, submit_prediction
from pydantic import BaseModel
from ml_model import predict_winner
from fastapi import FastAPI, Request, Body
from chatgpt_api import get_chatgpt_analysis

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionInput(BaseModel):
    team_1: str
    team_2: str
    date: str  # Optional, unused
    features: dict


@app.get("/games/upcoming")
def read_upcoming_games():
    return get_upcoming_games()

@app.post("/predict")
def make_prediction(input: PredictionInput):
    try:
        winner, confidence = predict_winner(input.features)
        return {
            "prediction": winner,
            "confidence": confidence
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/games/{sport_key}")
def get_games_by_sport(sport_key: str):
    return get_upcoming_games(sport_key)

@app.post("/analyze-bet/")
async def analyze_bet(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")

    if not prompt:
        return {"error": "Prompt is missing."}

    try:
        analysis = get_chatgpt_analysis(prompt)
        return {"analysis": analysis}
    except Exception as e:
        return {"error": str(e)}

