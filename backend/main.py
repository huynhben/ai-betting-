from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import PredictionInput
from crud import get_upcoming_games, submit_prediction

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/games/upcoming")
def read_upcoming_games():
    return get_upcoming_games()

@app.post("/predict")
def make_prediction(input: PredictionInput):
    return submit_prediction(input)
