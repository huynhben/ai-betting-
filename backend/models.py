from pydantic import BaseModel

class PredictionInput(BaseModel):
    team_1: str
    team_2: str
    date: str
    features: dict