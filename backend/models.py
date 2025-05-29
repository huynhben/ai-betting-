from sqlalchemy import Column, Integer, Float, String
from database import Base

class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True, index=True)
    game_date = Column(String)
    team_1 = Column(String)
    team_2 = Column(String)
    fg_pct_home = Column(Float)
    reb_home = Column(Integer)
    ast_home = Column(Integer)
    fg_pct_away = Column(Float)
    reb_away = Column(Integer)
    ast_away = Column(Integer)
    pts_home = Column(Integer)
    pts_away = Column(Integer)
    winner = Column(Integer)


from pydantic import BaseModel

class PredictionInput(BaseModel):
    # Example fields
    home_team: str
    away_team: str
    date: str
    home_pts: int
    away_pts: int
    # Add more as needed

