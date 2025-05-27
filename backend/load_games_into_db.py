# load_games_to_db.py
import pandas as pd
from sqlalchemy.orm import Session
from database import engine, SessionLocal
from models import Game, Base

Base.metadata.create_all(bind=engine)

df = pd.read_csv("/Users/huynhben/ai-betting-app/data/csv/game.csv").dropna(subset=[
    "team_name_home", "team_name_away", "pts_home", "pts_away",
    "fg_pct_home", "reb_home", "ast_home",
    "fg_pct_away", "reb_away", "ast_away"
])

db = SessionLocal()
for _, row in df.iterrows():
    db.add(Game(
        game_date=row["game_date"],
        team_1=row["team_name_home"],
        team_2=row["team_name_away"],
        fg_pct_home=row["fg_pct_home"],
        reb_home=row["reb_home"],
        ast_home=row["ast_home"],
        fg_pct_away=row["fg_pct_away"],
        reb_away=row["reb_away"],
        ast_away=row["ast_away"],
        pts_home=row["pts_home"],
        pts_away=row["pts_away"],
        winner=1 if row["pts_home"] > row["pts_away"] else 0
    ))
db.commit()
db.close()
print("✅ All games inserted into SQLite database.")