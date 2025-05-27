from odds_api import get_events_for_sport

def get_upcoming_games(sport_key: str):
    events = get_events_for_sport(sport_key)
    games = []

    for event in events:
        if "home_team" in event and "away_team" in event:
            game = {
                "team_1": event["home_team"],
                "team_2": event["away_team"],
                "date": event["commence_time"]
            }

            # Add odds from bookmakers if available
            if event.get("bookmakers"):
                outcomes = event["bookmakers"][0]["markets"][0]["outcomes"]
                odds = {outcome["name"]: outcome["price"] for outcome in outcomes}
                game["odds"] = odds

            games.append(game)

    return games


from ml_model import predict_winner

from ml_model import predict_winner

def submit_prediction(input):
    features = {
        "team_id_home": input.team_id_home,
        "pts_home": input.pts_home,
        "team_id_away": input.team_id_away,
        "pts_away": input.pts_away
    }

    winner, prob = predict_winner(features)
    return {
        "prediction": f"{winner} team wins",
        "confidence": prob
    }

