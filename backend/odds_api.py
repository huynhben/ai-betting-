import requests
import os  # if you want to import securely

def get_events_for_sport(sport_key: str):
    API_KEY = os.getenv('API_KEY')
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={API_KEY}&regions=us&markets=h2h,spreads&oddsFormat=american"
    params = {"api_key": API_KEY}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"Odds API error {response.status_code}: {response.text}")

    return response.json()
