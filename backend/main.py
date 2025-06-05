# main.py - Updated to work with your ESPN ML model and React frontend
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import traceback
import logging

# Import your ESPN ML system
try:
    from ml_model_espn import ESPNMLPipelineIntegration
    from complete_data_pipeline import CompleteSportsDataPipeline
    ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML model not available: {e}")
    ML_AVAILABLE = False

# Import your existing CRUD functions for odds API
from crud import get_upcoming_games
from chatgpt_api import get_chatgpt_analysis

app = FastAPI(title="AI Sports Betting API", description="ESPN ML-Powered Predictions")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global ML system - initialize once
ml_system = None

def initialize_ml_system():
    """Initialize the ESPN ML system"""
    global ml_system
    try:
        if ml_system is None and ML_AVAILABLE:
            print("🚀 Initializing ESPN ML system...")
            
            # Initialize your ESPN pipeline
            data_pipeline = CompleteSportsDataPipeline()
            
            # Initialize ML integration
            ml_system = ESPNMLPipelineIntegration(data_pipeline)
            
            # Try to load existing models
            try:
                import os
                import glob
                
                model_files = glob.glob("espn_ml_models_*.pkl")
                if model_files:
                    latest_model = sorted(model_files)[-1]
                    ml_system.ml_model.load_models(latest_model)
                    print(f"✅ Loaded existing models from {latest_model}")
                else:
                    print("⚠️ No trained models found. Use /model/train to train models.")
            except Exception as e:
                print(f"⚠️ Could not load models: {e}")
            
            print("✅ ESPN ML system initialized")
        
        return ml_system
    except Exception as e:
        print(f"❌ Failed to initialize ML system: {e}")
        return None

# Pydantic models for API
class PredictionInput(BaseModel):
    sport: str  # 'basketball_nba', 'americanfootball_nfl', etc.
    team_1: str  # Home team name from odds API
    team_2: str  # Away team name from odds API
    # Keep backward compatibility with your existing frontend
    team_id_home: Optional[str] = None
    team_id_away: Optional[str] = None
    fg_pct_home: Optional[float] = 45.0
    reb_home: Optional[int] = 43
    ast_home: Optional[int] = 25
    fg_pct_away: Optional[float] = 45.0
    reb_away: Optional[int] = 43
    ast_away: Optional[int] = 25

@app.on_event("startup")
async def startup_event():
    """Initialize ML system on startup"""
    initialize_ml_system()

@app.get("/")
def root():
    return {
        "message": "AI Sports Betting API - ESPN ML Powered",
        "model": "ESPN Ensemble (XGBoost + RF + GB + LR)" if ML_AVAILABLE else "Basic Model",
        "sports": ["NBA", "NFL", "MLB", "NHL"],
        "data_source": "ESPN API" if ML_AVAILABLE else "Mock Data",
        "games_trained_on": "6,717+" if ML_AVAILABLE else "N/A"
    }

@app.get("/games/{sport_key}")
def get_games_by_sport(sport_key: str):
    """Get upcoming games from your odds API"""
    return get_upcoming_games(sport_key)

def find_team_id_by_name(sport: str, team_name: str) -> Optional[str]:
    """Find team ID in your ESPN database by team name"""
    try:
        ml_sys = initialize_ml_system()
        if not ml_sys:
            return None
            
        # Clean team name (remove common suffixes/prefixes)
        clean_name = team_name.replace("NBA", "").replace("NFL", "").replace("MLB", "").replace("NHL", "").strip()
        
        # Try exact match first
        team = ml_sys.data_pipeline.mongo_manager.collections['teams'].find_one({
            'sport': sport,
            '$or': [
                {'team_name': {'$regex': f'^{team_name}$', '$options': 'i'}},
                {'full_name': {'$regex': f'^{team_name}$', '$options': 'i'}},
                {'team_name': {'$regex': f'^{clean_name}$', '$options': 'i'}},
                {'full_name': {'$regex': f'^{clean_name}$', '$options': 'i'}}
            ]
        })
        
        if team:
            return team['team_id']
        
        # Try fuzzy matching
        team = ml_sys.data_pipeline.mongo_manager.collections['teams'].find_one({
            'sport': sport,
            '$or': [
                {'team_name': {'$regex': clean_name, '$options': 'i'}},
                {'full_name': {'$regex': clean_name, '$options': 'i'}},
                {'alias': {'$regex': clean_name, '$options': 'i'}}
            ]
        })
        
        return team['team_id'] if team else None
        
    except Exception as e:
        logger.error(f"Error finding team {team_name}: {e}")
        return None

@app.post("/predict")
def make_prediction(input: PredictionInput):
    """Make prediction with your ESPN ML model"""
    try:
        print(f"🎯 Prediction request: {input}")
        
        # Determine sport and teams
        sport = input.sport if hasattr(input, 'sport') else 'basketball_nba'
        team_1 = input.team_1 if hasattr(input, 'team_1') else input.team_id_home
        team_2 = input.team_2 if hasattr(input, 'team_2') else input.team_id_away
        
        print(f"📊 Parsed: Sport={sport}, Team1={team_1}, Team2={team_2}")
        
        # Initialize ML system
        ml_sys = initialize_ml_system()
        
        if not ml_sys or not ML_AVAILABLE:
            # Fallback to basic prediction if ML not available
            print("⚠️ ESPN ML not available, using fallback")
            return {
                "prediction": "home team wins",
                "confidence": 0.55,
                "home_win_probability": 0.55,
                "away_win_probability": 0.45,
                "model_info": {"type": "Fallback Model"},
                "teams": {"home": team_1, "away": team_2},
                "note": "ESPN ML model not available - using fallback"
            }
        
        # Check if sport is trained
        if sport not in ml_sys.ml_model.trained_sports:
            available_sports = list(ml_sys.ml_model.trained_sports)
            return {
                "error": f"No trained model for {sport}",
                "available_sports": available_sports,
                "confidence": 0.5,
                "home_win_probability": 0.5,
                "away_win_probability": 0.5
            }
        
        # Find team IDs in your ESPN database
        home_team_id = find_team_id_by_name(sport, team_1)
        away_team_id = find_team_id_by_name(sport, team_2)
        
        print(f"🔍 Team lookup: {team_1} -> {home_team_id}, {team_2} -> {away_team_id}")
        
        if not home_team_id or not away_team_id:
            # Get available teams for debugging
            available_teams = list(ml_sys.data_pipeline.mongo_manager.collections['teams'].find(
                {'sport': sport},
                {'team_name': 1, 'full_name': 1, '_id': 0}
            ).limit(10))
            
            return {
                "error": f"Teams not found in database: {team_1}, {team_2}",
                "available_teams_sample": [team.get('team_name') or team.get('full_name') for team in available_teams],
                "confidence": 0.5,
                "home_win_probability": 0.5,
                "away_win_probability": 0.5,
                "debug": {
                    "searched_for": [team_1, team_2],
                    "sport": sport
                }
            }
        
        # Make ESPN ML prediction
        print(f"🤖 Making prediction for {sport}: {home_team_id} vs {away_team_id}")
        prediction = ml_sys.ml_model.predict_game(sport, home_team_id, away_team_id)
        
        if 'error' in prediction:
            return {
                "error": prediction['error'],
                "confidence": 0.5,
                "home_win_probability": 0.5,
                "away_win_probability": 0.5
            }
        
        # Format response for your frontend
        response = {
            "prediction": f"{prediction['prediction']} team wins",
            "confidence": float(prediction["confidence"]),
            "home_win_probability": float(prediction["home_win_probability"]),
            "away_win_probability": float(prediction["away_win_probability"]),
            "model_info": {
                "type": prediction.get("model_type", "ESPN Ensemble"),
                "features_used": prediction.get("feature_count", 0)
            },
            "teams": {
                "home": prediction.get("home_team", team_1),
                "away": prediction.get("away_team", team_2)
            },
            "sport": sport,
            "timestamp": prediction.get("prediction_timestamp")
        }
        
        print(f"✅ Prediction successful: {response['prediction']} ({response['confidence']:.1%})")
        return response
        
    except Exception as e:
        error_response = {
            "error": f"Prediction failed: {str(e)}",
            "confidence": 0.5,
            "home_win_probability": 0.5,
            "away_win_probability": 0.5,
            "debug": str(traceback.format_exc())
        }
        print(f"❌ Prediction error: {error_response}")
        return error_response

@app.post("/analyze-bet/")
async def analyze_bet(request: Request):
    """ChatGPT analysis - unchanged"""
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        
        if not prompt:
            return {"error": "Prompt is missing."}
        
        analysis = get_chatgpt_analysis(prompt)
        return {"analysis": analysis}
        
    except Exception as e:
        print(f"ChatGPT analysis error: {str(e)}")
        return {"error": f"Analysis failed: {str(e)}"}

@app.get("/model/status")
def get_model_status():
    """Check ML model status"""
    try:
        ml_sys = initialize_ml_system()
        
        if not ml_sys or not ML_AVAILABLE:
            return {
                "model_available": False,
                "status": "not_available",
                "trained_sports": [],
                "message": "ESPN ML system not available"
            }
        
        trained_sports = list(ml_sys.ml_model.trained_sports)
        
        return {
            "model_available": True,
            "status": "ready" if trained_sports else "needs_training",
            "trained_sports": trained_sports,
            "available_sports": ml_sys.data_pipeline.available_sports,
            "total_games": "6,717+",
            "data_source": "ESPN API"
        }
    except Exception as e:
        return {
            "model_available": False,
            "error": str(e),
            "status": "error"
        }

@app.post("/model/train")
def trigger_model_training():
    """Endpoint to trigger model training"""
    try:
        ml_sys = initialize_ml_system()
        
        if not ml_sys or not ML_AVAILABLE:
            return {
                "status": "error",
                "message": "ESPN ML system not available"
            }
        
        print("🤖 Starting model training...")
        results = ml_sys.train_all_sports()
        
        successful_sports = [sport for sport, result in results.items() 
                           if 'error' not in result and 'ensemble' in result]
        
        return {
            "status": "success",
            "message": f"Training completed for {len(successful_sports)} sports",
            "trained_sports": successful_sports,
            "training_results": results
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Training failed: {str(e)}"
        }

@app.get("/teams/{sport}")
def get_teams_by_sport_debug(sport: str):
    """Debug endpoint to see available teams"""
    try:
        ml_sys = initialize_ml_system()
        
        if not ml_sys:
            return {"error": "ML system not available"}
        
        teams = list(ml_sys.data_pipeline.mongo_manager.collections['teams'].find(
            {'sport': sport},
            {'_id': 0, 'team_id': 1, 'team_name': 1, 'full_name': 1, 'wins': 1, 'losses': 1}
        ).limit(20))
        
        return {
            "sport": sport,
            "total_teams": len(teams),
            "teams": teams
        }
        
    except Exception as e:
        return {"error": str(e)}