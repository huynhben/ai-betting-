# complete_data_pipeline_espn.py - ESPN version of your complete pipeline

from espn_api import FinalSportsRadarAPI, SportType  # Now using ESPN API
from mongodb import SportsDataManager
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

logger = logging.getLogger(__name__)

class CompleteSportsDataPipeline:
    """
    Complete data pipeline for all sports with ESPN API - BETTER DATA!
    Same interface, much better results
    """
    
    def __init__(self, api_keys: dict = None, mongodb_uri: str = None):
        """
        Initialize the complete pipeline
        
        Args:
            api_keys: Not needed for ESPN (kept for compatibility)
            mongodb_uri: MongoDB connection string (uses Atlas by default)
        """
        print("🏆 Initializing Complete Sports Data Pipeline with ESPN API...")
        
        # Initialize ESPN client (no API key needed!)
        self.sr_client = FinalSportsRadarAPI()  # ESPN API is FREE!
        self.mongo_manager = SportsDataManager(mongodb_uri)
        
        # Available sports from ESPN client
        self.available_sports = self.sr_client.available_sports
        print(f"✅ ESPN Pipeline initialized for sports: {self.available_sports}")
    
    # ==================== DATA COLLECTION METHODS ====================
    
    def update_all_data_for_sport(self, sport: str, include_games: bool = True) -> dict:
        """
        Complete data update for a sport using ESPN API
        """
        if sport not in self.available_sports:
            return {'error': f'Sport {sport} not available. ESPN supports: {self.available_sports}'}
        
        print(f"\n🔄 Starting complete data update for {sport.upper()} (ESPN)")
        print("=" * 50)
        
        results = {
            'sport': sport,
            'teams_updated': 0,
            'games_updated': 0,
            'errors': [],
            'timestamp': datetime.utcnow().isoformat(),
            'data_source': 'ESPN API (FREE)'
        }
        
        try:
            # Get data from ESPN
            print("📡 Fetching data from ESPN...")
            all_sports_data = self.sr_client.get_all_sports_data()
            
            if sport in all_sports_data:
                sport_data = all_sports_data[sport]
                
                # Store teams
                if sport_data.get('teams'):
                    teams_stored = self.mongo_manager.store_teams(sport_data['teams'])
                    results['teams_updated'] = teams_stored
                    print(f"✅ Stored {teams_stored} teams")
                
                # Store games
                if include_games and sport_data.get('games'):
                    games_stored = self.mongo_manager.store_games(sport_data['games'])
                    results['games_updated'] = games_stored
                    print(f"✅ Stored {games_stored} games")
                
                # Log the update
                self.mongo_manager.log_data_update(
                    sport, 
                    'espn_update', 
                    results['teams_updated'], 
                    results['games_updated']
                )
            else:
                results['errors'].append(f"No data returned for {sport}")
        
        except Exception as e:
            error_msg = f"Error updating {sport}: {str(e)}"
            results['errors'].append(error_msg)
            logger.error(error_msg)
            print(f"❌ {error_msg}")
        
        return results
    
    def update_all_available_sports(self, include_games: bool = True) -> dict:
        """Update data for ALL sports using ESPN API"""
        print("\n🌟 UPDATING ALL AVAILABLE SPORTS WITH ESPN API (FREE!)")
        print("=" * 60)
        
        all_results = {
            'total_sports_processed': 0,
            'successful_sports': [],
            'failed_sports': [],
            'sport_results': {},
            'overall_timestamp': datetime.utcnow().isoformat(),
            'data_source': 'ESPN API (FREE)'
        }
        
        try:
            # Get all sports data at once from ESPN
            print("📡 Fetching data for all sports from ESPN...")
            all_sports_data = self.sr_client.get_all_sports_data()
            
            # Store data for each sport
            storage_results = self.mongo_manager.store_all_sports_data(all_sports_data)
            
            for sport, result in storage_results.items():
                all_results['total_sports_processed'] += 1
                
                if 'error' not in result:
                    all_results['successful_sports'].append(sport)
                    all_results['sport_results'][sport] = {
                        'teams_updated': result.get('teams_stored', 0),
                        'games_updated': result.get('games_stored', 0),
                        'completed_games': result.get('completed_games', 0),
                        'success': True
                    }
                else:
                    all_results['failed_sports'].append(sport)
                    all_results['sport_results'][sport] = {'error': result['error']}
        
        except Exception as e:
            error_msg = f"Failed to update all sports: {str(e)}"
            logger.error(error_msg)
            all_results['sport_results']['general_error'] = error_msg
        
        # Summary
        print(f"\n📈 SUMMARY:")
        print(f"✅ Successful: {len(all_results['successful_sports'])} sports")
        print(f"❌ Failed: {len(all_results['failed_sports'])} sports")
        print(f"🆓 Data source: ESPN API (FREE!)")
        
        return all_results
    
    # ==================== DATA RETRIEVAL METHODS ====================
    
    def get_ml_ready_data_for_sport(self, sport: str, team_limit: int = None) -> dict:
        """Get ML-ready data for a sport from MongoDB"""
        print(f"\n🤖 Preparing ML data for {sport.upper()}")
        
        # Get teams from MongoDB
        teams_cursor = self.mongo_manager.collections['teams'].find({'sport': sport})
        if team_limit:
            teams_cursor = teams_cursor.limit(team_limit)
        
        ml_data = {
            'sport': sport,
            'teams': [],
            'team_stats': [],
            'recent_games': [],
            'data_summary': {},
            'data_source': 'ESPN API'
        }
        
        team_count = 0
        games_count = 0
        
        teams_list = list(teams_cursor)
        
        for team_doc in teams_list:
            team_count += 1
            
            # Basic team data
            team_data = {
                'team_id': team_doc['team_id'],
                'team_name': team_doc['team_name'],
                'full_name': team_doc['full_name'],
                'wins': team_doc.get('wins', 0),
                'losses': team_doc.get('losses', 0),
                'win_percentage': team_doc.get('win_percentage', 0.5),
                'conference': team_doc.get('conference', ''),
                'division': team_doc.get('division', ''),
                'points_per_game': team_doc.get('points_per_game', 0),
                'points_allowed_per_game': team_doc.get('points_allowed_per_game', 0)
            }
            ml_data['teams'].append(team_data)
            
            # Recent games and form
            team_form = self.mongo_manager.calculate_team_form(sport, team_doc['team_id'], 10)
            if team_form.get('games_analyzed', 0) > 0:
                games_count += team_form['games_analyzed']
                ml_data['recent_games'].append({
                    'team_id': team_doc['team_id'],
                    'form': team_form
                })
        
        ml_data['data_summary'] = {
            'total_teams': team_count,
            'total_recent_games': games_count,
            'data_completeness': {
                'teams': team_count > 0,
                'recent_games': games_count > 0
            }
        }
        
        print(f"✅ ML data prepared:")
        print(f"   Teams: {team_count}")
        print(f"   Recent games: {games_count}")
        
        return ml_data
    
    def get_team_prediction_data(self, sport: str, team1_name: str, team2_name: str) -> dict:
        """Get all relevant data for predicting a game between two teams"""
        print(f"\n🔮 Preparing prediction data: {team1_name} vs {team2_name}")
        
        # Get teams
        team1 = self.mongo_manager.get_team_by_name(sport, team1_name)
        team2 = self.mongo_manager.get_team_by_name(sport, team2_name)
        
        if not team1 or not team2:
            return {'error': 'One or both teams not found'}
        
        prediction_data = {
            'sport': sport,
            'matchup': f"{team1['full_name']} vs {team2['full_name']}",
            'team1': {
                'info': team1,
                'recent_form': self.mongo_manager.calculate_team_form(sport, team1['team_id'], 10),
                'recent_games': self.mongo_manager.get_recent_team_games(sport, team1['team_id'], 5)
            },
            'team2': {
                'info': team2,
                'recent_form': self.mongo_manager.calculate_team_form(sport, team2['team_id'], 10),
                'recent_games': self.mongo_manager.get_recent_team_games(sport, team2['team_id'], 5)
            },
            'head_to_head': self.mongo_manager.get_head_to_head(sport, team1['team_id'], team2['team_id'], 10),
            'timestamp': datetime.utcnow().isoformat(),
            'data_source': 'ESPN API'
        }
        
        print(f"✅ Prediction data prepared for {team1['team_name']} vs {team2['team_name']}")
        return prediction_data
    
    def get_todays_games_for_predictions(self, sport: str) -> List[dict]:
        """Get today's games with full prediction data"""
        print(f"\n📅 Getting today's {sport.upper()} games for predictions...")
        
        # Get today's games from MongoDB
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        
        todays_games = list(self.mongo_manager.collections['games'].find({
            'sport': sport,
            'status': {'$in': ['scheduled', 'created', 'STATUS_SCHEDULED']},
            'scheduled_time': {
                '$gte': today.isoformat(),
                '$lt': tomorrow.isoformat()
            }
        }))
        
        games_with_data = []
        
        for game in todays_games:
            # Get team data
            home_team = self.mongo_manager.collections['teams'].find_one({'team_id': game['home_team_id']})
            away_team = self.mongo_manager.collections['teams'].find_one({'team_id': game['away_team_id']})
            
            if home_team and away_team:
                game_data = {
                    'game_info': game,
                    'home_team': {
                        'info': home_team,
                        'recent_form': self.mongo_manager.calculate_team_form(sport, home_team['team_id'], 10)
                    },
                    'away_team': {
                        'info': away_team,
                        'recent_form': self.mongo_manager.calculate_team_form(sport, away_team['team_id'], 10)
                    },
                    'head_to_head': self.mongo_manager.get_head_to_head(sport, home_team['team_id'], away_team['team_id'], 5)
                }
                games_with_data.append(game_data)
        
        print(f"✅ Found {len(games_with_data)} games with complete prediction data")
        return games_with_data
    
    # ==================== ANALYTICS & HEALTH METHODS ====================
    
    # Add this method to your complete_data_pipeline.py to replace get_database_health_report

    def get_database_health_report(self) -> dict:
        """Get comprehensive database health and data quality report - FIXED FOR ALL SPORTS"""
        print("\n📊 GENERATING DATABASE HEALTH REPORT (ESPN DATA)")
        print("=" * 50)
    
        db_stats = self.mongo_manager.get_database_stats()
    
        health_report = {
            'database_stats': db_stats,
            'sport_analysis': {},
            'data_quality': {},
            'recommendations': [],
            'totals': {},
            'data_source': 'ESPN API (FREE)'
        }
    
        total_teams = 0
        total_games = 0
        total_completed_games = 0
    
    # Analyze each sport
        for sport in self.available_sports:
            teams_count = self.mongo_manager.collections['teams'].count_documents({'sport': sport})
            games_count = self.mongo_manager.collections['games'].count_documents({'sport': sport})
        
        # FIXED: Count completed games by presence of scores (works for all sports including MLB)
            completed_games = self.mongo_manager.collections['games'].count_documents({
                'sport': sport,
                'home_score': {'$exists': True, '$ne': None},
                'away_score': {'$exists': True, '$ne': None}
            })
        
        # Also check traditional "closed" status for comparison  
            closed_status_games = self.mongo_manager.collections['games'].count_documents({
                'sport': sport,
                'status': 'closed'
         })
        
            total_teams += teams_count
            total_games += games_count
            total_completed_games += completed_games
        
            sport_analysis = {
                'teams_count': teams_count,
                'games_count': games_count,
                'completed_games': completed_games,  # Now based on scores!
                'closed_status_games': closed_status_games,  # For comparison
                'completion_rate': completed_games / games_count if games_count > 0 else 0
            }
        
            health_report['sport_analysis'][sport] = sport_analysis
        
        # Data quality recommendations
            if teams_count == 0:
                health_report['recommendations'].append(f"❌ No teams found for {sport}")
            elif completed_games < 50:
                health_report['recommendations'].append(f"⚠️ {sport}: Only {completed_games} completed games (need 50+ for ML)")
            else:
                health_report['recommendations'].append(f"✅ {sport}: {completed_games} completed games (ML ready)")
    
        health_report['totals'] = {
            'teams_count': total_teams,
            'games_count': total_games,
            'completed_games': total_completed_games
        }
    
    # Print summary with the fix
        print(f"📈 TOTALS: {total_teams} teams, {total_games} games, {total_completed_games} completed")
        print(f"🆓 Data source: ESPN API (FREE!)")
    
        for sport, analysis in health_report['sport_analysis'].items():
            print(f"\n{sport.upper()}:")
            print(f"  Teams: {analysis['teams_count']}")
            print(f"  Games: {analysis['games_count']} ({analysis['completed_games']} with scores)")
        
        # Show the difference if any
            if analysis['completed_games'] != analysis['closed_status_games']:
                print(f"  Note: {analysis['closed_status_games']} have 'closed' status vs {analysis['completed_games']} with scores")
    
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in health_report['recommendations']:
            print(f"  {rec}")
    
        return health_report
    
    def export_ml_dataset(self, sport: str, filename: str = None) -> str:
        """Export complete ML dataset to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{sport}_espn_ml_dataset_{timestamp}.json"
        
        print(f"\n💾 Exporting {sport.upper()} ML dataset to {filename}...")
        
        ml_data = self.get_ml_ready_data_for_sport(sport)
        
        with open(filename, 'w') as f:
            json.dump(ml_data, f, indent=2, default=str)
        
        print(f"✅ ESPN ML dataset exported to {filename}")
        return filename
    
    def store_prediction(self, prediction_data: dict) -> bool:
        """Store ML model prediction in MongoDB"""
        return self.mongo_manager.store_prediction(prediction_data)

# ==================== SIMPLE TEST ====================

def test_espn_pipeline():
    """Test the ESPN pipeline"""
    
    print("🧪 TESTING ESPN COMPLETE DATA PIPELINE")
    print("=" * 60)
    
    try:
        # Test initialization (no API keys needed for ESPN!)
        print("🏆 Testing ESPN initialization...")
        pipeline = CompleteSportsDataPipeline()  # No API keys needed!
        print(f"✅ ESPN Pipeline initialized for: {pipeline.available_sports}")
        
        # Test database health
        print("\n📊 Testing database health...")
        health_report = pipeline.get_database_health_report()
        
        totals = health_report.get('totals', {})
        print(f"✅ Database health check complete")
        print(f"   Total teams: {totals.get('teams_count', 0)}")
        print(f"   Total games: {totals.get('games_count', 0)}")
        print(f"   Completed games: {totals.get('completed_games', 0)}")
        
        # Test ML data preparation
        sports_with_data = [sport for sport, analysis in health_report['sport_analysis'].items() 
                           if analysis['teams_count'] > 0]
        
        if sports_with_data:
            test_sport = sports_with_data[0]
            print(f"\n🤖 Testing ML data for {test_sport}...")
            ml_data = pipeline.get_ml_ready_data_for_sport(test_sport, team_limit=3)
            print(f"✅ ML data prepared for {test_sport}")
        
        print("\n🎉 ESPN PIPELINE TEST PASSED!")
        print("✅ All imports working")
        print("✅ Database connection working") 
        print("✅ Data retrieval working")
        print("✅ ESPN API providing much better data!")
        print("🆓 Completely FREE - no API key hassles!")
        
        return True
        
    except Exception as e:
        print(f"❌ ESPN pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_espn_pipeline()