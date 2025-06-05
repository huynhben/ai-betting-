# fixed_mongodb_manager.py - MongoDB integration for the final four sports client
from pymongo import MongoClient, IndexModel, ASCENDING, DESCENDING
from pymongo.server_api import ServerApi
from pymongo.errors import DuplicateKeyError, ConnectionFailure
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import os
import json
from dataclasses import asdict

# FIXED: Import the correct final four sports client
from espn_api import FinalSportsRadarAPI, TeamData, GameData, SportType

logger = logging.getLogger(__name__)

# Your Atlas connection URI
ATLAS_URI = os.getenv("ATLAS_URI")

class SportsDataManager:
    """MongoDB manager for sports betting data - FOUR SPORTS (NFL, NBA, MLB, NHL)"""
    
    def __init__(self, mongodb_uri: str = None, database_name: str = "sports_betting"):
        """
        Initialize MongoDB connection
        
        Args:
            mongodb_uri: MongoDB connection string (defaults to Atlas URI)
            database_name: Database name to use
        """
        # Use the Atlas URI if no URI is provided
        if mongodb_uri is None:
            mongodb_uri = ATLAS_URI
            
        try:
            # Initialize with server API and longer timeout for Atlas
            self.client = MongoClient(
                mongodb_uri, 
                server_api=ServerApi('1'),
                serverSelectionTimeoutMS=30000,  # 30 second timeout for Atlas
                connectTimeoutMS=20000,
                socketTimeoutMS=20000
            )
            self.db = self.client[database_name]
            
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB Atlas: {database_name}")
            print(f"✅ Successfully connected to MongoDB Atlas database: {database_name}")
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            print(f"❌ Failed to connect to MongoDB: {e}")
            raise
        
        # Define collections
        self.collections = {
            'teams': self.db.teams,
            'games': self.db.games,
            'team_stats': self.db.team_stats,
            'predictions': self.db.predictions,
            'model_performance': self.db.model_performance,
            'data_updates': self.db.data_updates
        }
        
        # Initialize database with indexes
        self._setup_indexes()
        logger.info("MongoDB collections and indexes initialized")
    
    def _setup_indexes(self):
        """Create database indexes for optimal performance"""
        try:
            # Teams collection indexes
            self.collections['teams'].create_indexes([
                IndexModel([("sport", ASCENDING), ("team_id", ASCENDING)], unique=True),
                IndexModel([("sport", ASCENDING), ("team_name", ASCENDING)]),
                IndexModel([("sport", ASCENDING), ("win_percentage", DESCENDING)]),
                IndexModel([("last_updated", DESCENDING)])
            ])
            
            # Games collection indexes
            self.collections['games'].create_indexes([
                IndexModel([("sport", ASCENDING), ("game_id", ASCENDING)], unique=True),
                IndexModel([("sport", ASCENDING), ("scheduled_time", DESCENDING)]),
                IndexModel([("home_team_id", ASCENDING), ("away_team_id", ASCENDING)]),
                IndexModel([("status", ASCENDING), ("scheduled_time", DESCENDING)])
            ])
            
            # Team stats collection indexes
            self.collections['team_stats'].create_indexes([
                IndexModel([("sport", ASCENDING), ("team_id", ASCENDING), ("stat_date", DESCENDING)], unique=True),
                IndexModel([("sport", ASCENDING), ("stat_date", DESCENDING)])
            ])
            
            # Predictions collection indexes
            self.collections['predictions'].create_indexes([
                IndexModel([("game_id", ASCENDING), ("model_version", ASCENDING)], unique=True),
                IndexModel([("sport", ASCENDING), ("prediction_date", DESCENDING)]),
                IndexModel([("confidence", DESCENDING)])
            ])
            
            logger.info("Database indexes created successfully")
            print("✅ Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            print(f"⚠️ Warning: Error creating indexes: {e}")
    
    def store_teams(self, teams: List[TeamData]) -> int:
        """
        Store team data in MongoDB
        
        Args:
            teams: List of TeamData objects
            
        Returns:
            Number of teams stored/updated
        """
        if not teams:
            return 0
        
        stored_count = 0
        
        try:
            for team in teams:
                team_doc = {
                    'sport': team.sport,
                    'team_id': team.team_id,
                    'team_name': team.team_name,
                    'full_name': team.full_name,
                    'alias': team.alias,
                    'wins': team.wins,
                    'losses': team.losses,
                    'win_percentage': team.win_percentage,
                    'conference': team.conference,
                    'division': team.division,
                    'points_per_game': team.points_per_game,
                    'points_allowed_per_game': team.points_allowed_per_game,
                    'detailed_stats': team.detailed_stats or {},
                    'last_updated': datetime.utcnow()
                }
                
                # Upsert team document
                result = self.collections['teams'].replace_one(
                    {'sport': team.sport, 'team_id': team.team_id},
                    team_doc,
                    upsert=True
                )
                
                if result.upserted_id or result.modified_count > 0:
                    stored_count += 1
                    
        except Exception as e:
            logger.error(f"Error storing teams: {e}")
            
        logger.info(f"Stored/updated {stored_count} teams")
        return stored_count
    
    def store_games(self, games: List[GameData]) -> int:
        """
        Store game data in MongoDB
        
        Args:
            games: List of GameData objects
            
        Returns:
            Number of games stored/updated
        """
        if not games:
            return 0
        
        stored_count = 0
        
        try:
            for game in games:
                game_doc = {
                    'sport': game.sport,
                    'game_id': game.game_id,
                    'scheduled_time': game.scheduled_time,
                    'status': game.status,
                    'home_team_id': game.home_team_id,
                    'home_team_name': game.home_team_name,
                    'away_team_id': game.away_team_id,
                    'away_team_name': game.away_team_name,
                    'venue': game.venue,
                    'home_score': game.home_score,
                    'away_score': game.away_score,
                    'game_stats': game.game_stats or {},
                    'last_updated': datetime.utcnow()
                }
                
                # Upsert game document
                result = self.collections['games'].replace_one(
                    {'sport': game.sport, 'game_id': game.game_id},
                    game_doc,
                    upsert=True
                )
                
                if result.upserted_id or result.modified_count > 0:
                    stored_count += 1
                    
        except Exception as e:
            logger.error(f"Error storing games: {e}")
            
        logger.info(f"Stored/updated {stored_count} games")
        return stored_count
    
    def store_all_sports_data(self, all_sports_data: Dict[str, Dict[str, List]]) -> Dict[str, int]:
        """
        Store complete data from all sports
        
        Args:
            all_sports_data: Dictionary from FinalSportsRadarAPI.get_all_sports_data()
            
        Returns:
            Dictionary with storage counts per sport
        """
        storage_results = {}
        
        print("\n💾 STORING ALL SPORTS DATA IN MONGODB...")
        print("=" * 50)
        
        for sport, data in all_sports_data.items():
            print(f"\n📊 Storing {sport.upper()} data...")
            
            try:
                # Store teams
                teams_stored = self.store_teams(data['teams'])
                
                # Store games
                games_stored = self.store_games(data['games'])
                
                # Log the update
                self.log_data_update(sport, 'bulk_import', teams_stored, games_stored)
                
                storage_results[sport] = {
                    'teams_stored': teams_stored,
                    'games_stored': games_stored,
                    'total_teams': len(data['teams']),
                    'total_games': len(data['games']),
                    'completed_games': len(data['completed_games'])
                }
                
                print(f"   ✅ {sport}: {teams_stored} teams, {games_stored} games stored")
                
            except Exception as e:
                print(f"   ❌ {sport} failed: {e}")
                storage_results[sport] = {'error': str(e)}
        
        return storage_results
    
    def get_team_by_name(self, sport: str, team_name: str) -> Optional[Dict]:
        """
        Get team data by name (fuzzy matching)
        
        Args:
            sport: Sport type
            team_name: Team name to search for
            
        Returns:
            Team document or None
        """
        try:
            # Try exact match first
            team = self.collections['teams'].find_one({
                'sport': sport,
                'team_name': {'$regex': f'^{team_name}$', '$options': 'i'}
            })
            
            if team:
                return team
            
            # Try fuzzy match on team_name and full_name
            team = self.collections['teams'].find_one({
                'sport': sport,
                '$or': [
                    {'team_name': {'$regex': team_name, '$options': 'i'}},
                    {'full_name': {'$regex': team_name, '$options': 'i'}},
                    {'alias': {'$regex': team_name, '$options': 'i'}}
                ]
            })
            
            return team
            
        except Exception as e:
            logger.error(f"Error getting team {team_name}: {e}")
            return None
    
    def get_recent_team_games(self, sport: str, team_id: str, limit: int = 10) -> List[Dict]:
        """
        Get recent games for a team
        
        Args:
            sport: Sport type
            team_id: Team ID
            limit: Number of games to return
            
        Returns:
            List of recent games
        """
        try:
            games = list(self.collections['games'].find(
                {
                    'sport': sport,
                    '$or': [
                        {'home_team_id': team_id},
                        {'away_team_id': team_id}
                    ],
                    'status': 'closed'  # Only completed games
                },
                sort=[('scheduled_time', DESCENDING)],
                limit=limit
            ))
            
            return games
            
        except Exception as e:
            logger.error(f"Error getting recent games for {team_id}: {e}")
            return []
    
    def get_head_to_head(self, sport: str, team1_id: str, team2_id: str, limit: int = 20) -> List[Dict]:
        """
        Get head-to-head games between two teams
        
        Args:
            sport: Sport type
            team1_id: First team ID
            team2_id: Second team ID
            limit: Number of games to return
            
        Returns:
            List of head-to-head games
        """
        try:
            games = list(self.collections['games'].find(
                {
                    'sport': sport,
                    'status': 'closed',
                    '$or': [
                        {'home_team_id': team1_id, 'away_team_id': team2_id},
                        {'home_team_id': team2_id, 'away_team_id': team1_id}
                    ]
                },
                sort=[('scheduled_time', DESCENDING)],
                limit=limit
            ))
            
            return games
            
        except Exception as e:
            logger.error(f"Error getting H2H for {team1_id} vs {team2_id}: {e}")
            return []
    
    def calculate_team_form(self, sport: str, team_id: str, num_games: int = 10) -> Dict[str, Any]:
        """
        Calculate team's recent form and performance metrics - works for all four sports
        
        Args:
            sport: Sport type
            team_id: Team ID
            num_games: Number of recent games to analyze
            
        Returns:
            Dictionary with form metrics
        """
        try:
            recent_games = self.get_recent_team_games(sport, team_id, num_games)
            
            if not recent_games:
                return {
                    'games_analyzed': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_percentage': 0.5,
                    'avg_points_for': 0,
                    'avg_points_against': 0,
                    'point_differential': 0
                }
            
            wins = 0
            points_for = []
            points_against = []
            
            for game in recent_games:
                # Determine if team was home or away
                is_home = game['home_team_id'] == team_id
                
                # Get scores (all four sports use same format now)
                if is_home:
                    team_score = game.get('home_score', 0)
                    opp_score = game.get('away_score', 0)
                else:
                    team_score = game.get('away_score', 0)
                    opp_score = game.get('home_score', 0)
                
                if team_score is not None and opp_score is not None:  # Only count games with scores
                    points_for.append(team_score)
                    points_against.append(opp_score)
                    
                    if team_score > opp_score:
                        wins += 1
            
            games_with_scores = len(points_for)
            
            if games_with_scores == 0:
                return {
                    'games_analyzed': len(recent_games),
                    'wins': 0,
                    'losses': 0,
                    'win_percentage': 0.5,
                    'avg_points_for': 0,
                    'avg_points_against': 0,
                    'point_differential': 0
                }
            
            avg_points_for = sum(points_for) / len(points_for)
            avg_points_against = sum(points_against) / len(points_against)
            
            return {
                'games_analyzed': games_with_scores,
                'wins': wins,
                'losses': games_with_scores - wins,
                'win_percentage': wins / games_with_scores,
                'avg_points_for': avg_points_for,
                'avg_points_against': avg_points_against,
                'point_differential': avg_points_for - avg_points_against,
                'recent_games': recent_games
            }
            
        except Exception as e:
            logger.error(f"Error calculating team form for {team_id}: {e}")
            return {}
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics and health metrics"""
        try:
            stats = {}
            
            # Collection counts
            for name, collection in self.collections.items():
                stats[f'{name}_count'] = collection.count_documents({})
            
            # Sport breakdown for teams
            sport_pipeline = [
                {'$group': {'_id': '$sport', 'count': {'$sum': 1}}},
                {'$sort': {'count': -1}}
            ]
            sport_breakdown = list(self.collections['teams'].aggregate(sport_pipeline))
            
            # Games breakdown for each sport
            games_pipeline = [
                {'$group': {'_id': '$sport', 'total_games': {'$sum': 1}, 'completed_games': {'$sum': {'$cond': [{'$eq': ['$status', 'closed']}, 1, 0]}}}},
                {'$sort': {'total_games': -1}}
            ]
            games_breakdown = list(self.collections['games'].aggregate(games_pipeline))
            
            # Recent update info
            recent_updates = list(self.collections['teams'].find(
                {},
                {'sport': 1, 'last_updated': 1},
                sort=[('last_updated', DESCENDING)],
                limit=5
            ))
            
            return {
                'collection_counts': stats,
                'sport_breakdown': sport_breakdown,
                'games_breakdown': games_breakdown,
                'recent_updates': recent_updates,
                'database_name': self.db.name,
                'last_checked': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def log_data_update(self, sport: str, update_type: str, teams_count: int = 0, games_count: int = 0):
        """Log data update activity"""
        try:
            update_doc = {
                'sport': sport,
                'update_type': update_type,
                'teams_updated': teams_count,
                'games_updated': games_count,
                'timestamp': datetime.utcnow(),
                'success': True
            }
            
            self.collections['data_updates'].insert_one(update_doc)
            
        except Exception as e:
            logger.error(f"Error logging data update: {e}")
    
    def store_prediction(self, prediction_data: Dict) -> bool:
        """
        Store ML model prediction
        
        Args:
            prediction_data: Prediction data dictionary
            
        Returns:
            True if stored successfully
        """
        try:
            prediction_doc = {
                **prediction_data,
                'prediction_date': datetime.utcnow(),
                'created_at': datetime.utcnow()
            }
            
            result = self.collections['predictions'].replace_one(
                {
                    'game_id': prediction_data.get('game_id'),
                    'model_version': prediction_data.get('model_version')
                },
                prediction_doc,
                upsert=True
            )
            
            return result.upserted_id is not None or result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
            return False

class CompletePipeline:
    """Complete pipeline combining SportsRadar API with MongoDB storage"""
    
    def __init__(self, api_key: str):
        """Initialize the complete pipeline"""
        self.sportsradar_client = FinalSportsRadarAPI(api_key)
        self.mongo_manager = SportsDataManager()
        
        print("🚀 Complete Pipeline initialized!")
        print(f"📊 Available sports: {self.sportsradar_client.available_sports}")
    
    def collect_and_store_all_data(self) -> Dict[str, Any]:
        """Collect data from SportsRadar and store in MongoDB"""
        
        print("\n🌟 COLLECTING AND STORING ALL SPORTS DATA")
        print("=" * 60)
        
        # Step 1: Collect data from SportsRadar
        print("📡 Step 1: Collecting data from SportsRadar...")
        all_sports_data = self.sportsradar_client.get_all_sports_data()
        
        # Step 2: Store in MongoDB
        print("💾 Step 2: Storing data in MongoDB...")
        storage_results = self.mongo_manager.store_all_sports_data(all_sports_data)
        
        # Step 3: Show results
        print("\n📊 STORAGE RESULTS:")
        total_teams_stored = 0
        total_games_stored = 0
        
        for sport, result in storage_results.items():
            if 'error' not in result:
                teams_stored = result['teams_stored']
                games_stored = result['games_stored']
                total_teams_stored += teams_stored
                total_games_stored += games_stored
                
                print(f"   {sport}: {teams_stored} teams, {games_stored} games")
            else:
                print(f"   {sport}: ERROR - {result['error']}")
        
        print(f"\n🎯 TOTALS STORED:")
        print(f"   Teams: {total_teams_stored}")
        print(f"   Games: {total_games_stored}")
        
        # Step 4: Database health check
        print("\n🔍 Database health check...")
        db_stats = self.mongo_manager.get_database_stats()
        
        print(f"📈 Database Statistics:")
        for name, count in db_stats['collection_counts'].items():
            print(f"   {name}: {count}")
        
        return {
            'sports_data': all_sports_data,
            'storage_results': storage_results,
            'database_stats': db_stats,
            'totals': {
                'teams_stored': total_teams_stored,
                'games_stored': total_games_stored
            }
        }

def main():
    """Test the complete pipeline"""
    
    # Your API key
    api_key = 'XNBq3JcpGlgtWBe1VOdI2NIR5VP3GMZCIOACC8Sx'
    
    try:
        # Initialize the complete pipeline
        pipeline = CompletePipeline(api_key)
        
        # Collect and store all data
        results = pipeline.collect_and_store_all_data()
        
        print("\n✅ COMPLETE PIPELINE TEST SUCCESSFUL!")
        print("🚀 Your data is now in MongoDB Atlas!")
        print("📊 Ready for ML model training!")
        print("🎯 Ready for frontend development!")
        
        # Test some queries
        print("\n🧪 Testing MongoDB queries...")
        
        # Find a sample team
        sample_team = pipeline.mongo_manager.collections['teams'].find_one()
        if sample_team:
            print(f"✅ Sample team found: {sample_team['full_name']} ({sample_team['sport']})")
            
            # Test team form calculation
            form = pipeline.mongo_manager.calculate_team_form(
                sample_team['sport'], 
                sample_team['team_id']
            )
            if form.get('games_analyzed', 0) > 0:
                print(f"📊 Team form: {form['wins']}-{form['losses']} ({form['win_percentage']:.1%})")
        
        return results
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        logger.error(f"Pipeline test error: {e}")
        return None

if __name__ == "__main__":
    main()