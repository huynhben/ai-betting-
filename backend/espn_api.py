# sportsradar_api.py - ESPN API REPLACEMENT (same interface, better data!)
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
from dataclasses import dataclass
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SportType(Enum):
    NBA = "basketball_nba"
    NFL = "americanfootball_nfl"
    MLB = "baseball_mlb"
    NHL = "icehockey_nhl"

@dataclass
class TeamData:
    """Standardized team data structure - SAME AS BEFORE"""
    sport: str
    team_id: str
    team_name: str
    full_name: str
    alias: str
    wins: int
    losses: int
    win_percentage: float
    conference: str = ""
    division: str = ""
    
    # Basic stats
    points_per_game: float = 0.0
    points_allowed_per_game: float = 0.0
    
    # Sport-specific stats
    detailed_stats: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.detailed_stats is None:
            self.detailed_stats = {}

@dataclass
class GameData:
    """Standardized game data structure - SAME AS BEFORE"""
    game_id: str
    sport: str
    scheduled_time: str
    status: str
    home_team_id: str
    home_team_name: str
    away_team_id: str
    away_team_name: str
    venue: str = ""
    
    # Scores
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    
    # Additional game info
    game_stats: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.game_stats is None:
            self.game_stats = {}

class FinalSportsRadarAPI:
    """ESPN-powered API client with SportsRadar interface - BETTER DATA, SAME CODE!"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize ESPN client (api_key ignored - ESPN is free!)
        
        Args:
            api_key: Ignored - ESPN doesn't need API keys!
        """
        if api_key:
            print("ℹ️ API key provided but not needed - ESPN API is completely FREE!")
        
        # ESPN API endpoints
        self.base_urls = {
            SportType.NBA.value: 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba',
            SportType.NFL.value: 'https://site.api.espn.com/apis/site/v2/sports/football/nfl', 
            SportType.MLB.value: 'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb',
            SportType.NHL.value: 'https://site.api.espn.com/apis/site/v2/sports/hockey/nhl'
        }
        
        # Sport name mapping
        self.sport_mapping = {
            SportType.NBA.value: 'nba',
            SportType.NFL.value: 'nfl',
            SportType.MLB.value: 'mlb', 
            SportType.NHL.value: 'nhl'
        }
        
        # HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Available sports - same as before
        self.available_sports = [sport.value for sport in SportType]
        
        logger.info(f"ESPN API initialized for: {self.available_sports}")
        print(f"🏆 ESPN API initialized (FREE!) - Sports: {self.available_sports}")
    
    def _make_request(self, url: str, params: Dict = None, retries: int = 3) -> Optional[Dict]:
        """Make request to ESPN API"""
        for attempt in range(retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    logger.warning(f"Rate limit hit, waiting...")
                    time.sleep(2)
                    continue
                else:
                    logger.error(f"ESPN API error {response.status_code}")
                    if attempt < retries:
                        time.sleep(1)
                        continue
                    return None
                    
            except Exception as e:
                logger.error(f"Request failed: {e}")
                if attempt < retries:
                    time.sleep(1)
                    continue
                return None
        
        return None
    
    def _get_sport_teams(self, sport: str) -> List[TeamData]:
        """Get teams for a sport from ESPN"""
        espn_sport = self.sport_mapping.get(sport)
        if not espn_sport:
            return []
        
        url = f"{self.base_urls[sport]}/teams"
        data = self._make_request(url)
        
        if not data:
            return []
        
        teams = []
        
        try:
            for team_data in data.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', []):
                team = team_data.get('team', {})
                
                # Get record
                record = team.get('record', {}).get('items', [{}])[0] if team.get('record', {}).get('items') else {}
                wins = 0
                losses = 0
                win_pct = 0.5
                
                if record and record.get('stats'):
                    try:
                        stats = record['stats']
                        if len(stats) >= 3:
                            wins = int(stats[0].get('value', 0))
                            losses = int(stats[1].get('value', 0))
                            win_pct = float(stats[2].get('value', 0.5))
                    except (IndexError, ValueError, TypeError):
                        pass
                
                team_obj = TeamData(
                    sport=sport,
                    team_id=team.get('id', ''),
                    team_name=team.get('name', ''),
                    full_name=team.get('displayName', ''),
                    alias=team.get('abbreviation', ''),
                    wins=wins,
                    losses=losses,
                    win_percentage=win_pct,
                    conference=team.get('groups', {}).get('parent', {}).get('name', ''),
                    division=team.get('groups', {}).get('name', ''),
                    detailed_stats=team
                )
                
                teams.append(team_obj)
        
        except Exception as e:
            logger.error(f"Error parsing {sport} teams: {e}")
        
        return teams
    
    def _get_sport_games(self, sport: str, days_back: int = 180, days_forward: int = 30) -> List[GameData]:
        """Get games for a sport from ESPN"""
        espn_sport = self.sport_mapping.get(sport)
        if not espn_sport:
            return []
        
        all_games = []
        
        # Get games from past and future
        start_date = datetime.now() - timedelta(days=days_back)
        end_date = datetime.now() + timedelta(days=days_forward)
        
        current_date = start_date
        
        print(f"   📅 Fetching {sport} games from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            
            # Get scoreboard for this date
            url = f"{self.base_urls[sport]}/scoreboard"
            params = {'dates': date_str}
            
            data = self._make_request(url, params)
            
            if data:
                try:
                    for event in data.get('events', []):
                        competition = event.get('competitions', [{}])[0]
                        competitors = competition.get('competitors', [])
                        
                        if len(competitors) >= 2:
                            # Find home and away teams
                            home_team = None
                            away_team = None
                            
                            for comp in competitors:
                                if comp.get('homeAway') == 'home':
                                    home_team = comp
                                elif comp.get('homeAway') == 'away':
                                    away_team = comp
                            
                            if home_team and away_team:
                                # Extract scores
                                home_score = None
                                away_score = None
                                
                                if 'score' in home_team:
                                    try:
                                        home_score = int(home_team['score'])
                                    except (ValueError, TypeError):
                                        pass
                                
                                if 'score' in away_team:
                                    try:
                                        away_score = int(away_team['score'])
                                    except (ValueError, TypeError):
                                        pass
                                
                                # Map ESPN status to your expected format
                                espn_status = competition.get('status', {}).get('type', {}).get('name', 'unknown')
                                status = 'closed' if espn_status in ['STATUS_FINAL', 'STATUS_FINAL_OVERTIME'] else espn_status.lower()
                                
                                # Create game object
                                game = GameData(
                                    game_id=event.get('id', ''),
                                    sport=sport,
                                    scheduled_time=event.get('date', ''),
                                    status=status,
                                    home_team_id=home_team.get('team', {}).get('id', ''),
                                    home_team_name=home_team.get('team', {}).get('displayName', ''),
                                    away_team_id=away_team.get('team', {}).get('id', ''),
                                    away_team_name=away_team.get('team', {}).get('displayName', ''),
                                    venue=competition.get('venue', {}).get('fullName', ''),
                                    home_score=home_score,
                                    away_score=away_score,
                                    game_stats=event
                                )
                                
                                all_games.append(game)
                
                except Exception as e:
                    logger.error(f"Error parsing games for {date_str}: {e}")
            
            current_date += timedelta(days=1)
            
            # Rate limiting and progress
            if current_date.day % 10 == 0:
                time.sleep(0.5)
        
        return all_games
    
    # ==================== SAME INTERFACE AS BEFORE ====================
    
    def get_nfl_schedule(self) -> List[GameData]:
        """Get NFL schedule - SAME METHOD NAME"""
        logger.info("Fetching NFL schedule from ESPN...")
        return self._get_sport_games(SportType.NFL.value)
    
    def get_nba_schedule(self) -> List[GameData]:
        """Get NBA schedule - SAME METHOD NAME"""
        logger.info("Fetching NBA schedule from ESPN...")
        return self._get_sport_games(SportType.NBA.value)
    
    def get_mlb_schedule(self, max_detailed_games: int = 300) -> List[GameData]:
        """Get MLB schedule - SAME METHOD NAME (max_detailed_games ignored - ESPN gives all scores!)"""
        logger.info("Fetching MLB schedule from ESPN...")
        return self._get_sport_games(SportType.MLB.value)
    
    def get_nhl_schedule(self) -> List[GameData]:
        """Get NHL schedule - SAME METHOD NAME"""
        logger.info("Fetching NHL schedule from ESPN...")
        return self._get_sport_games(SportType.NHL.value)
    
    def extract_teams_from_games(self, games: List[GameData]) -> List[TeamData]:
        """Extract teams from games - SAME METHOD"""
        teams = {}
        
        for game in games:
            # Add home team
            if game.home_team_id and game.home_team_id not in teams:
                teams[game.home_team_id] = TeamData(
                    sport=game.sport,
                    team_id=game.home_team_id,
                    team_name=game.home_team_name,
                    full_name=game.home_team_name,
                    alias=game.home_team_name[:3].upper() if game.home_team_name else '',
                    wins=0,
                    losses=0,
                    win_percentage=0.5
                )
            
            # Add away team
            if game.away_team_id and game.away_team_id not in teams:
                teams[game.away_team_id] = TeamData(
                    sport=game.sport,
                    team_id=game.away_team_id,
                    team_name=game.away_team_name,
                    full_name=game.away_team_name,
                    alias=game.away_team_name[:3].upper() if game.away_team_name else '',
                    wins=0,
                    losses=0,
                    win_percentage=0.5
                )
        
        return list(teams.values())
    
    def get_all_sports_data(self) -> Dict[str, Dict[str, List]]:
        """Get complete data for all sports - SAME METHOD NAME, BETTER DATA!"""
        logger.info("🏆 Fetching data for NFL, NBA, MLB, NHL from ESPN... (FREE & BETTER!)")
        
        all_data = {}
        
        # NFL
        print("\n🏈 Fetching NFL data from ESPN...")
        try:
            nfl_games = self.get_nfl_schedule()
            nfl_teams = self._get_sport_teams(SportType.NFL.value)  # Get actual team data
            all_data[SportType.NFL.value] = {
                'games': nfl_games,
                'teams': nfl_teams,
                'completed_games': [g for g in nfl_games if g.status == 'closed'],
                'upcoming_games': [g for g in nfl_games if g.status in ['scheduled', 'created']]
            }
            scored_games = [g for g in nfl_games if g.status == 'closed' and g.home_score is not None]
            print(f"   ✅ NFL: {len(nfl_games)} games, {len(nfl_teams)} teams, {len(scored_games)} with scores")
        except Exception as e:
            print(f"   ❌ NFL failed: {e}")
            all_data[SportType.NFL.value] = {'games': [], 'teams': [], 'completed_games': [], 'upcoming_games': []}
        
        # NBA
        print("\n🏀 Fetching NBA data from ESPN...")
        try:
            nba_games = self.get_nba_schedule()
            nba_teams = self._get_sport_teams(SportType.NBA.value)
            all_data[SportType.NBA.value] = {
                'games': nba_games,
                'teams': nba_teams,
                'completed_games': [g for g in nba_games if g.status == 'closed'],
                'upcoming_games': [g for g in nba_games if g.status in ['scheduled', 'created']]
            }
            scored_games = [g for g in nba_games if g.status == 'closed' and g.home_score is not None]
            print(f"   ✅ NBA: {len(nba_games)} games, {len(nba_teams)} teams, {len(scored_games)} with scores")
        except Exception as e:
            print(f"   ❌ NBA failed: {e}")
            all_data[SportType.NBA.value] = {'games': [], 'teams': [], 'completed_games': [], 'upcoming_games': []}
        
        # MLB
        print("\n⚾ Fetching MLB data from ESPN...")
        try:
            mlb_games = self.get_mlb_schedule()
            mlb_teams = self._get_sport_teams(SportType.MLB.value)
            all_data[SportType.MLB.value] = {
                'games': mlb_games,
                'teams': mlb_teams,
                'completed_games': [g for g in mlb_games if g.status == 'closed'],
                'upcoming_games': [g for g in mlb_games if g.status in ['scheduled', 'created']]
            }
            scored_games = [g for g in mlb_games if g.status == 'closed' and g.home_score is not None]
            print(f"   ✅ MLB: {len(mlb_games)} games, {len(mlb_teams)} teams, {len(scored_games)} with scores")
        except Exception as e:
            print(f"   ❌ MLB failed: {e}")
            all_data[SportType.MLB.value] = {'games': [], 'teams': [], 'completed_games': [], 'upcoming_games': []}
        
        # NHL
        print("\n🏒 Fetching NHL data from ESPN...")
        try:
            nhl_games = self.get_nhl_schedule()
            nhl_teams = self._get_sport_teams(SportType.NHL.value)
            all_data[SportType.NHL.value] = {
                'games': nhl_games,
                'teams': nhl_teams,
                'completed_games': [g for g in nhl_games if g.status == 'closed'],
                'upcoming_games': [g for g in nhl_games if g.status in ['scheduled', 'created']]
            }
            scored_games = [g for g in nhl_games if g.status == 'closed' and g.home_score is not None]
            print(f"   ✅ NHL: {len(nhl_games)} games, {len(nhl_teams)} teams, {len(scored_games)} with scores")
        except Exception as e:
            print(f"   ❌ NHL failed: {e}")
            all_data[SportType.NHL.value] = {'games': [], 'teams': [], 'completed_games': [], 'upcoming_games': []}
        
        return all_data
    
    def show_sample_games_all_sports(self, limit: int = 3):
        """Show sample games - SAME METHOD"""
        all_data = self.get_all_sports_data()
        
        print("\n🎯 SAMPLE GAMES FROM ESPN:")
        print("=" * 60)
        
        sport_icons = {
            SportType.NFL.value: "🏈",
            SportType.NBA.value: "🏀", 
            SportType.MLB.value: "⚾",
            SportType.NHL.value: "🏒"
        }
        
        for sport, data in all_data.items():
            completed_games = data['completed_games']
            icon = sport_icons.get(sport, "🏆")
            print(f"\n{icon} {sport.upper()}:")
            
            if completed_games:
                count = 0
                for game in completed_games:
                    if count >= limit:
                        break
                    if game.home_score is not None and game.away_score is not None:
                        winner = game.home_team_name if game.home_score > game.away_score else game.away_team_name
                        print(f"   {count+1}. {game.away_team_name} {game.away_score} - {game.home_score} {game.home_team_name} (Winner: {winner})")
                        count += 1
                
                if count == 0:
                    print("   No completed games with scores found")
            else:
                print("   No completed games found")
    
    def get_games_for_ml_training(self) -> Dict[str, List[GameData]]:
        """Get games with scores for ML training - SAME METHOD"""
        logger.info("🤖 Collecting games with scores for ML training from ESPN...")
        
        ml_games = {}
        
        # Get games from each sport
        sports_methods = {
            SportType.NFL.value: self.get_nfl_schedule,
            SportType.NBA.value: self.get_nba_schedule,
            SportType.MLB.value: self.get_mlb_schedule,
            SportType.NHL.value: self.get_nhl_schedule
        }
        
        for sport, method in sports_methods.items():
            try:
                logger.info(f"📊 Processing {sport.upper()}...")
                all_games = method()
                scored_games = []
                
                for game in all_games:
                    if (game.status == 'closed' and 
                        game.home_score is not None and 
                        game.away_score is not None):
                        scored_games.append(game)
                
                ml_games[sport] = scored_games
                logger.info(f"✅ Found {len(scored_games)} {sport.upper()} games with scores for ML training")
                
            except Exception as e:
                logger.error(f"Error getting {sport} games for ML: {e}")
                ml_games[sport] = []
        
        total_ml_games = sum(len(games) for games in ml_games.values())
        logger.info(f"🎯 Total ML training games from ESPN: {total_ml_games}")
        
        return ml_games
    
    def get_games_with_scores(self) -> Dict[str, List[GameData]]:
        """Get games with scores - SAME METHOD"""
        return self.get_games_for_ml_training()
    
    def export_ml_training_data(self, filename: str = None) -> str:
        """Export ML training data - SAME METHOD"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"espn_ml_training_data_{timestamp}.json"
        
        print(f"\n🤖 Exporting ESPN ML training data to {filename}...")
        
        ml_games = self.get_games_for_ml_training()
        
        # Convert to dict format
        export_data = {}
        for sport, games in ml_games.items():
            export_data[sport] = [game.__dict__ for game in games]
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        total_games = sum(len(games) for games in ml_games.values())
        print(f"✅ ESPN ML training data exported: {total_games} games with scores")
        return filename

# For backward compatibility - SAME CLASS NAMES
OptimizedSportsRadarAPI = FinalSportsRadarAPI

def main():
    """Test the ESPN replacement"""
    
    # Your API key (will be ignored)
    api_key = 'XNBq3JcpGlgtWBe1VOdI2NIR5VP3GMZCIOACC8Sx' 
    
    # Initialize client - SAME INTERFACE!
    client = FinalSportsRadarAPI(api_key)
    
    print("⚡ ESPN-POWERED SPORTS DATA COLLECTION")
    print("=" * 60)
    print("🏈 NFL | 🏀 NBA | ⚾ MLB | 🏒 NHL")
    print("🆓 FREE ESPN API - No limits, better data!")
    
    # Show time tracking
    start_time = time.time()
    
    # Get games with scores (for ML) - SAME METHOD CALL!
    games_with_scores = client.get_games_with_scores()
    
    end_time = time.time()
    
    print(f"\n🤖 GAMES WITH SCORES (ML READY) - Completed in {end_time - start_time:.1f} seconds:")
    total_ml_games = 0
    for sport, games in games_with_scores.items():
        print(f"   {sport}: {len(games)} games with final scores")
        total_ml_games += len(games)
    
    print(f"\n🎯 TOTAL ML TRAINING GAMES: {total_ml_games}")
    
    # Show sample games - SAME METHOD CALL!
    client.show_sample_games_all_sports(3)
    
    # Export ML training data - SAME METHOD CALL!
    client.export_ml_training_data()
    
    print(f"\n✅ ESPN data collection finished in {end_time - start_time:.1f} seconds!")
    print("🎉 MUCH MORE DATA THAN SPORTSRADAR!")
    print(f"🤖 {total_ml_games} games ready for machine learning training!")

if __name__ == "__main__":
    main()