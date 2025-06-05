# ml_model_espn.py - FIXED ML model for your ESPN pipeline with 6,717+ games!
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# FIXED imports for your ESPN pipeline
from mongodb import SportsDataManager
from espn_api import SportType

logger = logging.getLogger(__name__)

class ESPNSportsMLModel:
    """Enhanced ML model for your ESPN data: NBA, NFL, MLB, NHL"""
    
    def __init__(self, mongo_manager: SportsDataManager):
        self.mongo = mongo_manager
        
        # Model ensemble with different algorithms
        self.models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'logistic': LogisticRegression(
                C=1.0,
                random_state=42,
                max_iter=1000
            )
        }
        
        # Model weights for ensemble
        self.model_weights = {
            'xgboost': 0.4,
            'gradient_boost': 0.25,
            'random_forest': 0.2,
            'logistic': 0.15
        }
        
        self.scaler = StandardScaler()
        self.feature_names = []
        self.trained_sports = set()
        
        print("🤖 ESPN ML model initialized for NBA, NFL, MLB, NHL")
    
    def extract_team_features(self, sport: str, team_id: str) -> Dict[str, float]:
        """Extract features for a team using your ESPN data - FIXED ERROR HANDLING"""
        try:
            # Get team basic data
            team = self.mongo.collections['teams'].find_one({
                'sport': sport,
                'team_id': team_id
            })
            
            if not team:
                logger.debug(f"Team {team_id} not found for {sport}, using defaults")
                return self._get_default_features(sport)
            
            # Get team form (recent games performance)
            form_data = self.mongo.calculate_team_form(sport, team_id, 15)
            
            # Basic features from team record - SAFE extraction
            features = {}
            
            try:
                features.update({
                    'win_percentage': float(team.get('win_percentage', 0.5)),
                    'games_played': float(team.get('wins', 0) + team.get('losses', 0)),
                    'wins': float(team.get('wins', 0)),
                    'losses': float(team.get('losses', 0))
                })
            except (TypeError, ValueError):
                features.update({
                    'win_percentage': 0.5,
                    'games_played': 50.0,
                    'wins': 25.0,
                    'losses': 25.0
                })
            
            # Form-based features (from recent games) - SAFE extraction
            try:
                features.update({
                    'recent_win_percentage': float(form_data.get('win_percentage', 0.5)),
                    'recent_games_analyzed': float(form_data.get('games_analyzed', 10)),
                    'avg_points_for': float(form_data.get('avg_points_for', 0)),
                    'avg_points_against': float(form_data.get('avg_points_against', 0)),
                    'point_differential': float(form_data.get('point_differential', 0)),
                    'recent_wins': float(form_data.get('wins', 5)),
                    'recent_losses': float(form_data.get('losses', 5))
                })
            except (TypeError, ValueError):
                # Use sport defaults if form data is bad
                sport_defaults = self._get_default_features(sport)
                features.update({
                    'recent_win_percentage': 0.5,
                    'recent_games_analyzed': 10.0,
                    'avg_points_for': sport_defaults.get('avg_points_for', 100.0),
                    'avg_points_against': sport_defaults.get('avg_points_against', 100.0),
                    'point_differential': 0.0,
                    'recent_wins': 5.0,
                    'recent_losses': 5.0
                })
            
            # Sport-specific features - FIXED to handle missing data
            avg_for = features.get('avg_points_for', 0)
            avg_against = features.get('avg_points_against', 0)
            
            if sport == SportType.NBA.value:
                # Use defaults if no data
                if avg_for == 0: avg_for = 110.0
                if avg_against == 0: avg_against = 110.0
                
                expected_total = avg_for + avg_against
                features.update({
                    'expected_total_score': expected_total,
                    'offensive_efficiency': avg_for / 110.0,
                    'defensive_efficiency': 110.0 / max(avg_against, 1),
                    'pace_factor': expected_total / 220.0
                })
            elif sport == SportType.NFL.value:
                # Use defaults if no data
                if avg_for == 0: avg_for = 22.0
                if avg_against == 0: avg_against = 22.0
                
                expected_total = avg_for + avg_against
                features.update({
                    'expected_total_score': expected_total,
                    'offensive_efficiency': avg_for / 22.0,
                    'defensive_efficiency': 22.0 / max(avg_against, 1),
                    'scoring_variance': abs(avg_for - 22.0)
                })
            elif sport == SportType.MLB.value:
                # Use defaults if no data
                if avg_for == 0: avg_for = 4.5
                if avg_against == 0: avg_against = 4.5
                
                expected_total = avg_for + avg_against
                features.update({
                    'expected_total_score': expected_total,
                    'offensive_efficiency': avg_for / 4.5,
                    'defensive_efficiency': 4.5 / max(avg_against, 1),
                    'run_differential_rate': features.get('point_differential', 0) / max(features.get('recent_games_analyzed', 1), 1)
                })
            elif sport == SportType.NHL.value:
                # Use defaults if no data
                if avg_for == 0: avg_for = 3.0
                if avg_against == 0: avg_against = 3.0
                
                expected_total = avg_for + avg_against
                features.update({
                    'expected_total_score': expected_total,
                    'offensive_efficiency': avg_for / 3.0,
                    'defensive_efficiency': 3.0 / max(avg_against, 1),
                    'goal_differential_rate': features.get('point_differential', 0) / max(features.get('recent_games_analyzed', 1), 1)
                })
            
            # Momentum features - SAFE calculation
            try:
                recent_games = form_data.get('recent_games', [])
                if recent_games and len(recent_games) >= 3:
                    features['momentum'] = self._calculate_momentum(recent_games, team_id)
                    features['consistency'] = self._calculate_consistency(recent_games, team_id)
                else:
                    features['momentum'] = 0.0
                    features['consistency'] = 0.5
            except Exception:
                features['momentum'] = 0.0
                features['consistency'] = 0.5
            
            # Validate all features are numbers
            for key, value in features.items():
                if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                    logger.debug(f"Invalid feature {key}={value} for team {team_id}, using default")
                    features[key] = 0.0 if 'differential' in key or 'momentum' in key else 0.5
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extracting features for team {team_id}: {e}")
            return self._get_default_features(sport)
    
    def _get_default_features(self, sport: str) -> Dict[str, float]:
        """Get default feature values"""
        base_features = {
            'win_percentage': 0.5,
            'games_played': 50.0,
            'wins': 25.0,
            'losses': 25.0,
            'recent_win_percentage': 0.5,
            'recent_games_analyzed': 10.0,
            'avg_points_for': 0.0,
            'avg_points_against': 0.0,
            'point_differential': 0.0,
            'recent_wins': 5.0,
            'recent_losses': 5.0,
            'momentum': 0.0,
            'consistency': 0.5,
            'expected_total_score': 0.0,
            'offensive_efficiency': 1.0,
            'defensive_efficiency': 1.0
        }
        
        # Sport-specific defaults
        if sport == SportType.NBA.value:
            base_features.update({
                'avg_points_for': 110.0,
                'avg_points_against': 110.0,
                'expected_total_score': 220.0,
                'pace_factor': 1.0
            })
        elif sport == SportType.NFL.value:
            base_features.update({
                'avg_points_for': 22.0,
                'avg_points_against': 22.0,
                'expected_total_score': 44.0,
                'scoring_variance': 0.0
            })
        elif sport == SportType.MLB.value:
            base_features.update({
                'avg_points_for': 4.5,
                'avg_points_against': 4.5,
                'expected_total_score': 9.0,
                'run_differential_rate': 0.0
            })
        elif sport == SportType.NHL.value:
            base_features.update({
                'avg_points_for': 3.0,
                'avg_points_against': 3.0,
                'expected_total_score': 6.0,
                'goal_differential_rate': 0.0
            })
        
        return base_features
    
    def _calculate_momentum(self, recent_games: List[Dict], team_id: str) -> float:
        """Calculate team momentum from recent games"""
        if not recent_games or len(recent_games) < 3:
            return 0.0
        
        try:
            # Get last 10 games results
            results = []
            for game in recent_games[-10:]:
                is_home = game.get('home_team_id') == team_id
                
                if is_home:
                    team_score = game.get('home_score')
                    opp_score = game.get('away_score')
                else:
                    team_score = game.get('away_score')
                    opp_score = game.get('home_score')
                
                if team_score is not None and opp_score is not None:
                    if team_score > opp_score:
                        results.append(1)  # Win
                    else:
                        results.append(0)  # Loss
            
            if len(results) < 3:
                return 0.0
            
            # Weight recent games more heavily
            weights = np.linspace(0.5, 1.5, len(results))
            momentum = np.average(results, weights=weights) - 0.5  # Center around 0
            
            return float(momentum)
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return 0.0
    
    def _calculate_consistency(self, recent_games: List[Dict], team_id: str) -> float:
        """Calculate team consistency (lower variance = higher consistency)"""
        if not recent_games or len(recent_games) < 3:
            return 0.5
        
        try:
            score_diffs = []
            for game in recent_games[-10:]:
                is_home = game.get('home_team_id') == team_id
                
                if is_home:
                    team_score = game.get('home_score')
                    opp_score = game.get('away_score')
                else:
                    team_score = game.get('away_score')
                    opp_score = game.get('home_score')
                
                if team_score is not None and opp_score is not None:
                    score_diffs.append(team_score - opp_score)
            
            if len(score_diffs) < 3:
                return 0.5
            
            # Lower standard deviation = higher consistency
            std_dev = np.std(score_diffs)
            consistency = 1.0 / (1.0 + std_dev / 10.0)  # Normalize
            
            return float(consistency)
            
        except Exception as e:
            logger.error(f"Error calculating consistency: {e}")
            return 0.5
    
    def create_matchup_features(self, sport: str, home_team_id: str, away_team_id: str) -> Dict[str, float]:
        """Create features for a specific matchup"""
        try:
            # Get features for both teams
            home_features = self.extract_team_features(sport, home_team_id)
            away_features = self.extract_team_features(sport, away_team_id)
            
            # Combine features
            matchup_features = {}
            
            # Add individual team features
            for feature, value in home_features.items():
                matchup_features[f'{feature}_home'] = value
                
            for feature, value in away_features.items():
                matchup_features[f'{feature}_away'] = value
            
            # Add differential features (home advantage)
            for feature in home_features.keys():
                if feature in away_features:
                    matchup_features[f'{feature}_diff'] = home_features[feature] - away_features[feature]
            
            # Add head-to-head features
            h2h_games = self.mongo.get_head_to_head(sport, home_team_id, away_team_id, 10)
            h2h_features = self._extract_h2h_features(h2h_games, home_team_id, sport)
            matchup_features.update(h2h_features)
            
            # Add contextual features
            context_features = self._extract_context_features(sport)
            matchup_features.update(context_features)
            
            return matchup_features
            
        except Exception as e:
            logger.error(f"Error creating matchup features: {e}")
            return {}
    
    def _extract_h2h_features(self, h2h_games: List[Dict], home_team_id: str, sport: str) -> Dict[str, float]:
        """Extract head-to-head features"""
        if not h2h_games:
            return {
                'h2h_games': 0.0,
                'h2h_home_wins': 0.5,
                'h2h_avg_total': self._get_sport_avg_total(sport),
                'h2h_avg_margin': 0.0
            }
        
        try:
            home_wins = 0
            total_scores = []
            margins = []
            
            for game in h2h_games:
                home_score = game.get('home_score')
                away_score = game.get('away_score')
                
                if home_score is not None and away_score is not None:
                    total_scores.append(home_score + away_score)
                    
                    # Determine margin from perspective of current home team
                    if game.get('home_team_id') == home_team_id:
                        if home_score > away_score:
                            home_wins += 1
                        margins.append(home_score - away_score)
                    else:
                        if away_score > home_score:
                            home_wins += 1
                        margins.append(away_score - home_score)
            
            games_count = len(total_scores)
            
            return {
                'h2h_games': float(min(games_count, 10)),
                'h2h_home_wins': home_wins / games_count if games_count > 0 else 0.5,
                'h2h_avg_total': sum(total_scores) / len(total_scores) if total_scores else self._get_sport_avg_total(sport),
                'h2h_avg_margin': sum(margins) / len(margins) if margins else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error extracting H2H features: {e}")
            return {
                'h2h_games': 0.0,
                'h2h_home_wins': 0.5,
                'h2h_avg_total': self._get_sport_avg_total(sport),
                'h2h_avg_margin': 0.0
            }
    
    def _get_sport_avg_total(self, sport: str) -> float:
        """Get average total score for sport"""
        averages = {
            SportType.NBA.value: 220.0,
            SportType.NFL.value: 44.0,
            SportType.MLB.value: 9.0,
            SportType.NHL.value: 6.0
        }
        return averages.get(sport, 100.0)
    
    def _extract_context_features(self, sport: str) -> Dict[str, float]:
        """Extract contextual features"""
        now = datetime.now()
        
        features = {
            'day_of_week': float(now.weekday()) / 6.0,
            'month': float(now.month) / 12.0,
            'is_weekend': 1.0 if now.weekday() >= 5 else 0.0,
            'home_field_advantage': 0.54  # Historical home win rate across sports
        }
        
        # Sport-specific home advantage
        home_advantages = {
            SportType.NBA.value: 0.57,
            SportType.NFL.value: 0.52,
            SportType.MLB.value: 0.54,
            SportType.NHL.value: 0.55
        }
        features['sport_home_advantage'] = home_advantages.get(sport, 0.54)
        
        return features
    
    def prepare_training_data(self, sport: str, min_games: int = 50) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data from your ESPN data"""
        print(f"📊 Preparing training data for {sport}...")
        
        # Get completed games with scores
        games = list(self.mongo.collections['games'].find({
            'sport': sport,
            'home_score': {'$exists': True, '$ne': None},
            'away_score': {'$exists': True, '$ne': None}
        }).limit(2000))
        
        if len(games) < min_games:
            print(f"⚠️ Only {len(games)} games found for {sport}, minimum is {min_games}")
            return np.array([]), np.array([]), []
        
        print(f"   Processing {len(games)} games...")
        
        X_data = []
        y_data = []
        feature_names = []
        
        for i, game in enumerate(games):
            if i % 100 == 0:
                print(f"   Progress: {i+1}/{len(games)} games processed...")
            
            try:
                # Create features for this matchup
                features = self.create_matchup_features(
                    sport,
                    game['home_team_id'],
                    game['away_team_id']
                )
                
                if not features:
                    continue
                
                # Determine winner (1 if home wins, 0 if away wins)
                home_score = game['home_score']
                away_score = game['away_score']
                home_wins = 1 if home_score > away_score else 0
                
                X_data.append(list(features.values()))
                y_data.append(home_wins)
                
                if not feature_names:
                    feature_names = list(features.keys())
                
            except Exception as e:
                logger.error(f"Error processing game {game.get('game_id')}: {e}")
                continue
        
        if len(X_data) == 0:
            print(f"❌ No valid training data created for {sport}")
            return np.array([]), np.array([]), []
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        print(f"✅ Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Home team win rate: {y.mean():.1%}")
        
        return X, y, feature_names
    
    def train_models(self, sport: str) -> Dict[str, Any]:
        """Train ensemble models for a sport"""
        print(f"\n🤖 Training models for {sport.upper()}...")
        
        # Prepare training data
        X, y, feature_names = self.prepare_training_data(sport, min_games=50)
        
        if len(X) == 0:
            return {'error': f'No training data available for {sport}'}
        
        self.feature_names = feature_names
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        results = {}
        
        for model_name, model in self.models.items():
            print(f"   Training {model_name}...")
            
            try:
                # Train
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                accuracy = accuracy_score(y_test, y_pred)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'auc_score': auc_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                print(f"      {model_name}: Accuracy={accuracy:.3f}, AUC={auc_score:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        # Ensemble performance
        try:
            ensemble_pred = self._ensemble_predict_proba(X_test_scaled)
            ensemble_accuracy = accuracy_score(y_test, (ensemble_pred > 0.5).astype(int))
            ensemble_auc = roc_auc_score(y_test, ensemble_pred)
            
            results['ensemble'] = {
                'accuracy': ensemble_accuracy,
                'auc_score': ensemble_auc
            }
            
            print(f"   ✅ Ensemble: Accuracy={ensemble_accuracy:.3f}, AUC={ensemble_auc:.3f}")
            
        except Exception as e:
            logger.error(f"Error calculating ensemble performance: {e}")
            results['ensemble'] = {'error': str(e)}
        
        # Mark sport as trained
        self.trained_sports.add(sport)
        
        results.update({
            'sport': sport,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': len(feature_names),
            'training_date': datetime.now().isoformat()
        })
        
        return results
    
    def _ensemble_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble prediction"""
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                pred_proba = model.predict_proba(X)[:, 1]
                predictions[model_name] = pred_proba
            except Exception as e:
                logger.warning(f"Error getting prediction from {model_name}: {e}")
                continue
        
        if not predictions:
            return np.full(len(X), 0.5)
        
        # Weighted average
        ensemble_pred = np.zeros(len(X))
        total_weight = 0
        
        for model_name, pred in predictions.items():
            weight = self.model_weights.get(model_name, 0.25)
            ensemble_pred += pred * weight
            total_weight += weight
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return ensemble_pred
    
    def predict_game(self, sport: str, home_team_id: str, away_team_id: str) -> Dict[str, Any]:
        """Predict outcome of a specific game"""
        try:
            if sport not in self.trained_sports:
                return {'error': f'No trained model available for {sport}'}
            
            # Create features for this matchup
            features = self.create_matchup_features(sport, home_team_id, away_team_id)
            
            if not features:
                return {'error': 'Could not create features for this matchup'}
            
            # Ensure feature order matches training
            feature_vector = np.array([features.get(name, 0) for name in self.feature_names]).reshape(1, -1)
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Ensemble prediction
            home_win_prob = self._ensemble_predict_proba(feature_vector_scaled)[0]
            away_win_prob = 1 - home_win_prob
            
            prediction = "home" if home_win_prob > 0.5 else "away"
            confidence = max(home_win_prob, away_win_prob)
            
            # Get team names
            home_team = self.mongo.collections['teams'].find_one({'sport': sport, 'team_id': home_team_id})
            away_team = self.mongo.collections['teams'].find_one({'sport': sport, 'team_id': away_team_id})
            
            home_team_name = home_team['team_name'] if home_team else home_team_id
            away_team_name = away_team['team_name'] if away_team else away_team_id
            
            result = {
                'sport': sport,
                'prediction': prediction,
                'home_win_probability': float(home_win_prob),
                'away_win_probability': float(away_win_prob),
                'confidence': float(confidence),
                'home_team': home_team_name,
                'away_team': away_team_name,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'model_type': 'ESPN Enhanced Ensemble',
                'feature_count': len(features),
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {'error': f'Prediction failed: {str(e)}'}
    
    def save_models(self, filepath: str):
        """Save trained models"""
        try:
            model_data = {
                'models': self.models,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_weights': self.model_weights,
                'trained_sports': list(self.trained_sports),
                'saved_at': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, filepath)
            print(f"💾 Models saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, filepath: str) -> bool:
        """Load trained models"""
        try:
            model_data = joblib.load(filepath)
            
            self.models = model_data['models']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_weights = model_data.get('model_weights', self.model_weights)
            self.trained_sports = set(model_data.get('trained_sports', []))
            
            print(f"✅ Models loaded from {filepath}")
            print(f"   Trained sports: {list(self.trained_sports)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

# ==================== INTEGRATION WITH YOUR ESPN PIPELINE ====================

class ESPNMLPipelineIntegration:
    """Integration class for your ESPN pipeline + ML model"""
    
    def __init__(self, data_pipeline):
        """Initialize with your ESPN data pipeline"""
        self.data_pipeline = data_pipeline
        self.ml_model = ESPNSportsMLModel(data_pipeline.mongo_manager)
        
        print("🚀 ESPN ML Pipeline Integration initialized!")
    
    def train_all_sports(self) -> Dict[str, Any]:
        """Train ML models for all sports with your ESPN data"""
        
        print("\n🏆 TRAINING ML MODELS FOR ALL ESPN SPORTS")
        print("=" * 60)
        
        training_results = {}
        
        for sport in self.data_pipeline.available_sports:
            print(f"\n🔄 Training {sport.upper()} model...")
            
            # Check available data
            completed_games = self.data_pipeline.mongo_manager.collections['games'].count_documents({
                'sport': sport,
                'home_score': {'$exists': True, '$ne': None},
                'away_score': {'$exists': True, '$ne': None}
            })
            
            print(f"   Available data: {completed_games} completed games")
            
            if completed_games < 50:
                print(f"   ⚠️ Not enough games for {sport} (need 50+), skipping...")
                training_results[sport] = {'error': f'Insufficient data: only {completed_games} games'}
                continue
            
            try:
                # Train model
                results = self.ml_model.train_models(sport)
                
                if 'error' not in results and 'ensemble' in results:
                    print(f"   ✅ {sport.upper()} model trained successfully!")
                    print(f"      Ensemble accuracy: {results['ensemble']['accuracy']:.1%}")
                    print(f"      Training samples: {results['training_samples']}")
                    training_results[sport] = results
                else:
                    print(f"   ❌ {sport.upper()} training failed")
                    training_results[sport] = results
                    
            except Exception as e:
                error_msg = f"Training error for {sport}: {str(e)}"
                print(f"   ❌ {error_msg}")
                training_results[sport] = {'error': error_msg}
        
        # Save models
        successful_sports = [sport for sport, result in training_results.items() 
                           if 'error' not in result and 'ensemble' in result]
        
        if successful_sports:
            model_filename = f"espn_ml_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            self.ml_model.save_models(model_filename)
            print(f"\n💾 Models saved to {model_filename}")
            print(f"✅ Successfully trained models for: {', '.join(successful_sports)}")
        else:
            print("\n❌ No models were successfully trained")
        
        return training_results
    
    def test_predictions(self) -> Dict[str, Any]:
        """Test predictions for all trained models"""
        
        print("\n🔮 TESTING PREDICTIONS FOR TRAINED MODELS")
        print("=" * 50)
        
        test_results = {}
        
        for sport in self.ml_model.trained_sports:
            print(f"\n🏟️ Testing {sport.upper()} predictions...")
            
            # Get two teams for testing
            teams = list(self.data_pipeline.mongo_manager.collections['teams'].find({'sport': sport}).limit(2))
            
            if len(teams) >= 2:
                try:
                    prediction = self.ml_model.predict_game(
                        sport,
                        teams[0]['team_id'], 
                        teams[1]['team_id']
                    )
                    
                    if 'error' not in prediction:
                        print(f"   ✅ {prediction['home_team']} vs {prediction['away_team']}")
                        print(f"      Winner: {prediction['prediction']} ({prediction['confidence']:.1%})")
                        print(f"      Probabilities: Home {prediction['home_win_probability']:.1%}, Away {prediction['away_win_probability']:.1%}")
                        
                        test_results[sport] = {
                            'success': True,
                            'prediction': prediction
                        }
                    else:
                        print(f"   ❌ Prediction failed: {prediction['error']}")
                        test_results[sport] = {
                            'success': False,
                            'error': prediction['error']
                        }
                        
                except Exception as e:
                    print(f"   ❌ Test failed: {e}")
                    test_results[sport] = {
                        'success': False,
                        'error': str(e)
                    }
            else:
                print(f"   ⚠️ Not enough teams found for testing")
                test_results[sport] = {
                    'success': False,
                    'error': 'Insufficient teams for testing'
                }
        
        return test_results
    
    def predict_todays_games(self) -> Dict[str, List]:
        """Get predictions for today's games"""
        
        print("\n📅 PREDICTING TODAY'S GAMES")
        print("=" * 40)
        
        all_predictions = {}
        
        for sport in self.ml_model.trained_sports:
            print(f"\n🏟️ Getting {sport.upper()} predictions...")
            
            # Get today's games
            today = datetime.now().date()
            tomorrow = today + timedelta(days=1)
            
            todays_games = list(self.data_pipeline.mongo_manager.collections['games'].find({
                'sport': sport,
                'status': {'$in': ['scheduled', 'created', 'STATUS_SCHEDULED']},
                'scheduled_time': {
                    '$gte': today.isoformat(),
                    '$lt': tomorrow.isoformat()
                }
            }))
            
            if not todays_games:
                print(f"   No games scheduled today for {sport}")
                all_predictions[sport] = []
                continue
            
            sport_predictions = []
            
            for game in todays_games:
                try:
                    prediction = self.ml_model.predict_game(
                        sport,
                        game['home_team_id'],
                        game['away_team_id']
                    )
                    
                    if 'error' not in prediction:
                        # Add game context
                        prediction['game_time'] = game['scheduled_time']
                        prediction['venue'] = game.get('venue', 'TBD')
                        
                        sport_predictions.append(prediction)
                        
                        print(f"   ✅ {prediction['home_team']} vs {prediction['away_team']}")
                        print(f"      Prediction: {prediction['prediction']} ({prediction['confidence']:.1%})")
                    else:
                        print(f"   ❌ Prediction failed: {prediction['error']}")
                        
                except Exception as e:
                    print(f"   ❌ Error predicting game: {e}")
                    continue
            
            all_predictions[sport] = sport_predictions
            print(f"   📊 Generated {len(sport_predictions)} predictions for {sport}")
        
        return all_predictions

def main():
    """Test the ESPN ML model with your data"""
    
    try:
        # Import your complete ESPN pipeline
        from complete_data_pipeline import CompleteSportsDataPipeline
        
        print("🚀 TESTING ESPN ML MODEL WITH YOUR 6,717+ GAMES")
        print("=" * 60)
        
        # Initialize your ESPN pipeline (no API keys needed!)
        data_pipeline = CompleteSportsDataPipeline()
        
        # Initialize ML integration
        ml_integration = ESPNMLPipelineIntegration(data_pipeline)
        
        # Check your data
        print("\n📊 Checking your ESPN data...")
        health_report = data_pipeline.get_database_health_report()
        
        print(f"\n🎯 YOUR DATA SUMMARY:")
        for sport, analysis in health_report['sport_analysis'].items():
            completed_games = analysis.get('completed_games', 0)
            teams_count = analysis['teams_count']
            
            print(f"   {sport.upper()}: {completed_games} games, {teams_count} teams", end="")
            if completed_games >= 50:
                print(" ✅ Ready for ML")
            else:
                print(f" ⚠️ Need {50 - completed_games} more games")
        
        # Train models
        print("\n🤖 Training ML models...")
        training_results = ml_integration.train_all_sports()
        
        # Test predictions
        successful_sports = [sport for sport, result in training_results.items() 
                           if 'error' not in result and 'ensemble' in result]
        
        if successful_sports:
            print(f"\n🎉 SUCCESS! Trained models for: {', '.join(successful_sports)}")
            
            # Test predictions
            test_results = ml_integration.test_predictions()
            
            # Try predicting today's games
            todays_predictions = ml_integration.predict_todays_games()
            
            print(f"\n🏆 ESPN ML SYSTEM READY!")
            print(f"✅ Trained models: {len(successful_sports)} sports")
            print(f"✅ Using {health_report['totals']['completed_games']} completed games")
            print(f"✅ Ready for predictions!")
            print(f"🚀 Much better than limited SportsRadar data!")
            
        else:
            print(f"\n❌ No models were successfully trained")
            print(f"💡 Check that you have enough games in your database")
        
    except Exception as e:
        print(f"❌ ESPN ML test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()