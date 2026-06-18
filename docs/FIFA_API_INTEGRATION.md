# 🌐 External APIs Integration

## Overview

This document explains how to integrate external APIs to enhance the World Cup 2026 prediction model with real-time data and rich statistics.

## Available APIs

### 1. FIFA Official API (Free)
- **Base URL**: `https://givevoicetofootball.fifa.com/api/v1`
- **Cost**: Free
- **Documentation**: https://givevoicetofootball.github.io/api/

### 2. API-Football / API-Sports (Commercial)
- **Base URL**: `https://v3.football.api-sports.io`
- **Cost**: Free tier (100 requests/day), Paid tiers available
- **Documentation**: https://www.api-football.com/documentation-v3
- **Features**: More comprehensive data, predictions, odds, live scores

---

## 1. FIFA Official API

### Overview

## API Documentation

- **Base URL**: `https://givevoicetofootball.fifa.com/api/v1`
- **Documentation**: https://givevoicetofootball.github.io/api/
- **Swagger**: https://givevoicetofootball.fifa.com/ApiFdcpSwagger/

## Available Data

The FIFA API provides access to:

### 1. **Seasons Data**
- Competition information
- Season dates and status
- Tournament structure

### 2. **Match Details**
- Live scores and results
- Match statistics
- Team lineups
- Match events (goals, cards, substitutions)

### 3. **Team Statistics**
- Goals scored/conceded
- Possession statistics
- Shots on target
- Pass accuracy
- Defensive metrics

### 4. **Player Information**
- Squad lists
- Player statistics
- Performance metrics

### 5. **Competition Stages**
- Group stage standings
- Knockout bracket
- Match schedules

## Implementation

### Client Module

We've created [`fifa_api_client.py`](../src/data/fifa_api_client.py) with the following features:

```python
from src.data.fifa_api_client import FIFAAPIClient

# Initialize client
client = FIFAAPIClient()

# Get World Cup 2026 data
data = client.get_world_cup_2026_data()

# Access specific data
seasons = client.search_seasons("FIFA World Cup 2026")
matches = client.get_stage_matches(stage_id)
team_stats = client.get_team_statistics(team_id, season_id)
```

## How to Improve the Model

### 1. **Enhanced Match Predictions**

Use real-time team statistics to improve predictions:

```python
# Get team statistics
home_stats = client.get_team_statistics(home_team_id, season_id)
away_stats = client.get_team_statistics(away_team_id, season_id)

# Extract features
features = {
    'home_goals_avg': home_stats['goals_scored'] / home_stats['matches_played'],
    'away_goals_avg': away_stats['goals_scored'] / away_stats['matches_played'],
    'home_defense': home_stats['goals_conceded'] / home_stats['matches_played'],
    'away_defense': away_stats['goals_conceded'] / away_stats['matches_played'],
    'home_possession': home_stats['possession_avg'],
    'away_possession': away_stats['possession_avg']
}
```

### 2. **Form Analysis**

Track recent performance:

```python
# Get last 5 matches
recent_matches = client.get_stage_matches(stage_id)[-5:]

# Calculate form
form_score = sum(1 if match['result'] == 'W' else 0.5 if match['result'] == 'D' else 0 
                 for match in recent_matches)
```

### 3. **Head-to-Head History**

Analyze historical matchups:

```python
# Get all matches between two teams
h2h_matches = [m for m in all_matches 
               if (m['home_team'] == team_a and m['away_team'] == team_b) or
                  (m['home_team'] == team_b and m['away_team'] == team_a)]

# Calculate H2H statistics
h2h_wins = sum(1 for m in h2h_matches if winner(m) == team_a)
h2h_draws = sum(1 for m in h2h_matches if is_draw(m))
```

### 4. **Player Impact**

Consider key player availability:

```python
# Get squad information
squad = client.get_stage_squads(stage_id)

# Check for key players
key_players_available = check_player_availability(squad, key_player_ids)

# Adjust prediction based on availability
if not key_players_available:
    prediction_adjustment = -0.1  # Reduce win probability
```

### 5. **Live Updates**

Integrate real-time match data:

```python
# During match
match_details = client.get_match_details(match_id)

# Update predictions based on:
# - Current score
# - Time elapsed
# - Red cards
# - Momentum indicators
```

## Integration with Current Model

### Step 1: Extend EnsemblePredictor

```python
class EnhancedEnsemblePredictor(EnsemblePredictor):
    def __init__(self):
        super().__init__()
        self.fifa_client = FIFAAPIClient()
    
    def predict_match_enhanced(self, home_team, away_team):
        # Get base prediction
        base_prediction = self.predict_match(home_team, away_team)
        
        # Get FIFA API data
        home_stats = self.fifa_client.get_team_statistics(home_team, season_id)
        away_stats = self.fifa_client.get_team_statistics(away_team, season_id)
        
        # Adjust prediction with real data
        adjustment = calculate_adjustment(home_stats, away_stats)
        
        return adjust_probabilities(base_prediction, adjustment)
```

### Step 2: Add New Features

```python
def extract_fifa_features(team_id, season_id):
    """Extract features from FIFA API"""
    stats = fifa_client.get_team_statistics(team_id, season_id)
    
    return {
        'attack_strength': stats['goals_scored'] / stats['matches_played'],
        'defense_strength': stats['goals_conceded'] / stats['matches_played'],
        'possession_avg': stats['possession_percentage'],
        'shots_on_target_ratio': stats['shots_on_target'] / stats['total_shots'],
        'pass_accuracy': stats['successful_passes'] / stats['total_passes'],
        'form': calculate_form(stats['recent_results'])
    }
```

### Step 3: Update Auto-Updater

```python
# In auto_updater.py
from src.data.fifa_api_client import FIFAAPIClient

class AutoDataUpdater:
    def __init__(self):
        self.fifa_client = FIFAAPIClient()
    
    def update_from_fifa_api(self):
        """Update using official FIFA API"""
        data = self.fifa_client.get_world_cup_2026_data()
        
        # Update schedule with API data
        matches_df = convert_fifa_matches_to_dataframe(data['matches'])
        
        # Merge with existing schedule
        updated_schedule = merge_schedules(self.schedule, matches_df)
        
        return updated_schedule
```

## Benefits

### 1. **Real-Time Data**
- Always up-to-date match results
- Live statistics during matches
- Instant updates on team changes

### 2. **Rich Statistics**
- Detailed team performance metrics
- Player-level data
- Advanced analytics

### 3. **Official Source**
- Authoritative data from FIFA
- Consistent formatting
- Reliable updates

### 4. **Enhanced Predictions**
- More accurate forecasts
- Better understanding of team form
- Consideration of more factors

## Next Steps

1. **Test API Availability**
   ```bash
   python src/data/fifa_api_client.py
   ```

2. **Integrate with Auto-Updater**
   - Replace web scraping with API calls
   - Add fallback to scraping if API unavailable

3. **Enhance Model**
   - Add FIFA statistics as features
   - Train model with historical API data
   - Implement real-time adjustments

4. **Monitor Performance**
   - Track prediction accuracy
   - Compare API vs scraping results
   - Optimize feature weights

## Example Usage

```python
# Complete workflow
from src.data.fifa_api_client import FIFAAPIClient
from src.data.auto_updater import AutoDataUpdater

# Initialize
fifa_client = FIFAAPIClient()
updater = AutoDataUpdater()

# Get latest data
wc_data = fifa_client.get_world_cup_2026_data()

# Update schedule
updater.update_from_fifa_api()

# Make enhanced prediction
prediction = predictor.predict_match_enhanced('Brazil', 'Argentina')

print(f"Win probabilities:")
print(f"  Brazil: {prediction['home_win']:.1%}")
print(f"  Draw: {prediction['draw']:.1%}")
print(f"  Argentina: {prediction['away_win']:.1%}")
```

## Notes

- API may not be fully available until closer to the tournament
- Implement proper error handling and fallbacks
- Cache API responses to reduce load
- Respect rate limits and terms of service

## Resources

- [FIFA API Documentation](https://givevoicetofootball.github.io/api/)
- [Current Implementation](../src/data/fifa_api_client.py)
- [Auto-Updater](../src/data/auto_updater.py)

---

## 2. API-Football (API-Sports)

### Overview

API-Football is a comprehensive commercial API that provides extensive football data including live scores, statistics, predictions, and odds.

### Setup

1. **Get API Key**
   - Sign up at https://www.api-football.com/
   - Choose a plan (Free tier: 100 requests/day)
   - Get your API key from the dashboard

2. **Configure Environment**
   ```bash
   export API_FOOTBALL_KEY='your-api-key-here'
   ```

3. **Test Connection**
   ```bash
   python src/data/api_football_client.py
   ```

### Features

#### 1. **Live Scores**
```python
from src.data.api_football_client import APIFootballClient

client = APIFootballClient()

# Get live World Cup matches
live_matches = client.get_live_fixtures()

for match in live_matches:
    print(f"{match['teams']['home']['name']} {match['goals']['home']} - {match['goals']['away']} {match['teams']['away']['name']}")
```

#### 2. **Detailed Statistics**
```python
# Get match statistics
stats = client.get_fixture_statistics(fixture_id)

# Available stats:
# - Shots on Goal
# - Shots off Goal
# - Total Shots
# - Blocked Shots
# - Shots insidebox
# - Shots outsidebox
# - Fouls
# - Corner Kicks
# - Offsides
# - Ball Possession
# - Yellow Cards
# - Red Cards
# - Goalkeeper Saves
# - Total passes
# - Passes accurate
# - Passes %
```

#### 3. **AI Predictions**
```python
# Get API-Football's AI predictions
prediction = client.get_predictions(fixture_id)

print(f"Winner: {prediction['predictions']['winner']['name']}")
print(f"Win Probability: {prediction['predictions']['percent']['home']}% - {prediction['predictions']['percent']['away']}%")
print(f"Advice: {prediction['predictions']['advice']}")
```

#### 4. **Head-to-Head Analysis**
```python
# Get H2H matches
h2h_matches = client.get_head_to_head(team1_id, team2_id)

# Analyze historical performance
wins_team1 = sum(1 for m in h2h_matches if m['teams']['home']['winner'])
draws = sum(1 for m in h2h_matches if m['goals']['home'] == m['goals']['away'])
```

#### 5. **Team Statistics**
```python
# Get comprehensive team stats
team_stats = client.get_team_statistics(team_id, league_id=1, season=2026)

# Available metrics:
# - Form (last 5 matches)
# - Goals scored/conceded
# - Clean sheets
# - Failed to score
# - Average goals per match
# - Biggest win/loss
# - Penalty statistics
```

### Integration with Model

#### Enhanced Prediction System

```python
from src.data.api_football_client import APIFootballClient
from src.models.ensemble_predictor import EnsemblePredictor

class EnhancedPredictor:
    def __init__(self, api_key):
        self.base_predictor = EnsemblePredictor()
        self.api_client = APIFootballClient(api_key)
    
    def predict_with_api_data(self, home_team, away_team, fixture_id):
        # Get base prediction
        base_pred = self.base_predictor.predict_match(home_team, away_team)
        
        # Get API-Football prediction
        api_pred = self.api_client.get_predictions(fixture_id)
        
        # Get match statistics
        stats = self.api_client.get_fixture_statistics(fixture_id)
        
        # Combine predictions (weighted average)
        combined = {
            'home_win': 0.6 * base_pred['home_win'] + 0.4 * (api_pred['predictions']['percent']['home'] / 100),
            'draw': 0.6 * base_pred['draw'] + 0.4 * (api_pred['predictions']['percent']['draw'] / 100),
            'away_win': 0.6 * base_pred['away_win'] + 0.4 * (api_pred['predictions']['percent']['away'] / 100)
        }
        
        return combined
```

### Pricing

| Plan | Requests/Day | Cost/Month | Features |
|------|--------------|------------|----------|
| Free | 100 | $0 | Basic data |
| Basic | 1,000 | $15 | + Predictions |
| Pro | 10,000 | $50 | + Odds, Live |
| Ultra | 100,000 | $150 | + Premium support |

### Best Practices

1. **Cache Responses**
   ```python
   import functools
   from datetime import timedelta
   
   @functools.lru_cache(maxsize=128)
   def get_cached_fixture(fixture_id):
       return client.get_fixture_details(fixture_id)
   ```

2. **Rate Limiting**
   ```python
   import time
   
   def rate_limited_request(func):
       last_call = [0]
       min_interval = 1.0  # 1 second between requests
       
       def wrapper(*args, **kwargs):
           elapsed = time.time() - last_call[0]
           if elapsed < min_interval:
               time.sleep(min_interval - elapsed)
           result = func(*args, **kwargs)
           last_call[0] = time.time()
           return result
       
       return wrapper
   ```

3. **Error Handling**
   ```python
   def safe_api_call(func):
       def wrapper(*args, **kwargs):
           try:
               return func(*args, **kwargs)
           except Exception as e:
               logger.error(f"API call failed: {e}")
               return None
       return wrapper
   ```

### Comparison: FIFA API vs API-Football

| Feature | FIFA API | API-Football |
|---------|----------|--------------|
| Cost | Free | Free tier + Paid |
| Requests/Day | Unlimited | 100 (free) |
| Live Scores | ✓ | ✓ |
| Statistics | Basic | Comprehensive |
| Predictions | ✗ | ✓ |
| Odds | ✗ | ✓ |
| Historical Data | Limited | Extensive |
| Player Stats | Basic | Detailed |
| Reliability | Good | Excellent |

### Recommendation

**For Development:**
- Use FIFA API (free, unlimited)
- Implement API-Football as optional enhancement

**For Production:**
- Consider API-Football Pro plan
- More reliable and comprehensive data
- Better for serious predictions

### Implementation Files

- [`api_football_client.py`](../src/data/api_football_client.py) - API-Football client
- [`fifa_api_client.py`](../src/data/fifa_api_client.py) - FIFA API client
- [`auto_updater.py`](../src/data/auto_updater.py) - Automatic data updater
