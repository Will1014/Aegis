"""
Aegis Training Data Downloader
==============================
Downloads bulk data from Sportsmonks to train Manager DNA and Squad Fit models.

Usage:
    # Set API token
    import os
    os.environ["SPORTMONKS_API_TOKEN"] = "your_token"
    
    # Run download
    from training_data_downloader import TrainingDataDownloader
    downloader = TrainingDataDownloader()
    downloader.download_all()
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

# Premier League teams with Sportsmonks IDs
PREMIER_LEAGUE_TEAMS = {
    "Arsenal": 1,
    "Tottenham": 6,
    "Aston Villa": 7,
    "West Ham": 8,
    "Liverpool": 9,
    "Everton": 10,
    "Newcastle": 11,
    "Crystal Palace": 12,
    "Wolves": 13,
    "Man United": 14,
    "Brighton": 15,
    "Fulham": 16,
    "Man City": 17,
    "Chelsea": 18,
    "Bournemouth": 19,
    "Nottingham Forest": 21,
    "Brentford": 63,
    "Leicester": 26,
    "Ipswich": 57,
    "Southampton": 20,
}

# Seasons to download
SEASONS = {
    "2024/25": 23614,
    "2023/24": 21646,
    "2022/23": 19735,
}

# Output directory
TRAINING_DATA_DIR = Path("/content/drive/MyDrive/GaiaMetric/Clients/Aegis/data/raw")


# =============================================================================
# LIGHTWEIGHT API CLIENT (standalone, no package dependency)
# =============================================================================

class SimpleClient:
    """Minimal Sportsmonks client for training data download."""
    
    BASE_URL = "https://api.sportmonks.com/v3"
    
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.last_request = 0
        self.min_interval = 0.4  # 2.5 requests per second
        
        try:
            import requests
            self.requests = requests
            self.session = requests.Session()
        except ImportError:
            raise ImportError("requests library required: pip install requests")
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request = time.time()
    
    def _request(self, endpoint: str, params: Dict = None, include: List[str] = None) -> Dict:
        """Make API request."""
        self._rate_limit()
        
        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        params["api_token"] = self.api_token
        
        if include:
            params["include"] = ";".join(include)
        
        response = self.session.get(url, params=params)
        
        if response.status_code == 429:
            retry = int(response.headers.get("Retry-After", 60))
            print(f"  ⏳ Rate limited, waiting {retry}s...")
            time.sleep(retry)
            return self._request(endpoint, params, include)
        
        if response.status_code == 404:
            return {"data": None}
        
        response.raise_for_status()
        return response.json()
    
    def get_team_statistics(self, team_id: int, season_id: int) -> Dict:
        """Get team statistics for a specific season."""
        return self._request(
            f"/football/teams/{team_id}",
            params={"filters": f"teamstatisticSeasons:{season_id}"},
            include=["statistics.details.type", "coaches"]
        )
    
    def get_squad(self, team_id: int, season_id: int) -> Dict:
        """Get squad with player statistics."""
        return self._request(
            f"/football/squads/teams/{team_id}",
            params={"filters": f"seasonSquads:{season_id}"},
            include=["player.position", "player.detailedPosition", "player.statistics.details.type"]
        )
    
    def get_coach(self, coach_id: int) -> Dict:
        """Get coach details."""
        return self._request(
            f"/football/coaches/{coach_id}",
            include=["teams", "statistics.details.type"]
        )


# =============================================================================
# TRAINING DATA DOWNLOADER
# =============================================================================

class TrainingDataDownloader:
    """
    Downloads and organises training data for MTFI models.
    
    Creates two datasets:
    1. team_statistics.json - Team tactical profiles for Manager DNA clustering
    2. squad_profiles.json - Player stats for Squad Fit modelling
    """
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.environ.get("SPORTMONKS_API_TOKEN")
        if not self.api_token:
            raise ValueError("API token required. Set SPORTMONKS_API_TOKEN env var.")
        
        self.client = SimpleClient(self.api_token)
        self.output_dir = TRAINING_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.team_statistics = []
        self.squad_profiles = []
        self.coaches = {}
    
    def download_all(self, teams: Dict = None, seasons: Dict = None):
        """
        Download all training data.
        
        Args:
            teams: Dict of {name: id}. Defaults to PREMIER_LEAGUE_TEAMS
            seasons: Dict of {name: id}. Defaults to SEASONS
        """
        teams = teams or PREMIER_LEAGUE_TEAMS
        seasons = seasons or SEASONS
        
        print("=" * 60)
        print("AEGIS TRAINING DATA DOWNLOAD")
        print("=" * 60)
        print(f"Teams: {len(teams)}")
        print(f"Seasons: {len(seasons)}")
        print(f"Total combinations: {len(teams) * len(seasons)}")
        print(f"Output: {self.output_dir}")
        print("=" * 60)
        
        total = len(teams) * len(seasons)
        current = 0
        
        for season_name, season_id in seasons.items():
            print(f"\n📅 Season: {season_name}")
            print("-" * 40)
            
            for team_name, team_id in teams.items():
                current += 1
                print(f"  [{current}/{total}] {team_name}...", end=" ", flush=True)
                
                try:
                    # Download team statistics
                    team_data = self.client.get_team_statistics(team_id, season_id)
                    
                    if team_data.get("data"):
                        record = self._process_team_stats(
                            team_data["data"], 
                            team_name, 
                            season_name,
                            season_id
                        )
                        if record:
                            self.team_statistics.append(record)
                            print("✓ stats", end=" ")
                    
                    # Download squad
                    squad_data = self.client.get_squad(team_id, season_id)
                    
                    if squad_data.get("data"):
                        players = self._process_squad(
                            squad_data["data"],
                            team_name,
                            team_id,
                            season_name,
                            season_id
                        )
                        self.squad_profiles.extend(players)
                        print(f"✓ {len(players)} players")
                    else:
                        print("(no squad data)")
                        
                except Exception as e:
                    print(f"✗ Error: {e}")
        
        # Save data
        self._save_data()
        
        print("\n" + "=" * 60)
        print("✓ DOWNLOAD COMPLETE")
        print("=" * 60)
        print(f"  Team statistics: {len(self.team_statistics)} records")
        print(f"  Player profiles: {len(self.squad_profiles)} records")
        print(f"  Files saved to: {self.output_dir}")
    
    def _process_team_stats(
        self, 
        data: Dict, 
        team_name: str, 
        season_name: str,
        season_id: int
    ) -> Optional[Dict]:
        """Extract tactical metrics from team statistics."""
        
        stats = data.get("statistics", [])
        if not stats:
            return None
        
        # Find the correct season's stats
        season_stats = None
        for s in stats:
            if s.get("season_id") == season_id:
                season_stats = s
                break
        
        if not season_stats:
            season_stats = stats[0] if stats else {}
        
        details = season_stats.get("details", [])
        
        # Parse statistics into a flat dict
        parsed = {}
        for detail in details:
            type_info = detail.get("type", {})
            code = type_info.get("code", "")
            value = detail.get("value", {})
            
            if isinstance(value, dict):
                # Extract all nested values
                for key, val in value.items():
                    if isinstance(val, dict):
                        for subkey, subval in val.items():
                            parsed[f"{code}_{key}_{subkey}"] = subval
                    else:
                        parsed[f"{code}_{key}"] = val
            else:
                parsed[code] = value
        
        # Get coach info if available
        coach_name = "Unknown"
        coaches = data.get("coaches", [])
        if coaches:
            # Find active coach or most recent
            for coach in coaches:
                if coach.get("end") is None:
                    coach_name = coach.get("common_name", coach.get("name", "Unknown"))
                    break
            if coach_name == "Unknown" and coaches:
                coach_name = coaches[0].get("common_name", coaches[0].get("name", "Unknown"))
        
        return {
            "team_id": data.get("id"),
            "team_name": team_name,
            "season": season_name,
            "season_id": season_id,
            "coach": coach_name,
            
            # Key tactical metrics
            "possession_avg": parsed.get("ball-possession_all_average", 0),
            "possession_home": parsed.get("ball-possession_home_average", 0),
            "possession_away": parsed.get("ball-possession_away_average", 0),
            
            "pass_accuracy": parsed.get("passes-percentage_all_average", 0),
            
            "tackles_avg": parsed.get("tackles_all_average", 0),
            "interceptions_avg": parsed.get("interceptions_all_average", 0),
            
            "shots_avg": parsed.get("shots-total_all_average", 0),
            "shots_on_target_avg": parsed.get("shots-on-target_all_average", 0),
            "dangerous_attacks_avg": parsed.get("dangerous-attacks_all_average", 0),
            
            "goals_avg": parsed.get("goals_all_average", 0),
            "goals_home_avg": parsed.get("goals_home_average", 0),
            "goals_away_avg": parsed.get("goals_away_average", 0),
            
            "conceded_avg": parsed.get("goals-conceded_all_average", 0),
            "clean_sheets": parsed.get("cleansheets_all_count", 0),
            "clean_sheet_pct": parsed.get("cleansheets_all_percentage", 0),
            
            "wins": parsed.get("wins_all_count", 0),
            "draws": parsed.get("draws_all_count", 0),
            "losses": parsed.get("lost_all_count", 0),
            
            "win_rate": parsed.get("wins_all_percentage", 0),
            "win_rate_home": parsed.get("wins_home_percentage", 0),
            "win_rate_away": parsed.get("wins_away_percentage", 0),
            
            # Raw parsed data for exploration
            "_raw_stats": parsed
        }
    
    def _process_squad(
        self,
        data: List[Dict],
        team_name: str,
        team_id: int,
        season_name: str,
        season_id: int
    ) -> List[Dict]:
        """Extract player profiles from squad data."""
        
        players = []
        
        for entry in data:
            player = entry.get("player", {})
            if not player:
                continue
            
            # Basic info
            profile = {
                "player_id": player.get("id"),
                "name": player.get("common_name") or player.get("name", "Unknown"),
                "team_name": team_name,
                "team_id": team_id,
                "season": season_name,
                "season_id": season_id,
                "jersey_number": entry.get("jersey_number"),
                "date_of_birth": player.get("date_of_birth"),
            }
            
            # Position
            position = player.get("position", {})
            if isinstance(position, dict):
                profile["position"] = position.get("name", "Unknown")
            else:
                profile["position"] = "Unknown"
            
            detailed_pos = player.get("detailedPosition", {})
            if isinstance(detailed_pos, dict):
                profile["detailed_position"] = detailed_pos.get("name", profile["position"])
            else:
                profile["detailed_position"] = profile["position"]
            
            # Statistics
            statistics = player.get("statistics", [])
            if statistics:
                # Get most recent season stats
                season_stats = statistics[0] if statistics else {}
                details = season_stats.get("details", [])
                
                # Map statistics
                stat_mapping = {
                    "appearances": "appearances",
                    "minutes-played": "minutes",
                    "goals": "goals",
                    "assists": "assists",
                    "cleansheets": "clean_sheets",
                    "saves": "saves",
                    "tackles": "tackles",
                    "interceptions": "interceptions",
                    "clearances": "clearances",
                    "passes-accuracy": "pass_accuracy",
                    "key-passes": "key_passes",
                    "successful-dribbles": "dribbles",
                    "shots-total": "shots",
                    "shots-on-target": "shots_on_target",
                    "rating": "rating"
                }
                
                for detail in details:
                    type_info = detail.get("type", {})
                    code = type_info.get("code", "")
                    value = detail.get("value", {})
                    
                    if isinstance(value, dict):
                        value = value.get("total", value.get("count", value.get("value", 0)))
                    
                    if code in stat_mapping:
                        profile[stat_mapping[code]] = value
            
            # Fill missing stats with 0
            for stat in ["appearances", "minutes", "goals", "assists", "clean_sheets",
                        "saves", "tackles", "interceptions", "clearances", 
                        "pass_accuracy", "key_passes", "dribbles", "shots",
                        "shots_on_target", "rating"]:
                if stat not in profile:
                    profile[stat] = 0
            
            players.append(profile)
        
        return players
    
    def _save_data(self):
        """Save downloaded data to JSON files."""
        
        # Team statistics (for Manager DNA clustering)
        team_file = self.output_dir / "team_statistics.json"
        with open(team_file, "w") as f:
            json.dump(self.team_statistics, f, indent=2)
        
        # Squad profiles (for Squad Fit modelling)
        squad_file = self.output_dir / "squad_profiles.json"
        with open(squad_file, "w") as f:
            json.dump(self.squad_profiles, f, indent=2)
        
        # Metadata
        metadata = {
            "downloaded_at": datetime.now().isoformat(),
            "teams_count": len(set(t["team_name"] for t in self.team_statistics)),
            "seasons": list(SEASONS.keys()),
            "team_records": len(self.team_statistics),
            "player_records": len(self.squad_profiles)
        }
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Check for API token
    if not os.environ.get("SPORTMONKS_API_TOKEN"):
        print("ERROR: Set SPORTMONKS_API_TOKEN environment variable")
        print("Example: export SPORTMONKS_API_TOKEN='your_token_here'")
        exit(1)
    
    downloader = TrainingDataDownloader()
    downloader.download_all()