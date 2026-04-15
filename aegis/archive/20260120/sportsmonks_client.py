"""
Aegis Prototype - Sportsmonks API Client
Aligned with Sportsmonks API v3 documentation

This module handles all API interactions following Sportsmonks best practices:
- Cache static entities (types, states, countries)
- Use includes efficiently
- Handle rate limiting
- Paginate with has_more / idAfter
"""

import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Configuration
API_BASE_URL = "https://api.sportmonks.com/v3"
DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"
RAW_DIR = DATA_DIR / "raw"

for d in [CACHE_DIR, RAW_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class SportmonksClient:
    """
    Sportsmonks API v3 Client
    
    Key endpoints for MTFI:
    - /football/fixtures - Match data with formations, statistics
    - /football/coaches/{id} - Coach/manager data with latest fixtures
    - /football/teams/{id} - Team data with squad, statistics
    - /football/players/{id} - Player data with statistics
    - /football/expected/fixtures - xG data per fixture
    - /core/types - All statistic types (cache this!)
    """
    
    # Statistic Type IDs (from Sportsmonks docs)
    STAT_TYPES = {
        # Team/Match Statistics
        "BALL_POSSESSION": 45,
        "SHOTS_TOTAL": 42,
        "SHOTS_ON_TARGET": 86,
        "SHOTS_OFF_TARGET": 41,
        "SHOTS_BLOCKED": 58,
        "SHOTS_INSIDE_BOX": 59,
        "SHOTS_OUTSIDE_BOX": 60,
        "CORNERS": 34,
        "OFFSIDES": 37,
        "FOULS": 56,
        "YELLOW_CARDS": 84,
        "RED_CARDS": 83,
        "GOALKEEPER_SAVES": 57,
        "TOTAL_PASSES": 80,
        "PASSES_ACCURATE": 81,
        "PASSES_PERCENTAGE": 82,
        "TACKLES": 78,
        "INTERCEPTIONS": 79,
        "CLEARANCES": 53,
        "FREE_KICKS": 55,
        "THROW_INS": 89,
        "GOAL_KICKS": 54,
        "ATTACKS": 43,
        "DANGEROUS_ATTACKS": 44,
        
        # Player Statistics
        "PLAYER_GOALS": 52,
        "PLAYER_ASSISTS": 79,
        "PLAYER_MINUTES_PLAYED": 118,
        "PLAYER_RATING": 118,
        "PLAYER_SUCCESSFUL_DRIBBLES": 115,
        "PLAYER_DUELS_WON": 105,
        "PLAYER_DUELS_TOTAL": 104,
        "PLAYER_AERIAL_WON": 107,
        "PLAYER_KEY_PASSES": 117,
    }
    
    def __init__(self, api_token: Optional[str] = None):
        """Initialize client with API token from env or parameter"""
        self.api_token = api_token or os.environ.get("SPORTMONKS_API_TOKEN")
        self.session = requests.Session()
        self.request_count = 0
        self.last_request_time = 0
        
        # Cache for static entities
        self._types_cache = None
        self._states_cache = None
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make authenticated API request with rate limiting"""
        if not self.api_token:
            raise ValueError("API token not set. Set SPORTMONKS_API_TOKEN env var or pass to constructor.")
        
        # Rate limiting - max 3000/hour = ~0.83/second
        elapsed = time.time() - self.last_request_time
        if elapsed < 0.4:  # Conservative 2.5 req/s
            time.sleep(0.4 - elapsed)
        
        url = f"{API_BASE_URL}{endpoint}"
        params = params or {}
        params["api_token"] = self.api_token
        
        response = self.session.get(url, params=params)
        self.last_request_time = time.time()
        self.request_count += 1
        
        if response.status_code == 429:
            # Rate limited - wait and retry
            retry_after = int(response.headers.get("Retry-After", 60))
            print(f"Rate limited. Waiting {retry_after}s...")
            time.sleep(retry_after)
            return self._make_request(endpoint, params)
        
        response.raise_for_status()
        return response.json()
    
    # =========================================================================
    # STATIC ENTITY CACHING (Best Practice from docs)
    # =========================================================================
    
    def get_types(self, force_refresh: bool = False) -> List[Dict]:
        """
        Get all types (statistic types, event types, etc.)
        Cache these as they rarely change!
        """
        cache_file = CACHE_DIR / "types.json"
        
        if not force_refresh and cache_file.exists():
            # Check cache age (refresh if > 24h)
            if (time.time() - cache_file.stat().st_mtime) < 86400:
                with open(cache_file) as f:
                    return json.load(f)
        
        print("Fetching types from API (caching for 24h)...")
        all_types = []
        page = 1
        
        while True:
            result = self._make_request("/core/types", {"page": page, "per_page": 100})
            all_types.extend(result.get("data", []))
            
            if not result.get("pagination", {}).get("has_more", False):
                break
            page += 1
        
        with open(cache_file, "w") as f:
            json.dump(all_types, f, indent=2)
        
        return all_types
    
    def get_statistic_types(self) -> Dict[int, Dict]:
        """Get statistic types as lookup dict by ID"""
        if self._types_cache is None:
            types = self.get_types()
            self._types_cache = {
                t["id"]: t for t in types 
                if t.get("model_type") == "statistic"
            }
        return self._types_cache
    
    # =========================================================================
    # COACH/MANAGER DATA
    # =========================================================================
    
    def get_coach(self, coach_id: int, include: str = "teams,nationality") -> Dict:
        """
        Get coach by ID
        
        Includes: teams, nationality, statistics, sidelined, latest
        """
        result = self._make_request(f"/football/coaches/{coach_id}", {"include": include})
        return result.get("data", {})
    
    def search_coaches(self, name: str) -> List[Dict]:
        """Search for coaches by name"""
        result = self._make_request(f"/football/coaches/search/{name}")
        return result.get("data", [])
    
    def get_coach_fixtures(
        self, 
        coach_id: int, 
        team_id: int,
        season_ids: List[int] = None,
        include: str = "formations;statistics;scores;participants"
    ) -> List[Dict]:
        """
        Get fixtures for a coach's team
        
        Strategy: Get team's fixtures filtered by season, which coach managed
        """
        fixtures = []
        
        # Get fixtures by team and season
        if season_ids:
            for season_id in season_ids:
                result = self._make_request(
                    f"/football/fixtures",
                    {
                        "include": include,
                        "filters": f"fixtureSeasons:{season_id};fixtureTeams:{team_id}",
                        "per_page": 50
                    }
                )
                fixtures.extend(result.get("data", []))
        else:
            # Get recent fixtures for the team
            result = self._make_request(
                f"/football/teams/{team_id}/fixtures",
                {"include": include, "per_page": 100}
            )
            fixtures = result.get("data", [])
        
        return fixtures
    
    # =========================================================================
    # TEAM & SQUAD DATA
    # =========================================================================
    
    def get_team(self, team_id: int, include: str = "players;statistics;coaches") -> Dict:
        """
        Get team by ID with squad
        
        Includes: players, statistics, coaches, venue, leagues, seasons
        """
        result = self._make_request(f"/football/teams/{team_id}", {"include": include})
        return result.get("data", {})
    
    def get_team_squad(
        self, 
        team_id: int, 
        season_id: int = None,
        include: str = "player.position;player.statistics"
    ) -> List[Dict]:
        """
        Get team squad with player statistics
        
        Uses squads endpoint for season-specific roster
        """
        params = {"include": include}
        if season_id:
            params["filters"] = f"seasonSquads:{season_id}"
        
        result = self._make_request(f"/football/squads/teams/{team_id}", params)
        return result.get("data", [])
    
    def search_teams(self, name: str) -> List[Dict]:
        """Search for teams by name"""
        result = self._make_request(f"/football/teams/search/{name}")
        return result.get("data", [])
    
    # =========================================================================
    # PLAYER DATA
    # =========================================================================
    
    def get_player(
        self, 
        player_id: int, 
        include: str = "position;detailedPosition;statistics;teams;nationality"
    ) -> Dict:
        """
        Get player by ID with statistics
        
        Key includes for MTFI:
        - position, detailedPosition
        - statistics (season stats)
        - transfers, teams
        """
        result = self._make_request(f"/football/players/{player_id}", {"include": include})
        return result.get("data", {})
    
    def get_player_statistics(
        self, 
        player_id: int, 
        season_id: int = None
    ) -> Dict:
        """Get player season statistics"""
        params = {"include": "statistics.type"}
        if season_id:
            params["filters"] = f"playerStatisticSeasons:{season_id}"
        
        result = self._make_request(f"/football/players/{player_id}", params)
        return result.get("data", {})
    
    # =========================================================================
    # FIXTURE DATA
    # =========================================================================
    
    def get_fixture(
        self, 
        fixture_id: int,
        include: str = "participants;scores;formations;statistics;lineups;events;xGFixture"
    ) -> Dict:
        """
        Get single fixture with all tactical data
        
        Key includes:
        - formations: Team formations used
        - statistics: Match stats (possession, tackles, etc)
        - lineups: Player lineups with details
        - xGFixture: Expected goals data
        - events: Goals, cards, substitutions
        """
        result = self._make_request(f"/football/fixtures/{fixture_id}", {"include": include})
        return result.get("data", {})
    
    def get_fixtures_by_date_range(
        self,
        start_date: str,
        end_date: str,
        team_id: int = None,
        include: str = "formations;statistics;scores;participants"
    ) -> List[Dict]:
        """
        Get fixtures between dates
        
        Date format: YYYY-MM-DD
        """
        endpoint = f"/football/fixtures/between/{start_date}/{end_date}"
        if team_id:
            endpoint += f"/{team_id}"
        
        result = self._make_request(endpoint, {"include": include, "per_page": 50})
        return result.get("data", [])
    
    def get_fixtures_by_season(
        self,
        season_id: int,
        team_id: int = None,
        include: str = "formations;statistics;scores;participants"
    ) -> List[Dict]:
        """Get all fixtures for a season (optionally filtered by team)"""
        params = {
            "include": include,
            "per_page": 50,
            "filters": f"fixtureSeasons:{season_id}"
        }
        if team_id:
            params["filters"] += f";fixtureTeams:{team_id}"
        
        all_fixtures = []
        page = 1
        
        while True:
            params["page"] = page
            result = self._make_request("/football/fixtures", params)
            all_fixtures.extend(result.get("data", []))
            
            if not result.get("pagination", {}).get("has_more", False):
                break
            page += 1
            
            # Safety limit
            if page > 20:
                break
        
        return all_fixtures
    
    # =========================================================================
    # EXPECTED GOALS (xG) DATA
    # =========================================================================
    
    def get_xg_by_fixture(self, fixture_id: int) -> Dict:
        """Get expected goals data for a fixture"""
        result = self._make_request(f"/football/expected/fixtures/{fixture_id}")
        return result.get("data", {})
    
    # =========================================================================
    # LEAGUES & SEASONS
    # =========================================================================
    
    def get_league(self, league_id: int, include: str = "currentSeason;seasons") -> Dict:
        """Get league with seasons"""
        result = self._make_request(f"/football/leagues/{league_id}", {"include": include})
        return result.get("data", {})
    
    def get_season(self, season_id: int, include: str = "league;stages") -> Dict:
        """Get season details"""
        result = self._make_request(f"/football/seasons/{season_id}", {"include": include})
        return result.get("data", {})
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def save_response(self, data: Any, filename: str):
        """Save API response to raw data directory"""
        filepath = RAW_DIR / filename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {filepath}")
    
    def extract_team_statistics(self, fixture: Dict, team_id: int) -> Dict:
        """
        Extract statistics for a specific team from fixture data
        
        Returns dict mapping stat code -> value
        """
        stats = {}
        for stat in fixture.get("statistics", []):
            if stat.get("participant_id") == team_id:
                type_info = stat.get("type", {})
                code = type_info.get("code") or type_info.get("developer_name", "").lower().replace("_", "-")
                value = stat.get("data", {}).get("value")
                if code and value is not None:
                    stats[code] = value
        return stats
    
    def extract_formation(self, fixture: Dict, team_id: int) -> Optional[str]:
        """Extract formation for a team from fixture"""
        for formation in fixture.get("formations", []):
            if formation.get("participant_id") == team_id:
                return formation.get("formation")
        return None
    
    def extract_score(self, fixture: Dict, team_id: int) -> Dict:
        """Extract score for a team (current/final)"""
        for score in fixture.get("scores", []):
            if score.get("participant_id") == team_id and score.get("description") == "CURRENT":
                return score.get("score", {})
        return {}


# =============================================================================
# KNOWN IDs FOR PROTOTYPE (can be found via search endpoints)
# =============================================================================

KNOWN_IDS = {
    "leagues": {
        "premier_league": 8,
        "championship": 9,
        "la_liga": 564,
        "bundesliga": 82,
        "serie_a": 384,
        "ligue_1": 301,
    },
    "teams": {
        "tottenham": 6,
        "brentford": 63,
        "manchester_city": 17,
        "liverpool": 9,
        "arsenal": 1,
        "chelsea": 18,
    },
    "coaches": {
        # These need to be looked up via search
        # "thomas_frank": ???
    },
    "seasons": {
        "premier_league_2024_25": 23614,
        "premier_league_2023_24": 21646,
        "premier_league_2022_23": 19735,
    }
}


# =============================================================================
# CLI FOR TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    
    client = SportmonksClient()
    
    if len(sys.argv) < 2:
        print("Usage: python sportsmonks_client.py <command> [args]")
        print("\nCommands:")
        print("  search_coach <name>     Search for a coach")
        print("  search_team <name>      Search for a team")
        print("  get_team <id>           Get team with squad")
        print("  get_fixture <id>        Get fixture details")
        print("  cache_types             Cache all types")
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        if command == "search_coach" and len(sys.argv) > 2:
            results = client.search_coaches(sys.argv[2])
            for coach in results[:10]:
                print(f"ID: {coach['id']}, Name: {coach['common_name']}")
        
        elif command == "search_team" and len(sys.argv) > 2:
            results = client.search_teams(sys.argv[2])
            for team in results[:10]:
                print(f"ID: {team['id']}, Name: {team['name']}")
        
        elif command == "get_team" and len(sys.argv) > 2:
            team = client.get_team(int(sys.argv[2]))
            print(json.dumps(team, indent=2)[:2000])
        
        elif command == "get_fixture" and len(sys.argv) > 2:
            fixture = client.get_fixture(int(sys.argv[2]))
            print(json.dumps(fixture, indent=2)[:3000])
        
        elif command == "cache_types":
            types = client.get_types(force_refresh=True)
            print(f"Cached {len(types)} types")
        
        else:
            print(f"Unknown command: {command}")
            
    except ValueError as e:
        print(f"Error: {e}")
        print("Set SPORTMONKS_API_TOKEN environment variable")
