"""
Aegis API Clients
=================
API clients for Sportsmonks Football API v3 and Hudl StatsBomb API.

Sportsmonks Statistics Strategy:
- DEFAULT: Use dedicated /teams/{id}?include=statistics and /players/{id}?include=statistics.details
  These return pre-aggregated season stats with home/away splits and averages.
- OPTIONAL: Use /fixtures/{id}?include=statistics for match-level granularity when needed.

StatsBomb Statistics Strategy:
- Competitions → Matches → Events / Lineups / Player Stats / Team Stats
- Pre-computed season stats via Player Season Stats and Team Match Stats APIs
- Rich event-level data with xG, xA, OBV, PPDA, progressive actions
"""

import os
import time
import json
import requests
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime

from .config import Config


class SportsmonksClient:
    """
    Sportsmonks API v3 client with rate limiting and caching.
    
    Usage:
        from aegis import SportsmonksClient
        
        # Initialize (reads SPORTMONKS_API_TOKEN from environment)
        client = SportsmonksClient()
        
        # Or pass token directly
        client = SportsmonksClient(api_token="your_token")
        
        # Search for a coach
        coaches = client.search_coaches("Thomas Frank")
        
        # Get team details
        team = client.get_team(6)  # Tottenham
        
        # Get squad
        squad = client.get_squad(6)
    """
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize client.
        
        Args:
            api_token: Sportsmonks API token. If not provided, reads from 
                      SPORTMONKS_API_TOKEN environment variable.
        """
        self.api_token = api_token or os.environ.get("SPORTMONKS_API_TOKEN")
        if not self.api_token:
            raise ValueError(
                "API token required.\n"
                "Set it with: os.environ['SPORTMONKS_API_TOKEN'] = 'your_token'\n"
                "Or pass: SportsmonksClient(api_token='your_token')"
            )
        
        self.base_url = Config.BASE_URL
        self.session = requests.Session()
        self.last_request_time = 0
        self.min_request_interval = 1.0 / Config.REQUESTS_PER_SECOND
        
        # Ensure directories exist
        Config.setup()
    
    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _request(
        self, 
        endpoint: str, 
        params: Optional[Dict] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Make rate-limited API request.
        
        Args:
            endpoint: API endpoint (e.g., "/football/coaches/123")
            params: Additional query parameters
            include: List of includes (e.g., ["teams", "nationality"])
            
        Returns:
            API response JSON
        """
        self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        params["api_token"] = self.api_token
        
        if include:
            params["include"] = ";".join(include)
        
        response = self.session.get(url, params=params)
        
        # Handle rate limiting
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            print(f"â³ Rate limited. Waiting {retry_after}s...")
            time.sleep(retry_after)
            return self._request(endpoint, params, include)
        
        if response.status_code == 404:
            return {"data": None}
        
        response.raise_for_status()
        return response.json()
    
    def _paginate(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        include: Optional[List[str]] = None,
        max_pages: int = 10
    ) -> List[Dict]:
        """Fetch all pages of a paginated endpoint."""
        params = params or {}
        all_data = []
        page = 1
        
        while page <= max_pages:
            params["page"] = page
            result = self._request(endpoint, params, include)
            
            data = result.get("data", [])
            if not data:
                break
                
            all_data.extend(data)
            
            # Check for more pages
            pagination = result.get("pagination", {})
            if not pagination.get("has_more", False):
                break
            
            page += 1
        
        return all_data
    
    # =========================================================================
    # TYPES / REFERENCE DATA
    # =========================================================================
    
    def get_types(self, use_cache: bool = True) -> List[Dict]:
        """
        Get all statistic type definitions.
        
        Args:
            use_cache: Use cached types if available (recommended)
            
        Returns:
            List of type definitions with id, name, code, developer_name
        """
        cache_file = Config.CACHE_DIR / "types.json"
        
        # Check cache
        if use_cache and cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 86400:  # 24 hours
                with open(cache_file) as f:
                    return json.load(f)
        
        # Fetch from API
        print("Fetching statistic types...")
        types_data = self._paginate("/core/types", {"per_page": 100})
        
        # Cache
        Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(types_data, f, indent=2)
        
        print(f"âœ“ Cached {len(types_data)} types")
        return types_data
    
    def get_type_lookup(self) -> Dict[int, Dict]:
        """Get types as a lookup dictionary by ID."""
        types = self.get_types()
        return {t["id"]: t for t in types}
    
    # =========================================================================
    # COACHES
    # =========================================================================
    
    def search_coaches(self, name: str) -> List[Dict]:
        """
        Search for coaches by name.
        
        Args:
            name: Coach name to search for
            
        Returns:
            List of matching coach records
        """
        result = self._request(f"/football/coaches/search/{name}")
        return result.get("data", [])
    
    def get_coach(self, coach_id: int) -> Dict:
        """
        Get coach details by ID.
        
        Args:
            coach_id: Sportsmonks coach ID
            
        Returns:
            Coach record with teams history
        """
        result = self._request(
            f"/football/coaches/{coach_id}",
            include=["teams", "nationality"]
        )
        return result.get("data", {})
    
    # =========================================================================
    # TEAMS
    # =========================================================================
    
    def search_teams(self, name: str) -> List[Dict]:
        """
        Search for teams by name.
        
        Args:
            name: Team name to search for
            
        Returns:
            List of matching team records
        """
        result = self._request(f"/football/teams/search/{name}")
        return result.get("data", [])
    
    def get_team(self, team_id: int) -> Dict:
        """
        Get team details by ID.
        
        Args:
            team_id: Sportsmonks team ID
            
        Returns:
            Team record
        """
        result = self._request(
            f"/football/teams/{team_id}",
            include=["venue", "coaches"]
        )
        return result.get("data", {})
    
    def get_squad(self, team_id: int, season_id: Optional[int] = None) -> List[Dict]:
        """
        Get team squad with player statistics.
        
        Args:
            team_id: Sportsmonks team ID
            season_id: Optional season filter
            
        Returns:
            List of squad players with statistics
        """
        params = {}
        if season_id:
            params["filters"] = f"seasonSquads:{season_id}"
        
        result = self._request(
            f"/football/squads/teams/{team_id}",
            params=params,
            include=["player.position", "player.detailedPosition", "player.statistics.details.type"]
        )
        return result.get("data", [])
    
    # =========================================================================
    # FIXTURES
    # =========================================================================
    
    def get_fixture(self, fixture_id: int) -> Dict:
        """
        Get single fixture with full details.
        
        Args:
            fixture_id: Sportsmonks fixture ID
            
        Returns:
            Fixture record with formations, statistics, scores
        """
        result = self._request(
            f"/football/fixtures/{fixture_id}",
            include=["formations", "statistics", "scores", "participants", "lineups", "events"]
        )
        return result.get("data", {})
    
    def get_fixtures_by_team_and_season(
        self,
        team_id: int,
        season_id: int,
        max_fixtures: int = 76
    ) -> List[Dict]:
        """
        Get fixtures for a team in a specific season.
        
        Args:
            team_id: Sportsmonks team ID
            season_id: Sportsmonks season ID
            max_fixtures: Maximum fixtures to return
            
        Returns:
            List of fixture records
        """
        print(f"Fetching fixtures for team {team_id}, season {season_id}...")
        
        fixtures = self._paginate(
            "/football/fixtures",
            params={
                "filters": f"fixtureSeasons:{season_id};fixtureTeams:{team_id}",
                "per_page": 50
            },
            include=["formations", "statistics", "scores", "participants"]
        )
        
        # Sort by date descending
        fixtures.sort(key=lambda x: x.get("starting_at", ""), reverse=True)
        
        print(f"âœ“ Retrieved {len(fixtures[:max_fixtures])} fixtures")
        return fixtures[:max_fixtures]
    
    def get_fixtures_by_date_range(
        self,
        team_id: int,
        start_date: str,
        end_date: str
    ) -> List[Dict]:
        """
        Get fixtures for a team within a date range.
        
        Args:
            team_id: Sportsmonks team ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of fixture records
        """
        print(f"Fetching fixtures from {start_date} to {end_date}...")
        
        fixtures = self._paginate(
            f"/football/fixtures/between/{start_date}/{end_date}/{team_id}",
            include=["formations", "statistics", "scores", "participants"]
        )
        
        print(f"âœ“ Retrieved {len(fixtures)} fixtures")
        return fixtures
    
    # =========================================================================
    # PLAYERS
    # =========================================================================
    
    def get_player(self, player_id: int) -> Dict:
        """
        Get player details by ID.
        
        Args:
            player_id: Sportsmonks player ID
            
        Returns:
            Player record with statistics
        """
        result = self._request(
            f"/football/players/{player_id}",
            include=["position", "detailedPosition", "statistics.details.type", "teams"]
        )
        return result.get("data", {})
    
    # =========================================================================
    # DEDICATED STATISTICS ENDPOINTS (RECOMMENDED)
    # These return pre-aggregated stats with home/away splits and averages
    # =========================================================================
    
    def get_team_statistics(
        self, 
        team_id: int, 
        season_id: Optional[int] = None
    ) -> Dict:
        """
        Get team statistics with pre-aggregated season data.
        
        Returns home/away splits, averages, and percentages already calculated.
        Much more efficient than aggregating fixture-by-fixture.
        
        Args:
            team_id: Sportsmonks team ID
            season_id: Optional season filter (returns all seasons if not specified)
            
        Returns:
            Team record with nested statistics array containing:
            - goals, assists, clean sheets, cards
            - home/away breakdowns with percentages
            - per-game averages
        """
        params = {}
        if season_id:
            params["filters"] = f"teamstatisticSeasons:{season_id}"
        
        result = self._request(
            f"/football/teams/{team_id}",
            params=params,
            include=["statistics.details.type"]
        )
        return result.get("data", {})
    
    def get_season_statistics(self, season_id: int) -> Dict:
        """
        Get league-wide season statistics for context/normalisation.
        
        Returns aggregated stats across all teams in the season:
        - Total goals, average goals per game
        - Cards distribution
        - Clean sheets
        - League-wide benchmarks
        
        Args:
            season_id: Sportsmonks season ID
            
        Returns:
            Season record with statistics array
        """
        result = self._request(
            f"/football/seasons/{season_id}",
            include=["statistics.type", "league"]
        )
        return result.get("data", {})
    
    def get_player_statistics(
        self, 
        player_id: int, 
        season_id: Optional[int] = None
    ) -> Dict:
        """
        Get player statistics with season aggregates.
        
        Args:
            player_id: Sportsmonks player ID
            season_id: Optional season filter
            
        Returns:
            Player record with statistics.details containing:
            - goals, assists, appearances, minutes
            - cards, rating
            - per-season breakdown
        """
        params = {}
        if season_id:
            params["filters"] = f"playerStatisticSeasons:{season_id}"
        
        result = self._request(
            f"/football/players/{player_id}",
            params=params,
            include=["statistics.details.type", "position", "detailedPosition"]
        )
        return result.get("data", {})
    
    def get_coach_statistics(
        self, 
        coach_id: int, 
        season_id: Optional[int] = None
    ) -> Dict:
        """
        Get coach statistics with season aggregates.
        
        Args:
            coach_id: Sportsmonks coach ID
            season_id: Optional season filter
            
        Returns:
            Coach record with statistics including:
            - wins, draws, losses
            - goals for/against
            - points per game
        """
        params = {}
        if season_id:
            params["filters"] = f"coachstatisticSeasons:{season_id}"
        
        result = self._request(
            f"/football/coaches/{coach_id}",
            params=params,
            include=["statistics.details.type", "teams"]
        )
        return result.get("data", {})
    
    # =========================================================================
    # MATCH-LEVEL STATISTICS (OPTIONAL - for granular analysis)
    # Use these when you need per-fixture data or custom aggregations
    # =========================================================================
    
    def get_fixtures_for_tactical_analysis(
        self,
        team_id: int,
        season_id: int,
        max_fixtures: int = 76
    ) -> List[Dict]:
        """
        Fetch completed fixtures with statistics for tactical aggregation.

        Used by ManagerDNATrainer to compute pass_accuracy, which is not
        available from the team statistics endpoint.

        Confirmed fixture stat structure (API diagnostic Feb 2026):
            stat["participant_id"]   team filter
            stat["type"]["code"]     stat code string (requires include=statistics.type)
            stat["data"]["value"]    the numeric value (NOT stat["value"])

        Args:
            team_id:      Sportsmonks team ID
            season_id:    Sportsmonks season ID
            max_fixtures: Maximum fixtures to return

        Returns:
            List of completed fixture records with statistics and scores.
        """
        fixtures = self._paginate(
            "/football/fixtures",
            params={
                "filters": f"fixtureSeasons:{season_id};fixtureTeams:{team_id}",
                "per_page": 50
            },
            include=["statistics.type", "scores", "participants"]
        )

        # Only completed fixtures (those with scores populated)
        completed = [f for f in fixtures if f.get("scores")]
        completed.sort(key=lambda x: x.get("starting_at", ""), reverse=True)
        return completed[:max_fixtures]

    def get_fixture_statistics(self, fixture_id: int) -> Dict:
        """
        Get detailed match-level statistics for a single fixture.
        
        Use this when you need:
        - Per-match granularity
        - Custom time-windowed analysis (e.g., last 5 games)
        - Form trend analysis
        
        For season aggregates, use get_team_statistics() instead.
        
        Args:
            fixture_id: Sportsmonks fixture ID
            
        Returns:
            Fixture with statistics array
        """
        result = self._request(
            f"/football/fixtures/{fixture_id}",
            include=["statistics.type", "formations", "scores", "participants"]
        )
        return result.get("data", {})
    
    def get_fixtures_with_statistics(
        self,
        team_id: int,
        season_id: int,
        max_fixtures: int = 76
    ) -> List[Dict]:
        """
        Get fixtures with match-level statistics (for custom aggregation).
        
        Note: For most use cases, get_team_statistics() is more efficient.
        Use this method when you need:
        - Per-match data for form analysis
        - Rolling averages (e.g., last 5 games)
        - Custom aggregation logic
        
        Args:
            team_id: Sportsmonks team ID
            season_id: Sportsmonks season ID
            max_fixtures: Maximum fixtures to return
            
        Returns:
            List of fixtures with statistics
        """
        return self.get_fixtures_by_team_and_season(team_id, season_id, max_fixtures)
    
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    
    def fetch_scenario(
        self,
        coach_name: str,
        target_club_name: str,
        season_id: int = 23614,  # 2024/25 default
        use_aggregated_stats: bool = True
    ) -> Dict:
        """
        Fetch all data for an Aegis analysis scenario.
        
        Args:
            coach_name: Manager name (e.g., "Thomas Frank")
            target_club_name: Target club (e.g., "Tottenham")
            season_id: Season to analyse
            use_aggregated_stats: If True (default), use efficient pre-aggregated 
                                  statistics endpoints. If False, fetch match-level
                                  data for custom analysis.
            
        Returns:
            Dictionary with coach, team_stats, season_stats, team, squad data
        """
        print("\n" + "=" * 50)
        print("AEGIS DATA FETCH")
        print("=" * 50)
        
        # 1. Find coach
        print(f"\n[1/5] Finding coach: {coach_name}")
        coaches = self.search_coaches(coach_name)
        if not coaches:
            raise ValueError(f"Coach not found: {coach_name}")
        coach = self.get_coach(coaches[0]["id"])
        print(f"      âœ“ {coach.get('common_name', coach.get('name'))}")
        
        # Get manager's current team
        manager_team_id = None
        if coach.get("teams"):
            for team in coach["teams"]:
                if team.get("end") is None:
                    manager_team_id = team.get("team_id")
                    break
            if not manager_team_id:
                manager_team_id = coach["teams"][0].get("team_id")
        
        # 2. Get manager's team statistics
        print(f"\n[2/5] Fetching manager's team statistics")
        manager_team_stats = None
        fixtures = []
        
        if manager_team_id:
            if use_aggregated_stats:
                # Use efficient aggregated endpoint
                manager_team_stats = self.get_team_statistics(manager_team_id, season_id)
                print(f"      âœ“ Team statistics (aggregated)")
            else:
                # Fetch match-level data for custom analysis
                fixtures = self.get_fixtures_by_team_and_season(
                    team_id=manager_team_id,
                    season_id=season_id,
                    max_fixtures=76
                )
                print(f"      âœ“ {len(fixtures)} fixtures (match-level)")
        
        # 3. Get season statistics (for league context)
        print(f"\n[3/5] Fetching season statistics (league context)")
        season_stats = self.get_season_statistics(season_id)
        league_name = season_stats.get("league", {}).get("name", "Unknown")
        print(f"      âœ“ {league_name} {season_stats.get('name', '')}")
        
        # 4. Find target club
        print(f"\n[4/5] Finding target club: {target_club_name}")
        teams = self.search_teams(target_club_name)
        if not teams:
            raise ValueError(f"Team not found: {target_club_name}")
        team = self.get_team(teams[0]["id"])
        print(f"      âœ“ {team.get('name')}")
        
        # 5. Get squad with player statistics
        print(f"\n[5/5] Fetching squad with player statistics")
        squad = self.get_squad(team["id"], season_id)
        print(f"      âœ“ {len(squad)} players")
        
        # Save to files
        self._save_scenario_data(
            coach=coach,
            manager_team_stats=manager_team_stats,
            fixtures=fixtures,
            season_stats=season_stats,
            team=team,
            squad=squad,
            use_aggregated=use_aggregated_stats
        )
        
        print("\n" + "=" * 50)
        print("âœ“ DATA FETCH COMPLETE")
        print("=" * 50)
        
        return {
            "coach": coach,
            "manager_team_stats": manager_team_stats,
            "fixtures": fixtures,
            "season_stats": season_stats,
            "team": team,
            "squad": squad
        }
    
    def _save_scenario_data(
        self, 
        coach, 
        manager_team_stats, 
        fixtures, 
        season_stats, 
        team, 
        squad,
        use_aggregated: bool
    ):
        """Save fetched data to files."""
        Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(Config.DATA_DIR / "coach.json", "w") as f:
            json.dump(coach, f, indent=2)
        
        if use_aggregated and manager_team_stats:
            with open(Config.DATA_DIR / "manager_team_stats.json", "w") as f:
                json.dump(manager_team_stats, f, indent=2)
        
        if fixtures:
            with open(Config.DATA_DIR / "fixtures.json", "w") as f:
                json.dump(fixtures, f, indent=2)
        
        with open(Config.DATA_DIR / "season_stats.json", "w") as f:
            json.dump(season_stats, f, indent=2)
        
        with open(Config.DATA_DIR / "team.json", "w") as f:
            json.dump(team, f, indent=2)
        
        with open(Config.DATA_DIR / "squad.json", "w") as f:
            json.dump(squad, f, indent=2)
        
        # Save metadata
        metadata = {
            "fetch_type": "aggregated" if use_aggregated else "match_level",
            "timestamp": datetime.now().isoformat()
        }
        with open(Config.DATA_DIR / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n      Files saved to: {Config.DATA_DIR}")


# =============================================================================
# STATSBOMB CLIENT
# =============================================================================

class StatsBombClient:
    """
    Hudl StatsBomb API client with rate limiting and caching.
    
    Provides access to all 9 StatsBomb API endpoints:
    - Competitions, Matches, Lineups, Events, 360 Frames
    - Player Match Stats, Player Season Stats
    - Team Match Stats, Player Mapping
    
    Usage:
        from aegis import StatsBombClient
        
        # Initialize with credentials
        client = StatsBombClient(username="you@email.com", password="your_password")
        
        # Or set environment variables SB_USERNAME / SB_PASSWORD
        client = StatsBombClient()
        
        # Get competitions you have access to
        comps = client.get_competitions()
        
        # Get matches for a competition/season
        matches = client.get_matches(competition_id=2, season_id=282)
        
        # Full scenario fetch (like SportsmonksClient.fetch_scenario)
        data = client.fetch_scenario(
            competition_id=2,       # Premier League
            season_id=282,          # 2024/25
            team_name="Tottenham"
        )
    """
    
    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize StatsBomb client.
        
        Args:
            username: StatsBomb username (email). If not provided, reads SB_USERNAME env var.
            password: StatsBomb password. If not provided, reads SB_PASSWORD env var.
        """
        self.username = username or os.environ.get("SB_USERNAME")
        self.password = password or os.environ.get("SB_PASSWORD")
        
        if not self.username or not self.password:
            raise ValueError(
                "StatsBomb credentials required.\n"
                "Set with:\n"
                '  os.environ["SB_USERNAME"] = "your_email"\n'
                '  os.environ["SB_PASSWORD"] = "your_password"\n'
                "Or pass: StatsBombClient(username='...', password='...')"
            )
        
        self.base_url = Config.STATSBOMB_BASE_URL
        self.alt_url = Config.STATSBOMB_ALT_URL
        self.session = requests.Session()
        self.session.auth = (self.username, self.password)
        self.last_request_time = 0
        self.min_request_interval = 1.0 / Config.STATSBOMB_REQUESTS_PER_SECOND
        self.versions = Config.STATSBOMB_API_VERSIONS
        
        # Ensure directories exist
        Config.setup()
    
    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _request(self, url: str) -> Any:
        """
        Make rate-limited, authenticated API request.
        
        Args:
            url: Full API URL
            
        Returns:
            Parsed JSON response
        """
        self._rate_limit()
        
        response = self.session.get(url)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            print(f"⏳ Rate limited. Waiting {retry_after}s...")
            time.sleep(retry_after)
            return self._request(url)
        
        if response.status_code == 404:
            return []
        
        if response.status_code == 401:
            raise ValueError(
                "StatsBomb authentication failed. Check your username and password."
            )
        
        response.raise_for_status()
        return response.json()
    
    # =========================================================================
    # COMPETITIONS
    # =========================================================================
    
    def get_competitions(self, use_cache: bool = True) -> List[Dict]:
        """
        Get all competition/season combinations the user has access to.
        
        Returns:
            List of {competition_id, season_id, competition_name, 
                     season_name, country_name, ...}
        """
        cache_file = Config.CACHE_DIR / "sb_competitions.json"
        
        if use_cache and cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 86400:
                with open(cache_file) as f:
                    return json.load(f)
        
        v = self.versions["competitions"]
        url = f"{self.base_url}/api/{v}/competitions"
        data = self._request(url)
        
        Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)
        
        return data
    
    def find_competition(
        self,
        competition_name: str = None,
        country_name: str = None,
        season_name: str = None
    ) -> List[Dict]:
        """
        Search available competitions by name/country/season.
        
        Args:
            competition_name: e.g. "Premier League"
            country_name: e.g. "England"
            season_name: e.g. "2024/2025"
            
        Returns:
            Matching competition-season entries
        """
        comps = self.get_competitions()
        results = []
        for c in comps:
            if competition_name and competition_name.lower() not in c.get("competition_name", "").lower():
                continue
            if country_name and country_name.lower() not in c.get("country_name", "").lower():
                continue
            if season_name and season_name not in c.get("season_name", ""):
                continue
            results.append(c)
        return results
    
    # =========================================================================
    # MATCHES
    # =========================================================================
    
    def get_matches(
        self,
        competition_id: int,
        season_id: int,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Get all matches for a competition/season.
        
        Returns:
            List of match objects with match_id, home_team, away_team,
            home_score, away_score, match_date, managers, etc.
        """
        cache_file = Config.CACHE_DIR / f"sb_matches_{competition_id}_{season_id}.json"
        
        if use_cache and cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 3600:
                with open(cache_file) as f:
                    return json.load(f)
        
        v = self.versions["matches"]
        url = f"{self.base_url}/api/{v}/competitions/{competition_id}/seasons/{season_id}/matches"
        data = self._request(url)
        
        Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)
        
        return data
    
    def get_matches_for_team(
        self,
        competition_id: int,
        season_id: int,
        team_name: str
    ) -> Tuple[List[Dict], Dict]:
        """
        Get all matches for a specific team in a competition/season.
        
        Args:
            competition_id: StatsBomb competition ID
            season_id: StatsBomb season ID
            team_name: Team name to filter by (partial match)
            
        Returns:
            Tuple of (matching_matches, team_info_dict)
        """
        all_matches = self.get_matches(competition_id, season_id)
        
        if not all_matches:
            print(f"      ⚠ No matches returned for comp={competition_id}, season={season_id}")
            print(f"        Check your licensed competitions with get_competitions()")
            return [], {}
        
        team_matches = []
        team_info = {}
        
        # Collect all team names for diagnostic if no match found
        all_team_names = set()
        
        for match in all_matches:
            home = match.get("home_team", {})
            away = match.get("away_team", {})
            
            # StatsBomb uses multiple possible key names depending on API version
            home_name = (
                home.get("home_team_name") or 
                home.get("name") or 
                home.get("team_name") or 
                ""
            )
            away_name = (
                away.get("away_team_name") or 
                away.get("name") or 
                away.get("team_name") or 
                ""
            )
            
            home_id = home.get("home_team_id") or home.get("id") or home.get("team_id")
            away_id = away.get("away_team_id") or away.get("id") or away.get("team_id")
            
            all_team_names.add(home_name)
            all_team_names.add(away_name)
            
            if team_name.lower() in home_name.lower():
                team_matches.append(match)
                if not team_info:
                    team_info = {"id": home_id, "name": home_name}
            elif team_name.lower() in away_name.lower():
                team_matches.append(match)
                if not team_info:
                    team_info = {"id": away_id, "name": away_name}
        
        if not team_matches:
            print(f"\n      ⚠ No matches found for '{team_name}'")
            print(f"        {len(all_matches)} matches returned, teams found:")
            for name in sorted(all_team_names):
                if name:
                    print(f"          • {name}")
            
            # Dump first match structure for debugging
            if all_matches:
                sample = all_matches[0]
                print(f"\n        Sample match keys: {list(sample.keys())}")
                print(f"        home_team structure: {sample.get('home_team', {})}")
                print(f"        away_team structure: {sample.get('away_team', {})}")
        
        return team_matches, team_info
    
    # =========================================================================
    # LINEUPS
    # =========================================================================
    
    def get_lineups(self, match_id: int) -> List[Dict]:
        """
        Get team lineups for a match.
        
        Returns:
            List of team lineup objects with team_id, team_name, lineup array,
            formations array, events array.
        """
        v = self.versions["lineups"]
        url = f"{self.alt_url}/api/{v}/lineups/{match_id}"
        return self._request(url)
    
    # =========================================================================
    # EVENTS
    # =========================================================================
    
    def get_events(self, match_id: int) -> List[Dict]:
        """
        Get all events for a match (passes, shots, tackles, etc.).
        
        Returns rich event-level data including xG, OBV, locations,
        freeze frames, and more.
        """
        v = self.versions["events"]
        url = f"{self.base_url}/api/{v}/events/{match_id}"
        return self._request(url)
    
    # =========================================================================
    # 360 FRAMES
    # =========================================================================
    
    def get_360_frames(self, match_id: int) -> List[Dict]:
        """
        Get 360 freeze frame data for a match.
        
        Returns positional data for visible players at each event.
        """
        v = self.versions["360_frames"]
        url = f"{self.base_url}/api/{v}/360-frames/{match_id}"
        return self._request(url)
    
    # =========================================================================
    # PLAYER MATCH STATS
    # =========================================================================
    
    def get_player_match_stats(self, match_id: int) -> List[Dict]:
        """
        Get pre-computed player statistics for a match.
        
        Returns per-player stats including: goals, assists, xG, xA,
        OBV, passes, tackles, interceptions, pressures, dribbles,
        key passes, and many more (~100+ metrics per player).
        """
        v = self.versions["player_match_stats"]
        url = f"{self.base_url}/api/{v}/matches/{match_id}/player-stats"
        return self._request(url)
    
    # =========================================================================
    # PLAYER SEASON STATS
    # =========================================================================
    
    def get_player_season_stats(
        self,
        competition_id: int,
        season_id: int,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Get pre-computed per-90 player statistics for a full season.
        
        This is the primary endpoint for building squad profiles.
        Returns ~100+ per-90 metrics per player including:
        - Goals, assists, xG, xA, OBV
        - Passes, passing ratio, key passes
        - Tackles, interceptions, pressures
        - Dribbles, carries, progressive actions
        - Aerial duels, clearances, etc.
        """
        cache_file = Config.CACHE_DIR / f"sb_player_season_{competition_id}_{season_id}.json"
        
        if use_cache and cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 3600:
                with open(cache_file) as f:
                    return json.load(f)
        
        v = self.versions["player_season_stats"]
        url = f"{self.base_url}/api/{v}/competitions/{competition_id}/seasons/{season_id}/player-stats"
        data = self._request(url)
        
        Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)
        
        return data
    
    # =========================================================================
    # TEAM MATCH STATS
    # =========================================================================
    
    def get_team_match_stats(self, match_id: int) -> List[Dict]:
        """
        Get pre-computed team statistics for a match.
        
        Returns rich team-level metrics including:
        - Possession, passing ratio, PPDA
        - xG, xG conceded, shots, shot distance
        - OBV (overall and by action type)
        - Pressing, counterpressing, defensive distance
        - Deep completions, deep progressions
        - Set piece breakdowns (corners, free kicks, throw-ins)
        """
        v = self.versions["team_match_stats"]
        url = f"{self.alt_url}/api/{v}/matches/{match_id}/team-stats"
        return self._request(url)
    
    # =========================================================================
    # PLAYER MAPPING
    # =========================================================================
    
    def get_player_mapping(
        self,
        competition_id: int = None,
        season_id: int = None,
        offline_player_id: int = None,
        all_account_data: bool = False
    ) -> List[Dict]:
        """
        Get player ID mapping across StatsBomb systems.
        
        Useful for cross-referencing players across different data sources.
        """
        v = self.versions["player_mapping"]
        params = []
        
        if competition_id:
            params.append(f"competition-id={competition_id}")
        if season_id:
            params.append(f"season-id={season_id}")
        if offline_player_id:
            params.append(f"offline-player-id={offline_player_id}")
        if all_account_data:
            params.append("all-account-data=true")
        
        param_str = "&".join(params)
        url = f"{self.alt_url}/api/{v}/player-mapping?{param_str}"
        return self._request(url)
    
    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================
    
    def diagnose(self):
        """
        Print a diagnostic summary of all competitions/seasons you have access to,
        organised by country and competition. Run this first to find the right IDs.
        """
        comps = self.get_competitions()
        
        print("=" * 70)
        print("STATSBOMB ACCOUNT DIAGNOSTICS")
        print("=" * 70)
        print(f"\nTotal competition-season combinations: {len(comps)}\n")
        
        # Group by country → competition → seasons
        from collections import defaultdict
        grouped = defaultdict(lambda: defaultdict(list))
        
        for c in comps:
            country = c.get("country_name", "Unknown")
            comp_name = c.get("competition_name", "Unknown")
            grouped[country][comp_name].append(c)
        
        for country in sorted(grouped.keys()):
            print(f"\n🏴 {country}")
            for comp_name in sorted(grouped[country].keys()):
                seasons = grouped[country][comp_name]
                seasons.sort(key=lambda x: x.get("season_name", ""), reverse=True)
                
                print(f"   📋 {comp_name}")
                for s in seasons:
                    print(f"      competition_id={s['competition_id']}, "
                          f"season_id={s['season_id']}, "
                          f"season={s.get('season_name', '?')}")
        
        print("\n" + "=" * 70)
        print("Use these IDs in fetch_scenario() or run_full_analysis_statsbomb()")
        print("=" * 70)
        
        return comps

    # =========================================================================
    # CONVENIENCE: FETCH SCENARIO
    # =========================================================================
    
    def fetch_scenario(
        self,
        competition_id: int,
        season_id: int,
        team_name: str,
        max_matches: int = 50
    ) -> Dict:
        """
        Fetch all data for an MTFI analysis scenario using StatsBomb data.
        
        This mirrors SportsmonksClient.fetch_scenario() but uses StatsBomb APIs.
        It fetches team match stats, player season stats, lineups, and match data
        to build a complete picture of a team's tactical profile and squad.
        
        Args:
            competition_id: StatsBomb competition ID (e.g., 2 for Premier League)
            season_id: StatsBomb season ID (e.g., 282 for 2024/25)
            team_name: Team name to analyse (e.g., "Tottenham")
            max_matches: Maximum matches to fetch detailed stats for
            
        Returns:
            Dictionary with matches, team_info, team_match_stats,
            player_season_stats, lineups, manager_info
        """
        print("\n" + "=" * 50)
        print("AEGIS DATA FETCH (StatsBomb)")
        print("=" * 50)
        
        # 1. Get matches for team
        print(f"\n[1/5] Finding matches for: {team_name}")
        matches, team_info = self.get_matches_for_team(
            competition_id, season_id, team_name
        )
        if not matches:
            # Give a helpful error - if matches were returned but team not found,
            # the diagnostic was already printed by get_matches_for_team
            raise ValueError(
                f"No matches found for '{team_name}' in competition {competition_id}, "
                f"season {season_id}.\n"
                f"Run sb.diagnose() to see your available competitions and correct IDs.\n"
                f"Or try: sb.get_matches({competition_id}, {season_id}) to inspect raw match data."
            )
        
        # Filter to available matches only
        available = [m for m in matches if m.get("match_status") == "available"]
        if not available:
            available = matches  # fall back to all
        available = available[:max_matches]
        
        print(f"      ✓ {team_info.get('name', team_name)}: {len(available)} matches")
        
        # 2. Extract manager info from matches
        print(f"\n[2/5] Extracting manager info")
        manager_info = self._extract_manager_from_matches(available, team_info)
        print(f"      ✓ Manager: {manager_info.get('name', 'Unknown')}")
        
        # 3. Fetch team match stats for all matches
        print(f"\n[3/5] Fetching team match stats ({len(available)} matches)")
        team_match_stats = []
        for i, match in enumerate(available):
            mid = match.get("match_id")
            try:
                stats = self.get_team_match_stats(mid)
                if stats:
                    team_match_stats.extend(stats)
                if (i + 1) % 10 == 0:
                    print(f"      ... {i + 1}/{len(available)}")
            except Exception as e:
                print(f"      ⚠ Match {mid}: {e}")
        print(f"      ✓ {len(team_match_stats)} team-match stat records")
        
        # 4. Fetch player season stats
        print(f"\n[4/5] Fetching player season stats")
        player_season_stats = self.get_player_season_stats(competition_id, season_id)
        print(f"      ✓ {len(player_season_stats)} players in season")
        
        # 5. Fetch lineups for formation analysis (sample of matches)
        print(f"\n[5/5] Fetching lineups for formation analysis")
        lineups = []
        sample = available[:min(20, len(available))]
        for match in sample:
            mid = match.get("match_id")
            try:
                lineup_data = self.get_lineups(mid)
                lineups.append({"match_id": mid, "lineups": lineup_data})
            except Exception:
                pass
        print(f"      ✓ {len(lineups)} match lineups")
        
        # Save to files
        self._save_scenario_data(
            matches=available,
            team_info=team_info,
            team_match_stats=team_match_stats,
            player_season_stats=player_season_stats,
            lineups=lineups,
            manager_info=manager_info,
            competition_id=competition_id,
            season_id=season_id
        )
        
        print("\n" + "=" * 50)
        print("✓ DATA FETCH COMPLETE (StatsBomb)")
        print("=" * 50)
        
        return {
            "matches": available,
            "team_info": team_info,
            "team_match_stats": team_match_stats,
            "player_season_stats": player_season_stats,
            "lineups": lineups,
            "manager_info": manager_info,
            "competition_id": competition_id,
            "season_id": season_id,
        }
    
    def _extract_manager_from_matches(
        self,
        matches: List[Dict],
        team_info: Dict
    ) -> Dict:
        """Extract manager info from match data."""
        team_id = team_info.get("id")
        team_name = team_info.get("name", "")
        
        for match in matches:
            home = match.get("home_team", {})
            away = match.get("away_team", {})
            
            home_id = home.get("home_team_id") or home.get("id") or home.get("team_id")
            away_id = away.get("away_team_id") or away.get("id") or away.get("team_id")
            
            # Also match by name as fallback
            home_name = home.get("home_team_name") or home.get("name") or ""
            away_name = away.get("away_team_name") or away.get("name") or ""
            
            is_home = (home_id == team_id) or (team_name and team_name.lower() in home_name.lower())
            is_away = (away_id == team_id) or (team_name and team_name.lower() in away_name.lower())
            
            manager_data = None
            if is_home:
                # Manager lives INSIDE the team object at .managers (not top-level)
                manager_data = (
                    home.get("managers") or 
                    match.get("home_team_manager") or
                    match.get("home_managers")
                )
            elif is_away:
                manager_data = (
                    away.get("managers") or
                    match.get("away_team_manager") or
                    match.get("away_managers")
                )
            
            if manager_data:
                if isinstance(manager_data, list):
                    mgr = manager_data[0] if manager_data else None
                elif isinstance(manager_data, dict):
                    mgr = manager_data
                else:
                    mgr = None
                
                if mgr:
                    country = mgr.get("country", {})
                    country_name = country.get("name", "Unknown") if isinstance(country, dict) else str(country)
                    return {
                        "id": mgr.get("id"),
                        "name": mgr.get("name") or mgr.get("nickname") or "Unknown",
                        "nickname": mgr.get("nickname"),
                        "dob": mgr.get("dob"),
                        "country": country_name
                    }
        
        return {"name": "Unknown", "id": None}
    
    def _save_scenario_data(
        self,
        matches,
        team_info,
        team_match_stats,
        player_season_stats,
        lineups,
        manager_info,
        competition_id,
        season_id
    ):
        """Save fetched StatsBomb data to files."""
        Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(Config.DATA_DIR / "sb_matches.json", "w") as f:
            json.dump(matches, f, indent=2)
        
        with open(Config.DATA_DIR / "sb_team_info.json", "w") as f:
            json.dump(team_info, f, indent=2)
        
        with open(Config.DATA_DIR / "sb_team_match_stats.json", "w") as f:
            json.dump(team_match_stats, f, indent=2)
        
        with open(Config.DATA_DIR / "sb_player_season_stats.json", "w") as f:
            json.dump(player_season_stats, f, indent=2)
        
        with open(Config.DATA_DIR / "sb_lineups.json", "w") as f:
            json.dump(lineups, f, indent=2)
        
        with open(Config.DATA_DIR / "sb_manager_info.json", "w") as f:
            json.dump(manager_info, f, indent=2)
        
        metadata = {
            "data_source": "statsbomb",
            "fetch_type": "statsbomb_full",
            "competition_id": competition_id,
            "season_id": season_id,
            "timestamp": datetime.now().isoformat()
        }
        with open(Config.DATA_DIR / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n      Files saved to: {Config.DATA_DIR}")
