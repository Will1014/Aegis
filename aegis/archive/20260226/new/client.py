"""
Aegis Sportsmonks Client
========================
API client for Sportsmonks Football API v3.

Statistics Strategy:
- DEFAULT: Use dedicated /teams/{id}?include=statistics and /players/{id}?include=statistics.details
  These return pre-aggregated season stats with home/away splits and averages.
- OPTIONAL: Use /fixtures/{id}?include=statistics for match-level granularity when needed.
"""

import os
import time
import json
import requests
from typing import Optional, Dict, List, Any
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
