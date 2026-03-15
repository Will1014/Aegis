"""
Aegis Analysis Engine
=====================
Core tactical fit modelling using K-Means clustering and weighted distance scoring.

Two-phase approach:
1. Manager DNA Training: Cluster managers by tactical profile (run once)
2. Squad Fit Scoring: Score players against position-specific ideals, weighted by archetype
"""

import json
import csv
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np

from .config import Config


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PlayerFit:
    """Player fit score result."""
    name: str
    position: str
    detailed_position: str
    position_group: str
    age: int
    fit_score: float
    classification: str
    stats: Dict


@dataclass 
class ManagerProfile:
    """Manager tactical profile."""
    coach_id: int
    coach_name: str
    team_id: int
    team_name: str
    cluster: int
    cluster_name: str
    features: Dict[str, float]


# =============================================================================
# CONSTANTS
# =============================================================================

# Features for Manager DNA clustering (from team statistics)
MANAGER_DNA_FEATURES = [
    "possession",
    "pass_accuracy",
    "shots",
    "goals_scored",
    "goals_conceded",
    "tackles",
    "interceptions",
    "clean_sheet_pct",
    "win_rate"
]

# ── Gary's 8-Pillar StatsBomb DNA Features (for clustering) ──
# Each feature maps to one of the 8 tactical DNA pillars.
# Computed from StatsBomb team_match_stats aggregates.
STATSBOMB_DNA_FEATURES = [
    "pressing_intensity",       # Pillar 4: Press (PPDA inverted, poss-adjusted)
    "counterpress_rate",        # Pillar 4: Counterpress tendency
    "build_up_patience",        # Pillar 2: Passes per possession / sequence length
    "directness",               # Pillar 2: Forward progression speed
    "chance_quality",           # Pillar 3: xG per shot
    "defensive_line_height",    # Pillar 5: Average defensive action distance
    "width_usage",              # Pillar 7: Cross + wide action share
    "set_piece_emphasis",       # Pillar 8: Set-piece xG share
    "transition_threat",        # Pillar 6: Counter-attacking shots
    "defensive_solidity",       # Pillar 5: xGA quality (inverted)
]

# The 8 Tactical DNA Pillars (Gary's framework)
# Each pillar has: display name, sub-metrics, and which DNA feature(s) drive it
DNA_PILLARS = {
    "shape_occupation": {
        "name": "Shape & Occupation",
        "number": 1,
        "metrics": ["defensive_line_height", "possession"],
        "description": "Base shape (IP/OOP), rest defence, role clarity",
    },
    "build_up": {
        "name": "Build-up vs Direct",
        "number": 2,
        "metrics": ["build_up_patience", "directness", "pass_accuracy"],
        "description": "Progression routes, pass/carry mix, tempo & patience",
    },
    "chance_creation": {
        "name": "Chance Creation",
        "number": 3,
        "metrics": ["chance_quality", "np_xg_pg", "deep_completions_pg"],
        "description": "Shot profile, assist types, zone entry patterns",
    },
    "press_counterpress": {
        "name": "Press & Counterpress",
        "number": 4,
        "metrics": ["pressing_intensity", "counterpress_rate", "high_regain_rate"],
        "description": "Pressure volume, counterpress, high regains",
    },
    "block_line_height": {
        "name": "Block & Line Height",
        "number": 5,
        "metrics": ["defensive_line_height", "defensive_solidity", "xga_pg"],
        "description": "Def action height, box protection, balls in behind",
    },
    "transitions": {
        "name": "Transitions",
        "number": 6,
        "metrics": ["transition_threat", "high_press_shots_pg", "turnover_shots_conceded_pg"],
        "description": "Time-to-shot, turnover risk, counter-attack",
    },
    "width_overloads": {
        "name": "Width & Overloads",
        "number": 7,
        "metrics": ["width_usage", "cross_rate", "box_cross_ratio"],
        "description": "Wing usage, switches, cross/cutback mix",
    },
    "set_pieces": {
        "name": "Set Pieces",
        "number": 8,
        "metrics": ["set_piece_emphasis", "sp_xg_pg", "sp_xg_against_pg"],
        "description": "Set-piece xG share, routines, second-phase control",
    },
}



# ─────────────────────────────────────────────────────────────────────
# PILLAR → PLAYER DEMAND TRANSLATION MATRIX
# ─────────────────────────────────────────────────────────────────────
# Maps each of Gary's 8 pillars to the player features it demands,
# broken down by position group. Values are demand strengths (0.0–1.0).
#
# Read as: "If a manager scores HIGH on this pillar, how important
# is each player feature for each position?"
#
# The manager's actual pillar score (0-100) scales these demands,
# producing unique weight profiles for every manager × position.

PILLAR_PLAYER_DEMANDS = {
    # ── Pillar 1: Shape & Occupation ──
    # High = structured, disciplined positional play
    "shape_occupation": {
        "GK":  {"pass_accuracy": 0.5},
        "DEF": {"interceptions_per90": 0.5, "pass_accuracy": 0.6, "tackles_per90": 0.3},
        "MID": {"pass_accuracy": 0.7, "key_passes_per90": 0.4, "interceptions_per90": 0.3},
        "ATT": {"shots_per90": 0.3, "goals_per90": 0.3},
    },
    # ── Pillar 2: Build-up (patience) ──
    # High = patient possession build from back
    "build_up": {
        "GK":  {"pass_accuracy": 0.8},
        "DEF": {"pass_accuracy": 0.8, "dribbles_per90": 0.4, "key_passes_per90": 0.3},
        "MID": {"pass_accuracy": 0.7, "key_passes_per90": 0.6, "dribbles_per90": 0.4},
        "ATT": {"assists_per90": 0.4, "key_passes_per90": 0.5, "pass_accuracy": 0.3},
    },
    # ── Pillar 3: Chance Creation ──
    # High = creates high-quality chances (xG per shot)
    "chance_creation": {
        "GK":  {},
        "DEF": {"key_passes_per90": 0.3, "assists_per90": 0.3},
        "MID": {"key_passes_per90": 0.8, "assists_per90": 0.6, "dribbles_per90": 0.4, "shots_per90": 0.3},
        "ATT": {"goals_per90": 0.7, "shots_per90": 0.8, "key_passes_per90": 0.3, "dribbles_per90": 0.4},
    },
    # ── Pillar 4: Press & Counterpress ──
    # High = intense pressing, aggressive counterpressing
    "press_counterpress": {
        "GK":  {"interceptions_per90": 0.3},
        "DEF": {"tackles_per90": 0.7, "interceptions_per90": 0.7},
        "MID": {"tackles_per90": 0.8, "interceptions_per90": 0.6},
        "ATT": {"tackles_per90": 0.5, "interceptions_per90": 0.3},
    },
    # ── Pillar 5: Block & Line Height ──
    # High = high line with proactive defending
    "block_line_height": {
        "GK":  {"pass_accuracy": 0.3},
        "DEF": {"tackles_per90": 0.6, "interceptions_per90": 0.7, "pass_accuracy": 0.3},
        "MID": {"tackles_per90": 0.5, "interceptions_per90": 0.5},
        "ATT": {"tackles_per90": 0.2},
    },
    # ── Pillar 6: Transitions ──
    # High = dangerous in transition / counter-attacking
    "transitions": {
        "GK":  {"pass_accuracy": 0.3},
        "DEF": {"dribbles_per90": 0.3, "pass_accuracy": 0.4},
        "MID": {"dribbles_per90": 0.5, "key_passes_per90": 0.5, "shots_per90": 0.3},
        "ATT": {"goals_per90": 0.6, "dribbles_per90": 0.7, "shots_per90": 0.5},
    },
    # ── Pillar 7: Width & Overloads ──
    # High = heavy use of width, crossing, overlaps
    "width_overloads": {
        "GK":  {},
        "DEF": {"assists_per90": 0.5, "dribbles_per90": 0.5, "key_passes_per90": 0.4},
        "MID": {"key_passes_per90": 0.5, "assists_per90": 0.4, "dribbles_per90": 0.3},
        "ATT": {"dribbles_per90": 0.7, "assists_per90": 0.5, "key_passes_per90": 0.4},
    },
    # ── Pillar 8: Set Pieces ──
    # High = relies on set pieces for xG share
    "set_pieces": {
        "GK":  {},
        "DEF": {"goals_per90": 0.6},
        "MID": {"assists_per90": 0.5, "key_passes_per90": 0.5},
        "ATT": {"goals_per90": 0.6, "shots_per90": 0.4},
    },
}

# Archetype weights for player fit scoring (unchanged — works with both feature sets)
# The archetype names are assigned by _name_clusters() based on feature z-scores

# Features for Player Fit scoring (per-90 metrics)
PLAYER_FIT_FEATURES = [
    "goals_per90",
    "assists_per90",
    "tackles_per90",
    "interceptions_per90",
    "pass_accuracy",
    "key_passes_per90",
    "dribbles_per90",
    "shots_per90"
]

# Position group mappings
POSITION_GROUPS = {
    "GK": ["Goalkeeper"],
    "DEF": ["Defender", "Centre-Back", "Right-Back", "Left-Back", "Wing-Back"],
    "MID": ["Midfielder", "Defensive Midfield", "Central Midfield", "Attacking Midfield"],
    "ATT": ["Attacker", "Winger", "Left Winger", "Right Winger", "Forward", "Centre-Forward", "Striker"]
}

# Classification thresholds
FIT_THRESHOLDS = {
    "key_enabler": 75,
    "good_fit": 60,
    "system_dependent": 45
}

# Default managers for training set (Premier League 2024/25)
# Updated January 2026 with current appointments
DEFAULT_MANAGERS = [
    {"name": "Pep Guardiola", "search": "Guardiola"},
    {"name": "Mikel Arteta", "search": "Arteta"},
    {"name": "Arne Slot", "search": "Slot"},
    {"name": "Enzo Maresca", "search": "Maresca"},
    {"name": "Ruben Amorim", "search": "Amorim"},  # Replaced ten Hag at Man Utd
    {"name": "Unai Emery", "search": "Emery"},
    {"name": "Thomas Frank", "search": "Thomas Frank"},
    {"name": "Marco Silva", "search": "Marco Silva"},
    {"name": "Nuno Espirito Santo", "search": "Nuno Espirito"},
    {"name": "Oliver Glasner", "search": "Glasner"},
    {"name": "Gary O'Neil", "search": "Gary O'Neil"},  # Now at Wolves
    {"name": "Andoni Iraola", "search": "Iraola"},
    {"name": "Fabian Hurzeler", "search": "Hurzeler"},  # At Brighton
    {"name": "Eddie Howe", "search": "Eddie Howe"},
    {"name": "Ange Postecoglou", "search": "Postecoglou"},
    {"name": "Sean Dyche", "search": "Dyche"},
    {"name": "Julen Lopetegui", "search": "Lopetegui"},
    {"name": "Ruud van Nistelrooy", "search": "Van Nistelrooy"},  # Leicester
    {"name": "Roberto De Zerbi", "search": "De Zerbi"},
    {"name": "Mauricio Pochettino", "search": "Pochettino"},
]

# Ideal player profiles per position group (per-90 benchmarks)
# These represent what a "good" player looks like in each position
IDEAL_PLAYER_PROFILES = {
    "GK": {
        "goals_per90": 0.0,
        "assists_per90": 0.02,
        "tackles_per90": 0.1,
        "interceptions_per90": 0.2,
        "pass_accuracy": 75.0,
        "key_passes_per90": 0.1,
        "dribbles_per90": 0.0,
        "shots_per90": 0.0
    },
    "DEF": {
        "goals_per90": 0.05,
        "assists_per90": 0.08,
        "tackles_per90": 2.5,
        "interceptions_per90": 1.8,
        "pass_accuracy": 85.0,
        "key_passes_per90": 0.5,
        "dribbles_per90": 0.4,
        "shots_per90": 0.4
    },
    "MID": {
        "goals_per90": 0.15,
        "assists_per90": 0.18,
        "tackles_per90": 2.0,
        "interceptions_per90": 1.2,
        "pass_accuracy": 87.0,
        "key_passes_per90": 1.5,
        "dribbles_per90": 1.2,
        "shots_per90": 1.5
    },
    "ATT": {
        "goals_per90": 0.45,
        "assists_per90": 0.22,
        "tackles_per90": 0.8,
        "interceptions_per90": 0.4,
        "pass_accuracy": 78.0,
        "key_passes_per90": 1.2,
        "dribbles_per90": 2.0,
        "shots_per90": 3.0
    }
}

# Feature weights for each manager archetype
# Higher values = more important for that style
ARCHETYPE_WEIGHTS = {
    "Possession-Based": {
        "pass_accuracy": 1.5,
        "key_passes_per90": 1.3,
        "dribbles_per90": 1.2,
        "tackles_per90": 0.8,
        "interceptions_per90": 0.8
    },
    "High-Press": {
        "tackles_per90": 1.5,
        "interceptions_per90": 1.5,
        "pass_accuracy": 1.0,
        "shots_per90": 1.2
    },
    "Counter-Attack": {
        "dribbles_per90": 1.4,
        "shots_per90": 1.3,
        "goals_per90": 1.3,
        "pass_accuracy": 0.9
    },
    "Defensive": {
        "tackles_per90": 1.4,
        "interceptions_per90": 1.4,
        "pass_accuracy": 1.1,
        "goals_per90": 0.8
    },
    "Attacking": {
        "goals_per90": 1.4,
        "assists_per90": 1.3,
        "shots_per90": 1.3,
        "key_passes_per90": 1.2
    },
    "Balanced": {}  # No adjustments
}


# =============================================================================
# MANAGER DNA TRAINER
# =============================================================================

class ManagerDNATrainer:
    """
    Train Manager DNA clustering model.
    
    This class fetches tactical data for multiple managers, extracts features,
    and clusters them into tactical archetypes using K-Means.
    
    Usage:
        from aegis import ManagerDNATrainer
        
        trainer = ManagerDNATrainer()
        trainer.fetch_manager_data()  # Pass fetch_all=True to fetch from database
        trainer.extract_features()
        trainer.fit()
        trainer.save()
    """
    
    def __init__(
        self, 
        client=None,
        training_dir: Optional[Path] = None,
        season_id: int = 23614
    ):
        """
        Initialize trainer.
        
        Args:
            client: SportsmonksClient instance (created if not provided)
            training_dir: Directory for saving model files
            season_id: Season to analyse (default: 2024/25)
        """
        self.client = client
        self.training_dir = Path(training_dir) if training_dir else Config.PROCESSED_DIR / "training"
        self.season_id = season_id
        
        # Data containers
        self.manager_tenures = []
        self.manager_features = []
        self.feature_names = None  # Set by fetch_manager_data_statsbomb; defaults to MANAGER_DNA_FEATURES
        
        # Model components
        self.kmeans = None
        self.scaler = None
        self.pca = None
        self.n_clusters = None
        self.cluster_names = {}
        
        # Results
        self.df_managers = None
        self.df_centroids = None
    

    def fetch_all_coaches_from_database(
        self,
        league_ids = None,
        max_coaches: int = 500,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Fetch all coaches from the Sportsmonks database.
        
        Args:
            league_ids: "top5", None (all), or list of IDs
            max_coaches: Maximum coaches to fetch
            verbose: Print progress
        """
        from .client import SportsmonksClient
        import re
        
        if self.client is None:
            self.client = SportsmonksClient()
        
        # Handle league_ids
        if league_ids == "top5":
            league_ids = [8, 564, 82, 384, 301]
        elif league_ids is None:
            league_ids = self._fetch_all_league_ids(verbose)
        
        if verbose:
            print("=" * 60)
            print("FETCHING ALL COACHES FROM DATABASE")
            print("=" * 60)
            print(f"Leagues: {len(league_ids)}")
        
        all_coaches = []
        
        for league_id in league_ids:
            if len(all_coaches) >= max_coaches:
                break
            
            try:
                # Get league name
                league_result = self.client._request(f"/football/leagues/{league_id}")
                league_name = league_result.get("data", {}).get("name", f"League {league_id}")
                
                # Get seasons for this league
                seasons_result = self.client._request(
                    "/football/seasons",
                    params={"filters": f"seasonLeagues:{league_id}"}
                )
                seasons = seasons_result.get("data", [])
                
                if not seasons:
                    if verbose:
                        print(f"  âœ— {league_name}: No seasons")
                    continue
                
                # Find most recent season by year in name
                def get_year(s):
                    years = re.findall(r'20\d{2}', s.get("name", ""))
                    return max(int(y) for y in years) if years else 0
                
                seasons.sort(key=get_year, reverse=True)
                season_id = seasons[0].get("id")
                season_name = seasons[0].get("name", "")
                
                if verbose:
                    print(f"\n  {league_name} ({season_name}):")
                
                # Get teams for this season
                teams_result = self.client._request(
                    f"/football/teams/seasons/{season_id}"
                )
                teams = teams_result.get("data", [])
                
                if not teams:
                    if verbose:
                        print(f"    No teams found")
                    continue
                
                # Fetch coaches for each team
                for team in teams:
                    if len(all_coaches) >= max_coaches:
                        break
                    
                    team_id = team.get("id")
                    team_name = team.get("name", "Unknown")
                    
                    # Fetch team with coaches
                    team_detail = self.client._request(
                        f"/football/teams/{team_id}",
                        include=["coaches"]
                    )
                    coaches = team_detail.get("data", {}).get("coaches", [])
                    
                    # Find active coach
                    for coach in coaches:
                        if coach.get("active") == True:
                            coach_id = coach.get("coach_id") or coach.get("id")
                            if coach_id:
                                all_coaches.append({
                                    "coach_id": coach_id,
                                    "team_id": team_id,
                                    "team_name": team_name,
                                    "league": league_name
                                })
                                if verbose:
                                    print(f"    âœ“ {team_name}")
                                break
                
            except Exception as e:
                if verbose:
                    print(f"  âœ— League {league_id}: {e}")
        
        if verbose:
            print(f"\n  Total: {len(all_coaches)} coaches")
        
        return all_coaches[:max_coaches]
    
    def _fetch_all_league_ids(self, verbose: bool = True) -> List[int]:
        """
        Fetch all available league IDs from the database.
        
        Returns:
            List of league IDs
        """
        if verbose:
            print("Fetching all leagues from database...")
        
        try:
            # Fetch all leagues
            leagues = self.client._paginate(
                "/football/leagues",
                params={"per_page": 100}
            )
            
            # Filter to only active leagues with current seasons
            league_ids = []
            for league in leagues:
                league_id = league.get("id")
                if league_id:
                    league_ids.append(league_id)
            
            if verbose:
                print(f"  Found {len(league_ids)} leagues")
            
            return league_ids
            
        except Exception as e:
            if verbose:
                print(f"  Error fetching leagues: {e}")
                print("  Falling back to top 5 European leagues")
            return [8, 564, 82, 384, 301]
    
    def fetch_manager_data(
        self, 
        managers: List[Dict] = None,
        fetch_all: bool = False,
        league_ids = None,
        verbose: bool = True
    ) -> "ManagerDNATrainer":
        """
        Fetch tactical data for managers.
        
        Args:
            managers: List of manager dicts with 'name' and 'search' keys.
                     If None and fetch_all=False, uses DEFAULT_MANAGERS.
                     If None and fetch_all=True, fetches from database.
            fetch_all: If True, fetch all coaches from database
            league_ids: League filter (only used if fetch_all=True). Options:
                       - None: ALL leagues in database
                       - "top5": Top 5 European leagues (PL, La Liga, etc.)
                       - [8, 564, ...]: Specific league IDs
            verbose: Print progress
            
        Returns:
            self for chaining
        """
        from .client import SportsmonksClient
        
        if self.client is None:
            self.client = SportsmonksClient()
        
        # Determine which managers to fetch
        if managers is None and fetch_all:
            # Fetch all from database
            db_coaches = self.fetch_all_coaches_from_database(
                league_ids=league_ids,
                verbose=verbose
            )
            self._process_database_coaches(db_coaches, verbose)
            return self
        
        managers = managers or DEFAULT_MANAGERS
        
        if verbose:
            print("=" * 60)
            print("FETCHING MANAGER DATA")
            print("=" * 60)
        
        self.manager_tenures = []
        errors = []
        
        for mgr in managers:
            try:
                # Search for coach
                results = self.client.search_coaches(mgr["search"])
                if not results:
                    if verbose:
                        print(f"  âœ— {mgr['name']}: Not found")
                    errors.append(mgr['name'])
                    continue
                
                # Get coach details
                coach = self.client.get_coach(results[0]["id"])
                if not coach:
                    if verbose:
                        print(f"  âœ— {mgr['name']}: No data")
                    errors.append(mgr['name'])
                    continue
                
                # Get team history
                teams = coach.get("teams", [])
                
                # Sort by start date (most recent first) to find correct current team
                def parse_date(team):
                    start = team.get("start")
                    return start[:10] if start and len(start) >= 10 else "0000-00-00"
                
                sorted_teams = sorted(teams, key=parse_date, reverse=True)
                
                # Find current team (end is None) from sorted list
                current_team = None
                for team in sorted_teams:
                    if team.get("end") is None:
                        current_team = team
                        break
                if not current_team and sorted_teams:
                    current_team = sorted_teams[0]
                
                if current_team:
                    team_id = current_team.get("team_id")
                    
                    # Fetch team name separately
                    team_name = "Unknown"
                    if team_id:
                        try:
                            team_data = self.client.get_team(team_id)
                            team_name = team_data.get("name", "Unknown")
                        except:
                            pass
                    
                    self.manager_tenures.append({
                        "coach_id": coach.get("id"),
                        "coach_name": coach.get("common_name", coach.get("name")),
                        "team_id": team_id,
                        "team_name": team_name,
                        "active": current_team.get("end") is None
                    })
                    
                    if verbose:
                        print(f"  âœ“ {mgr['name']}: {team_name}")
                else:
                    if verbose:
                        print(f"  âœ— {mgr['name']}: No team found")
                    errors.append(mgr['name'])
                    
            except Exception as e:
                if verbose:
                    print(f"  âœ— {mgr['name']}: Error - {e}")
                errors.append(mgr['name'])
        
        if verbose:
            print("-" * 60)
            print(f"Found: {len(self.manager_tenures)} | Errors: {len(errors)}")
        
        return self
    
    def _process_database_coaches(self, db_coaches: List[Dict], verbose: bool = True):
        """Process coaches fetched from database."""
        if verbose:
            print("\n" + "=" * 60)
            print("FETCHING COACH DETAILS")
            print("=" * 60)
        
        self.manager_tenures = []
        
        for coach_info in db_coaches:
            try:
                coach = self.client.get_coach(coach_info["coach_id"])
                if coach:
                    self.manager_tenures.append({
                        "coach_id": coach_info["coach_id"],
                        "coach_name": coach.get("common_name", coach.get("name", "Unknown")),
                        "team_id": coach_info["team_id"],
                        "team_name": coach_info["team_name"],
                        "league": coach_info.get("league", "Unknown"),
                        "active": True
                    })
                    
                    if verbose:
                        print(f"  âœ“ {coach.get('common_name', 'Unknown')}: {coach_info['team_name']}")
                        
            except Exception as e:
                if verbose:
                    print(f"  âœ— Coach {coach_info['coach_id']}: {e}")
        
        if verbose:
            print("-" * 60)
            print(f"Processed: {len(self.manager_tenures)} coaches")
    
    def extract_features(self, verbose: bool = True) -> "ManagerDNATrainer":
        """
        Extract tactical features for each manager's team.

        Two data sources (confirmed by API diagnostic Feb 2026):
          - Team stats endpoint: provides 8/9 features directly
          - Fixtures:            provides pass_accuracy (not in team stats endpoint)

        Returns:
            self for chaining
        """
        if verbose:
            print("\n" + "=" * 60)
            print("EXTRACTING FEATURES")
            print("=" * 60)

        self.manager_features = []

        for tenure in self.manager_tenures:
            try:
                # Step 1: 8 features from team stats endpoint
                team_stats = self.client.get_team_statistics(
                    team_id=tenure["team_id"],
                    season_id=self.season_id
                )

                if not team_stats:
                    if verbose:
                        print(f"  ✗ {tenure['coach_name']}: No stats")
                    continue

                features = self._parse_team_stats(team_stats)

                if not features:
                    if verbose:
                        print(f"  ✗ {tenure['coach_name']}: Could not parse stats")
                    continue

                # Step 2: pass_accuracy from fixtures
                # Not available from the team stats endpoint.
                # Confirmed at fixture level as "successful-passes-percentage"
                # at stat["data"]["value"].
                features["pass_accuracy"] = self._fetch_pass_accuracy_from_fixtures(
                    team_id=tenure["team_id"],
                    season_id=self.season_id,
                    verbose=verbose
                )

                features["coach_id"]   = tenure["coach_id"]
                features["coach_name"] = tenure["coach_name"]
                features["team_id"]    = tenure["team_id"]
                features["team_name"]  = tenure["team_name"]
                features["league"]     = tenure.get("league", "Unknown")
                self.manager_features.append(features)

                if verbose:
                    print(
                        f"  ✓ {tenure['coach_name']}: {len(MANAGER_DNA_FEATURES)} features  "
                        f"poss={features['possession']:.1f} "
                        f"pass%={features['pass_accuracy']:.1f} "
                        f"shots={features['shots']:.1f} "
                        f"tkl={features['tackles']:.1f} "
                        f"int={features['interceptions']:.1f} "
                        f"wr={features['win_rate']:.1f}%"
                    )

            except Exception as e:
                if verbose:
                    print(f"  ✗ {tenure['coach_name']}: Error - {e}")

        if verbose:
            print("-" * 60)
            print(f"Managers with features: {len(self.manager_features)}")

        return self

    def fetch_manager_data_statsbomb(
        self,
        sb_client,
        competition_id: int,
        season_id: int,
        competition_ids: List[int] = None,
        verbose: bool = True
    ) -> "ManagerDNATrainer":
        """
        Fetch tactical data for ALL managers in a StatsBomb competition/season.
        
        This replaces the Sportsmonks fetch_manager_data() + extract_features()
        pipeline — it populates manager_features directly from StatsBomb team
        match stats, so you can call fit() immediately after.
        
        Args:
            sb_client: StatsBombClient instance
            competition_id: Primary competition (e.g. 2 for Premier League)
            season_id: Season (e.g. 317 for 2024/25)
            competition_ids: Additional competitions to include (optional).
                            Adds more managers for richer clustering.
                            e.g. [2, 3, 6] for PL + Championship + Eredivisie
            verbose: Print progress
            
        Returns:
            self for chaining — manager_features is populated ready for fit()
        """
        from collections import defaultdict
        
        all_comp_ids = competition_ids or [competition_id]
        if competition_id not in all_comp_ids:
            all_comp_ids.insert(0, competition_id)
        
        if verbose:
            print("=" * 60)
            print("FETCHING MANAGER DATA (StatsBomb)")
            print("=" * 60)
            print(f"Competitions: {all_comp_ids}")
            print(f"Season: {season_id}")
        
        self.manager_tenures = []
        self.manager_features = []
        
        for comp_id in all_comp_ids:
            if verbose:
                print(f"\n── Competition {comp_id} ──")
            
            # 1. Get all matches to extract teams and managers
            matches = sb_client.get_matches(comp_id, season_id)
            if not matches:
                if verbose:
                    print(f"  ⚠ No matches for comp={comp_id}, season={season_id}")
                continue
            
            if verbose:
                print(f"  {len(matches)} matches found")
            
            # Extract team→manager mapping and results
            team_managers = {}   # team_id → {name, manager_name, ...}
            team_results = defaultdict(lambda: {"w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "cs": 0})
            
            for match in matches:
                home = match.get("home_team", {})
                away = match.get("away_team", {})
                
                h_id = home.get("home_team_id") or home.get("id") or home.get("team_id")
                a_id = away.get("away_team_id") or away.get("id") or away.get("team_id")
                h_name = home.get("home_team_name") or home.get("name") or ""
                a_name = away.get("away_team_name") or away.get("name") or ""
                
                hs = match.get("home_score", 0) or 0
                aws = match.get("away_score", 0) or 0
                
                # Extract managers (nested inside team object at .managers)
                h_mgr = home.get("managers") or match.get("home_team_manager")
                a_mgr = away.get("managers") or match.get("away_team_manager")
                
                def _mgr_name(mgr):
                    if isinstance(mgr, list):
                        mgr = mgr[0] if mgr else None
                    if isinstance(mgr, dict):
                        return mgr.get("name") or mgr.get("nickname") or "Unknown"
                    return "Unknown"
                
                if h_id and h_id not in team_managers:
                    team_managers[h_id] = {
                        "team_name": h_name,
                        "manager_name": _mgr_name(h_mgr),
                    }
                if a_id and a_id not in team_managers:
                    team_managers[a_id] = {
                        "team_name": a_name,
                        "manager_name": _mgr_name(a_mgr),
                    }
                
                # Track results
                if h_id:
                    r = team_results[h_id]
                    r["gf"] += hs; r["ga"] += aws
                    if aws == 0: r["cs"] += 1
                    if hs > aws: r["w"] += 1
                    elif hs < aws: r["l"] += 1
                    else: r["d"] += 1
                
                if a_id:
                    r = team_results[a_id]
                    r["gf"] += aws; r["ga"] += hs
                    if hs == 0: r["cs"] += 1
                    if aws > hs: r["w"] += 1
                    elif aws < hs: r["l"] += 1
                    else: r["d"] += 1
            
            if verbose:
                print(f"  {len(team_managers)} teams with managers")
            
            # 2. Fetch team match stats for all matches
            # Aggregate per team
            team_stats_agg = defaultdict(list)  # team_id → [stat_dicts]
            
            available = [m for m in matches if m.get("match_status") == "available"]
            if not available:
                available = matches
            
            if verbose:
                print(f"  Fetching team match stats ({len(available)} matches)...")
            
            for i, match in enumerate(available):
                mid = match.get("match_id")
                try:
                    stats_list = sb_client.get_team_match_stats(mid)
                    for stat in (stats_list or []):
                        tid = stat.get("team_id")
                        if tid:
                            team_stats_agg[tid].append(stat)
                except Exception:
                    pass
                
                if verbose and (i + 1) % 50 == 0:
                    print(f"    ... {i + 1}/{len(available)}")
            
            if verbose:
                print(f"  Stats collected for {len(team_stats_agg)} teams")
            
            # 3. Build features for each team
            for team_id, info in team_managers.items():
                stats_list = team_stats_agg.get(team_id, [])
                results = team_results.get(team_id, {})
                
                if not stats_list:
                    continue
                
                n_matches = len(stats_list)
                total_matches = (results.get("w", 0) + results.get("d", 0) + results.get("l", 0)) or 1
                
                def avg_stat(key, default=0):
                    vals = [s.get(key) for s in stats_list if s.get(key) is not None]
                    return round(sum(vals) / len(vals), 2) if vals else default
                
                possession = avg_stat("team_match_possession", 50.0)
                passing_ratio = avg_stat("team_match_passing_ratio", 80.0)
                np_shots = avg_stat("team_match_np_shots", 12.0)
                ppda = avg_stat("team_match_ppda", 10.0)
                np_xg = avg_stat("team_match_np_xg", 1.5)
                np_xg_conceded = avg_stat("team_match_np_xg_conceded", 1.2)
                pressures = avg_stat("team_match_pressures", 150)
                counterpressures = avg_stat("team_match_counterpressures", 30)
                pressure_regains = avg_stat("team_match_pressure_regains", 20)
                defensive_distance = avg_stat("team_match_defensive_distance", 40)
                directness_raw = avg_stat("team_match_directness", 0.3)
                deep_completions = avg_stat("team_match_deep_completions", 5)
                deep_progressions = avg_stat("team_match_deep_progressions", 20)
                crosses_into_box = avg_stat("team_match_crosses_into_box", 5)
                box_cross_ratio = avg_stat("team_match_box_cross_ratio", 30)
                sp_xg = avg_stat("team_match_sp_xg", 0.3)
                op_xg = avg_stat("team_match_op_xg", 1.0)
                counter_shots = avg_stat("team_match_counter_attacking_shots", 1.0)
                high_press_shots = avg_stat("team_match_high_press_shots", 0.5)
                possessions_pg = avg_stat("team_match_possessions", 50)
                obv = avg_stat("team_match_obv", 0.0)
                
                goals_scored_pg = round(results["gf"] / total_matches, 2) if results["gf"] else 1.5
                goals_conceded_pg = round(results["ga"] / total_matches, 2) if results["ga"] else 1.2
                clean_sheet_pct = round(results["cs"] / total_matches * 100, 1)
                win_rate = round(results["w"] / total_matches * 100, 1)
                
                # ── 8-Pillar Feature Computation ──
                
                # P4: Pressing intensity (PPDA inverted — lower PPDA = more pressing)
                pressing_intensity = round(max(0, 30 - ppda), 1)
                
                # P4: Counterpress rate
                counterpress_rate = round(
                    counterpressures / max(pressures, 1) * 100, 1
                )
                
                # P2: Build-up patience (passes per possession proxy)
                build_up_patience = round(100 - (directness_raw * 100), 1)
                
                # P2: Directness (0–100 scale)
                directness_score = round(directness_raw * 100, 1)
                
                # P3: Chance creation quality (xG per shot)
                chance_quality = round(np_xg / max(np_shots, 1), 3)
                
                # P5: Defensive line height
                defensive_line_height = round(defensive_distance, 1)
                
                # P7: Width usage (crosses into box per game)
                width_usage = round(crosses_into_box, 1)
                
                # P8: Set-piece emphasis (SP xG as % of total xG)
                total_xg = sp_xg + op_xg
                set_piece_emphasis = round(
                    sp_xg / max(total_xg, 0.01) * 100, 1
                )
                
                # P6: Transition threat (counter-attacking shots per game)
                transition_threat = round(counter_shots, 2)
                
                # P5: Defensive solidity (inverted xGA, scaled)
                defensive_solidity = round(max(0, 3.0 - np_xg_conceded) * 33.3, 1)
                
                # Build feature dict matching STATSBOMB_DNA_FEATURES
                features = {
                    # Clustering features (STATSBOMB_DNA_FEATURES)
                    "pressing_intensity": pressing_intensity,
                    "counterpress_rate": counterpress_rate,
                    "build_up_patience": build_up_patience,
                    "directness": directness_score,
                    "chance_quality": chance_quality,
                    "defensive_line_height": defensive_line_height,
                    "width_usage": width_usage,
                    "set_piece_emphasis": set_piece_emphasis,
                    "transition_threat": transition_threat,
                    "defensive_solidity": defensive_solidity,
                    # Metadata
                    "coach_id": team_id,
                    "coach_name": info["manager_name"],
                    "team_id": team_id,
                    "team_name": info["team_name"],
                    "league": f"Competition {comp_id}",
                    # Extended metrics (not used for clustering, used for DNA profile)
                    "_possession": possession,
                    "_pass_accuracy": passing_ratio,
                    "_np_xg_pg": np_xg,
                    "_np_xg_conceded_pg": np_xg_conceded,
                    "_ppda": ppda,
                    "_np_shots_pg": np_shots,
                    "_pressures_pg": pressures,
                    "_deep_completions_pg": deep_completions,
                    "_deep_progressions_pg": deep_progressions,
                    "_sp_xg_pg": sp_xg,
                    "_counter_shots_pg": counter_shots,
                    "_high_press_shots_pg": high_press_shots,
                    "_obv_pg": obv,
                    "_goals_scored_pg": goals_scored_pg,
                    "_goals_conceded_pg": goals_conceded_pg,
                    "_clean_sheet_pct": clean_sheet_pct,
                    "_win_rate": win_rate,
                }
                
                self.manager_tenures.append({
                    "coach_id": team_id,
                    "coach_name": info["manager_name"],
                    "team_id": team_id,
                    "team_name": info["team_name"],
                })
                self.manager_features.append(features)
                
                if verbose:
                    print(f"    ✓ {info['manager_name']:25s} ({info['team_name']:25s}) "
                          f"press={pressing_intensity:.0f} patience={build_up_patience:.0f} "
                          f"xG/sh={chance_quality:.3f} def_h={defensive_line_height:.0f} "
                          f"width={width_usage:.1f} SP%={set_piece_emphasis:.0f}")
        
        # Set feature names for fit()
        self.feature_names = STATSBOMB_DNA_FEATURES
        
        if verbose:
            print("\n" + "-" * 60)
            print(f"Total managers with features: {len(self.manager_features)}")
            print(f"Feature set: 8-Pillar StatsBomb DNA ({len(STATSBOMB_DNA_FEATURES)} features)")
        
        return self

    def _fetch_pass_accuracy_from_fixtures(
        self,
        team_id: int,
        season_id: int,
        verbose: bool = False,
        default: float = 80.0
    ) -> float:
        """
        Average pass accuracy from fixture-level statistics.

        The team stats endpoint has no pass accuracy figure.
        Confirmed present in fixtures as "successful-passes-percentage"
        at stat["data"]["value"]  (participant_id filters to this team).

        Requires include=statistics.type on the fixture request.
        """
        try:
            fixtures = self.client.get_fixtures_for_tactical_analysis(
                team_id=team_id,
                season_id=season_id
            )
        except Exception as e:
            if verbose:
                print(f"    ⚠  Pass accuracy fixture fetch failed: {e}")
            return default

        if not fixtures:
            return default

        values = []
        for fixture in fixtures:
            stats = fixture.get("statistics", [])
            if isinstance(stats, dict):
                stats = stats.get("data", [])

            for stat in stats:
                if stat.get("participant_id") != team_id:
                    continue
                type_info = stat.get("type", {})
                code = type_info.get("code", "") if isinstance(type_info, dict) else ""
                if code == "successful-passes-percentage":
                    raw = stat.get("data", {})
                    val = raw.get("value") if isinstance(raw, dict) else None
                    if val is not None:
                        try:
                            values.append(float(val))
                        except (TypeError, ValueError):
                            pass

        if not values:
            if verbose:
                print(f"    ⚠  No pass accuracy found in fixtures, using default {default}")
            return default
        return round(sum(values) / len(values), 1)

    def _parse_team_stats(self, team_data: Dict) -> Optional[Dict]:
        """
        Parse team statistics into feature dictionary.

        All codes and value structures confirmed by API diagnostic (Feb 2026).
        Requires include=statistics.details.type so type is a dict with "code".

        Code              Value structure                        Example
        ─────────────────────────────────────────────────────────────────────
        ball-possession   {"average": X, ...}           flat    53.55
        shots             {"average": X, ...}           flat    13.89
        tackles           {"average": X, ...}           flat    21.5
        interception-stats{"interceptions_per_game": X} flat     9.58
        goals             {"all": {"average": X}}       nested   1.16
        goals-conceded    {"all": {"average": X}}       nested   1.42
        cleansheets       {"all": {"percentage": X}}    nested  26.32
        team-wins         {"all": {"percentage": X}}    nested  28.95
        pass_accuracy     NOT IN TEAM STATS — fetched separately
        ─────────────────────────────────────────────────────────────────────
        """
        stats = team_data.get("statistics", [])
        if not stats:
            return None

        season_stats = stats[0] if stats else {}
        details = season_stats.get("details", [])

        # Build lookup: code -> value dict
        # Only works when include=statistics.details.type (type is a dict)
        parsed: Dict[str, Any] = {}
        for detail in details:
            type_info = detail.get("type", {})
            if not isinstance(type_info, dict):
                continue  # integer ID without .type sub-include — skip
            code  = type_info.get("code", "")
            value = detail.get("value", {})
            if code and isinstance(value, dict):
                parsed[code] = value

        if not parsed:
            return None

        features: Dict[str, Any] = {}

        # Flat: value = {"average": X, ...}
        features["possession"]    = float(parsed.get("ball-possession",   {}).get("average",                  50.0))
        features["shots"]         = float(parsed.get("shots",             {}).get("average",                  12.0))
        features["tackles"]       = float(parsed.get("tackles",           {}).get("average",                  15.0))

        # Flat: value = {"interceptions_per_game": X, ...}
        features["interceptions"] = float(parsed.get("interception-stats",{}).get("interceptions_per_game",   10.0))

        # Nested: value = {"all": {"average": X}}
        features["goals_scored"]  = float(parsed.get("goals",             {}).get("all", {}).get("average",   1.5))
        features["goals_conceded"]= float(parsed.get("goals-conceded",    {}).get("all", {}).get("average",   1.2))

        # Nested: value = {"all": {"percentage": X}}
        features["clean_sheet_pct"] = float(parsed.get("cleansheets",    {}).get("all", {}).get("percentage", 30.0))
        features["win_rate"]      = float(parsed.get("team-wins",         {}).get("all", {}).get("percentage", 33.0))

        # Placeholder — overwritten immediately by extract_features()
        features["pass_accuracy"] = 80.0

        return features

    def fit(
        self, 
        n_clusters: int = None,
        verbose: bool = True
    ) -> "ManagerDNATrainer":
        """
        Fit K-Means clustering model.
        
        Args:
            n_clusters: Number of clusters (auto-determined if None)
            verbose: Print progress
            
        Returns:
            self for chaining
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from sklearn.metrics import silhouette_score
            import pandas as pd
        except ImportError:
            raise ImportError("scikit-learn and pandas required. Run: pip install scikit-learn pandas")
        
        if not self.manager_features:
            raise ValueError("No manager features. Run fetch_manager_data() and extract_features() first.")
        
        if verbose:
            print("\n" + "=" * 60)
            print("FITTING CLUSTERING MODEL")
            print("=" * 60)
        
        # Resolve feature columns (StatsBomb 8-pillar or Sportsmonks legacy)
        feature_cols = self.feature_names or MANAGER_DNA_FEATURES
        
        # Create DataFrame
        self.df_managers = pd.DataFrame(self.manager_features)
        
        # Extract feature matrix
        X = self.df_managers[feature_cols].values
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine optimal K if not specified
        if n_clusters is None:
            if verbose:
                print("\nDetermining optimal cluster count...")
            
            best_k = 3
            best_score = -1
            
            for k in range(2, min(7, len(X))):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)
                
                if verbose:
                    print(f"  K={k}: silhouette={score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_k = k
            
            n_clusters = best_k
            if verbose:
                print(f"\n  Selected K={n_clusters} (silhouette={best_score:.3f})")
        
        self.n_clusters = n_clusters
        
        # Fit final model
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df_managers["cluster"] = self.kmeans.fit_predict(X_scaled)
        
        # PCA for visualisation
        self.pca = PCA(n_components=2)
        X_pca = self.pca.fit_transform(X_scaled)
        self.df_managers["pca_1"] = X_pca[:, 0]
        self.df_managers["pca_2"] = X_pca[:, 1]
        
        # Create centroids DataFrame
        centroids = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        self.df_centroids = pd.DataFrame(centroids, columns=feature_cols)
        self.df_centroids["cluster"] = range(n_clusters)
        
        # Auto-name clusters
        self._name_clusters(verbose)
        
        if verbose:
            print("\n" + "-" * 60)
            print("CLUSTER ASSIGNMENTS:")
            for c in range(n_clusters):
                managers = self.df_managers[self.df_managers["cluster"] == c]["coach_name"].tolist()
                print(f"\n  {self.cluster_names[c]}:")
                for m in managers:
                    print(f"    â€¢ {m}")
        
        return self
    
    def _name_clusters(self, verbose: bool = True):
        """
        Name clusters using only the six canonical ARCHETYPE_WEIGHTS keys.

        Every name produced here MUST be a key in ARCHETYPE_WEIGHTS so that
        _calculate_weighted_fit() can look up the correct feature weights.
        Valid names: Possession-Based, High-Press, Counter-Attack,
                     Defensive, Attacking, Balanced.
        
        Supports both Sportsmonks features and StatsBomb 8-pillar features.
        """
        import pandas as pd

        feature_cols = self.feature_names or MANAGER_DNA_FEATURES
        overall_mean = self.df_managers[feature_cols].mean()
        overall_std  = self.df_managers[feature_cols].std()

        # Features where lower = better (z-scores get inverted)
        negative_traits = ["goals_conceded"]
        used_names: set = set()

        STRONG = 0.5
        WEAK   = 0.2
        
        # Feature → archetype mapping (covers both Sportsmonks and StatsBomb)
        FEAT_TO_ARCHETYPE_POS = {
            # Sportsmonks features
            "possession": "Possession-Based",
            "pass_accuracy": "Possession-Based",
            "shots": "Attacking",
            "goals_scored": "Attacking",
            "tackles": "High-Press",
            "interceptions": "High-Press",
            "goals_conceded": "Defensive",  # inverted: low conceded = Defensive
            "clean_sheet_pct": "Defensive",
            "win_rate": "Balanced",
            # StatsBomb 8-pillar features
            "pressing_intensity": "High-Press",
            "counterpress_rate": "High-Press",
            "build_up_patience": "Possession-Based",
            "directness": "Counter-Attack",
            "chance_quality": "Attacking",
            "defensive_line_height": "High-Press",
            "width_usage": "Attacking",
            "set_piece_emphasis": "Balanced",
            "transition_threat": "Counter-Attack",
            "defensive_solidity": "Defensive",
        }
        FEAT_TO_ARCHETYPE_NEG = {
            "possession": "Counter-Attack",
            "pass_accuracy": "Counter-Attack",
            "shots": "Defensive",
            "goals_scored": "Defensive",
            "tackles": "Defensive",
            "interceptions": "Balanced",
            "goals_conceded": "Attacking",
            "clean_sheet_pct": "Attacking",
            "pressing_intensity": "Defensive",
            "counterpress_rate": "Balanced",
            "build_up_patience": "Counter-Attack",
            "directness": "Possession-Based",
            "chance_quality": "Defensive",
            "defensive_line_height": "Defensive",
            "width_usage": "Balanced",
            "set_piece_emphasis": "Balanced",
            "transition_threat": "Balanced",
            "defensive_solidity": "Attacking",
        }

        STRONG = 0.5
        WEAK   = 0.2

        for c in range(self.n_clusters):
            centroid = self.df_centroids.loc[c]

            z_scores = {}
            for feat in feature_cols:
                if overall_std[feat] > 0:
                    z = (centroid[feat] - overall_mean[feat]) / overall_std[feat]
                    if feat in negative_traits:
                        z = -z  # low conceded -> positive
                    z_scores[feat] = z
                else:
                    z_scores[feat] = 0

            sorted_feats = sorted(z_scores.items(), key=lambda x: abs(x[1]), reverse=True)
            top_feat, top_z = sorted_feats[0]
            sec_feat, sec_z = sorted_feats[1] if len(sorted_feats) > 1 else ("", 0)

            # Lookup-based naming using the archetype maps
            if top_z >= STRONG:
                name = FEAT_TO_ARCHETYPE_POS.get(top_feat, "Balanced")
            elif top_z <= -STRONG:
                name = FEAT_TO_ARCHETYPE_NEG.get(top_feat, "Balanced")
            elif abs(top_z) >= WEAK:
                if top_z > 0:
                    name = FEAT_TO_ARCHETYPE_POS.get(top_feat, "Balanced")
                else:
                    name = FEAT_TO_ARCHETYPE_NEG.get(top_feat, "Balanced")
            else:
                name = "Balanced"

            # Deduplicate: pick next-best canonical name by archetype signal score
            if name in used_names:
                # Build archetype signals from available features (works for both feature sets)
                archetype_signals = {"Balanced": 0}
                for feat, z in z_scores.items():
                    pos_arch = FEAT_TO_ARCHETYPE_POS.get(feat)
                    if pos_arch and pos_arch != "Balanced":
                        archetype_signals[pos_arch] = archetype_signals.get(pos_arch, 0) + max(z, 0)
                    neg_arch = FEAT_TO_ARCHETYPE_NEG.get(feat)
                    if neg_arch and neg_arch != "Balanced":
                        archetype_signals[neg_arch] = archetype_signals.get(neg_arch, 0) + max(-z, 0)
                
                ranked = sorted(
                    [a for a in archetype_signals if a not in used_names],
                    key=lambda a: archetype_signals[a],
                    reverse=True
                )
                name = ranked[0] if ranked else "Balanced"

            used_names.add(name)
            self.cluster_names[c] = name

            if verbose:
                print(f"  Cluster {c}: {name}  (top={top_feat} z={top_z:+.2f})")

        self.df_managers["cluster_name"] = self.df_managers["cluster"].map(self.cluster_names)
        self.df_centroids["cluster_name"] = self.df_centroids["cluster"].map(self.cluster_names)

    def save(self, verbose: bool = True) -> "ManagerDNATrainer":
        """Save model and data to files."""
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model components
        model_data = {
            "kmeans": self.kmeans,
            "scaler": self.scaler,
            "pca": self.pca,
            "feature_cols": self.feature_names or MANAGER_DNA_FEATURES,
            "cluster_names": self.cluster_names,
            "n_clusters": self.n_clusters,
        }
        
        with open(self.training_dir / "manager_dna_model.pkl", "wb") as f:
            pickle.dump(model_data, f)
        
        self.df_managers.to_csv(self.training_dir / "manager_profiles.csv", index=False)
        self.df_centroids.to_csv(self.training_dir / "cluster_centroids.csv", index=False)
        
        summary = {
            "created_at": datetime.now().isoformat(),
            "season_id": self.season_id,
            "n_managers": len(self.df_managers),
            "n_clusters": self.n_clusters,
            "cluster_names": self.cluster_names,
            "feature_cols": self.feature_names or MANAGER_DNA_FEATURES,
            "managers_by_cluster": {
                self.cluster_names[c]: self.df_managers[self.df_managers["cluster"] == c]["coach_name"].tolist()
                for c in range(self.n_clusters)
            }
        }
        
        with open(self.training_dir / "manager_dna_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        if verbose:
            print("\n" + "=" * 60)
            print("MODEL SAVED")
            print("=" * 60)
            print(f"  â€¢ {self.training_dir}/manager_dna_model.pkl")
            print(f"  â€¢ {self.training_dir}/manager_profiles.csv")
            print(f"  â€¢ {self.training_dir}/cluster_centroids.csv")
            print(f"  â€¢ {self.training_dir}/manager_dna_summary.json")
        
        return self
    
    def plot_clusters(self, save_path: Optional[Path] = None):
        """Generate PCA cluster visualisation."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
        
        for c in range(self.n_clusters):
            mask = self.df_managers["cluster"] == c
            ax.scatter(
                self.df_managers.loc[mask, "pca_1"],
                self.df_managers.loc[mask, "pca_2"],
                c=colors[c % len(colors)],
                s=150,
                label=self.cluster_names[c],
                alpha=0.7,
                edgecolors='white',
                linewidth=2
            )
        
        for idx, row in self.df_managers.iterrows():
            ax.annotate(
                row["coach_name"],
                (row["pca_1"], row["pca_2"]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                alpha=0.8
            )
        
        ax.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        ax.set_title('Manager Tactical Archetypes', fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  âœ“ Saved: {save_path}")
        
        plt.show()


# =============================================================================
# SQUAD FIT ANALYZER
# =============================================================================

class SquadFitAnalyzer:
    """
    Analyse squad fit against a manager's tactical profile.
    
    Uses weighted comparison to position-specific ideal profiles,
    adjusted for the manager's tactical archetype.
    """
    
    def __init__(
        self,
        client=None,
        training_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        season_id: int = 23614
    ):
        self.client = client
        self.training_dir = Path(training_dir) if training_dir else Config.PROCESSED_DIR / "training"
        self.output_dir = Path(output_dir) if output_dir else Config.OUTPUT_DIR
        self.season_id = season_id
        
        self.model = None
        self.df_managers = None
        self.df_centroids = None
        
        self.target_manager = None
        self.target_cluster = None
        self.target_cluster_name = None
        self.target_club = None
        self.target_squad = None
        
        self.squad_fit = []
        self.ideal_xi = []
        self.squad_stats = []
        self.manager_pillar_scores = None  # 8-pillar scores from DNA
        self.league_percentiles = None     # Percentile distributions by position
    
    def load_model(self, verbose: bool = True) -> "SquadFitAnalyzer":
        """Load trained Manager DNA model."""
        import pandas as pd
        
        model_path = self.training_dir / "manager_dna_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Run ManagerDNATrainer first to create the model."
            )
        
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        
        self.df_managers = pd.read_csv(self.training_dir / "manager_profiles.csv")
        self.df_centroids = pd.read_csv(self.training_dir / "cluster_centroids.csv")
        
        if verbose:
            print("âœ“ Loaded Manager DNA model")
            print(f"  Clusters: {self.model['n_clusters']}")
            print(f"  Cluster names: {list(self.model['cluster_names'].values())}")
            print(f"  Managers: {len(self.df_managers)}")
        
        return self
    
    def set_target_manager(self, manager_name: str, verbose: bool = True) -> "SquadFitAnalyzer":
        """
        Set the target manager by looking them up in the trained model.
        
        Uses the manager's ACTUAL tactical features from training to compute
        pillar scores. This is the correct method for hypothetical scenarios
        ("what if Arteta managed Chelsea?").
        """
        mask = self.df_managers["coach_name"].str.contains(manager_name, case=False, na=False)
        matches = self.df_managers[mask]
        
        if matches.empty:
            raise ValueError(
                f"Manager '{manager_name}' not found in model.\n"
                f"Available: {self.df_managers['coach_name'].tolist()}"
            )
        
        manager = matches.iloc[0]
        self.target_manager = manager["coach_name"]
        self.target_cluster = int(manager["cluster"])
        self.target_cluster_name = self.model["cluster_names"][self.target_cluster]
        
        # Compute pillar scores from the manager's actual training features
        # These reflect their real tactical profile (e.g., Arteta from Arsenal data)
        if all(f in manager.index for f in STATSBOMB_DNA_FEATURES):
            self.manager_pillar_scores = {
                "shape_occupation": min(100, round(
                    float(manager.get("defensive_line_height", 40)) * 1.5 + 
                    float(manager.get("defensive_solidity", 50)) * 0.3, 0)),
                "build_up": min(100, round(float(manager.get("build_up_patience", 50)), 0)),
                "chance_creation": min(100, round(float(manager.get("chance_quality", 0.1)) * 500, 0)),
                "press_counterpress": min(100, round(float(manager.get("pressing_intensity", 15)) * 4, 0)),
                "block_line_height": min(100, round(
                    float(manager.get("defensive_line_height", 40)) * 1.5 + 
                    float(manager.get("defensive_solidity", 50)) * 0.5, 0)),
                "transitions": min(100, round(float(manager.get("transition_threat", 1)) * 25, 0)),
                "width_overloads": min(100, round(float(manager.get("width_usage", 5)) * 10, 0)),
                "set_pieces": min(100, round(float(manager.get("set_piece_emphasis", 20)) * 2, 0)),
            }
            
            if verbose:
                print(f"\nTarget Manager: {self.target_manager}")
                print(f"  Team: {manager.get('team_name', '?')}")
                print(f"  Tactical Archetype: {self.target_cluster_name}")
                print(f"  Pillar scores (from training data):")
                for k, v in self.manager_pillar_scores.items():
                    print(f"    {k}: {v}")
        else:
            self.manager_pillar_scores = None
            if verbose:
                print(f"\nTarget Manager: {self.target_manager}")
                print(f"  Tactical Archetype: {self.target_cluster_name}")
        
        if verbose:
            same_cluster = self.df_managers[self.df_managers["cluster"] == self.target_cluster]
            others = [n for n in same_cluster["coach_name"].tolist() if n != self.target_manager]
            if others:
                print(f"  Similar Managers: {', '.join(others)}")
        
        return self
    
    def set_target_manager_from_dna(
        self,
        manager_dna: dict,
        verbose: bool = True
    ) -> "SquadFitAnalyzer":
        """
        Set the target manager using a pre-built manager DNA dict.
        
        Predicts which cluster the DNA belongs to using the trained model.
        Works with DNA from either ETLPipeline or StatsBombETL.
        """
        import numpy as np
        
        self.target_manager = manager_dna.get("manager", "Unknown")
        self.target_club = manager_dna.get("team", "Unknown")
        
        # Store pillar scores for dynamic weight computation
        self.manager_pillar_scores = manager_dna.get("pillar_scores", None)
        
        tp = manager_dna.get("tactical_profile", {})
        rp = manager_dna.get("results_profile", {})
        sb = manager_dna.get("statsbomb_enhanced", {})
        
        # Build features for BOTH feature sets — the model picks which to use
        features = {
            # Sportsmonks legacy features
            "possession": tp.get("possession", {}).get("avg", 50.0),
            "pass_accuracy": tp.get("build_up", {}).get("pass_accuracy", 80.0),
            "shots": tp.get("attacking", {}).get("shots_pg", 12.0),
            "goals_scored": rp.get("goals_per_game", 1.5),
            "goals_conceded": rp.get("conceded_per_game", 1.2),
            "tackles": tp.get("pressing", {}).get("intensity", {}).get("avg", 20.0) * 0.6,
            "interceptions": tp.get("pressing", {}).get("intensity", {}).get("avg", 20.0) * 0.4,
            "clean_sheet_pct": rp.get("clean_sheet_pct", 30.0),
            "win_rate": rp.get("win_rate", 33.0),
            # StatsBomb 8-pillar features
            "pressing_intensity": max(0, 30 - sb.get("ppda", tp.get("pressing", {}).get("ppda", 10))),
            "counterpress_rate": round(
                sb.get("pressures_per_game", 0) and 
                (tp.get("pressing", {}).get("counterpressures_per_game", 30) / 
                 max(sb.get("pressures_per_game", 150), 1) * 100) or 20, 1),
            "build_up_patience": 100 - (sb.get("directness", 30) if "directness" in sb else 
                                        max(0, 100 - tp.get("build_up", {}).get("pass_accuracy", 80))),
            "directness": sb.get("directness", 30),
            "chance_quality": round(
                sb.get("np_xg_per_game", tp.get("attacking", {}).get("np_xg_pg", 1.5)) / 
                max(tp.get("attacking", {}).get("shots_pg", 12), 1), 3),
            "defensive_line_height": tp.get("pressing", {}).get("defensive_distance", 40),
            "width_usage": sb.get("width_usage", 5),
            "set_piece_emphasis": sb.get("set_piece_emphasis", 20),
            "transition_threat": sb.get("counter_attacking_shots_pg", 
                                       tp.get("attacking", {}).get("counter_attacking_shots_pg", 1.0)),
            "defensive_solidity": round(max(0, 3.0 - rp.get("conceded_per_game", 1.2)) * 33.3, 1),
        }
        
        if self.model and self.model.get("scaler") and self.model.get("kmeans"):
            feature_cols = self.model.get("feature_cols", MANAGER_DNA_FEATURES)
            feature_vector = np.array([[features.get(f, 0) for f in feature_cols]])
            scaled = self.model["scaler"].transform(feature_vector)
            self.target_cluster = int(self.model["kmeans"].predict(scaled)[0])
            self.target_cluster_name = self.model["cluster_names"].get(
                self.target_cluster, "Balanced"
            )
        else:
            self.target_cluster = 0
            poss = features["possession"]
            if poss >= 55:
                self.target_cluster_name = "Possession-Based"
            elif features["tackles"] >= 14:
                self.target_cluster_name = "High-Press"
            elif features["shots"] >= 14 and features["goals_scored"] >= 1.8:
                self.target_cluster_name = "Attacking"
            elif features["goals_conceded"] <= 1.0:
                self.target_cluster_name = "Defensive"
            elif poss <= 45:
                self.target_cluster_name = "Counter-Attack"
            else:
                self.target_cluster_name = "Balanced"
        
        if verbose:
            print(f"\n\u2713 Target Manager: {self.target_manager}")
            print(f"  Team: {self.target_club}")
            print(f"  Tactical Archetype: {self.target_cluster_name}")
            print(f"  (Assigned from DNA, data mode: {manager_dna.get('data_mode', '?')})")
        
        return self
    
    def calculate_fit_scores_from_profiles(
        self,
        squad_profiles: list,
        club_name: str = None,
        league_player_stats: list = None,
        verbose: bool = True
    ) -> "SquadFitAnalyzer":
        """
        Calculate fit scores using pillar-driven, position-specific, percentile-based scoring.
        
        New scoring engine (v2):
        1. Manager's 8-pillar DNA scores -> position-specific weight profiles
        2. All league players -> percentile distributions by position group
        3. Each player's fit = weighted average of their percentile ranks
        
        Falls back to legacy archetype scoring if pillar scores or league data
        are not available (Sportsmonks compatibility).
        
        Args:
            squad_profiles: Player profile dicts (from ETL)
            club_name: Club name override
            league_player_stats: Full league player_season_stats for percentile computation.
            verbose: Print progress
        """
        if self.target_cluster is None and self.target_cluster_name is None:
            raise ValueError("No target manager set.")
        
        if club_name:
            self.target_club = club_name
        
        # Decide scoring mode
        use_pillar_scoring = (
            self.manager_pillar_scores is not None and 
            league_player_stats is not None and
            len(league_player_stats) > 20
        )
        
        if verbose:
            print("\n" + "=" * 60)
            if use_pillar_scoring:
                print("CALCULATING FIT SCORES (8-Pillar Percentile Engine)")
            else:
                print("CALCULATING FIT SCORES (Legacy Archetype Mode)")
            print("=" * 60)
        
        if use_pillar_scoring:
            self._build_league_percentiles(league_player_stats, verbose)
        
        self.squad_fit = []
        
        for profile in squad_profiles:
            minutes = profile.get("minutes", 0)
            if minutes < 90:
                continue
            
            per90_factor = 90.0 / max(minutes, 90)
            
            features = {
                "goals_per90": float(profile.get("goals", 0) or 0) * per90_factor,
                "assists_per90": float(profile.get("assists", 0) or 0) * per90_factor,
                "tackles_per90": float(profile.get("tackles", 0) or 0) * per90_factor,
                "interceptions_per90": float(profile.get("interceptions", 0) or 0) * per90_factor,
                "pass_accuracy": float(profile.get("pass_accuracy", 0) or 0) or 75.0,
                "key_passes_per90": float(profile.get("key_passes", 0) or 0) * per90_factor,
                "dribbles_per90": float(profile.get("dribbles", 0) or 0) * per90_factor,
                "shots_per90": float(profile.get("shots", 0) or 0) * per90_factor,
            }
            
            position = profile.get("position", "Unknown")
            detailed_position = profile.get("detailed_position", position)
            position_group = self._get_position_group_from_name(position, detailed_position)
            
            if use_pillar_scoring:
                fit_score = self._calculate_pillar_fit(features, position_group)
            else:
                fit_score = self._calculate_weighted_fit(features, position_group)
            
            classification = self._classify_score(fit_score)
            
            self.squad_fit.append(PlayerFit(
                name=profile.get("name", "Unknown"),
                position=position,
                detailed_position=detailed_position,
                position_group=position_group,
                age=profile.get("age", 25),
                fit_score=round(fit_score, 1),
                classification=classification,
                stats=features
            ))
        
        self.squad_fit.sort(key=lambda x: x.fit_score, reverse=True)
        
        if verbose and self.squad_fit:
            counts = {}
            for p in self.squad_fit:
                counts[p.classification] = counts.get(p.classification, 0) + 1
            
            print("\n\u2713 Classifications:")
            for cls in ["Key Enabler", "Good Fit", "System Dependent", "Potentially Marginalised"]:
                count = counts.get(cls, 0)
                if count > 0:
                    print(f"    {cls}: {count}")
            
            avg_fit = sum(p.fit_score for p in self.squad_fit) / len(self.squad_fit)
            print(f"\n  Average Fit Score: {avg_fit:.1f}")
            
            if use_pillar_scoring:
                print(f"  Scoring: 8-Pillar Percentile Engine (league={len(league_player_stats)} players)")
            else:
                print(f"  Scoring: Legacy archetype ({self.target_cluster_name})")
        
        self._generate_ideal_xi(verbose)
        return self
    
    # =========================================================================
    # NEW SCORING ENGINE: Pillar-Driven Percentile Scoring
    # =========================================================================
    
    def _build_league_percentiles(self, league_player_stats: list, verbose: bool = True):
        """Pre-compute percentile distributions from all league players by position group."""
        from collections import defaultdict
        from .etl import StatsBombETL
        
        if verbose:
            print(f"\n  Building league percentiles from {len(league_player_stats)} players...")
        
        position_features = defaultdict(lambda: defaultdict(list))
        
        for p in league_player_stats:
            minutes = p.get("player_season_minutes", 0) or 0
            if minutes < 270:
                continue
            
            primary_pos = p.get("primary_position")
            pos = StatsBombETL.POSITION_MAP.get(primary_pos, None)
            if not pos or pos == "Unknown":
                save_ratio = p.get("player_season_save_ratio", 0) or 0
                goals_faced = p.get("player_season_goals_faced_90", 0) or 0
                tackles_90 = p.get("player_season_tackles_90", 0) or 0
                interceptions_90 = p.get("player_season_interceptions_90", 0) or 0
                clearance_90 = p.get("player_season_clearance_90", 0) or 0
                goals_90 = p.get("player_season_goals_90", 0) or 0
                np_xg_90 = p.get("player_season_np_xg_90", 0) or 0
                
                if save_ratio > 0 or goals_faced > 0:
                    pos = "Goalkeeper"
                elif (tackles_90 + interceptions_90 + clearance_90) > 4.0 and goals_90 < 0.15:
                    pos = "Defender"
                elif goals_90 > 0.25 or np_xg_90 > 0.25:
                    pos = "Attacker"
                else:
                    pos = "Midfielder"
            
            pos_lower = pos.lower()
            if "goal" in pos_lower: group = "GK"
            elif any(k in pos_lower for k in ["back", "defender"]): group = "DEF"
            elif any(k in pos_lower for k in ["winger", "forward", "striker", "attacker"]): group = "ATT"
            else: group = "MID"
            
            features = {
                "goals_per90": float(p.get("player_season_goals_90", 0) or 0),
                "assists_per90": float(p.get("player_season_assists_90", 0) or 0),
                "tackles_per90": float(p.get("player_season_tackles_90", 0) or 0),
                "interceptions_per90": float(p.get("player_season_interceptions_90", 0) or 0),
                "pass_accuracy": float(p.get("player_season_passing_ratio", 0) or 0),
                "key_passes_per90": float(p.get("player_season_key_passes_90", 0) or 0),
                "dribbles_per90": float(p.get("player_season_dribbles_90", 0) or 0),
                "shots_per90": float(p.get("player_season_np_shots_90", 0) or 0),
            }
            
            for feat, val in features.items():
                position_features[group][feat].append(val)
        
        self.league_percentiles = {}
        for group in ["GK", "DEF", "MID", "ATT"]:
            self.league_percentiles[group] = {}
            for feat in PLAYER_FIT_FEATURES:
                vals = sorted(position_features[group].get(feat, [0]))
                self.league_percentiles[group][feat] = vals
        
        if verbose:
            for group in ["GK", "DEF", "MID", "ATT"]:
                n = len(position_features[group].get("pass_accuracy", []))
                print(f"    {group}: {n} players")
    
    def _get_percentile(self, value: float, position_group: str, feature: str) -> float:
        """Get percentile rank (0-100) of a value within its position group."""
        import bisect
        if not self.league_percentiles:
            return 50.0
        sorted_vals = self.league_percentiles.get(position_group, {}).get(feature, [])
        if not sorted_vals:
            return 50.0
        rank = bisect.bisect_left(sorted_vals, value)
        return round(rank / len(sorted_vals) * 100, 1)
    
    def _compute_dynamic_weights(self, position_group: str) -> Dict[str, float]:
        """
        Compute position-specific feature weights from manager's 8-pillar DNA.
        
        Each pillar's demand matrix is scaled by the manager's actual score
        on that pillar (0-100 -> 0-1). Base weight of 0.5 ensures all features
        contribute, preventing any feature from being zeroed out.
        """
        weights = {f: 0.5 for f in PLAYER_FIT_FEATURES}
        
        if not self.manager_pillar_scores:
            return weights
        
        for pillar_key, demands_by_pos in PILLAR_PLAYER_DEMANDS.items():
            pillar_score = self.manager_pillar_scores.get(pillar_key, 50) / 100.0
            demands = demands_by_pos.get(position_group, {})
            for feature, demand_strength in demands.items():
                if feature in weights:
                    weights[feature] += pillar_score * demand_strength
        
        return weights
    
    def _calculate_pillar_fit(self, player_stats: Dict, position_group: str) -> float:
        """
        Calculate fit score using pillar-driven weights and league percentiles.
        
        Score = weighted average of percentile ranks, producing a 0-100 score
        unique to this specific manager x position combination.
        """
        weights = self._compute_dynamic_weights(position_group)
        
        total_weighted_pct = 0
        total_weight = 0
        
        for feature in PLAYER_FIT_FEATURES:
            value = player_stats.get(feature, 0)
            weight = weights.get(feature, 0.5)
            percentile = self._get_percentile(value, position_group, feature)
            
            total_weighted_pct += percentile * weight
            total_weight += weight
        
        raw_score = total_weighted_pct / total_weight if total_weight > 0 else 50.0
        
        # Spread adjustment to avoid scores bunching around 45-55
        if raw_score >= 65:
            adjusted = 65 + (raw_score - 65) * 1.3
        elif raw_score <= 35:
            adjusted = 35 - (35 - raw_score) * 1.3
        else:
            adjusted = raw_score
        
        return max(0, min(100, adjusted))
    
    def _get_position_group_from_name(self, position: str, detailed_position: str = "") -> str:
        """Map position name string to group (GK/DEF/MID/ATT)."""
        check = (detailed_position or position or "").lower()
        if "goal" in check: return "GK"
        if any(kw in check for kw in ["back", "defender", "centre-back", "center-back"]): return "DEF"
        if any(kw in check for kw in ["winger", "forward", "striker", "attacker", "centre-forward"]): return "ATT"
        if "attacking" in check and "midfield" in check: return "ATT"
        if "wing" in check and "back" not in check: return "ATT"
        return "MID"

    def fetch_squad(self, club_name: str, verbose: bool = True) -> "SquadFitAnalyzer":
        """Fetch target club's squad."""
        from .client import SportsmonksClient
        
        if self.client is None:
            self.client = SportsmonksClient()
        
        if verbose:
            print(f"\nFetching squad for: {club_name}")
        
        teams = self.client.search_teams(club_name)
        if not teams:
            raise ValueError(f"Team not found: {club_name}")
        
        team = self.client.get_team(teams[0]["id"])
        self.target_club = team.get("name", club_name)
        
        squad = self.client.get_squad(team["id"], self.season_id)
        self.target_squad = squad
        
        if verbose:
            print(f"  âœ“ {self.target_club}: {len(squad)} players")
        
        return self
    
    def calculate_fit_scores(self, verbose: bool = True) -> "SquadFitAnalyzer":
        """Calculate fit scores using weighted comparison to ideal profiles."""
        if self.target_squad is None:
            raise ValueError("No squad loaded. Run fetch_squad() first.")
        
        if self.target_cluster is None:
            raise ValueError("No target manager set. Run set_target_manager() first.")
        
        if verbose:
            print("\n" + "=" * 60)
            print("CALCULATING FIT SCORES")
            print("=" * 60)
        
        self.squad_stats = []
        player_metadata = []
        
        for entry in self.target_squad:
            player = entry.get("player", {})
            if not player:
                continue
            
            features = self._extract_player_features(player, entry)
            position_group = self._get_position_group(player)
            
            self.squad_stats.append(features)
            player_metadata.append({
                "name": player.get("common_name") or player.get("name", "Unknown"),
                "position": self._extract_position(player),
                "detailed_position": self._extract_detailed_position(player),
                "position_group": position_group,
                "age": self._calculate_age(player.get("date_of_birth")),
            })
        
        if not self.squad_stats:
            if verbose:
                print("  âš  No player stats extracted")
            return self
        
        self.squad_fit = []
        
        for i, (stats, meta) in enumerate(zip(self.squad_stats, player_metadata)):
            fit_score = self._calculate_weighted_fit(
                player_stats=stats,
                position_group=meta["position_group"]
            )
            
            classification = self._classify_score(fit_score)
            
            self.squad_fit.append(PlayerFit(
                name=meta["name"],
                position=meta["position"],
                detailed_position=meta["detailed_position"],
                position_group=meta["position_group"],
                age=meta["age"],
                fit_score=round(fit_score, 1),
                classification=classification,
                stats=stats
            ))
        
        self.squad_fit.sort(key=lambda x: x.fit_score, reverse=True)
        
        if verbose:
            counts = {}
            for p in self.squad_fit:
                counts[p.classification] = counts.get(p.classification, 0) + 1
            
            print("\nâœ“ Classifications:")
            for cls in ["Key Enabler", "Good Fit", "System Dependent", "Potentially Marginalised"]:
                count = counts.get(cls, 0)
                if count > 0:
                    print(f"    {cls}: {count}")
            
            avg_fit = sum(p.fit_score for p in self.squad_fit) / len(self.squad_fit) if self.squad_fit else 0
            print(f"\n  Average Fit Score: {avg_fit:.1f}")
        
        self._generate_ideal_xi(verbose)
        
        return self
    
    def _calculate_weighted_fit(self, player_stats: Dict, position_group: str) -> float:
        """Calculate fit score using weighted comparison to ideal profile."""
        ideal = IDEAL_PLAYER_PROFILES.get(position_group, IDEAL_PLAYER_PROFILES["MID"])
        weights = ARCHETYPE_WEIGHTS.get(self.target_cluster_name, {})
        
        total_score = 0
        total_weight = 0
        
        for feat in PLAYER_FIT_FEATURES:
            player_val = player_stats.get(feat, 0)
            ideal_val = ideal.get(feat, 0)
            weight = weights.get(feat, 1.0)
            
            if ideal_val > 0:
                # How much of the ideal is achieved (cap at 150%)
                ratio = min(player_val / ideal_val, 1.5)
                feature_score = ratio * 100 * weight
            elif player_val > 0:
                # Player has stat but ideal is 0
                feature_score = 50 * weight
            else:
                # Both zero - perfect for this feature
                feature_score = 100 * weight
            
            total_score += feature_score
            total_weight += weight
        
        if total_weight > 0:
            raw_score = total_score / total_weight
        else:
            raw_score = 50
        
        # Apply diminishing returns to spread out scores
        if raw_score > 80:
            adjusted_score = 80 + (raw_score - 80) * 0.4
        elif raw_score > 60:
            adjusted_score = 60 + (raw_score - 60) * 0.7
        else:
            adjusted_score = raw_score
        
        return max(0, min(100, adjusted_score))
    
    def _extract_player_features(self, player: Dict, entry: Dict) -> Dict:
        """Extract per-90 features from player stats."""
        stats = player.get("statistics", [])
        features = {f: 0.0 for f in PLAYER_FIT_FEATURES}
        
        if not stats:
            return features
        
        season_stats = stats[0] if stats else {}
        details = season_stats.get("details", [])
        
        parsed = {}
        for detail in details:
            type_info = detail.get("type", {})
            code = type_info.get("code", "")
            value = detail.get("value", {})
            
            if isinstance(value, dict):
                parsed[code] = value.get("total", value.get("count", value.get("average", 0)))
            else:
                parsed[code] = value if value else 0
        
        minutes = parsed.get("minutes-played", 0)
        if isinstance(minutes, dict):
            minutes = minutes.get("total", 0)
        minutes = float(minutes) if minutes else 0
        
        per90_factor = 90 / max(minutes, 90) if minutes > 0 else 0
        
        features["goals_per90"] = float(parsed.get("goals", 0) or 0) * per90_factor
        features["assists_per90"] = float(parsed.get("assists", 0) or 0) * per90_factor
        features["tackles_per90"] = float(parsed.get("tackles", 0) or 0) * per90_factor
        features["interceptions_per90"] = float(parsed.get("interceptions", 0) or 0) * per90_factor
        
        pass_acc = parsed.get("passes-accuracy", parsed.get("passes-percentage", 0))
        if isinstance(pass_acc, dict):
            pass_acc = pass_acc.get("average", 0)
        features["pass_accuracy"] = float(pass_acc) if pass_acc else 75.0
        
        features["key_passes_per90"] = float(parsed.get("key-passes", 0) or 0) * per90_factor
        features["dribbles_per90"] = float(parsed.get("successful-dribbles", parsed.get("dribbles", 0)) or 0) * per90_factor
        features["shots_per90"] = float(parsed.get("shots-total", 0) or 0) * per90_factor
        
        return features
    
    def _get_position_group(self, player: Dict) -> str:
        """Map player position to group with improved keyword matching."""
        position = self._extract_detailed_position(player) or self._extract_position(player)
        if not position:
            return "MID"
        
        position_lower = position.lower()
        
        # Goalkeeper
        if "goal" in position_lower:
            return "GK"
        
        # Defenders - check before midfield keywords
        if any(kw in position_lower for kw in ["back", "defender", "centre-back", "center-back"]):
            return "DEF"
        
        # Attackers - explicit attacking positions
        if any(kw in position_lower for kw in ["winger", "forward", "striker", "attacker", "centre-forward"]):
            return "ATT"
        
        # Attacking midfielders should be classified as ATT (they play in attack)
        if "attacking" in position_lower and "midfield" in position_lower:
            return "ATT"
        
        # Wing positions without "back" are attackers
        if "wing" in position_lower and "back" not in position_lower:
            return "ATT"
        
        # Default: MID (includes defensive/central midfielders)
        return "MID"
    
    def _extract_position(self, player: Dict) -> str:
        pos = player.get("position", {})
        if isinstance(pos, dict):
            return pos.get("name", "Unknown")
        return "Unknown"
    
    def _extract_detailed_position(self, player: Dict) -> str:
        pos = player.get("detailedPosition", {})
        if isinstance(pos, dict):
            return pos.get("name", "")
        return ""
    
    def _calculate_age(self, dob: str) -> int:
        if not dob:
            return 25
        try:
            from datetime import datetime
            birth = datetime.strptime(dob[:10], "%Y-%m-%d")
            today = datetime.now()
            return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        except:
            return 25
    
    def _classify_score(self, score: float) -> str:
        if score >= FIT_THRESHOLDS["key_enabler"]:
            return "Key Enabler"
        elif score >= FIT_THRESHOLDS["good_fit"]:
            return "Good Fit"
        elif score >= FIT_THRESHOLDS["system_dependent"]:
            return "System Dependent"
        else:
            return "Potentially Marginalised"
    
    def _generate_ideal_xi(self, verbose: bool = True):
        """Generate ideal starting XI based on fit scores and actual positions."""
        
        # Define formation slots with position group and preferred positions
        formation_slots = {
            "GK": {"group": "GK", "pref": ["Goalkeeper"]},
            "CB1": {"group": "DEF", "pref": ["Centre-Back"]},
            "CB2": {"group": "DEF", "pref": ["Centre-Back"]},
            "RB": {"group": "DEF", "pref": ["Right-Back", "Right Wing-Back"]},
            "LB": {"group": "DEF", "pref": ["Left-Back", "Left Wing-Back"]},
            "DM": {"group": "MID", "pref": ["Defensive Midfield", "Defensive Midfielder"]},
            "CM": {"group": "MID", "pref": ["Central Midfield", "Central Midfielder", "Midfielder"]},
            "AM": {"group": "MID", "pref": ["Attacking Midfield", "Attacking Midfielder"]},
            "LW": {"group": "ATT", "pref": ["Left Winger", "Left Wing", "Left Midfield"]},
            "RW": {"group": "ATT", "pref": ["Right Winger", "Right Wing", "Right Midfield"]},
            "CF": {"group": "ATT", "pref": ["Centre-Forward", "Forward", "Striker", "Second Striker"]}
        }
        
        self.ideal_xi = []
        used_players = set()
        unfilled_slots = []
        
        # FIRST PASS: Assign players who exactly match slot preferences
        for slot, config in formation_slots.items():
            candidates = [
                p for p in self.squad_fit 
                if p.position_group == config["group"] and p.name not in used_players
            ]
            
            # Only consider players whose actual position matches this slot
            if "pref" in config and candidates:
                preferred = [
                    p for p in candidates
                    if any(pref.lower() in (p.detailed_position or p.position).lower() 
                           for pref in config["pref"])
                ]
                
                if preferred:
                    # Found exact match - assign best one
                    best = max(preferred, key=lambda x: x.fit_score)
                    self.ideal_xi.append({
                        "slot": slot,
                        "name": best.name,
                        "position": best.detailed_position or best.position,
                        "fit_score": best.fit_score,
                        "classification": best.classification
                    })
                    used_players.add(best.name)
                else:
                    # No exact match - mark for second pass
                    unfilled_slots.append((slot, config))
            elif candidates:
                # No preferences defined (e.g., generic positions)
                best = max(candidates, key=lambda x: x.fit_score)
                self.ideal_xi.append({
                    "slot": slot,
                    "name": best.name,
                    "position": best.detailed_position or best.position,
                    "fit_score": best.fit_score,
                    "classification": best.classification
                })
                used_players.add(best.name)
            else:
                unfilled_slots.append((slot, config))
        
        # SECOND PASS: Fill remaining slots with best available from position group
        for slot, config in unfilled_slots:
            candidates = [
                p for p in self.squad_fit 
                if p.position_group == config["group"] and p.name not in used_players
            ]
            
            if candidates:
                best = max(candidates, key=lambda x: x.fit_score)
                self.ideal_xi.append({
                    "slot": slot,
                    "name": best.name,
                    "position": best.detailed_position or best.position,  # Show ACTUAL position
                    "fit_score": best.fit_score,
                    "classification": best.classification
                })
                used_players.add(best.name)
        
        # Sort by slot order for consistent display
        slot_order = ["GK", "LB", "CB1", "CB2", "RB", "DM", "CM", "AM", "LW", "CF", "RW"]
        self.ideal_xi.sort(key=lambda x: slot_order.index(x["slot"]) if x["slot"] in slot_order else 99)
        
        if verbose and self.ideal_xi:
            avg_fit = sum(p["fit_score"] for p in self.ideal_xi) / len(self.ideal_xi)
            print(f"\n✓ Ideal XI average fit: {avg_fit:.1f}")
    
    def save(self, verbose: bool = True) -> "SquadFitAnalyzer":
        """Save analysis results."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_dir / "squad_fit_scores.csv", "w", newline="") as f:
            if self.squad_fit:
                writer = csv.DictWriter(f, fieldnames=[
                    "name", "position", "detailed_position", "position_group",
                    "age", "fit_score", "classification"
                ])
                writer.writeheader()
                for p in self.squad_fit:
                    writer.writerow({
                        "name": p.name,
                        "position": p.position,
                        "detailed_position": p.detailed_position,
                        "position_group": p.position_group,
                        "age": p.age,
                        "fit_score": p.fit_score,
                        "classification": p.classification
                    })
        
        if self.ideal_xi:
            with open(self.output_dir / "ideal_xi.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.ideal_xi[0].keys())
                writer.writeheader()
                writer.writerows(self.ideal_xi)
        
        avg_fit = sum(p.fit_score for p in self.squad_fit) / len(self.squad_fit) if self.squad_fit else 0
        avg_xi_fit = sum(p["fit_score"] for p in self.ideal_xi) / len(self.ideal_xi) if self.ideal_xi else 0
        
        summary = {
            "analysis_date": datetime.now().isoformat(),
            "target_manager": self.target_manager,
            "target_club": self.target_club,
            "tactical_archetype": self.target_cluster_name,
            "cluster": self.target_cluster,
            "squad_size": len(self.squad_fit),
            "average_fit_score": round(avg_fit, 1),
            "classification_breakdown": {
                "key_enablers": sum(1 for p in self.squad_fit if p.classification == "Key Enabler"),
                "good_fit": sum(1 for p in self.squad_fit if p.classification == "Good Fit"),
                "system_dependent": sum(1 for p in self.squad_fit if p.classification == "System Dependent"),
                "potentially_marginalised": sum(1 for p in self.squad_fit if p.classification == "Potentially Marginalised"),
            },
            "ideal_xi": self.ideal_xi,
            "ideal_xi_avg_fit": round(avg_xi_fit, 1)
        }
        
        with open(self.output_dir / "squad_fit_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        if verbose:
            print("\n" + "=" * 60)
            print("RESULTS SAVED")
            print("=" * 60)
            print(f"  â€¢ {self.output_dir}/squad_fit_scores.csv")
            print(f"  â€¢ {self.output_dir}/ideal_xi.csv")
            print(f"  â€¢ {self.output_dir}/squad_fit_summary.json")
        
        return self
    
    def get_results(self) -> Dict:
        """Return analysis results as dictionary."""
        avg_fit = sum(p.fit_score for p in self.squad_fit) / len(self.squad_fit) if self.squad_fit else 0
        
        return {
            "manager": self.target_manager,
            "club": self.target_club,
            "archetype": self.target_cluster_name,
            "squad_fit": [asdict(p) for p in self.squad_fit],
            "ideal_xi": self.ideal_xi,
            "average_fit": round(avg_fit, 1),
            "classification_counts": {
                cls: sum(1 for p in self.squad_fit if p.classification == cls)
                for cls in ["Key Enabler", "Good Fit", "System Dependent", "Potentially Marginalised"]
            }
        }


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

class AegisAnalyzer:
    """Legacy wrapper for backward compatibility."""
    
    THRESHOLDS = FIT_THRESHOLDS
    
    def __init__(self, processed_dir: Optional[Path] = None):
        self.processed_dir = Path(processed_dir) if processed_dir else Config.PROCESSED_DIR
        self.output_dir = Config.OUTPUT_DIR
        self.manager_dna = None
        self.squad_profiles = None
        self.dna_dimensions = None
        self.squad_fit = []
        self.ideal_xi = []
        self.recruitment = []
    
    def run(self) -> Dict:
        print("\n" + "=" * 50)
        print("AEGIS ANALYSIS ENGINE (Legacy Mode)")
        print("=" * 50)
        print("Note: For ML-based analysis, use SquadFitAnalyzer")
        
        self.load_data()
        self.calculate_dna_dimensions()
        self.calculate_squad_fit()
        self.generate_ideal_xi()
        self.analyse_recruitment()
        self.save_results()
        
        return {
            "manager": self.manager_dna.get("manager", "Unknown") if self.manager_dna else "Unknown",
            "dna_dimensions": self.dna_dimensions,
            "squad_fit": self.squad_fit,
            "ideal_xi": self.ideal_xi,
            "recruitment": self.recruitment
        }
    
    def load_data(self):
        print("\n[1/5] Loading processed data...")
        
        dna_file = self.processed_dir / "manager_dna.json"
        if dna_file.exists():
            with open(dna_file) as f:
                self.manager_dna = json.load(f)
            print(f"      âœ“ Manager: {self.manager_dna.get('manager', 'Unknown')}")
        
        squad_file = self.processed_dir / "squad_profiles.csv"
        if squad_file.exists():
            with open(squad_file) as f:
                reader = csv.DictReader(f)
                self.squad_profiles = list(reader)
            print(f"      âœ“ Squad: {len(self.squad_profiles)} players")
        
        return self
    
    def calculate_dna_dimensions(self):
        print("\n[2/5] Calculating DNA dimensions...")
        
        if not self.manager_dna:
            self.dna_dimensions = {}
            return self
        
        tactical = self.manager_dna.get("tactical_profile", {})
        results = self.manager_dna.get("results_profile", {})
        formation = self.manager_dna.get("formation_profile", {})
        
        poss = tactical.get("possession", {}).get("avg", 50)
        press = tactical.get("pressing", {}).get("intensity", {}).get("avg", 20)
        pass_acc = tactical.get("build_up", {}).get("pass_accuracy", 80)
        shots = tactical.get("attacking", {}).get("shots_pg", 12)
        conceded = results.get("conceded_per_game", 1.5)
        ppg = results.get("points_per_game", 1.5)
        flexibility = formation.get("flexibility_score", 3)
        
        self.dna_dimensions = {
            "Possession": self._clamp((poss - 30) * 2.5),
            "Pressing": self._clamp((press - 15) * 4),
            "Build-up": self._clamp(pass_acc),
            "Attacking Threat": self._clamp(shots * 6),
            "Defensive Solidity": self._clamp(100 - conceded * 30),
            "Set Piece": 50,
            "Flexibility": self._clamp(flexibility * 20),
            "Results": self._clamp(ppg * 33)
        }
        
        print("      âœ“ Dimensions calculated")
        return self
    
    def _clamp(self, value: float, min_val: float = 0, max_val: float = 100) -> float:
        return round(max(min_val, min(max_val, value)), 1)
    
    def calculate_squad_fit(self):
        print("\n[3/5] Calculating squad fit scores...")
        if not self.squad_profiles:
            return self
        self.squad_fit = []
        return self
    
    def generate_ideal_xi(self):
        print("\n[4/5] Generating ideal XI...")
        self.ideal_xi = []
        return self
    
    def analyse_recruitment(self):
        print("\n[5/5] Analysing recruitment needs...")
        self.recruitment = []
        return self
    
    def save_results(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n      Results saved to: {self.output_dir}")
        return self
