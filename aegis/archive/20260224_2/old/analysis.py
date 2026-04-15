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
                # Get team statistics
                team_stats = self.client.get_team_statistics(
                    team_id=tenure["team_id"],
                    season_id=self.season_id
                )
                
                if not team_stats:
                    if verbose:
                        print(f"  âœ— {tenure['coach_name']}: No stats")
                    continue
                
                # Parse statistics
                features = self._parse_team_stats(team_stats)
                
                if features:
                    features["coach_id"] = tenure["coach_id"]
                    features["coach_name"] = tenure["coach_name"]
                    features["team_id"] = tenure["team_id"]
                    features["team_name"] = tenure["team_name"]
                    features["league"] = tenure.get("league", "Unknown")
                    self.manager_features.append(features)
                    
                    if verbose:
                        print(f"  âœ“ {tenure['coach_name']}: {len(MANAGER_DNA_FEATURES)} features")
                else:
                    if verbose:
                        print(f"  âœ— {tenure['coach_name']}: Could not parse stats")
                        
            except Exception as e:
                if verbose:
                    print(f"  âœ— {tenure['coach_name']}: Error - {e}")
        
        if verbose:
            print("-" * 60)
            print(f"Managers with features: {len(self.manager_features)}")
        
        return self
    
    def _parse_team_stats(self, team_data: Dict, verbose: bool = False) -> Optional[Dict]:
        """
        Parse team statistics into feature dictionary.

        Sportsmonks API codes can differ across plan tiers and API versions.
        Each feature is looked up via an ordered list of known aliases so that
        a code rename or typo in any one alias does not silently zero-out the
        whole feature.  When a stat falls back to its default value a warning
        is emitted so the issue is visible rather than silent.
        """
        stats = team_data.get("statistics", [])
        if not stats:
            return None

        # Most-recent season is the first element
        season_stats = stats[0] if stats else {}
        details = season_stats.get("details", [])

        # Build lookup: code → raw value dict
        parsed: Dict[str, Any] = {}
        for detail in details:
            type_info = detail.get("type", {})
            if isinstance(type_info, str):
                code = type_info
            elif isinstance(type_info, dict):
                code = type_info.get("code", "")
            else:
                code = ""

            value = detail.get("value", {})
            if not isinstance(value, dict):
                value = {}
            if code:
                parsed[code] = value

        def _get_stat(aliases: List[str], path: List[str], default):
            """
            Try each alias in order, walk the returned dict along `path`,
            and return the first non-None value found.
            e.g. _get_stat(["ball-possession","possession"], ["all","average"], 50)
            """
            for alias in aliases:
                node = parsed.get(alias)
                if node is None:
                    continue
                for key in path:
                    if not isinstance(node, dict):
                        node = None
                        break
                    node = node.get(key)
                if node is not None:
                    return node
            return default

        features: Dict[str, Any] = {}
        defaults_used: List[str] = []

        # ── Possession ────────────────────────────────────────────────────────
        val = _get_stat(["ball-possession", "possession", "ball_possession"],
                        ["all", "average"], None)
        if val is not None:
            features["possession"] = float(val)
        else:
            features["possession"] = 50.0
            defaults_used.append("possession")

        # ── Pass accuracy ─────────────────────────────────────────────────────
        val = _get_stat(
            ["passes-percentage", "passes-accuracy", "pass-accuracy",
             "pass_accuracy", "passes_percentage"],
            ["all", "average"], None
        )
        if val is not None:
            features["pass_accuracy"] = float(val)
        else:
            features["pass_accuracy"] = 80.0
            defaults_used.append("pass_accuracy")

        # ── Shots ─────────────────────────────────────────────────────────────
        val = _get_stat(
            ["shots-total", "shots", "total-shots", "shots_total"],
            ["all", "average"], None
        )
        if val is not None:
            features["shots"] = float(val)
        else:
            features["shots"] = 12.0
            defaults_used.append("shots")

        # ── Goals scored ──────────────────────────────────────────────────────
        val = _get_stat(
            ["goals", "goals-scored", "goals_scored"],
            ["all", "average"], None
        )
        if val is not None:
            features["goals_scored"] = float(val)
        else:
            features["goals_scored"] = 1.5
            defaults_used.append("goals_scored")

        # ── Goals conceded ────────────────────────────────────────────────────
        # API has a known typo "goals-condeded" in some response versions
        val = _get_stat(
            ["goals-conceded", "goals-condeded", "goals_conceded",
             "conceded", "goals-against"],
            ["all", "average"], None
        )
        if val is not None:
            features["goals_conceded"] = float(val)
        else:
            features["goals_conceded"] = 1.2
            defaults_used.append("goals_conceded")

        # ── Tackles ───────────────────────────────────────────────────────────
        val = _get_stat(
            ["tackles", "tackles-total", "total-tackles"],
            ["all", "average"], None
        )
        if val is not None:
            features["tackles"] = float(val)
        else:
            features["tackles"] = 15.0
            defaults_used.append("tackles")

        # ── Interceptions ─────────────────────────────────────────────────────
        val = _get_stat(
            ["interceptions", "interceptions-total"],
            ["all", "average"], None
        )
        if val is not None:
            features["interceptions"] = float(val)
        else:
            features["interceptions"] = 10.0
            defaults_used.append("interceptions")

        # ── Clean sheet % ─────────────────────────────────────────────────────
        val = _get_stat(
            ["cleansheets", "clean-sheets", "clean_sheets", "cleansheet"],
            ["all", "percentage"], None
        )
        if val is not None:
            features["clean_sheet_pct"] = float(val)
        else:
            features["clean_sheet_pct"] = 30.0
            defaults_used.append("clean_sheet_pct")

        # ── Win rate ──────────────────────────────────────────────────────────
        val = _get_stat(
            ["wins", "win", "wins-total"],
            ["all", "percentage"], None
        )
        if val is not None:
            features["win_rate"] = float(val)
        else:
            features["win_rate"] = 33.0
            defaults_used.append("win_rate")

        # ── Diagnostic warning ────────────────────────────────────────────────
        if defaults_used:
            available_codes = sorted(parsed.keys())
            print(f"  ⚠  Stats using defaults (codes not found in API response): "
                  f"{defaults_used}")
            if verbose:
                print(f"     Available codes: {available_codes}")

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
        
        # Create DataFrame
        self.df_managers = pd.DataFrame(self.manager_features)
        
        # Extract feature matrix
        X = self.df_managers[MANAGER_DNA_FEATURES].values
        
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
        self.df_centroids = pd.DataFrame(centroids, columns=MANAGER_DNA_FEATURES)
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
        Auto-name clusters using the six canonical ARCHETYPE_WEIGHTS keys.

        Every cluster name produced here MUST be a key in ARCHETYPE_WEIGHTS so
        that _calculate_weighted_fit() can look up the correct feature weights.
        The six valid names are:
            "Possession-Based", "High-Press", "Counter-Attack",
            "Defensive", "Attacking", "Balanced"
        """
        import pandas as pd

        overall_mean = self.df_managers[MANAGER_DNA_FEATURES].mean()
        overall_std  = self.df_managers[MANAGER_DNA_FEATURES].std()

        # Features where a LOWER value signals the archetype
        # (z-score is inverted so "defensively solid" shows as positive)
        negative_traits = ["goals_conceded"]

        used_names = set()

        for c in range(self.n_clusters):
            centroid = self.df_centroids.loc[c]

            # z-score each feature against the training population
            z_scores = {}
            for feat in MANAGER_DNA_FEATURES:
                if overall_std[feat] > 0:
                    z = (centroid[feat] - overall_mean[feat]) / overall_std[feat]
                    if feat in negative_traits:
                        z = -z          # low conceded → positive z
                    z_scores[feat] = z
                else:
                    z_scores[feat] = 0

            # Rank features by how distinctive they are for this cluster
            sorted_feats = sorted(z_scores.items(), key=lambda x: abs(x[1]), reverse=True)
            top_feat, top_z   = sorted_feats[0]
            sec_feat, sec_z   = sorted_feats[1]

            # ----------------------------------------------------------------
            # Map feature signals → canonical archetype names
            # ----------------------------------------------------------------
            STRONG = 0.5    # threshold for a "clear" signal
            WEAK   = 0.2    # threshold for a "moderate" signal

            if top_z >= STRONG:
                # Cluster is distinctively HIGH on this feature
                if top_feat == "possession":
                    name = "Possession-Based"
                elif top_feat in ("tackles", "interceptions"):
                    name = "High-Press"
                elif top_feat in ("shots", "goals_scored"):
                    name = "Attacking"
                elif top_feat in ("goals_conceded", "clean_sheet_pct"):
                    name = "Defensive"
                elif top_feat == "pass_accuracy":
                    # High pass accuracy with possession → Possession-Based
                    # Without possession context → Balanced
                    name = "Possession-Based" if z_scores.get("possession", 0) >= 0 else "Balanced"
                elif top_feat == "win_rate":
                    # High win-rate clusters could be Attacking or Balanced;
                    # check secondary signal for discrimination
                    if sec_z >= WEAK and sec_feat in ("shots", "goals_scored"):
                        name = "Attacking"
                    elif sec_z >= WEAK and sec_feat in ("tackles", "interceptions"):
                        name = "High-Press"
                    else:
                        name = "Balanced"
                else:
                    name = "Balanced"

            elif top_z <= -STRONG:
                # Cluster is distinctively LOW on this feature
                if top_feat == "possession":
                    name = "Counter-Attack"
                elif top_feat in ("tackles", "interceptions"):
                    # Low pressing intensity → sitting deep
                    name = "Defensive"
                elif top_feat in ("goals_conceded",):
                    # goals_conceded z was INVERTED, so a strong negative z here
                    # means LOTS of goals conceded → vulnerable, open style
                    name = "Attacking"
                elif top_feat == "clean_sheet_pct":
                    name = "Attacking"   # rarely keeps clean sheets
                elif top_feat in ("shots", "goals_scored"):
                    # Low attacking output → defensive / counter
                    name = "Defensive" if sec_z <= -WEAK else "Counter-Attack"
                elif top_feat == "win_rate":
                    name = "Balanced"   # low win-rate is circumstantial
                else:
                    name = "Balanced"

            else:
                # No strong signal – use moderate signals to break ties
                if abs(top_z) >= WEAK:
                    if top_z > 0:
                        if top_feat in ("possession", "pass_accuracy"):
                            name = "Possession-Based"
                        elif top_feat in ("tackles", "interceptions"):
                            name = "High-Press"
                        elif top_feat in ("shots", "goals_scored"):
                            name = "Attacking"
                        elif top_feat in ("goals_conceded", "clean_sheet_pct"):
                            name = "Defensive"
                        else:
                            name = "Balanced"
                    else:
                        if top_feat == "possession":
                            name = "Counter-Attack"
                        elif top_feat in ("tackles", "interceptions"):
                            name = "Defensive"
                        else:
                            name = "Balanced"
                else:
                    name = "Balanced"

            # ----------------------------------------------------------------
            # Deduplicate: when two clusters get the same canonical name,
            # use the secondary signal to pick the nearest alternative.
            # ----------------------------------------------------------------
            if name in used_names:
                # Ordered preference list excluding already-used name
                all_archetypes = [
                    "Possession-Based", "High-Press", "Counter-Attack",
                    "Attacking", "Defensive", "Balanced"
                ]
                # Build a score for each archetype based on z-score alignment
                archetype_signals = {
                    "Possession-Based": z_scores.get("possession", 0) + z_scores.get("pass_accuracy", 0),
                    "High-Press":       z_scores.get("tackles", 0)    + z_scores.get("interceptions", 0),
                    "Counter-Attack":   -z_scores.get("possession", 0) + z_scores.get("goals_scored", 0),
                    "Attacking":        z_scores.get("shots", 0)       + z_scores.get("goals_scored", 0),
                    "Defensive":        z_scores.get("goals_conceded", 0) + z_scores.get("clean_sheet_pct", 0),
                    "Balanced":         0
                }
                ranked = sorted(
                    [a for a in all_archetypes if a not in used_names],
                    key=lambda a: archetype_signals[a],
                    reverse=True
                )
                name = ranked[0] if ranked else "Balanced"

            used_names.add(name)
            self.cluster_names[c] = name

            if verbose:
                print(f"  Cluster {c}: {name}  "
                      f"(top signal: {top_feat} z={top_z:+.2f})")

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
            "feature_cols": MANAGER_DNA_FEATURES,
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
            "feature_cols": MANAGER_DNA_FEATURES,
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
        """Set the target manager for analysis."""
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
        
        if verbose:
            print(f"\nâœ“ Target Manager: {self.target_manager}")
            print(f"  Tactical Archetype: {self.target_cluster_name}")
            
            same_cluster = self.df_managers[self.df_managers["cluster"] == self.target_cluster]
            others = [n for n in same_cluster["coach_name"].tolist() if n != self.target_manager]
            if others:
                print(f"  Similar Managers: {', '.join(others)}")
        
        return self
    
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
        """
        Calculate fit score using weighted comparison to ideal profile.

        Looks up archetype weights by cluster name.  The training model now
        guarantees cluster names are canonical ARCHETYPE_WEIGHTS keys, but a
        safety mapping is kept here so that any model trained on an older
        codebase also works without re-training.
        """
        # Safety mapping: old / non-canonical names → canonical ARCHETYPE_WEIGHTS key
        _ARCHETYPE_NORMALISE = {
            # Possession family
            "Technical":        "Possession-Based",
            "Possession":       "Possession-Based",
            # High-press family
            "Pressing":         "High-Press",
            "Intercepting":     "High-Press",
            # Counter-attack family
            "Direct":           "Counter-Attack",
            "Low-Scoring":      "Counter-Attack",
            "Clinical":         "Counter-Attack",
            # Attacking family
            "Shot-Heavy":       "Attacking",
            "High-Scoring":     "Attacking",
            "Winning":          "Attacking",
            "Results-Driven":   "Attacking",
            "Expansive":        "Attacking",
            "High-Risk":        "Attacking",
            # Defensive family
            "Low-Block":        "Defensive",
            "Passive":          "Defensive",
            "Reactive":         "Defensive",
            "Physical":         "Defensive",
            # Balanced / catch-all
            "Pragmatic":        "Balanced",
            "Struggling":       "Balanced",
        }

        canonical = _ARCHETYPE_NORMALISE.get(self.target_cluster_name, self.target_cluster_name)
        ideal   = IDEAL_PLAYER_PROFILES.get(position_group, IDEAL_PLAYER_PROFILES["MID"])
        weights = ARCHETYPE_WEIGHTS.get(canonical, {})

        total_score = 0
        total_weight = 0

        for feat in PLAYER_FIT_FEATURES:
            player_val = player_stats.get(feat, 0)
            ideal_val  = ideal.get(feat, 0)
            weight     = weights.get(feat, 1.0)

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

            total_score  += feature_score
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
