"""
Aegis ETL Pipeline
==================
Transform raw API data into analysis-ready structures.

Supports two data modes:
- AGGREGATED (default): Uses pre-aggregated team/season statistics
- MATCH_LEVEL: Uses fixture-by-fixture data for custom analysis
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime

from .config import Config


class ETLPipeline:
    """
    ETL Pipeline for transforming Sportsmonks data.
    
    Usage:
        from aegis import ETLPipeline
        
        # Run the full pipeline
        etl = ETLPipeline()
        manager_dna, squad_profiles = etl.run()
        
        # Or step by step
        etl = ETLPipeline()
        etl.load_raw_data()
        etl.extract_manager_dna()
        etl.extract_squad_profiles()
        etl.save()
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize ETL pipeline.
        
        Args:
            data_dir: Directory containing raw JSON files.
                     Defaults to Config.DATA_DIR
        """
        self.data_dir = Path(data_dir) if data_dir else Config.DATA_DIR
        self.output_dir = Config.PROCESSED_DIR
        
        # Data containers
        self.coach = None
        self.manager_team_stats = None
        self.fixtures = None
        self.season_stats = None
        self.team = None
        self.squad = None
        self.metadata = None
        
        # Processed outputs
        self.manager_dna = None
        self.squad_profiles = None
        self.league_context = None
    
    def run(self) -> Tuple[Dict, List]:
        """
        Run full ETL pipeline.
        
        Returns:
            Tuple of (manager_dna dict, squad_profiles list)
        """
        print("\n" + "=" * 50)
        print("AEGIS ETL PIPELINE")
        print("=" * 50)
        
        self.load_raw_data()
        self.extract_league_context()
        self.extract_manager_dna()
        self.extract_squad_profiles()
        self.save()
        
        print("\n" + "=" * 50)
        print("✓ ETL COMPLETE")
        print("=" * 50)
        
        return self.manager_dna, self.squad_profiles
    
    # =========================================================================
    # LOAD RAW DATA
    # =========================================================================
    
    def load_raw_data(self):
        """Load raw JSON files from data directory."""
        print("\n[1/5] Loading raw data...")
        
        # Load metadata to determine fetch type
        metadata_file = self.data_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                self.metadata = json.load(f)
            fetch_type = self.metadata.get("fetch_type", "unknown")
            print(f"      Data mode: {fetch_type}")
        else:
            self.metadata = {"fetch_type": "match_level"}  # Legacy default
        
        # Load coach
        coach_file = self.data_dir / "coach.json"
        if coach_file.exists():
            with open(coach_file) as f:
                self.coach = json.load(f)
            print(f"      ✓ Coach: {self.coach.get('common_name', 'Unknown')}")
        
        # Load manager team stats (aggregated mode)
        stats_file = self.data_dir / "manager_team_stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                self.manager_team_stats = json.load(f)
            print(f"      ✓ Manager team statistics (aggregated)")
        
        # Load fixtures (match-level mode)
        fixtures_file = self.data_dir / "fixtures.json"
        if fixtures_file.exists():
            with open(fixtures_file) as f:
                self.fixtures = json.load(f)
            if self.fixtures:
                print(f"      ✓ Fixtures: {len(self.fixtures)}")
        
        # Load season stats (league context)
        season_file = self.data_dir / "season_stats.json"
        if season_file.exists():
            with open(season_file) as f:
                self.season_stats = json.load(f)
            print(f"      ✓ Season statistics (league context)")
        
        # Load team
        team_file = self.data_dir / "team.json"
        if team_file.exists():
            with open(team_file) as f:
                self.team = json.load(f)
            print(f"      ✓ Team: {self.team.get('name', 'Unknown')}")
        
        # Load squad
        squad_file = self.data_dir / "squad.json"
        if squad_file.exists():
            with open(squad_file) as f:
                self.squad = json.load(f)
            print(f"      ✓ Squad: {len(self.squad)} players")
        
        return self
    
    # =========================================================================
    # EXTRACT LEAGUE CONTEXT
    # =========================================================================
    
    def extract_league_context(self):
        """Extract league-wide benchmarks from season statistics."""
        print("\n[2/5] Extracting league context...")
        
        if not self.season_stats:
            print("      ⚠ No season statistics available")
            self.league_context = self._default_league_context()
            return self
        
        stats = self.season_stats.get("statistics", [])
        
        context = {
            "season_name": self.season_stats.get("name", "Unknown"),
            "league": self.season_stats.get("league", {}).get("name", "Unknown"),
            "avg_goals_per_game": 2.5,  # default
            "avg_possession_home": 52,
            "avg_possession_away": 48,
        }
        
        # Parse season statistics
        for stat in stats:
            type_info = stat.get("type", {})
            code = type_info.get("code", "")
            value = stat.get("value", {})
            
            if code == "number-of-goals":
                context["avg_goals_per_game"] = value.get("average", 2.5)
                context["total_goals"] = value.get("total", 0)
                context["home_goal_pct"] = value.get("home", {}).get("percentage", 58)
            
            elif code == "matches":
                context["total_matches"] = value.get("total", 0)
                context["matches_played"] = value.get("played", 0)
            
            elif code == "cards":
                context["avg_yellows"] = value.get("average_yellowcards", 4)
                context["avg_reds"] = value.get("average_redcards", 0.2)
        
        self.league_context = context
        print(f"      ✓ League: {context['league']}")
        print(f"      ✓ Avg goals/game: {context['avg_goals_per_game']}")
        
        return self
    
    def _default_league_context(self) -> Dict:
        """Return sensible defaults when no season data available."""
        return {
            "season_name": "Unknown",
            "league": "Unknown",
            "avg_goals_per_game": 2.5,
            "avg_possession_home": 52,
            "total_matches": 380,
        }
    
    # =========================================================================
    # EXTRACT MANAGER DNA
    # =========================================================================
    
    def extract_manager_dna(self):
        """Extract manager tactical profile from statistics."""
        print("\n[3/5] Extracting manager DNA...")
        
        # Check which data mode we're in
        if self.manager_team_stats:
            self._extract_dna_from_aggregated()
        elif self.fixtures:
            self._extract_dna_from_fixtures()
        else:
            print("      ⚠ No manager data available")
            self.manager_dna = self._default_manager_dna()
        
        return self
    
    def _extract_dna_from_aggregated(self):
        """Extract DNA from pre-aggregated team statistics."""
        print("      Mode: Aggregated statistics")
        
        stats = self.manager_team_stats.get("statistics", [])
        if not stats:
            self.manager_dna = self._default_manager_dna()
            return
        
        # Find the current season stats (first entry is usually most recent)
        season_stats = stats[0] if stats else {}
        details = season_stats.get("details", [])
        
        # Parse statistics
        parsed = {}
        for detail in details:
            type_info = detail.get("type", {})
            code = type_info.get("code", "")
            value = detail.get("value", {})
            parsed[code] = value
        
        # Extract key metrics with home/away splits
        wins = parsed.get("wins", {})
        goals = parsed.get("goals", {})
        goals_conceded = parsed.get("goals-condeded", {})  # Note: API typo
        clean_sheets = parsed.get("cleansheets", {})
        
        total_matches = wins.get("all", {}).get("count", 0) + \
                       parsed.get("draws", {}).get("all", {}).get("count", 0) + \
                       parsed.get("lost", {}).get("all", {}).get("count", 0)
        
        if total_matches == 0:
            total_matches = 1  # Avoid division by zero
        
        # Calculate tactical metrics
        win_count = wins.get("all", {}).get("count", 0)
        draw_count = parsed.get("draws", {}).get("all", {}).get("count", 0)
        
        self.manager_dna = {
            "manager": self.coach.get("common_name", "Unknown") if self.coach else "Unknown",
            "team": self.manager_team_stats.get("name", "Unknown"),
            "matches_analysed": total_matches,
            "data_mode": "aggregated",
            
            "formation_profile": {
                "primary": "Unknown",  # Not available in aggregated stats
                "flexibility_score": 3  # Default
            },
            
            "results_profile": {
                "wins": win_count,
                "draws": draw_count,
                "losses": parsed.get("lost", {}).get("all", {}).get("count", 0),
                "win_rate": (win_count / total_matches * 100) if total_matches else 0,
                "win_rate_home": wins.get("home", {}).get("percentage", 0),
                "win_rate_away": wins.get("away", {}).get("percentage", 0),
                "points_per_game": (win_count * 3 + draw_count) / total_matches,
                "goals_per_game": goals.get("all", {}).get("average", 0),
                "goals_per_game_home": goals.get("home", {}).get("average", 0),
                "goals_per_game_away": goals.get("away", {}).get("average", 0),
                "conceded_per_game": goals_conceded.get("all", {}).get("average", 0),
                "conceded_per_game_home": goals_conceded.get("home", {}).get("average", 0),
                "conceded_per_game_away": goals_conceded.get("away", {}).get("average", 0),
                "clean_sheet_pct": clean_sheets.get("all", {}).get("percentage", 0),
            },
            
            "tactical_profile": {
                "possession": {
                    "avg": parsed.get("ball-possession", {}).get("all", {}).get("average", 50),
                    "home": parsed.get("ball-possession", {}).get("home", {}).get("average", 52),
                    "away": parsed.get("ball-possession", {}).get("away", {}).get("average", 48),
                },
                "pressing": {
                    "intensity": {
                        "avg": parsed.get("tackles", {}).get("all", {}).get("average", 15) + 
                               parsed.get("interceptions", {}).get("all", {}).get("average", 10)
                    }
                },
                "build_up": {
                    "pass_accuracy": parsed.get("passes-percentage", {}).get("all", {}).get("average", 80)
                },
                "attacking": {
                    "shots_pg": parsed.get("shots-total", {}).get("all", {}).get("average", 12),
                    "shots_on_target_pg": parsed.get("shots-on-target", {}).get("all", {}).get("average", 4),
                    "dangerous_attacks": parsed.get("dangerous-attacks", {}).get("all", {}).get("average", 50)
                }
            }
        }
        
        print(f"      ✓ Win rate: {self.manager_dna['results_profile']['win_rate']:.1f}%")
        print(f"      ✓ Home: {self.manager_dna['results_profile']['win_rate_home']:.1f}% | Away: {self.manager_dna['results_profile']['win_rate_away']:.1f}%")
    
    def _extract_dna_from_fixtures(self):
        """Extract DNA from match-level fixture data (legacy mode)."""
        print("      Mode: Match-level fixtures")
        
        if not self.fixtures:
            self.manager_dna = self._default_manager_dna()
            return
        
        # Get manager's team ID
        team_id = self._get_manager_team_id()
        
        # Analyse formations
        formation_profile = self._analyse_formations(team_id)
        
        # Analyse results
        results_profile = self._analyse_results(team_id)
        
        # Analyse tactical metrics
        tactical_profile = self._analyse_tactics(team_id)
        
        self.manager_dna = {
            "manager": self.coach.get("common_name", "Unknown") if self.coach else "Unknown",
            "matches_analysed": len(self.fixtures),
            "data_mode": "match_level",
            "formation_profile": formation_profile,
            "results_profile": results_profile,
            "tactical_profile": tactical_profile
        }
        
        print(f"      ✓ Primary formation: {formation_profile['primary']}")
        print(f"      ✓ Win rate: {results_profile['win_rate']:.1f}%")
    
    def _default_manager_dna(self) -> Dict:
        """Return default manager DNA structure."""
        return {
            "manager": "Unknown",
            "matches_analysed": 0,
            "data_mode": "none",
            "formation_profile": {"primary": "4-4-2", "flexibility_score": 1},
            "results_profile": {"win_rate": 33, "points_per_game": 1.33, "conceded_per_game": 1.2},
            "tactical_profile": {
                "possession": {"avg": 50},
                "pressing": {"intensity": {"avg": 20}},
                "build_up": {"pass_accuracy": 80},
                "attacking": {"shots_pg": 12}
            }
        }
    
    def _get_manager_team_id(self) -> int:
        """Determine manager's team ID from fixtures or coach data."""
        if self.coach and self.coach.get("teams"):
            for team in self.coach["teams"]:
                if team.get("end") is None:
                    return team.get("team_id")
            return self.coach["teams"][0].get("team_id")
        
        # Fallback: most common team in fixtures
        if self.fixtures:
            team_counts = defaultdict(int)
            for fixture in self.fixtures:
                participants = fixture.get("participants", [])
                for p in participants:
                    team_counts[p.get("id")] += 1
            if team_counts:
                return max(team_counts, key=team_counts.get)
        
        return None
    
    def _analyse_formations(self, team_id: int) -> Dict:
        """Analyse formation usage patterns."""
        formations = defaultdict(int)
        
        for fixture in self.fixtures:
            formation_data = fixture.get("formations", [])
            if isinstance(formation_data, dict):
                formation_data = formation_data.get("data", [])
            
            for f in formation_data:
                if f.get("participant_id") == team_id or f.get("location") == "home":
                    formation = f.get("formation", "Unknown")
                    formations[formation] += 1
        
        total = sum(formations.values()) or 1
        formation_pcts = {k: round(v / total * 100, 1) for k, v in formations.items()}
        
        primary = max(formations, key=formations.get) if formations else "4-4-2"
        
        return {
            "primary": primary,
            "usage": dict(formations),
            "percentages": formation_pcts,
            "flexibility_score": len(formations)
        }
    
    def _analyse_results(self, team_id: int) -> Dict:
        """Analyse match results."""
        wins, draws, losses = 0, 0, 0
        goals_for, goals_against = 0, 0
        
        for fixture in self.fixtures:
            scores = fixture.get("scores", [])
            if isinstance(scores, dict):
                scores = scores.get("data", [])
            
            team_goals = 0
            opponent_goals = 0
            
            for score in scores:
                if score.get("description") == "CURRENT":
                    if score.get("participant_id") == team_id:
                        team_goals = score.get("score", {}).get("goals", 0)
                    else:
                        opponent_goals = score.get("score", {}).get("goals", 0)
            
            goals_for += team_goals
            goals_against += opponent_goals
            
            if team_goals > opponent_goals:
                wins += 1
            elif team_goals < opponent_goals:
                losses += 1
            else:
                draws += 1
        
        total = wins + draws + losses or 1
        
        return {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": wins / total * 100,
            "points_per_game": (wins * 3 + draws) / total,
            "goals_per_game": goals_for / total,
            "conceded_per_game": goals_against / total
        }
    
    def _analyse_tactics(self, team_id: int) -> Dict:
        """Analyse tactical metrics from match statistics."""
        metrics = defaultdict(list)
        
        for fixture in self.fixtures:
            stats = fixture.get("statistics", [])
            if isinstance(stats, dict):
                stats = stats.get("data", [])
            
            for stat in stats:
                if stat.get("participant_id") == team_id or stat.get("location") == "home":
                    type_info = stat.get("type", {})
                    code = type_info.get("code", "")
                    value = stat.get("data", {}).get("value", 0)
                    
                    if code:
                        metrics[code].append(value)
        
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0
        
        possession = avg(metrics.get("ball-possession", [0]))
        tackles = avg(metrics.get("tackles", [0]))
        interceptions = avg(metrics.get("interceptions", [0]))
        pass_acc = avg(metrics.get("passes-percentage", [0]))
        shots = avg(metrics.get("shots-total", [0]))
        shots_ot = avg(metrics.get("shots-on-target", [0]))
        
        return {
            "possession": {"avg": round(possession, 1)},
            "pressing": {"intensity": {"avg": round(tackles + interceptions, 1)}},
            "build_up": {"pass_accuracy": round(pass_acc, 1)},
            "attacking": {
                "shots_pg": round(shots, 1),
                "shots_on_target_pg": round(shots_ot, 1)
            }
        }
    
    # =========================================================================
    # EXTRACT SQUAD PROFILES
    # =========================================================================
    
    def extract_squad_profiles(self):
        """Extract player profiles from squad data."""
        print("\n[4/5] Extracting squad profiles...")
        
        if not self.squad:
            print("      ⚠ No squad loaded")
            return self
        
        profiles = []
        
        for entry in self.squad:
            player = entry.get("player", {})
            if not player:
                continue
            
            # Basic info
            profile = {
                "id": player.get("id"),
                "name": player.get("common_name") or player.get("name", "Unknown"),
                "age": self._calculate_age(player.get("date_of_birth")),
                "position": self._extract_position(player),
                "detailed_position": self._extract_detailed_position(player),
                "jersey_number": entry.get("jersey_number"),
            }
            
            # Extract statistics
            stats = self._extract_player_stats(player)
            profile.update(stats)
            
            profiles.append(profile)
        
        self.squad_profiles = profiles
        print(f"      ✓ Processed {len(profiles)} players")
        
        return self
    
    def _calculate_age(self, dob: str) -> int:
        """Calculate age from date of birth string."""
        if not dob:
            return 25
        try:
            birth = datetime.strptime(dob[:10], "%Y-%m-%d")
            today = datetime.now()
            return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        except:
            return 25
    
    def _extract_position(self, player: Dict) -> str:
        """Extract position name from player data."""
        pos = player.get("position", {})
        if isinstance(pos, dict):
            return pos.get("name", "Unknown")
        return "Unknown"
    
    def _extract_detailed_position(self, player: Dict) -> str:
        """Extract detailed position from player data."""
        pos = player.get("detailedPosition", {})
        if isinstance(pos, dict):
            return pos.get("name", "Unknown")
        return self._extract_position(player)
    
    def _extract_player_stats(self, player: Dict) -> Dict:
        """Extract season statistics for a player."""
        stats = {
            "appearances": 0,
            "minutes": 0,
            "goals": 0,
            "assists": 0,
            "clean_sheets": 0,
            "saves": 0,
            "tackles": 0,
            "interceptions": 0,
            "clearances": 0,
            "pass_accuracy": 0,
            "key_passes": 0,
            "dribbles": 0,
            "shots": 0,
            "shots_on_target": 0,
            "rating": 0
        }
        
        statistics = player.get("statistics", [])
        if not statistics:
            return stats
        
        # Get most recent season stats
        season_stats = statistics[0] if statistics else {}
        details = season_stats.get("details", [])
        
        # Map type codes to our stat names
        code_mapping = {
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
            
            # Handle different value structures
            if isinstance(value, dict):
                value = value.get("total", value.get("count", value.get("value", 0)))
            
            if code in code_mapping:
                stats[code_mapping[code]] = value
        
        return stats
    
    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    
    def save(self):
        """Save processed data to files."""
        print("\n[5/5] Saving outputs...")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save manager DNA
        if self.manager_dna:
            with open(self.output_dir / "manager_dna.json", "w") as f:
                json.dump(self.manager_dna, f, indent=2)
            print(f"      ✓ manager_dna.json")
        
        # Save league context
        if self.league_context:
            with open(self.output_dir / "league_context.json", "w") as f:
                json.dump(self.league_context, f, indent=2)
            print(f"      ✓ league_context.json")
        
        # Save squad profiles as CSV
        if self.squad_profiles:
            csv_path = self.output_dir / "squad_profiles.csv"
            with open(csv_path, "w", newline="") as f:
                if self.squad_profiles:
                    writer = csv.DictWriter(f, fieldnames=self.squad_profiles[0].keys())
                    writer.writeheader()
                    writer.writerows(self.squad_profiles)
            print(f"      ✓ squad_profiles.csv")
        
        print(f"\n      Output directory: {self.output_dir}")
        
        return self
