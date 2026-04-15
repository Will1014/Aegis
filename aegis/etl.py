"""
Aegis ETL Pipeline
==================
Transform raw API data into analysis-ready structures.

Supports three data modes:
- AGGREGATED (default Sportsmonks): Uses pre-aggregated team/season statistics
- MATCH_LEVEL (Sportsmonks): Uses fixture-by-fixture data for custom analysis
- STATSBOMB: Uses StatsBomb team match stats + player season stats

Both Sportsmonks (ETLPipeline) and StatsBomb (StatsBombETL) pipelines
produce the same output format: manager_dna dict + squad_profiles list.
This means analysis.py and visualizations.py work identically with either source.
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
        print("âœ“ ETL COMPLETE")
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
            print(f"      âœ“ Coach: {self.coach.get('common_name', 'Unknown')}")
        
        # Load manager team stats (aggregated mode)
        stats_file = self.data_dir / "manager_team_stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                self.manager_team_stats = json.load(f)
            print(f"      âœ“ Manager team statistics (aggregated)")
        
        # Load fixtures (match-level mode)
        fixtures_file = self.data_dir / "fixtures.json"
        if fixtures_file.exists():
            with open(fixtures_file) as f:
                self.fixtures = json.load(f)
            if self.fixtures:
                print(f"      âœ“ Fixtures: {len(self.fixtures)}")
        
        # Load season stats (league context)
        season_file = self.data_dir / "season_stats.json"
        if season_file.exists():
            with open(season_file) as f:
                self.season_stats = json.load(f)
            print(f"      âœ“ Season statistics (league context)")
        
        # Load team
        team_file = self.data_dir / "team.json"
        if team_file.exists():
            with open(team_file) as f:
                self.team = json.load(f)
            print(f"      âœ“ Team: {self.team.get('name', 'Unknown')}")
        
        # Load squad
        squad_file = self.data_dir / "squad.json"
        if squad_file.exists():
            with open(squad_file) as f:
                self.squad = json.load(f)
            print(f"      âœ“ Squad: {len(self.squad)} players")
        
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
        
        # Handle league being either a dict or string
        league_data = self.season_stats.get("league", {})
        if isinstance(league_data, dict):
            league_name = league_data.get("name", "Unknown")
        elif isinstance(league_data, str):
            league_name = league_data
        else:
            league_name = "Unknown"
        
        context = {
            "season_name": self.season_stats.get("name", "Unknown"),
            "league": league_name,
            "avg_goals_per_game": 2.5,  # default
            "avg_possession_home": 52,
            "avg_possession_away": 48,
        }
        
        # Parse season statistics
        for stat in stats:
            # Handle type being either a dict or a string
            type_info = stat.get("type", {})
            if isinstance(type_info, str):
                code = type_info
            elif isinstance(type_info, dict):
                code = type_info.get("code", "")
            else:
                code = ""
            
            value = stat.get("value", {})
            if not isinstance(value, dict):
                value = {}
            
            if code == "number-of-goals":
                context["avg_goals_per_game"] = value.get("average", 2.5)
                context["total_goals"] = value.get("total", 0)
                home_data = value.get("home", {})
                if isinstance(home_data, dict):
                    context["home_goal_pct"] = home_data.get("percentage", 58)
                else:
                    context["home_goal_pct"] = 58
            
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
            print("      âš  No manager data available")
            self.manager_dna = self._default_manager_dna()
        
        return self
    
    def _extract_dna_from_aggregated(self):
        """
        Extract DNA from pre-aggregated team statistics.

        All codes and value paths confirmed by API diagnostic (Feb 2026).
        include=statistics.details.type must be used so that type is a dict
        with a "code" key rather than a bare integer.

        Confirmed codes and value structures:
          "team-wins"          {"all": {"count": X, "percentage": X}, "home": {...}}
          "team-draws"         {"all": {"count": X, "percentage": X}, ...}
          "team-lost"          {"all": {"count": X, "percentage": X}, ...}
          "goals"              {"all": {"count": X, "average": X}, "home": {...}}
          "goals-conceded"     {"all": {"count": X, "average": X}, "home": {...}}
          "cleansheets"        {"all": {"count": X, "percentage": X}, ...}
          "ball-possession"    {"average": X}           FLAT (no "all" wrapper)
          "shots"              {"average": X, "on_target": X, ...}  FLAT
          "tackles"            {"average": X, ...}      FLAT
          "interception-stats" {"interceptions_per_game": X}  FLAT, different key name
          "dangerous-attacks"  {"average": X, ...}      FLAT
          pass_accuracy        NOT IN TEAM STATS — read from self.fixtures if available
        """
        print("      Mode: Aggregated statistics")

        stats = self.manager_team_stats.get("statistics", [])
        if not stats:
            self.manager_dna = self._default_manager_dna()
            return

        season_stats = stats[0] if stats else {}
        details = season_stats.get("details", [])

        # Build lookup: code -> value dict
        # Only works when include=statistics.details.type (type is a dict with "code")
        parsed = {}
        for detail in details:
            type_info = detail.get("type", {})
            if isinstance(type_info, dict):
                code = type_info.get("code", "")
            elif isinstance(type_info, str):
                code = type_info
            else:
                code = ""
            value = detail.get("value", {})
            if not isinstance(value, dict):
                value = {}
            if code:
                parsed[code] = value

        # Results (nested {"all": {"count": X, "percentage": X}} structure)
        wins       = parsed.get("team-wins",  {})   # was "wins"  — wrong code
        draws      = parsed.get("team-draws", {})   # was "draws" — wrong code
        losses_d   = parsed.get("team-lost",  {})   # was "lost"  — wrong code
        goals      = parsed.get("goals",       {})
        goals_conc = parsed.get("goals-conceded", {})  # was "goals-condeded" (typo)
        clean_sh   = parsed.get("cleansheets", {})

        win_count  = wins.get("all",  {}).get("count", 0)
        draw_count = draws.get("all", {}).get("count", 0)
        loss_count = losses_d.get("all", {}).get("count", 0)
        total_matches = (win_count + draw_count + loss_count) or 1

        # Tactical — flat structure (no "all" wrapper)
        # possession: {"average": X}  (was wrongly reading ["all"]["average"])
        possession_avg  = parsed.get("ball-possession", {}).get("average",      50.0)
        possession_home = parsed.get("ball-possession", {}).get("home_average", 52.0)
        possession_away = parsed.get("ball-possession", {}).get("away_average", 48.0)

        # shots: {"average": X, "on_target": X, ...}  (was wrong code "shots-total")
        shots_avg       = parsed.get("shots", {}).get("average",   12.0)
        shots_on_tgt    = parsed.get("shots", {}).get("on_target",  0)
        shots_on_tgt_pg = round(shots_on_tgt / total_matches, 1) if shots_on_tgt else 4.0

        # tackles: {"average": X}  (was wrongly reading ["all"]["average"])
        tackles_avg = parsed.get("tackles", {}).get("average", 15.0)

        # interceptions: different code AND different key name
        # was: "interceptions" -> ["all"]["average"]
        # now: "interception-stats" -> ["interceptions_per_game"]
        interceptions_pg = parsed.get("interception-stats", {}).get("interceptions_per_game", 10.0)

        # dangerous attacks: flat {"average": X}  (was wrongly reading ["all"]["average"])
        dangerous_attacks = parsed.get("dangerous-attacks", {}).get("average", 50.0)

        # pass_accuracy — not available in team stats endpoint at all
        # Read from self.fixtures if available (fixture code: "successful-passes-percentage")
        pass_accuracy = self._get_pass_accuracy_from_fixtures()

        self.manager_dna = {
            "manager": self.coach.get("common_name", "Unknown") if self.coach else "Unknown",
            "team": self.manager_team_stats.get("name", "Unknown"),
            "matches_analysed": total_matches,
            "data_mode": "aggregated",

            "formation_profile": {
                "primary": "Unknown",
                "flexibility_score": 3
            },

            "results_profile": {
                "wins":                win_count,
                "draws":               draw_count,
                "losses":              loss_count,
                "win_rate":            wins.get("all",  {}).get("percentage", 0),
                "win_rate_home":       wins.get("home", {}).get("percentage", 0),
                "win_rate_away":       wins.get("away", {}).get("percentage", 0),
                "points_per_game":     (win_count * 3 + draw_count) / total_matches,
                "goals_per_game":      goals.get("all",  {}).get("average", 0),
                "goals_per_game_home": goals.get("home", {}).get("average", 0),
                "goals_per_game_away": goals.get("away", {}).get("average", 0),
                "conceded_per_game":      goals_conc.get("all",  {}).get("average", 0),
                "conceded_per_game_home": goals_conc.get("home", {}).get("average", 0),
                "conceded_per_game_away": goals_conc.get("away", {}).get("average", 0),
                "clean_sheet_pct":     clean_sh.get("all", {}).get("percentage", 0),
            },

            "tactical_profile": {
                "possession": {
                    "avg":  possession_avg,
                    "home": possession_home,
                    "away": possession_away,
                },
                "pressing": {
                    "intensity": {"avg": round(tackles_avg + interceptions_pg, 1)}
                },
                "build_up": {
                    "pass_accuracy": pass_accuracy
                },
                "attacking": {
                    "shots_pg":          shots_avg,
                    "shots_on_target_pg": shots_on_tgt_pg,
                    "dangerous_attacks": dangerous_attacks,
                }
            }
        }

        print(f"      ✓ Win rate: {self.manager_dna['results_profile']['win_rate']:.1f}%")
        print(f"      ✓ Home: {self.manager_dna['results_profile']['win_rate_home']:.1f}% | Away: {self.manager_dna['results_profile']['win_rate_away']:.1f}%")
        print(f"      ✓ Possession: {possession_avg:.1f}%  Pass acc: {pass_accuracy:.1f}%  Shots/g: {shots_avg:.1f}")

    def _get_pass_accuracy_from_fixtures(self, default: float = 80.0) -> float:
        """
        Average pass accuracy from self.fixtures.

        Pass accuracy is not available in the team stats endpoint. At fixture
        level it is available as "successful-passes-percentage" at
        stat["data"]["value"], filtered by stat["participant_id"].

        Returns default if no fixtures are loaded or no values are found.
        """
        if not self.fixtures:
            return default

        # We need team_id to filter correctly
        team_id = None
        try:
            team_id = self._get_manager_team_id()
        except Exception:
            pass

        values = []
        for fixture in self.fixtures:
            stats = fixture.get("statistics", [])
            if isinstance(stats, dict):
                stats = stats.get("data", [])

            for stat in stats:
                # Filter by team if we know the ID; otherwise accept all entries
                if team_id is not None and stat.get("participant_id") != team_id:
                    continue

                type_info = stat.get("type", {})
                code = (
                    type_info.get("code", "")
                    if isinstance(type_info, dict)
                    else str(type_info) if isinstance(type_info, str) else ""
                )

                if code == "successful-passes-percentage":
                    data = stat.get("data", {})
                    val = data.get("value") if isinstance(data, dict) else None
                    if val is not None:
                        try:
                            values.append(float(val))
                        except (TypeError, ValueError):
                            pass

        return round(sum(values) / len(values), 1) if values else default

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
        
        print(f"      âœ“ Primary formation: {formation_profile['primary']}")
        print(f"      âœ“ Win rate: {results_profile['win_rate']:.1f}%")
    
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
        """
        Analyse tactical metrics from fixture-level match statistics.

        Fixture stat structure confirmed by API diagnostic (Feb 2026):
            stat["participant_id"]  — team filter (use this ONLY, not location)
            stat["type"]["code"]    — stat code string (requires include=statistics.type)
            stat["data"]["value"]   — the numeric value

        Confirmed fixture codes:
            "ball-possession"              → possession %
            "shots-total"                  → total shots
            "tackles"                      → tackles
            "interceptions"                → interceptions
            "successful-passes-percentage" → pass accuracy  (was "passes-percentage" — wrong)
            "shots-on-target"              → shots on target
        """
        metrics = defaultdict(list)

        for fixture in self.fixtures:
            stats = fixture.get("statistics", [])
            if isinstance(stats, dict):
                stats = stats.get("data", [])

            for stat in stats:
                # Filter strictly by participant_id — location alone is unreliable
                if stat.get("participant_id") != team_id:
                    continue

                type_info = stat.get("type", {})
                if isinstance(type_info, dict):
                    code = type_info.get("code", "")
                elif isinstance(type_info, str):
                    code = type_info
                else:
                    code = ""

                data = stat.get("data", {})
                value = data.get("value", 0) if isinstance(data, dict) else 0

                if code:
                    metrics[code].append(value)

        def avg(lst):
            return round(sum(lst) / len(lst), 1) if lst else 0

        possession    = avg(metrics.get("ball-possession", []))
        tackles       = avg(metrics.get("tackles",         []))
        interceptions = avg(metrics.get("interceptions",   []))
        pass_acc      = avg(metrics.get("successful-passes-percentage", []))  # was "passes-percentage"
        shots         = avg(metrics.get("shots-total",     []))
        shots_ot      = avg(metrics.get("shots-on-target", []))

        return {
            "possession": {"avg": possession or 50.0},
            "pressing":   {"intensity": {"avg": round((tackles or 15.0) + (interceptions or 10.0), 1)}},
            "build_up":   {"pass_accuracy": pass_acc or 80.0},
            "attacking":  {
                "shots_pg":          shots or 12.0,
                "shots_on_target_pg": shots_ot or 4.0,
            }
        }
    
    # =========================================================================
    # EXTRACT SQUAD PROFILES
    # =========================================================================
    
    def extract_squad_profiles(self):
        """Extract player profiles from squad data."""
        print("\n[4/5] Extracting squad profiles...")
        
        if not self.squad:
            print("      âš  No squad loaded")
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
        print(f"      âœ“ Processed {len(profiles)} players")
        
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
            # Handle type being either a dict or a string
            type_info = detail.get("type", {})
            if isinstance(type_info, str):
                code = type_info
            elif isinstance(type_info, dict):
                code = type_info.get("code", "")
            else:
                code = ""
            
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
            print(f"      âœ“ manager_dna.json")
        
        # Save league context
        if self.league_context:
            with open(self.output_dir / "league_context.json", "w") as f:
                json.dump(self.league_context, f, indent=2)
            print(f"      âœ“ league_context.json")
        
        # Save squad profiles as CSV
        if self.squad_profiles:
            csv_path = self.output_dir / "squad_profiles.csv"
            with open(csv_path, "w", newline="") as f:
                if self.squad_profiles:
                    writer = csv.DictWriter(f, fieldnames=self.squad_profiles[0].keys())
                    writer.writeheader()
                    writer.writerows(self.squad_profiles)
            print(f"      âœ“ squad_profiles.csv")
        
        print(f"\n      Output directory: {self.output_dir}")
        
        return self


# =============================================================================
# STATSBOMB ETL PIPELINE
# =============================================================================

class StatsBombETL:
    """
    ETL Pipeline for transforming StatsBomb data into analysis-ready structures.
    
    Produces the SAME output format as ETLPipeline (manager_dna + squad_profiles)
    so that analysis.py and visualizations.py work identically.
    
    Usage:
        from aegis import StatsBombETL
        
        etl = StatsBombETL()
        manager_dna, squad_profiles = etl.run()
    """
    
    # StatsBomb position_id to Aegis position name
    POSITION_MAP = {
        1: "Goalkeeper",
        2: "Defender", 3: "Defender", 4: "Defender",
        5: "Defender", 6: "Defender",
        7: "Defender", 8: "Defender",
        9: "Midfielder", 10: "Midfielder", 11: "Midfielder",
        12: "Midfielder", 13: "Midfielder", 14: "Midfielder",
        15: "Midfielder", 16: "Midfielder",
        17: "Attacker", 18: "Attacker", 19: "Attacker",
        20: "Attacker", 21: "Attacker",
        22: "Attacker", 23: "Attacker", 24: "Attacker", 25: "Attacker",
    }
    
    DETAILED_POSITION_MAP = {
        1: "Goalkeeper",
        2: "Right-Back", 3: "Centre-Back", 4: "Centre-Back",
        5: "Centre-Back", 6: "Left-Back",
        7: "Wing-Back", 8: "Wing-Back",
        9: "Defensive Midfield", 10: "Defensive Midfield", 11: "Defensive Midfield",
        12: "Right Midfield", 13: "Central Midfield", 14: "Central Midfield",
        15: "Central Midfield", 16: "Left Midfield",
        17: "Right Winger", 18: "Attacking Midfield", 19: "Attacking Midfield",
        20: "Attacking Midfield", 21: "Left Winger",
        22: "Centre-Forward", 23: "Striker", 24: "Centre-Forward", 25: "Striker",
    }
    
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir else Config.DATA_DIR
        self.output_dir = Config.PROCESSED_DIR
        
        self.matches = None
        self.team_info = None
        self.team_match_stats = None
        self.player_season_stats = None
        self.lineups = None
        self.manager_info = None
        self.metadata = None
        
        self.manager_dna = None
        self.squad_profiles = None
        self.league_context = None
    
    def run(self):
        """Run full StatsBomb ETL pipeline. Returns (manager_dna, squad_profiles)."""
        print("\n" + "=" * 50)
        print("AEGIS ETL PIPELINE (StatsBomb)")
        print("=" * 50)
        
        self.load_raw_data()
        self.extract_league_context()
        self.extract_manager_dna()
        self.extract_squad_profiles()
        self.save()
        
        print("\n" + "=" * 50)
        print("✓ ETL COMPLETE (StatsBomb)")
        print("=" * 50)
        
        return self.manager_dna, self.squad_profiles
    
    def load_raw_data(self):
        """Load raw StatsBomb JSON files."""
        print("\n[1/5] Loading raw StatsBomb data...")
        
        def _load(filename):
            path = self.data_dir / filename
            if path.exists():
                with open(path) as f:
                    return json.load(f)
            return None
        
        self.metadata = _load("metadata.json") or {"data_source": "statsbomb"}
        self.matches = _load("sb_matches.json") or []
        self.team_info = _load("sb_team_info.json") or {}
        self.team_match_stats = _load("sb_team_match_stats.json") or []
        self.player_season_stats = _load("sb_player_season_stats.json") or []
        self.lineups = _load("sb_lineups.json") or []
        self.manager_info = _load("sb_manager_info.json") or {}
        
        print(f"      ✓ Matches: {len(self.matches)}")
        print(f"      ✓ Team: {self.team_info.get('name', 'Unknown')}")
        print(f"      ✓ Team match stats: {len(self.team_match_stats)} records")
        print(f"      ✓ Player season stats: {len(self.player_season_stats)} players")
        print(f"      ✓ Manager: {self.manager_info.get('name', 'Unknown')}")
        return self
    
    def extract_league_context(self):
        """Extract league-wide benchmarks from match data."""
        print("\n[2/5] Extracting league context...")
        
        comp_id = self.metadata.get("competition_id", "")
        season_id = self.metadata.get("season_id", "")
        
        self.league_context = {
            "season_name": f"Season {season_id}",
            "league": f"Competition {comp_id}",
            "avg_goals_per_game": 2.5,
            "avg_possession_home": 52,
            "total_matches": len(self.matches),
        }
        
        total_goals = 0
        match_count = 0
        for match in self.matches:
            hs = match.get("home_score")
            aws = match.get("away_score")
            if hs is not None and aws is not None:
                total_goals += hs + aws
                match_count += 1
        
        if match_count > 0:
            self.league_context["avg_goals_per_game"] = round(total_goals / match_count, 2)
        
        print(f"      ✓ Matches: {match_count}")
        print(f"      ✓ Avg goals/game: {self.league_context['avg_goals_per_game']}")
        return self
    
    def extract_manager_dna(self):
        """
        Extract manager tactical profile from StatsBomb team match stats.
        
        Uses team_match_stats for: possession, passing ratio, PPDA,
        xG, OBV, pressures, deep completions, etc.
        """
        print("\n[3/5] Extracting manager DNA (StatsBomb)...")
        
        team_id = self.team_info.get("id")
        
        if not self.team_match_stats or not team_id:
            print("      ⚠ No team match stats available")
            self.manager_dna = self._default_manager_dna()
            return self
        
        our_stats = [s for s in self.team_match_stats if s.get("team_id") == team_id]
        
        if not our_stats:
            print(f"      ⚠ No stats found for team_id={team_id}")
            self.manager_dna = self._default_manager_dna()
            return self
        
        def avg(key, default=0):
            vals = [s.get(key) for s in our_stats if s.get(key) is not None]
            return round(sum(vals) / len(vals), 2) if vals else default
        
        # Results from match data
        wins = draws = losses = goals_for = goals_against = clean_sheets = 0
        for match in (self.matches or []):
            home = match.get("home_team", {})
            away = match.get("away_team", {})
            home_id = home.get("home_team_id", home.get("id"))
            away_id = away.get("away_team_id", away.get("id"))
            hs = match.get("home_score", 0) or 0
            aws = match.get("away_score", 0) or 0
            
            if home_id == team_id:
                goals_for += hs; goals_against += aws
                if aws == 0: clean_sheets += 1
                if hs > aws: wins += 1
                elif hs < aws: losses += 1
                else: draws += 1
            elif away_id == team_id:
                goals_for += aws; goals_against += hs
                if hs == 0: clean_sheets += 1
                if aws > hs: wins += 1
                elif aws < hs: losses += 1
                else: draws += 1
        
        total_matches = (wins + draws + losses) or 1
        
        possession = avg("team_match_possession", 50.0)
        passing_ratio = avg("team_match_passing_ratio", 80.0)
        ppda = avg("team_match_ppda", 10.0)
        defensive_distance = avg("team_match_defensive_distance", 40.0)
        np_xg = avg("team_match_np_xg", 1.5)
        np_xg_conceded = avg("team_match_np_xg_conceded", 1.2)
        np_shots = avg("team_match_np_shots", 12.0)
        obv = avg("team_match_obv", 0.0)
        deep_completions = avg("team_match_deep_completions", 5.0)
        deep_progressions = avg("team_match_deep_progressions", 20.0)
        pressures = avg("team_match_pressures", 150.0)
        counterpressures = avg("team_match_counterpressures", 30.0)
        counter_shots = avg("team_match_counter_attacking_shots", 1.0)
        high_press_shots = avg("team_match_high_press_shots", 0.5)
        
        # Additional pillar metrics
        directness_raw = avg("team_match_directness", 0.3)
        pace_towards_goal = avg("team_match_pace_towards_goal", 1.0)
        crosses_into_box = avg("team_match_crosses_into_box", 5.0)
        box_cross_ratio = avg("team_match_box_cross_ratio", 30.0)
        sp_xg = avg("team_match_sp_xg", 0.3)
        op_xg = avg("team_match_op_xg", 1.0)
        sp_xg_conceded = avg("team_match_sp_xg_conceded", 0.3)
        pressure_regains = avg("team_match_pressure_regains", 20)
        counterpressure_regains = avg("team_match_counterpressure_regains", 5)
        deep_completions_conceded = avg("team_match_deep_completions_conceded", 5.0)
        deep_progs_conceded = avg("team_match_deep_progressions_conceded", 20.0)
        possessions_pg = avg("team_match_possessions", 50)
        
        # ── NEW: Extended metrics for richer pillar differentiation ──
        # Shape & Occupation
        passes_inside_box = avg("team_match_passes_inside_box", 10)
        obv_defensive = avg("team_match_obv_defensive_action", 0.0)
        
        # Build-up
        gk_pass_distance = avg("team_match_gk_pass_distance", 25.0)
        gk_long_pass_ratio = avg("team_match_gk_long_pass_ratio", 40.0)
        obv_pass = avg("team_match_obv_pass", 0.0)
        successful_passes = avg("team_match_successful_passes", 350)
        
        # Chance Creation
        xg_per_shot = round(np_xg / max(np_shots, 1), 3)
        op_shot_distance = avg("team_match_op_shot_distance", 18.0)
        shots_in_clear = avg("team_match_shots_in_clear", 0.5)
        obv_dribble_carry = avg("team_match_obv_dribble_carry", 0.0)
        
        # Press & Counterpress
        fhalf_pressures_ratio = avg("team_match_fhalf_pressures_ratio", 40.0)
        aggressive_actions = avg("team_match_aggressive_actions", 40)
        aggression = avg("team_match_aggression", 0.15)
        defensive_action_regains = avg("team_match_defensive_action_regains", 15)
        
        # Block & Line Height
        np_xg_per_shot_conceded = avg("team_match_np_xg_per_shot_conceded", 0.10)
        op_shots_conceded = avg("team_match_op_shots_conceded", 10)
        passes_inside_box_conceded = avg("team_match_passes_inside_box_conceded", 8)
        defensive_distance_ppda = avg("team_match_defensive_distance_ppda", 38.0)
        
        # Transitions
        counter_shots_conceded = avg("team_match_counter_attacking_shots_conceded", 1.0)
        high_press_shots_conceded = avg("team_match_high_press_shots_conceded", 0.5)
        ball_in_play_time = avg("team_match_ball_in_play_time", 55.0)
        shots_in_clear_conceded = avg("team_match_shots_in_clear_conceded", 0.5)
        
        # Width & Overloads
        successful_crosses = avg("team_match_successful_crosses_into_box", 2.0)
        successful_box_cross_ratio = avg("team_match_successful_box_cross_ratio", 20.0)
        completed_dribbles = avg("team_match_completed_dribbles", 8.0)
        dribble_ratio = avg("team_match_dribble_ratio", 50.0)
        
        # Set Pieces (extended)
        sp_count = avg("team_match_sp", 10)
        xg_per_sp = avg("team_match_xg_per_sp", 0.03)
        sp_shot_ratio = avg("team_match_sp_shot_ratio", 0.2)
        corner_xg = avg("team_match_corner_xg", 0.15)
        xg_per_corner = avg("team_match_xg_per_corner", 0.03)
        xg_per_sp_conceded = avg("team_match_xg_per_sp_conceded", 0.03)
        sp_shot_ratio_conceded = avg("team_match_sp_shot_ratio_conceded", 0.2)
        
        # Derived pillar scores
        pressing_intensity = round(max(1, 30 - ppda) * 1.5, 1)
        counterpress_rate = round(counterpressures / max(pressures, 1) * 100, 1)
        total_xg = sp_xg + op_xg
        set_piece_emphasis = round(sp_xg / max(total_xg, 0.01) * 100, 1)
        
        formation_profile = self._analyse_formations_sb()
        
        self.manager_dna = {
            "manager": self.manager_info.get("name", "Unknown"),
            "team": self.team_info.get("name", "Unknown"),
            "matches_analysed": total_matches,
            "data_mode": "statsbomb",
            "formation_profile": formation_profile,
            "results_profile": {
                "wins": wins, "draws": draws, "losses": losses,
                "win_rate": round(wins / total_matches * 100, 1),
                "win_rate_home": 0, "win_rate_away": 0,
                "points_per_game": round((wins * 3 + draws) / total_matches, 2),
                "goals_per_game": round(goals_for / total_matches, 2),
                "conceded_per_game": round(goals_against / total_matches, 2),
                "clean_sheet_pct": round(clean_sheets / total_matches * 100, 1),
            },
            "tactical_profile": {
                "possession": {"avg": possession},
                "pressing": {
                    "intensity": {"avg": pressing_intensity},
                    "ppda": ppda,
                    "pressures_per_game": pressures,
                    "counterpressures_per_game": counterpressures,
                    "counterpress_rate": counterpress_rate,
                    "pressure_regains_pg": pressure_regains,
                    "defensive_distance": defensive_distance,
                    "high_press_shots_pg": high_press_shots,
                },
                "build_up": {
                    "pass_accuracy": passing_ratio,
                    "directness": round(directness_raw * 100, 1),
                    "pace_towards_goal": pace_towards_goal,
                    "deep_completions_pg": deep_completions,
                    "deep_progressions_pg": deep_progressions,
                },
                "attacking": {
                    "shots_pg": np_shots,
                    "shots_on_target_pg": round(np_shots * 0.33, 1),
                    "np_xg_pg": np_xg,
                    "xg_per_shot": xg_per_shot,
                    "np_xg_conceded_pg": np_xg_conceded,
                    "obv_pg": obv,
                    "counter_attacking_shots_pg": counter_shots,
                    "crosses_into_box_pg": crosses_into_box,
                    "box_cross_ratio": box_cross_ratio,
                }
            },
            
            # ── Gary's 8-pillar scores (0-100 scale for radar chart) ──
            # v3: retuned coefficients for better spread across league tiers.
            # Compared to v2: boosted build-up (wider GK threshold, lower pass_acc
            # floor), transitions (floors on rare-event metrics, higher multipliers),
            # press (regains amplified), width (crossing/dribble uplift).
            "pillar_scores": {
                # P1 Shape & Occupation
                "shape_occupation": min(100, round(
                    min(50, possession * 0.85) +
                    min(15, passes_inside_box * 0.8) +
                    min(15, max(0, 1.0 - deep_progs_conceded / 25) * 15) +
                    min(10, round(clean_sheets / total_matches * 100, 1) * 0.15) +
                    min(10, max(0, obv_defensive * 20 + 5)),
                0)),

                # P2 Build-up — widened GK distance window (50m), lowered
                # pass_acc floor (65), boosted deep progressions multiplier
                "build_up": min(100, round(
                    min(35, possession * 0.7) +
                    min(20, max(0, passing_ratio - 65) * 1.2) +
                    min(15, max(0, 50 - gk_pass_distance) * 0.4) +
                    min(15, deep_progressions * 0.7) +
                    min(10, max(0, obv_pass * 15 + 5)) +
                    min(5, (100 - round(directness_raw * 100, 1)) * 0.05),
                0)),

                # P3 Chance Creation
                "chance_creation": min(100, round(
                    min(30, np_xg * 18) +
                    min(25, xg_per_shot * 200) +
                    min(15, deep_completions * 1.2) +
                    min(15, shots_in_clear * 15) +
                    min(15, max(0, obv_dribble_carry * 25 + 5)),
                0)),

                # P4 Press & Counterpress — boosted regains & aggression
                "press_counterpress": min(100, round(
                    min(25, max(0, 30 - ppda) * 1.5) +
                    min(20, counterpress_rate * 1.2) +
                    min(20, fhalf_pressures_ratio * 0.4) +
                    min(15, pressure_regains * 0.7) +
                    min(10, defensive_action_regains * 0.5) +
                    min(10, aggression * 65),
                0)),

                # P5 Block & Line Height — boosted line height contribution
                "block_line_height": min(100, round(
                    min(30, defensive_distance * 0.85) +
                    min(20, max(0, 0.12 - np_xg_per_shot_conceded) * 250) +
                    min(20, max(0, 15 - passes_inside_box_conceded) * 1.3) +
                    min(15, max(0, 8 - deep_completions_conceded) * 1.8) +
                    min(15, max(0, 2.0 - np_xg_conceded) * 10),
                0)),

                # P6 Transitions — added floor values on rare-event metrics,
                # boosted pace & ball-in-play, widened defensive threshold
                "transitions": min(100, round(
                    min(25, 3 + counter_shots * 16) +
                    min(20, 2 + high_press_shots * 20) +
                    min(20, pace_towards_goal * 12) +
                    min(15, max(0, obv_dribble_carry * 25 + 5)) +
                    min(10, max(0, 2.5 - counter_shots_conceded) * 5) +
                    min(10, ball_in_play_time * 0.16),
                0)),

                # P7 Width & Overloads — boosted crosses & dribble multipliers
                "width_overloads": min(100, round(
                    min(25, crosses_into_box * 2.5) +
                    min(20, successful_crosses * 5.0) +
                    min(20, successful_box_cross_ratio * 0.6) +
                    min(20, completed_dribbles * 1.5) +
                    min(15, dribble_ratio * 0.25),
                0)),

                # P8 Set Pieces — boosted efficiency multipliers
                "set_pieces": min(100, round(
                    min(25, set_piece_emphasis * 0.8) +
                    min(20, xg_per_sp * 600) +
                    min(15, sp_shot_ratio * 50) +
                    min(15, corner_xg * 60) +
                    min(15, max(0, 0.04 - xg_per_sp_conceded) * 500) +
                    min(10, sp_xg * 20),
                0)),
            },
            
            "statsbomb_enhanced": {
                "ppda": ppda,
                "np_xg_per_game": np_xg,
                "np_xg_conceded_per_game": np_xg_conceded,
                "xg_difference": round(np_xg - np_xg_conceded, 2),
                "xg_per_shot": xg_per_shot,
                "obv_per_game": obv,
                "obv_pass_pg": obv_pass,
                "obv_dribble_carry_pg": obv_dribble_carry,
                "obv_defensive_action_pg": obv_defensive,
                "deep_completions_pg": deep_completions,
                "deep_progressions_pg": deep_progressions,
                "deep_completions_conceded_pg": deep_completions_conceded,
                "deep_progressions_conceded_pg": deep_progs_conceded,
                "pressures_per_game": pressures,
                "counterpress_rate": counterpress_rate,
                "pressure_regains_pg": pressure_regains,
                "counterpressure_regains_pg": counterpressure_regains,
                "defensive_action_regains_pg": defensive_action_regains,
                "fhalf_pressures_ratio": fhalf_pressures_ratio,
                "aggression": aggression,
                "aggressive_actions_pg": aggressive_actions,
                "counter_attacking_shots_pg": counter_shots,
                "counter_attacking_shots_conceded_pg": counter_shots_conceded,
                "high_press_shots_pg": high_press_shots,
                "high_press_shots_conceded_pg": high_press_shots_conceded,
                "shots_in_clear_pg": shots_in_clear,
                "shots_in_clear_conceded_pg": shots_in_clear_conceded,
                "directness": round(directness_raw * 100, 1),
                "pace_towards_goal": pace_towards_goal,
                "ball_in_play_time": ball_in_play_time,
                "gk_pass_distance": gk_pass_distance,
                "gk_long_pass_ratio": gk_long_pass_ratio,
                "width_usage": crosses_into_box,
                "successful_crosses_pg": successful_crosses,
                "box_cross_ratio": box_cross_ratio,
                "successful_box_cross_ratio": successful_box_cross_ratio,
                "completed_dribbles_pg": completed_dribbles,
                "dribble_ratio": dribble_ratio,
                "passes_inside_box_pg": passes_inside_box,
                "passes_inside_box_conceded_pg": passes_inside_box_conceded,
                "set_piece_emphasis": set_piece_emphasis,
                "sp_xg_pg": sp_xg,
                "sp_xg_conceded_pg": sp_xg_conceded,
                "xg_per_sp": xg_per_sp,
                "sp_shot_ratio": sp_shot_ratio,
                "corner_xg_pg": corner_xg,
                "xg_per_corner": xg_per_corner,
                "xg_per_sp_conceded": xg_per_sp_conceded,
                "np_xg_per_shot_conceded": np_xg_per_shot_conceded,
                "defensive_distance": defensive_distance,
                "defensive_distance_ppda": defensive_distance_ppda,
            }
        }
        
        print(f"      ✓ Win rate: {self.manager_dna['results_profile']['win_rate']:.1f}%")
        print(f"      ✓ Possession: {possession:.1f}%  Pass acc: {passing_ratio:.1f}%")
        print(f"      ✓ PPDA: {ppda:.1f}  xG: {np_xg:.2f}  xGA: {np_xg_conceded:.2f}")
        print(f"      ✓ OBV/g: {obv:.3f}  Deep prog: {deep_progressions:.1f}")
        return self
    
    def _analyse_formations_sb(self) -> Dict:
        """Analyse formation usage from StatsBomb lineups."""
        formations = defaultdict(int)
        for lineup_data in (self.lineups or []):
            for team_lineup in lineup_data.get("lineups", []):
                if team_lineup.get("team_id") != self.team_info.get("id"):
                    continue
                for formation_obj in team_lineup.get("formations", []):
                    formation = formation_obj.get("formation")
                    if formation:
                        f_str = str(formation)
                        if len(f_str) >= 3 and "-" not in f_str:
                            f_str = "-".join(f_str)
                        formations[f_str] += 1
        
        total = sum(formations.values()) or 1
        formation_pcts = {k: round(v / total * 100, 1) for k, v in formations.items()}
        primary = max(formations, key=formations.get) if formations else "4-3-3"
        
        return {
            "primary": primary,
            "usage": dict(formations),
            "percentages": formation_pcts,
            "flexibility_score": len(formations)
        }
    
    def _default_manager_dna(self) -> Dict:
        return {
            "manager": "Unknown", "matches_analysed": 0, "data_mode": "none",
            "formation_profile": {"primary": "4-4-2", "flexibility_score": 1},
            "results_profile": {"win_rate": 33, "points_per_game": 1.33, "conceded_per_game": 1.2},
            "tactical_profile": {
                "possession": {"avg": 50},
                "pressing": {"intensity": {"avg": 20}},
                "build_up": {"pass_accuracy": 80},
                "attacking": {"shots_pg": 12}
            }
        }
    
    def extract_squad_profiles(self):
        """
        Extract player profiles from StatsBomb player season stats.
        
        Position resolution strategy (primary_position is often None):
        1. Try primary_position from season stats
        2. Fall back to lineup position data (most reliable)
        3. Fall back to stat-based heuristic (GK saves, DEF tackles, ATT goals)
        """
        print("\n[4/5] Extracting squad profiles (StatsBomb)...")
        
        if not self.player_season_stats:
            print("      ⚠ No player season stats loaded")
            self.squad_profiles = []
            return self
        
        team_id = self.team_info.get("id")
        team_name = self.team_info.get("name", "")
        
        team_players = [
            p for p in self.player_season_stats
            if p.get("team_id") == team_id
        ]
        if not team_players:
            team_players = [
                p for p in self.player_season_stats
                if team_name.lower() in (p.get("team_name", "") or "").lower()
            ]
        
        # Build player_id → position lookup from lineups
        lineup_positions = self._build_position_lookup_from_lineups()
        
        profiles = []
        pos_resolved = {"season_stats": 0, "lineup": 0, "heuristic": 0, "unknown": 0}
        
        for p in team_players:
            minutes = p.get("player_season_minutes", 0) or 0
            if minutes < 90:
                continue
            
            nineties = minutes / 90.0
            player_id = p.get("player_id")
            
            # Position resolution chain
            primary_pos = p.get("primary_position")
            position = None
            detailed_position = None
            
            # 1. Try season stats primary_position
            if primary_pos and primary_pos in self.POSITION_MAP:
                position = self.POSITION_MAP[primary_pos]
                detailed_position = self.DETAILED_POSITION_MAP.get(primary_pos, position)
                pos_resolved["season_stats"] += 1
            
            # 2. Fall back to lineup data
            if not position or position == "Unknown":
                lineup_pos = lineup_positions.get(player_id)
                if lineup_pos:
                    pos_id = lineup_pos.get("position_id")
                    position = self.POSITION_MAP.get(pos_id, None)
                    detailed_position = self.DETAILED_POSITION_MAP.get(
                        pos_id, lineup_pos.get("position_name", position)
                    )
                    if position:
                        pos_resolved["lineup"] += 1
            
            # 3. Fall back to stat-based heuristic
            if not position or position == "Unknown":
                position, detailed_position = self._infer_position_from_stats(p, nineties)
                if position != "Unknown":
                    pos_resolved["heuristic"] += 1
                else:
                    pos_resolved["unknown"] += 1
            
            age = 25
            dob = p.get("birth_date")
            if dob:
                try:
                    birth = datetime.strptime(str(dob)[:10], "%Y-%m-%d")
                    today = datetime.now()
                    age = today.year - birth.year - (
                        (today.month, today.day) < (birth.month, birth.day)
                    )
                except (ValueError, TypeError):
                    pass
            
            profile = {
                "id": player_id,
                "name": p.get("player_name", "Unknown"),
                "age": age,
                "position": position or "Unknown",
                "detailed_position": detailed_position or position or "Unknown",
                "jersey_number": None,
                "appearances": int(p.get("player_season_appearances", 0) or 0),
                "minutes": minutes,
                "goals": round((p.get("player_season_goals_90", 0) or 0) * nineties),
                "assists": round((p.get("player_season_assists_90", 0) or 0) * nineties),
                "clean_sheets": 0,
                "saves": 0,
                "tackles": round((p.get("player_season_tackles_90", 0) or 0) * nineties),
                "interceptions": round((p.get("player_season_interceptions_90", 0) or 0) * nineties),
                "clearances": round((p.get("player_season_clearance_90", 0) or 0) * nineties),
                "pass_accuracy": p.get("player_season_passing_ratio", 0) or 0,
                "key_passes": round((p.get("player_season_key_passes_90", 0) or 0) * nineties),
                "dribbles": round((p.get("player_season_dribbles_90", 0) or 0) * nineties),
                "shots": round((p.get("player_season_np_shots_90", 0) or 0) * nineties),
                "shots_on_target": 0,
                "rating": 0,
                "np_xg": round((p.get("player_season_np_xg_90", 0) or 0) * nineties, 2),
                "xa": round((p.get("player_season_xa_90", 0) or 0) * nineties, 2),
                "obv": round((p.get("player_season_obv_90", 0) or 0) * nineties, 3),
                "obv_pass": round((p.get("player_season_obv_pass_90", 0) or 0) * nineties, 3),
                "obv_dribble_carry": round((p.get("player_season_obv_dribble_carry_90", 0) or 0) * nineties, 3),
                "obv_defensive": round((p.get("player_season_obv_defensive_action_90", 0) or 0) * nineties, 3),
                "pressures_90": p.get("player_season_pressures_90", 0) or 0,
                "deep_progressions_90": p.get("player_season_deep_progressions_90", 0) or 0,
                "xg_chain_90": p.get("player_season_xgchain_90", 0) or 0,
                "xg_buildup_90": p.get("player_season_xgbuildup_90", 0) or 0,
                "challenge_ratio": p.get("player_season_challenge_ratio", 0) or 0,
                "aerial_ratio": p.get("player_season_aerial_ratio", 0) or 0,
            }
            profiles.append(profile)
        
        self.squad_profiles = profiles
        print(f"      ✓ Processed {len(profiles)} players")
        print(f"      ✓ Positions: stats={pos_resolved['season_stats']}, "
              f"lineup={pos_resolved['lineup']}, heuristic={pos_resolved['heuristic']}, "
              f"unknown={pos_resolved['unknown']}")
        return self
    
    def _build_position_lookup_from_lineups(self) -> Dict:
        """Build player_id → {position_id, position_name} from lineup data."""
        from collections import Counter
        
        player_positions = defaultdict(list)
        player_pos_names = {}
        team_id = self.team_info.get("id")
        
        for lineup_data in (self.lineups or []):
            for team_lineup in lineup_data.get("lineups", []):
                if team_lineup.get("team_id") != team_id:
                    continue
                for player in team_lineup.get("lineup", []):
                    pid = player.get("player_id")
                    if not pid:
                        continue
                    for pos_entry in player.get("positions", []):
                        pos_id = pos_entry.get("position_id")
                        pos_name = pos_entry.get("position")
                        if pos_id:
                            player_positions[pid].append(pos_id)
                            if pos_name:
                                player_pos_names[(pid, pos_id)] = pos_name
        
        result = {}
        for pid, pos_ids in player_positions.items():
            most_common = Counter(pos_ids).most_common(1)[0][0]
            result[pid] = {
                "position_id": most_common,
                "position_name": player_pos_names.get((pid, most_common), "")
            }
        
        if result:
            print(f"      ✓ Position lookup from lineups: {len(result)} players")
        return result
    
    def _infer_position_from_stats(self, p: Dict, nineties: float) -> tuple:
        """Infer position from statistical profile when no position data exists."""
        tackles_90 = p.get("player_season_tackles_90", 0) or 0
        interceptions_90 = p.get("player_season_interceptions_90", 0) or 0
        goals_90 = p.get("player_season_goals_90", 0) or 0
        np_shots_90 = p.get("player_season_np_shots_90", 0) or 0
        np_xg_90 = p.get("player_season_np_xg_90", 0) or 0
        key_passes_90 = p.get("player_season_key_passes_90", 0) or 0
        save_ratio = p.get("player_season_save_ratio", 0) or 0
        goals_faced_90 = p.get("player_season_goals_faced_90", 0) or 0
        clearance_90 = p.get("player_season_clearance_90", 0) or 0
        dribbles_90 = p.get("player_season_dribbles_90", 0) or 0
        
        if save_ratio > 0 or goals_faced_90 > 0:
            return "Goalkeeper", "Goalkeeper"
        
        defensive = tackles_90 + interceptions_90 + clearance_90
        attacking = goals_90 + np_xg_90 + (np_shots_90 * 0.1) + dribbles_90
        
        if defensive > 4.0 and attacking < 0.5:
            return "Defender", "Centre-Back"
        if defensive > 3.0 and attacking < 1.0:
            return "Defender", "Defender"
        if goals_90 > 0.3 or np_xg_90 > 0.3 or np_shots_90 > 2.5:
            return "Attacker", "Winger" if dribbles_90 > 1.5 else "Centre-Forward"
        if (goals_90 > 0.15 or np_xg_90 > 0.15) and key_passes_90 > 1.0:
            return "Attacker", "Attacking Midfield"
        if defensive > 2.0:
            return "Midfielder", "Defensive Midfield"
        return "Midfielder", "Central Midfield"
    
    def save(self):
        """Save processed data (same format as ETLPipeline)."""
        print("\n[5/5] Saving outputs...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.manager_dna:
            with open(self.output_dir / "manager_dna.json", "w") as f:
                json.dump(self.manager_dna, f, indent=2)
            print(f"      ✓ manager_dna.json")
        
        if self.league_context:
            with open(self.output_dir / "league_context.json", "w") as f:
                json.dump(self.league_context, f, indent=2)
            print(f"      ✓ league_context.json")
        
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
