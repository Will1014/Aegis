"""
Aegis Prototype - ETL Pipeline
==============================
Extract, Transform, Load for Manager DNA and Squad Fit Analysis
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Optional

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MOCK_DIR = DATA_DIR / "mock"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


class AegisETL:
    """ETL Pipeline for Aegis MTFI Prototype"""
    
    def __init__(self, use_mock: bool = True, manager_team_id: int = 100):
        self.use_mock = use_mock
        self.manager_team_id = manager_team_id
        self.fixtures_raw = []
        self.squad_raw = []
        self.manager_raw = {}
        self.manager_dna = {}
        self.squad_df = None
        
    def extract(self) -> 'AegisETL':
        """Extract all data from source"""
        print("\n" + "=" * 60)
        print("EXTRACT PHASE")
        print("=" * 60)
        
        with open(MOCK_DIR / "fixtures_brentford.json") as f:
            self.fixtures_raw = json.load(f)["data"]
        with open(MOCK_DIR / "squad_tottenham.json") as f:
            self.squad_raw = json.load(f)["data"]
        with open(MOCK_DIR / "manager_thomas_frank.json") as f:
            self.manager_raw = json.load(f)
        
        print(f"  ✓ Extracted {len(self.fixtures_raw)} fixtures")
        print(f"  ✓ Extracted {len(self.squad_raw)} players")
        print(f"  ✓ Manager: {self.manager_raw.get('data', {}).get('name')}")
        return self
    
    def transform_manager_dna(self) -> 'AegisETL':
        """Transform fixture data into Manager DNA profile"""
        print("\n" + "=" * 60)
        print("TRANSFORM: MANAGER DNA")
        print("=" * 60)
        
        formation_profile = self._analyze_formations()
        results_profile = self._analyze_results()
        tactical_profile = self._analyze_tactical_metrics()
        
        self.manager_dna = {
            "manager_id": self.manager_raw.get("data", {}).get("id"),
            "manager_name": self.manager_raw.get("data", {}).get("name"),
            "team_analyzed": "Brentford",
            "matches_analyzed": len(self.fixtures_raw),
            "formation_profile": formation_profile,
            "results_profile": results_profile,
            "tactical_profile": tactical_profile,
            "generated_at": datetime.now().isoformat()
        }
        
        with open(PROCESSED_DIR / "manager_dna.json", "w") as f:
            json.dump(self.manager_dna, f, indent=2)
        
        print(f"  ✓ Manager DNA generated")
        print(f"    Formation: {formation_profile['primary_formation']}")
        print(f"    Win Rate: {results_profile['win_rate']:.1f}%")
        print(f"    Style: {tactical_profile['playing_style']}")
        return self
    
    def _analyze_formations(self) -> Dict:
        formations = []
        formation_results = defaultdict(lambda: {"W": 0, "D": 0, "L": 0})
        
        for fixture in self.fixtures_raw:
            # API v3: formations is a direct list, not nested {"data": [...]}
            fixture_formations = fixture.get("formations", [])
            if isinstance(fixture_formations, dict):
                fixture_formations = fixture_formations.get("data", [])
            
            for form in fixture_formations:
                if form.get("participant_id") == self.manager_team_id:
                    f = form.get("formation")
                    if f:
                        formations.append(f)
                        result = self._get_result(fixture)
                        formation_results[f][result] += 1
        
        total = len(formations)
        counts = Counter(formations)
        
        effectiveness = {}
        for f, res in formation_results.items():
            matches = sum(res.values())
            if matches >= 3:
                effectiveness[f] = {
                    "matches": matches,
                    "win_rate": round(res["W"]/matches*100, 1),
                    "ppg": round((res["W"]*3 + res["D"])/matches, 2)
                }
        
        return {
            "primary_formation": counts.most_common(1)[0][0] if counts else "Unknown",
            "formations_used": dict(counts),
            "formation_percentages": {k: round(v/total*100, 1) for k, v in counts.items()},
            "flexibility_score": len(counts),
            "formation_effectiveness": effectiveness
        }
    
    def _get_result(self, fixture: Dict) -> str:
        # API v3: scores is a direct list
        scores = fixture.get("scores", [])
        if isinstance(scores, dict):
            scores = scores.get("data", [])
        
        team_goals = opp_goals = 0
        for s in scores:
            if s.get("description") == "CURRENT":  # Use final score
                if s.get("participant_id") == self.manager_team_id:
                    team_goals = s.get("score", {}).get("goals", 0)
                else:
                    opp_goals = s.get("score", {}).get("goals", 0)
        
        if team_goals > opp_goals: return "W"
        if team_goals < opp_goals: return "L"
        return "D"
    
    def _analyze_results(self) -> Dict:
        wins = draws = losses = goals_for = goals_against = clean_sheets = 0
        
        for fixture in self.fixtures_raw:
            # API v3: scores is a direct list
            scores = fixture.get("scores", [])
            if isinstance(scores, dict):
                scores = scores.get("data", [])
            
            team_goals = opp_goals = 0
            for s in scores:
                if s.get("description") == "CURRENT":
                    if s.get("participant_id") == self.manager_team_id:
                        team_goals = s.get("score", {}).get("goals", 0)
                    else:
                        opp_goals = s.get("score", {}).get("goals", 0)
            
            goals_for += team_goals
            goals_against += opp_goals
            if opp_goals == 0: clean_sheets += 1
            
            if team_goals > opp_goals: wins += 1
            elif team_goals < opp_goals: losses += 1
            else: draws += 1
        
        total = wins + draws + losses
        return {
            "matches": total,
            "wins": wins, "draws": draws, "losses": losses,
            "win_rate": round(wins/total*100, 1) if total else 0,
            "points_per_game": round((wins*3+draws)/total, 2) if total else 0,
            "goals_scored": goals_for, "goals_conceded": goals_against,
            "goal_difference": goals_for - goals_against,
            "goals_per_game": round(goals_for/total, 2) if total else 0,
            "conceded_per_game": round(goals_against/total, 2) if total else 0,
            "clean_sheets": clean_sheets,
            "clean_sheet_rate": round(clean_sheets/total*100, 1) if total else 0
        }
    
    def _analyze_tactical_metrics(self) -> Dict:
        metrics = defaultdict(list)
        
        for fixture in self.fixtures_raw:
            # API v3: statistics is a direct list
            fixture_stats = fixture.get("statistics", [])
            if isinstance(fixture_stats, dict):
                fixture_stats = fixture_stats.get("data", [])
            
            for stat in fixture_stats:
                if stat.get("participant_id") == self.manager_team_id:
                    code = stat.get("type", {}).get("code", "")
                    value = stat.get("data", {}).get("value")
                    if value is not None:
                        metrics[code].append(float(value))
        
        def avg(vals): return round(np.mean(vals), 2) if vals else 0
        def std(vals): return round(np.std(vals), 2) if len(vals) > 1 else 0
        
        pressing = [t+i for t,i in zip(metrics.get("tackles",[]), metrics.get("interceptions",[]))]
        
        # API v3 stat codes
        poss = avg(metrics.get("ball-possession", metrics.get("possession-percentage", [50])))
        press = avg(pressing) if pressing else 25
        pass_acc = avg(metrics.get("passes-percentage", metrics.get("passes-accuracy", [80])))
        
        # Style classification
        poss_style = "Possession" if poss >= 54 else "Counter" if poss < 46 else "Balanced"
        press_style = "Gegenpress" if press >= 32 else "High" if press >= 26 else "Mid" if press >= 20 else "Low"
        build_style = "Short" if pass_acc >= 87 else "Direct" if pass_acc < 82 else "Mixed"
        
        return {
            "playing_style": f"{poss_style}, {press_style} Press, {build_style} Build-up",
            "possession": {"avg": poss, "std": std(metrics.get("ball-possession",[])), "style": poss_style},
            "pressing": {
                "intensity": {"avg": avg(pressing), "std": std(pressing)},
                "tackles_pg": avg(metrics.get("tackles", [])),
                "interceptions_pg": avg(metrics.get("interceptions", [])),
                "style": press_style
            },
            "build_up": {
                "pass_accuracy": avg(metrics.get("passes-percentage", [])),
                "passes_pg": avg(metrics.get("passes-total", [])),
                "style": build_style
            },
            "attacking": {
                "shots_pg": avg(metrics.get("shots-total", [])),
                "shots_ot_pg": avg(metrics.get("shots-on-target", [])),
                "corners_pg": avg(metrics.get("corners", [])),
                "dangerous_attacks": avg(metrics.get("dangerous-attacks", []))
            },
            "defensive": {
                "fouls_pg": avg(metrics.get("fouls", [])),
                "clearances_pg": avg(metrics.get("clearances", []))
            }
        }
    
    def transform_squad_profiles(self) -> 'AegisETL':
        """Transform squad data into player profiles"""
        print("\n" + "=" * 60)
        print("TRANSFORM: SQUAD PROFILES")
        print("=" * 60)
        
        players = []
        for p in self.squad_raw:
            # API v3: statistics is a list with details array
            player_stats = p.get("statistics", [])
            if isinstance(player_stats, dict):
                player_stats = player_stats.get("data", [])
            
            stats_dict = {}
            if player_stats:
                # Get the first season's statistics
                season_stats = player_stats[0] if player_stats else {}
                details = season_stats.get("details", [])
                
                # Convert details array to dict by code
                for detail in details:
                    code = detail.get("type", {}).get("code", "")
                    if code:
                        stats_dict[code] = detail.get("value", 0)
            
            # Get position info
            position = p.get("position", {})
            detailed_position = p.get("detailed_position", position)
            
            profile = {
                "player_id": p.get("id"),
                "name": p.get("name"),
                "age": self._calculate_age(p.get("date_of_birth")),
                "nationality_id": p.get("nationality_id"),
                "position": position.get("name", "Unknown") if isinstance(position, dict) else position,
                "detailed_position": detailed_position.get("name", "Unknown") if isinstance(detailed_position, dict) else detailed_position,
                "height_cm": p.get("height"),
                "preferred_foot": p.get("preferred_foot"),
                "market_value": p.get("market_value", 0),
                "games_played": stats_dict.get("appearances", 0),
                "minutes_played": stats_dict.get("minutes-played", 0),
                "goals": stats_dict.get("goals", 0),
                "assists": stats_dict.get("assists", 0),
                "rating": stats_dict.get("rating", 6.5),
                "tackles": stats_dict.get("tackles", 0),
                "interceptions": stats_dict.get("interceptions", 0),
                "clearances": stats_dict.get("clearances", 0),
                "aerials_won": stats_dict.get("aerials-won", 0),
                "duels_won": stats_dict.get("duels-won", 0),
                "pass_accuracy": stats_dict.get("passes-percentage", 0),
                "key_passes": stats_dict.get("key-passes", 0),
                "successful_dribbles": stats_dict.get("successful-dribbles", 0),
                "dribbles_attempted": stats_dict.get("dribbles-attempted", 0),
                "shots": stats_dict.get("shots-total", 0),
                "shots_on_target": stats_dict.get("shots-on-target", 0),
                "clean_sheets": stats_dict.get("cleansheets", 0),
                "saves": stats_dict.get("saves", 0),
                "goals_conceded": stats_dict.get("goals-conceded", 0),
            }
            players.append(profile)
        
        self.squad_df = pd.DataFrame(players)
        self.squad_df.to_csv(PROCESSED_DIR / "squad_profiles.csv", index=False)
        
        print(f"  ✓ Processed {len(self.squad_df)} players")
        for pos, count in self.squad_df["position"].value_counts().items():
            print(f"    {pos}: {count}")
        return self
    
    def _calculate_age(self, dob_str):
        """Calculate age from date of birth string"""
        if not dob_str:
            return 25  # Default
        try:
            dob = datetime.strptime(dob_str, "%Y-%m-%d")
            today = datetime.now()
            return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        except:
            return 25
    
    def load(self) -> 'AegisETL':
        """Generate summary"""
        print("\n" + "=" * 60)
        print("LOAD: OUTPUTS")
        print("=" * 60)
        
        summary = {
            "run_at": datetime.now().isoformat(),
            "manager": self.manager_dna.get("manager_name"),
            "matches_analyzed": len(self.fixtures_raw),
            "target_squad": "Tottenham Hotspur",
            "players_analyzed": len(self.squad_df) if self.squad_df is not None else 0,
            "outputs": ["manager_dna.json", "squad_profiles.csv"]
        }
        with open(PROCESSED_DIR / "etl_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"  ✓ ETL complete")
        print(f"  ✓ Outputs: {PROCESSED_DIR}")
        return self
    
    def run(self) -> 'AegisETL':
        """Execute full pipeline"""
        print("\n" + "#" * 60)
        print("#  AEGIS ETL PIPELINE")
        print("#" * 60)
        return self.extract().transform_manager_dna().transform_squad_profiles().load()


if __name__ == "__main__":
    AegisETL(use_mock=True, manager_team_id=63).run()  # 63 = Brentford
