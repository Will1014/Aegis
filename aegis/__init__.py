"""
Aegis - Manager Tactical Fit Intelligence
==========================================

A Python package for analyzing tactical fit between football managers
and club squads using K-Means clustering and cosine similarity.

Quick Start (Training):
-----------------------
from aegis import Config, ManagerDNATrainer

Config.set_base_dir("/content/aegis_data")
Config.setup()

trainer = ManagerDNATrainer()
trainer.fetch_manager_data()
trainer.extract_features()
trainer.fit()
trainer.save()

Quick Start (Analysis):
-----------------------
from aegis import Config, SquadFitAnalyzer

Config.set_base_dir("/content/aegis_data")
Config.setup()

analyzer = SquadFitAnalyzer()
analyzer.load_model()
analyzer.set_target_manager("Thomas Frank")
analyzer.fetch_squad("Tottenham")
analyzer.calculate_fit_scores()
analyzer.save()

results = analyzer.get_results()
"""

__version__ = "0.3.0"
__author__ = "Aegis Football Advisory Group"

from .config import Config
from .client import SportsmonksClient, StatsBombClient
from .etl import ETLPipeline, StatsBombETL
from .analysis import (
    # New ML-based classes
    ManagerDNATrainer,
    SquadFitAnalyzer,
    PlayerFit,
    ManagerProfile,
    # Constants
    MANAGER_DNA_FEATURES,
    STATSBOMB_DNA_FEATURES,
    DNA_PILLARS,
    PILLAR_PLAYER_DEMANDS,
    PLAYER_FIT_FEATURES,
    FIT_THRESHOLDS,
    DEFAULT_MANAGERS,
    POSITION_GROUPS,
    # Legacy compatibility
    AegisAnalyzer,
)
from .visualizations import AegisVisualizer

__all__ = [
    # Config & Clients
    "Config",
    "SportsmonksClient",
    "StatsBombClient",
    
    # ETL
    "ETLPipeline",
    "StatsBombETL",
    
    # Analysis (New)
    "ManagerDNATrainer",
    "SquadFitAnalyzer",
    "PlayerFit",
    "ManagerProfile",
    
    # Constants
    "MANAGER_DNA_FEATURES",
    "PLAYER_FIT_FEATURES",
    "FIT_THRESHOLDS",
    "DEFAULT_MANAGERS",
    "POSITION_GROUPS",
    
    # Legacy
    "AegisAnalyzer",
    
    # Visualization
    "AegisVisualizer",
    
    # Convenience functions
    "run_full_analysis",
    "run_full_analysis_statsbomb",
    "train_manager_dna",
    "analyse_squad_fit",
]


def train_manager_dna(
    api_token: str = None,
    base_dir: str = "/content/aegis_data",
    season_id: int = None,
    managers: list = None,
    n_clusters: int = None,
    fetch_all: bool = False,
    league_ids = "top5"
) -> dict:
    """
    Train the Manager DNA clustering model.
    
    Args:
        api_token: Sportsmonks API token (or set SPORTMONKS_API_TOKEN env var)
        base_dir: Base directory for data and outputs
        season_id: Season to analyse. Options:
                  - None (default): Fetches ALL seasons (2024/25, 2023/24, 2022/23)
                    Creates multiple entries per manager, maximizing training data
                  - Specific ID (e.g., 23614): Fetches only that season
        managers: Custom list of managers (uses DEFAULT_MANAGERS if None)
        n_clusters: Number of clusters (auto-determined if None)
        fetch_all: If True, fetch all coaches from database (ignores managers param)
        league_ids: League filter when fetch_all=True. Options:
                   "top5" = Top 5 European leagues (default)
                   None = ALL leagues
                   [8, 564, ...] = Specific league IDs
    
    Returns:
        Dictionary with training summary
    
    Example:
        # Fetch ALL seasons for maximum training data (recommended)
        summary = train_manager_dna(fetch_all=True, league_ids="top5")
        # Result: ~200+ data points (80 managers × 2-3 seasons each)
        
        # Fetch only 2023/24 season (for consistent timeframe)
        summary = train_manager_dna(fetch_all=True, league_ids="top5", season_id=21646)
        # Result: ~80 data points (one per manager)
        
        # Fetch ALL managers from ALL leagues and ALL seasons
        summary = train_manager_dna(fetch_all=True, league_ids=None)
        # Result: ~600+ data points (massive training set!)
        
    Season Behavior:
        season_id=None (default):
          - Fetches 2024/25, 2023/24, AND 2022/23 for EACH manager
          - Creates 1-3 entries per manager (one per available season)
          - Example: "Pep Guardiola 2024/25", "Pep Guardiola 2023/24"
          - Maximizes training data for better clustering
          - Model learns tactical evolution over time
        
        season_id=23614 (specific):
          - Fetches ONLY 2024/25 for each manager
          - Creates 1 entry per manager
          - Ensures consistent timeframe for fair comparison
    """
    import os
    
    if api_token:
        os.environ["SPORTMONKS_API_TOKEN"] = api_token
    
    Config.set_base_dir(base_dir)
    Config.setup()
    
    training_dir = Config.PROCESSED_DIR / "training"
    
    trainer = ManagerDNATrainer(
        training_dir=training_dir,
        season_id=season_id
    )
    
    trainer.fetch_manager_data(
        managers=managers,
        fetch_all=fetch_all,
        league_ids=league_ids
    )
    trainer.extract_features()
    trainer.fit(n_clusters=n_clusters)
    trainer.save()
    
    # Try to plot
    try:
        trainer.plot_clusters(save_path=training_dir / "manager_clusters_pca.png")
    except Exception as e:
        print(f"âš  Could not generate plot: {e}")
    
    return {
        "n_managers": len(trainer.df_managers),
        "n_clusters": trainer.n_clusters,
        "cluster_names": trainer.cluster_names,
        "training_dir": str(training_dir)
    }


def analyse_squad_fit(
    manager_name: str,
    club_name: str,
    api_token: str = None,
    base_dir: str = "/content/aegis_data",
    season_id: int = 23614
) -> dict:
    """
    Analyse squad fit for a manager-club pairing.
    
    Requires trained model (run train_manager_dna first).
    
    Args:
        manager_name: Target manager name
        club_name: Target club name
        api_token: Sportsmonks API token
        base_dir: Base directory
        season_id: Season to analyse
    
    Returns:
        Dictionary with analysis results
    
    Example:
        from aegis import analyse_squad_fit
        
        results = analyse_squad_fit(
            manager_name="Thomas Frank",
            club_name="Tottenham",
            api_token="your_token"
        )
    """
    import os
    
    if api_token:
        os.environ["SPORTMONKS_API_TOKEN"] = api_token
    
    Config.set_base_dir(base_dir)
    Config.setup()
    
    training_dir = Config.PROCESSED_DIR / "training"
    
    analyzer = SquadFitAnalyzer(
        training_dir=training_dir,
        output_dir=Config.OUTPUT_DIR,
        season_id=season_id
    )
    
    analyzer.load_model()
    analyzer.set_target_manager(manager_name)
    analyzer.fetch_squad(club_name)
    analyzer.calculate_fit_scores()
    analyzer.save()
    
    return analyzer.get_results()


def run_full_analysis(
    coach_name: str,
    target_club: str,
    api_token: str = None,
    base_dir: str = "/content/aegis_data",
    season_id: int = None,
    visualize: bool = True,
    train_model: bool = True,
    fetch_all_managers: bool = False,
    league_ids = "top5",
    output_filename: str = None
):
    """
    Run complete Aegis analysis pipeline (ML-based).
    
    Args:
        coach_name: Manager name (e.g., "Thomas Frank")
        target_club: Target club name (e.g., "Tottenham")
        api_token: Sportsmonks API token
        base_dir: Base directory for data and outputs
        season_id: Season to analyse. Options:
                  - None (default): Fetches ALL seasons (2024/25, 2023/24, 2022/23)
                    Creates multiple entries per manager, maximizing training data
                  - Specific ID (e.g., 23614): Fetches only that season
        visualize: Generate visualization charts
        train_model: Train Manager DNA model (set False if already trained)
        fetch_all_managers: If True, fetch all coaches from database for training
        league_ids: League filter when fetch_all_managers=True. Options:
                   "top5" = Top 5 European leagues (default)
                   None = ALL leagues  
                   [8, 564, ...] = Specific league IDs
        output_filename: HTML output filename (default: auto-generated from
                        coach and club names, e.g. "De_Zerbi_Chelsea.html").
                        Can include a path, e.g. "/content/Chelsea_1.html".
    
    Returns:
        Dictionary with analysis results
    
    Example:
        # Recommended: Fetch ALL seasons + top 5 leagues (maximum training data)
        results = run_full_analysis(
            coach_name="Thomas Frank",
            target_club="Tottenham", 
            fetch_all_managers=True,
            league_ids="top5"
        )
        # Result: ~200+ training data points (80 managers × 2-3 seasons)
        
        # Force 2023/24 only (consistent timeframe)
        results = run_full_analysis(
            coach_name="Thomas Frank",
            target_club="Tottenham",
            season_id=21646,
            fetch_all_managers=True
        )
        # Result: ~80 training data points (one per manager)
        
        # Subsequent runs (reuse model - fast!)
        results = run_full_analysis("Thomas Frank", "Arsenal", train_model=False)
    
    Season Behavior:
        season_id=None (default):
          - Trains model with ALL seasons: 2024/25, 2023/24, 2022/23
          - Each manager contributes 1-3 data points (one per season)
          - Example: Model includes "Pep 2024/25", "Pep 2023/24", "Pep 2022/23"
          - Maximizes training data: ~80 managers → ~200+ data points
          - Better clustering with more samples
        
        season_id=23614 (specific):
          - Trains model with ONLY 2024/25 season
          - Each manager contributes 1 data point
          - Consistent timeframe for fair comparison
          - ~80 managers → ~80 data points
    """
    import os
    import json
    from collections import defaultdict
    
    if api_token:
        os.environ["SPORTMONKS_API_TOKEN"] = api_token
    
    Config.set_base_dir(base_dir)
    Config.setup()
    
    training_dir = Config.PROCESSED_DIR / "training"
    
    # Phase 1: Train Manager DNA model (if needed)
    if train_model:
        print("\n" + "=" * 60)
        print("PHASE 1: TRAINING MANAGER DNA MODEL")
        print("=" * 60)
        
        if fetch_all_managers:
            print(f"\nFetching all managers from: {league_ids if league_ids else 'ALL LEAGUES'}")
        
        trainer = ManagerDNATrainer(training_dir=training_dir, season_id=season_id)
        trainer.fetch_manager_data(
            fetch_all=fetch_all_managers,
            league_ids=league_ids
        )
        trainer.extract_features()
        trainer.fit()
        trainer.save()
        
        try:
            trainer.plot_clusters(save_path=training_dir / "manager_clusters_pca.png")
        except:
            pass
    
    # Phase 2: Analyze Squad Fit
    print("\n" + "=" * 60)
    print("PHASE 2: SQUAD FIT ANALYSIS")
    print("=" * 60)
    
    analyzer = SquadFitAnalyzer(
        training_dir=training_dir,
        output_dir=Config.OUTPUT_DIR,
        season_id=season_id
    )
    
    analyzer.load_model()
    analyzer.set_target_manager(coach_name)
    analyzer.fetch_squad(target_club)
    analyzer.calculate_fit_scores()
    analyzer.save()
    
    results = analyzer.get_results()
    
    # Phase 3: Generate Visualizations
    if visualize:
        print("\n" + "=" * 60)
        print("PHASE 3: GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        try:
            # Create legacy-compatible results format for visualizer
            position_groups = defaultdict(list)
            for player in results["squad_fit"]:
                position_groups[player["position_group"]].append(player)
            
            # Generate recruitment priorities
            recruitment = []
            for pos_group, players in position_groups.items():
                if not players:
                    continue
                
                avg_fit = sum(p["fit_score"] for p in players) / len(players)
                
                if avg_fit < 60:
                    gap = 75 - avg_fit
                    cost_map = {"GK": (15, 35), "DEF": (25, 55), "MID": (30, 65), "ATT": (35, 75)}
                    cost_low, cost_high = cost_map.get(pos_group, (20, 50))
                    
                    urgency = "Critical" if avg_fit < 45 else "High" if avg_fit < 55 else "Medium"
                    timeline = "January" if urgency == "Critical" else "Summer"
                    
                    recruitment.append({
                        "position": pos_group,
                        "gap": round(gap, 1),
                        "urgency": urgency,
                        "timeline": timeline,
                        "cost_low": cost_low,
                        "cost_high": cost_high
                    })
            
            recruitment.sort(key=lambda x: x["gap"], reverse=True)
            
            # Create placeholder DNA dimensions
            dna_dimensions = {
                "Possession": 65,
                "Pressing": 72,
                "Build-up": 68,
                "Attacking": 70,
                "Defence": 62,
                "Flexibility": 60,
                "Results": 58
            }
            
            legacy_results = {
                "manager": results["manager"],
                "matches_analysed": 38,
                "primary_formation": "4-3-3",
                "dna_dimensions": dna_dimensions,
                "squad_summary": {
                    "total": len(results["squad_fit"]),
                    "key_enablers": results["classification_counts"]["Key Enabler"],
                    "good_fit": results["classification_counts"]["Good Fit"],
                    "system_dependent": results["classification_counts"]["System Dependent"],
                    "marginalised": results["classification_counts"]["Potentially Marginalised"],
                    "average_fit": results["average_fit"]
                },
                "ideal_xi": results["ideal_xi"],
                "recruitment": recruitment
            }
            
            # Save for visualizer
            with open(Config.OUTPUT_DIR / "aegis_analysis.json", "w") as f:
                json.dump(legacy_results, f, indent=2)
            
            # Save recruitment priorities as CSV
            import csv
            if recruitment:
                with open(Config.OUTPUT_DIR / "recruitment_priorities.csv", "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["position", "gap", "urgency", "timeline", "cost_low", "cost_high"])
                    writer.writeheader()
                    writer.writerows(recruitment)
            
            # Generate interactive dashboard (skip PNG charts)
            viz = AegisVisualizer()
            viz.load_results()

            # Build output filename: use caller's choice, or auto-generate
            # from coach + club so each scenario gets a unique file.
            if output_filename:
                dashboard_filename = output_filename
            else:
                safe_coach = coach_name.replace(" ", "_").replace(".", "")
                safe_club  = target_club.replace(" ", "_")
                dashboard_filename = f"{safe_coach}_{safe_club}.html"

            viz.generate_dashboard(filename=dashboard_filename)

            print(f"\n  * Interactive dashboard: {Config.OUTPUT_DIR / dashboard_filename}")
            print(f"  * All visualizations are embedded in the dashboard")
            
        except Exception as e:
            print(f"  * Visualization error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Manager: {results['manager']}")
    print(f"Club: {results['club']}")
    print(f"Tactical Archetype: {results['archetype']}")
    print(f"Average Squad Fit: {results['average_fit']:.1f}")
    print(f"\nClassifications:")
    print(f"  Key Enablers: {results['classification_counts']['Key Enabler']}")
    print(f"  Good Fit: {results['classification_counts']['Good Fit']}")
    print(f"  System Dependent: {results['classification_counts']['System Dependent']}")
    print(f"  Potentially Marginalised: {results['classification_counts']['Potentially Marginalised']}")
    
    return results
    return results


def run_full_analysis_statsbomb(
    competition_id: int,
    season_id: int,
    team_name: str,
    coach_name: str = None,
    username: str = None,
    password: str = None,
    base_dir: str = "/content/aegis_data",
    train_model: bool = True,
    training_competition_ids: list = None,
    visualize: bool = True,
    max_matches: int = 50
) -> dict:
    """
    Run full MTFI analysis using StatsBomb data.
    
    Run full MTFI analysis using StatsBomb data end-to-end.
    No Sportsmonks dependency.
    
    Args:
        competition_id: StatsBomb competition ID (e.g., 2 for Premier League)
        season_id: StatsBomb season ID (e.g., 317 for 2024/25)
        team_name: Target club name (e.g., "Chelsea")
        coach_name: Manager name (recommended, e.g. "Enzo Maresca")
        username: StatsBomb username (or set SB_USERNAME env var)
        password: StatsBomb password (or set SB_PASSWORD env var)
        base_dir: Base directory for data and outputs
        train_model: Whether to train Manager DNA model (set False after first run)
        training_competition_ids: List of competition IDs for training.
                    More = richer clustering. e.g. [2, 3, 6] for PL+Champ+Eredivisie
        visualize: Generate interactive dashboard
        max_matches: Maximum matches to fetch team stats for
        
    Returns:
        Dictionary with analysis results
        
    Example:
        results = run_full_analysis_statsbomb(
            competition_id=2,      # Premier League
            season_id=282,         # 2024/25
            team_name="Chelsea",
            username="you@email.com",
            password="your_password",
            train_model=True
        )
    """
    import os
    
    if username:
        os.environ["SB_USERNAME"] = username
    if password:
        os.environ["SB_PASSWORD"] = password
    
    Config.set_base_dir(base_dir)
    Config.setup()
    
    # ── Step 1: Train Manager DNA (StatsBomb) ──
    if train_model:
        print("\n" + "=" * 60)
        print("STEP 1: TRAINING MANAGER DNA MODEL (StatsBomb)")
        print("=" * 60)
        
        training_dir = Config.PROCESSED_DIR / "training"
        sb_trainer_client = StatsBombClient()
        
        trainer = ManagerDNATrainer(training_dir=training_dir)
        trainer.fetch_manager_data_statsbomb(
            sb_client=sb_trainer_client,
            competition_id=competition_id,
            season_id=season_id,
            competition_ids=training_competition_ids
        )
        # Note: extract_features() is NOT needed — fetch_manager_data_statsbomb
        # populates manager_features directly from team match stats
        trainer.fit()
        trainer.save()
    
    # ── Step 2: Fetch StatsBomb Data ──
    print("\n" + "=" * 60)
    print("STEP 2: FETCHING STATSBOMB DATA")
    print("=" * 60)
    
    sb_client = StatsBombClient()
    scenario = sb_client.fetch_scenario(
        competition_id=competition_id,
        season_id=season_id,
        team_name=team_name,
        max_matches=max_matches
    )
    
    # ── Step 3: Run StatsBomb ETL ──
    print("\n" + "=" * 60)
    print("STEP 3: STATSBOMB ETL")
    print("=" * 60)
    
    etl = StatsBombETL()
    manager_dna, squad_profiles = etl.run()
    
    # ── Step 4: Squad Fit Analysis ──
    print("\n" + "=" * 60)
    print("STEP 4: SQUAD FIT ANALYSIS")
    print("=" * 60)
    
    analyzer = SquadFitAnalyzer()
    analyzer.load_model()
    
    # Use explicit coach_name if provided, otherwise from match data
    manager_name = coach_name or scenario["manager_info"].get("name", "Unknown")
    
    # Inject the coach name into the DNA so downstream consumers have it
    if coach_name:
        manager_dna["manager"] = coach_name
    analyzer.set_target_manager_from_dna(manager_dna)
    analyzer.calculate_fit_scores_from_profiles(
        squad_profiles, 
        club_name=team_name,
        league_player_stats=scenario.get("player_season_stats")
    )
    analyzer.save()
    
    # ── Step 4b: Generate recruitment priorities + legacy JSON for visualiser ──
    import json
    import csv as _csv
    from collections import defaultdict as _dd
    
    position_groups = _dd(list)
    for p in analyzer.squad_fit:
        position_groups[p.position_group].append(p)
    
    recruitment = []
    for pos_group, players in position_groups.items():
        if not players:
            continue
        avg_fit = sum(p.fit_score for p in players) / len(players)
        if avg_fit < 60:
            gap = 75 - avg_fit
            cost_map = {"GK": (15, 35), "DEF": (25, 55), "MID": (30, 65), "ATT": (35, 75)}
            cost_low, cost_high = cost_map.get(pos_group, (20, 50))
            urgency = "Critical" if avg_fit < 45 else "High" if avg_fit < 55 else "Medium"
            timeline = "January" if urgency == "Critical" else "Summer"
            recruitment.append({
                "position": pos_group, "gap": round(gap, 1),
                "urgency": urgency, "timeline": timeline,
                "cost_low": cost_low, "cost_high": cost_high
            })
    recruitment.sort(key=lambda x: x["gap"], reverse=True)
    
    if recruitment:
        with open(Config.OUTPUT_DIR / "recruitment_priorities.csv", "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=["position", "gap", "urgency", "timeline", "cost_low", "cost_high"])
            writer.writeheader()
            writer.writerows(recruitment)
    
    # Build DNA dimensions for radar chart (Gary's 8 pillars)
    tp = manager_dna.get("tactical_profile", {})
    rp = manager_dna.get("results_profile", {})
    sb = manager_dna.get("statsbomb_enhanced", {})
    
    ppda = sb.get("ppda", tp.get("pressing", {}).get("ppda", 10))
    possession = tp.get("possession", {}).get("avg", 50)
    pass_acc = tp.get("build_up", {}).get("pass_accuracy", 80)
    
    dna_dimensions = {
        # Pillar 1: Shape & Occupation (possession + defensive structure)
        "Shape & Occupation": min(100, round(possession * 1.2 + rp.get("clean_sheet_pct", 30) * 0.3, 0)),
        # Pillar 2: Build-up vs Direct (pass accuracy + patience)
        "Build-up": min(100, round(pass_acc, 0)),
        # Pillar 3: Chance Creation (xG quality)
        "Chance Creation": min(100, round(sb.get("np_xg_per_game", rp.get("goals_per_game", 1.5)) * 40, 0)),
        # Pillar 4: Press & Counterpress (PPDA inverted)
        "Press & Counterpress": min(100, round(max(0, 30 - ppda) * 4, 0)),
        # Pillar 5: Block & Line Height (defensive distance + solidity)
        "Block & Line Height": min(100, round(
            tp.get("pressing", {}).get("defensive_distance", 40) * 1.5 + 
            max(0, 2.0 - rp.get("conceded_per_game", 1.2)) * 20, 0)),
        # Pillar 6: Transitions (counter-attacking threat)
        "Transitions": min(100, round(
            sb.get("counter_attacking_shots_pg", 
                   tp.get("attacking", {}).get("counter_attacking_shots_pg", 1)) * 25 +
            sb.get("high_press_shots_pg", 0) * 20, 0)),
        # Pillar 7: Width & Overloads
        "Width & Overloads": min(100, round(sb.get("width_usage", 5) * 10, 0)),
        # Pillar 8: Set Pieces
        "Set Pieces": min(100, round(sb.get("set_piece_emphasis", 20) * 2, 0)),
    }
    
    legacy_results = {
        "manager": manager_name,
        "matches_analysed": manager_dna.get("matches_analysed", 0),
        "primary_formation": manager_dna.get("formation_profile", {}).get("primary", "4-3-3"),
        "dna_dimensions": dna_dimensions,
        "squad_summary": {
            "total": len(analyzer.squad_fit),
            "key_enablers": sum(1 for p in analyzer.squad_fit if p.classification == "Key Enabler"),
            "good_fit": sum(1 for p in analyzer.squad_fit if p.classification == "Good Fit"),
            "system_dependent": sum(1 for p in analyzer.squad_fit if p.classification == "System Dependent"),
            "marginalised": sum(1 for p in analyzer.squad_fit if p.classification == "Potentially Marginalised"),
            "average_fit": round(sum(p.fit_score for p in analyzer.squad_fit) / max(len(analyzer.squad_fit), 1), 1)
        },
        "ideal_xi": analyzer.ideal_xi,
        "recruitment": recruitment
    }
    
    with open(Config.OUTPUT_DIR / "aegis_analysis.json", "w") as f:
        json.dump(legacy_results, f, indent=2)
    
    # ── Step 5: Visualise ──
    if visualize:
        print("\n" + "=" * 60)
        print("STEP 5: GENERATING DASHBOARD")
        print("=" * 60)
        try:
            viz = AegisVisualizer()
            viz.generate_dashboard()
        except Exception as e:
            print(f"⚠ Dashboard generation: {e}")
    
    # Build results dict
    avg_fit = sum(p.fit_score for p in analyzer.squad_fit) / len(analyzer.squad_fit) if analyzer.squad_fit else 0
    
    classification_counts = {}
    for cls_name in ["Key Enabler", "Good Fit", "System Dependent", "Potentially Marginalised"]:
        classification_counts[cls_name] = sum(
            1 for p in analyzer.squad_fit if p.classification == cls_name
        )
    
    results = {
        "manager": manager_name,
        "club": team_name,
        "data_source": "statsbomb",
        "archetype": analyzer.target_cluster_name or "Unknown",
        "average_fit": round(avg_fit, 1),
        "classification_counts": classification_counts,
        "squad_profiles": squad_profiles,
        "manager_dna": manager_dna,
    }
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE (StatsBomb)")
    print("=" * 60)
    print(f"Manager: {results['manager']}")
    print(f"Club: {results['club']}")
    print(f"Tactical Archetype: {results['archetype']}")
    print(f"Average Squad Fit: {results['average_fit']:.1f}")
    
    return results
