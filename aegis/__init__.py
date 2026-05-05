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
    PILLAR_PERCENTILE_MAP,
    PILLAR_PLAYER_DEMANDS,
    PLAYER_FIT_FEATURES,
    FIT_THRESHOLDS,
    DEFAULT_MANAGERS,
    POSITION_GROUPS,
    # Legacy compatibility
    AegisAnalyzer,
)
from .visualizations import AegisVisualizer
from .player_dossier import PlayerDossierGenerator, generate_player_dossier

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
    
    # Player Dossier
    "PlayerDossierGenerator",
    "generate_player_dossier",
    
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


def _discover_teams_and_managers(sb_client, competition_ids, season_id):
    """
    Discover all teams and their managers from StatsBomb match data.
    
    Args:
        sb_client: StatsBombClient instance
        competition_ids: List of competition IDs to scan
        season_id: StatsBomb season ID
        
    Returns:
        Tuple of (team_names: list[str], manager_names: list[str],
                  team_manager_map: dict[str, str])
        team_manager_map maps team_name → most-recent manager_name
    """
    team_manager_map = {}   # team_name → manager_name
    
    for comp_id in competition_ids:
        matches = sb_client.get_matches(comp_id, season_id)
        if not matches:
            continue
        
        for match in matches:
            home = match.get("home_team", {})
            away = match.get("away_team", {})
            
            h_name = (
                home.get("home_team_name") or home.get("name") or
                home.get("team_name") or ""
            )
            a_name = (
                away.get("away_team_name") or away.get("name") or
                away.get("team_name") or ""
            )
            
            def _mgr_name(side):
                mgrs = side.get("managers")
                if isinstance(mgrs, list) and mgrs:
                    m = mgrs[0]
                    return m.get("name") or m.get("nickname") or "Unknown"
                if isinstance(mgrs, dict):
                    return mgrs.get("name") or mgrs.get("nickname") or "Unknown"
                return "Unknown"
            
            if h_name and h_name not in team_manager_map:
                team_manager_map[h_name] = _mgr_name(home)
            if a_name and a_name not in team_manager_map:
                team_manager_map[a_name] = _mgr_name(away)
    
    team_names = sorted(team_manager_map.keys())
    manager_names = sorted(set(team_manager_map.values()) - {"Unknown"})
    
    return team_names, manager_names, team_manager_map


def run_full_analysis_statsbomb(
    target_league_id,
    season_id: int,
    team_name = "all",
    coach_name = None,
    username: str = None,
    password: str = None,
    base_dir: str = "/content/aegis_data",
    train_model: bool = True,
    training_league_ids: list = None,
    visualize: bool = True,
    max_matches: int = 50,
    output_file: str = None
):
    """
    Run full MTFI analysis using StatsBomb data end-to-end.
    No Sportsmonks dependency.
    
    Supports single values, lists, or "all" for both team_name and
    coach_name so you can batch-produce analyses in one call.
    
    Args:
        target_league_id: The league(s) the target club plays in.
            Single int (e.g. 2 for Premier League) or list of ints
            (e.g. [2, 6] to search PL + Eredivisie for cross-league scenarios).
        season_id: StatsBomb season ID (e.g., 317 for 2024/25)
        team_name: Target club(s). Accepts:
            - A single string  (e.g. "Chelsea")
            - A list of strings (e.g. ["Chelsea", "Arsenal"])
            - "all" — every team in the target league(s) for the season
        coach_name: Manager(s). Accepts:
            - A single string  (e.g. "Enzo Maresca")
            - A list of strings (e.g. ["Enzo Maresca", "Mikel Arteta"])
            - "all" — every manager discovered in the target league(s)
            - None — auto-detect the incumbent manager for each team
        username: StatsBomb username (or set SB_USERNAME env var)
        password: StatsBomb password (or set SB_PASSWORD env var)
        base_dir: Base directory for data and outputs
        train_model: Whether to train Manager DNA model (set False after first run)
        training_league_ids: League IDs for the training set (the pool of
            managers the model clusters). More leagues = richer clustering.
            e.g. [2, 3, 6] for PL + Championship + Eredivisie.
            Defaults to target_league_id if not set.
        visualize: Generate interactive dashboard
        max_matches: Maximum matches to fetch team stats for
        output_file: Dashboard filename override (only used for single-combo runs)
        
    Returns:
        - Single dict when only one (team, coach) combination is run
        - List of dicts when multiple combinations are run
        
    Examples:
        # ── Single scenario (backward-compatible) ──
        results = run_full_analysis_statsbomb(
            target_league_id=2, season_id=317,
            team_name="Chelsea", coach_name="Enzo Maresca"
        )
        
        # ── All teams, incumbent managers ──
        results = run_full_analysis_statsbomb(
            target_league_id=2, season_id=317,
            team_name="all", coach_name=None
        )
        
        # ── One coach at every club ──
        results = run_full_analysis_statsbomb(
            target_league_id=2, season_id=317,
            team_name="all", coach_name="Mikel Arteta"
        )
        
        # ── Every coach at one club ──
        results = run_full_analysis_statsbomb(
            target_league_id=2, season_id=317,
            team_name="Chelsea", coach_name="all"
        )
        
        # ── Explicit lists ──
        results = run_full_analysis_statsbomb(
            target_league_id=2, season_id=317,
            team_name=["Chelsea", "Arsenal"],
            coach_name=["Enzo Maresca", "Mikel Arteta"]
        )
    """
    import os
    
    if username:
        os.environ["SB_USERNAME"] = username
    if password:
        os.environ["SB_PASSWORD"] = password
    
    Config.set_base_dir(base_dir)
    Config.setup()
    
    # Normalise target_league_id to a list
    if isinstance(target_league_id, int):
        target_league_ids = [target_league_id]
    else:
        target_league_ids = list(target_league_id)
    
    # Primary league is the first in the list (used for squad fetching)
    primary_league_id = target_league_ids[0]
    
    # ── Step 0: Discover teams / managers if "all" requested ──
    sb_client = StatsBombClient()
    
    needs_discovery = (
        (isinstance(team_name, str) and team_name.lower() == "all") or
        (isinstance(coach_name, str) and coach_name.lower() == "all")
    )
    
    all_team_names = []
    all_manager_names = []
    team_manager_map = {}
    
    if needs_discovery:
        print("\n" + "=" * 60)
        print("STEP 0: DISCOVERING TEAMS & MANAGERS")
        print("=" * 60)
        all_team_names, all_manager_names, team_manager_map = \
            _discover_teams_and_managers(sb_client, target_league_ids, season_id)
        print(f"  Found {len(all_team_names)} teams, {len(all_manager_names)} managers")
        for t in all_team_names:
            print(f"    • {t:30s}  →  {team_manager_map.get(t, '?')}")
    
    # ── Resolve team_name list ──
    if isinstance(team_name, str) and team_name.lower() == "all":
        teams_to_run = all_team_names
    elif isinstance(team_name, list):
        teams_to_run = list(team_name)
    else:
        teams_to_run = [team_name]
    
    # ── Resolve coach_name list (per-team if None) ──
    # "all"  → every discovered manager against every team
    # list   → each listed manager against every team
    # str    → that single manager against every team
    # None   → auto-detect incumbent per team
    coaches_are_fixed = True   # same coach list for every team?
    
    if isinstance(coach_name, str) and coach_name.lower() == "all":
        coaches_to_run = all_manager_names
    elif isinstance(coach_name, list):
        coaches_to_run = list(coach_name)
    elif isinstance(coach_name, str):
        coaches_to_run = [coach_name]
    else:
        # None → auto-detect per team (handled inside the loop)
        coaches_to_run = None
        coaches_are_fixed = False
    
    total_combos = len(teams_to_run) * (len(coaches_to_run) if coaches_to_run else 1)
    is_batch = total_combos > 1
    
    if is_batch:
        coach_desc = (
            f"{len(coaches_to_run)} managers" if coaches_to_run
            else "incumbent manager per team"
        )
        print(f"\n{'=' * 60}")
        print(f"BATCH MODE: {len(teams_to_run)} teams × {coach_desc} = {total_combos} analyses")
        print(f"{'=' * 60}")
    
    # ── Step 1: Train Manager DNA (once) ──
    if train_model:
        print("\n" + "=" * 60)
        print("STEP 1: TRAINING MANAGER DNA MODEL (StatsBomb)")
        print("=" * 60)
        
        training_dir = Config.PROCESSED_DIR / "training"
        sb_trainer_client = StatsBombClient()
        
        # training_league_ids defaults to target_league_ids if not explicitly set
        effective_training_ids = training_league_ids or target_league_ids
        
        trainer = ManagerDNATrainer(training_dir=training_dir)
        trainer.fetch_manager_data_statsbomb(
            sb_client=sb_trainer_client,
            competition_id=primary_league_id,
            season_id=season_id,
            competition_ids=effective_training_ids
        )
        trainer.fit()
        trainer.save()
    
    # ── Step 2+: Loop over each (team, coach) combination ──
    all_results = []
    combo_num = 0
    
    for t_idx, current_team in enumerate(teams_to_run):
        # Determine coaches to pair with this team
        if coaches_to_run is not None:
            team_coaches = coaches_to_run
        else:
            # Auto-detect incumbent: use discovery map or let fetch_scenario find it
            team_coaches = [None]  # sentinel — resolved after fetch_scenario
        
        # ── Fetch scenario data for this team (once per team) ──
        print(f"\n{'=' * 60}")
        print(f"TEAM {t_idx + 1}/{len(teams_to_run)}: {current_team}")
        print(f"{'=' * 60}")
        
        print(f"\n  Fetching StatsBomb data for {current_team}...")
        try:
            scenario = sb_client.fetch_scenario(
                competition_id=primary_league_id,
                season_id=season_id,
                team_name=current_team,
                max_matches=max_matches
            )
        except ValueError as e:
            print(f"  ⚠ Skipping {current_team}: {e}")
            continue
        
        # ── Run ETL (once per team) ──
        print(f"\n  Running ETL for {current_team}...")
        etl = StatsBombETL()
        manager_dna, squad_profiles = etl.run()
        
        # If coach_name was None, resolve the incumbent from scenario data
        if not coaches_are_fixed and team_coaches == [None]:
            incumbent = scenario["manager_info"].get("name", "Unknown")
            team_coaches = [incumbent]
        
        # ── Score each coach against this team's squad ──
        for c_idx, current_coach in enumerate(team_coaches):
            combo_num += 1
            manager_name = current_coach or scenario["manager_info"].get("name", "Unknown")
            
            if is_batch:
                print(f"\n  ── Combo {combo_num}/{total_combos}: "
                      f"{manager_name} → {current_team} ──")
            else:
                print(f"\n  Manager: {manager_name}")
            
            result = _run_single_statsbomb_analysis(
                team_name=current_team,
                manager_name=manager_name,
                scenario=scenario,
                manager_dna=manager_dna,
                squad_profiles=squad_profiles,
                visualize=visualize,
                output_file=output_file if not is_batch else None,
                coach_name_override=current_coach,
            )
            
            if result:
                all_results.append(result)
    
    # ── Summary ──
    if is_batch:
        print(f"\n{'=' * 60}")
        print(f"BATCH COMPLETE: {len(all_results)}/{total_combos} analyses succeeded")
        print(f"{'=' * 60}")
        for r in all_results:
            print(f"  • {r['manager']:25s} → {r['club']:25s}  "
                  f"Fit: {r['average_fit']:.1f}  "
                  f"Archetype: {r['archetype']}")
        return all_results
    elif all_results:
        return all_results[0]
    else:
        return {}


def _run_single_statsbomb_analysis(
    team_name: str,
    manager_name: str,
    scenario: dict,
    manager_dna: dict,
    squad_profiles: list,
    visualize: bool = True,
    output_file: str = None,
    coach_name_override: str = None,
) -> dict:
    """
    Run squad-fit analysis + dashboard for a single (team, coach) combo.
    
    This is the inner workhorse called by run_full_analysis_statsbomb
    for each combination.  The model is already trained and data already
    fetched — this handles Steps 4 & 5 only.
    """
    import json
    import csv as _csv
    from collections import defaultdict as _dd

    analyzer = SquadFitAnalyzer()
    analyzer.load_model()
    
    # Look up the manager in the TRAINING DATA to get their actual tactical profile.
    # This is critical for hypothetical scenarios — "what if Arteta managed Chelsea?"
    # uses Arteta's profile from Arsenal, not Chelsea's current profile.
    try:
        analyzer.set_target_manager(manager_name)
    except ValueError:
        # Manager not in training data — fall back to ETL DNA (target team's profile)
        print(f"  ⚠ '{manager_name}' not in training data, using target team's profile")
        dna_copy = dict(manager_dna)
        if coach_name_override:
            dna_copy["manager"] = coach_name_override
        analyzer.set_target_manager_from_dna(dna_copy)
    analyzer.calculate_fit_scores_from_profiles(
        squad_profiles, 
        club_name=team_name,
        league_player_stats=scenario.get("player_season_stats")
    )
    analyzer.save()
    
    # ── Generate recruitment priorities + legacy JSON for visualiser ──
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
    # Use the manager's pillar scores from training data when available
    # (correct for hypothetical scenarios), fall back to ETL DNA
    if analyzer.manager_pillar_scores:
        ps = analyzer.manager_pillar_scores
        score_vals = list(ps.values())
        
        # Detect degenerate scores: all pillars within 5 points of each other
        # This means the training data had no variance (all-default feature values)
        is_degenerate = len(score_vals) > 0 and (max(score_vals) - min(score_vals)) <= 5
        
        if is_degenerate:
            print("  ⚠ Training-based pillar scores are degenerate — using ETL pillar scores")
            etl_scores = manager_dna.get("pillar_scores", {})
            if etl_scores and len(set(etl_scores.values())) > 1:
                # ETL scores have real variance — use them directly
                ps = etl_scores
            else:
                # Both degenerate — use neutral scores and warn
                print("  ⚠ ETL pillar scores also degenerate — radar will show neutral values")

        dna_dimensions = {
            "Shape & Occupation": ps.get("shape_occupation", 50),
            "Build-up": ps.get("build_up", 50),
            "Chance Creation": ps.get("chance_creation", 50),
            "Press & Counterpress": ps.get("press_counterpress", 50),
            "Block & Line Height": ps.get("block_line_height", 50),
            "Transitions": ps.get("transitions", 50),
            "Width & Overloads": ps.get("width_overloads", 50),
            "Set Pieces": ps.get("set_pieces", 50),
        }
    else:
        # Fallback: neutral scores (should rarely trigger now that
        # percentile-based pillar scores are computed in analysis.py)
        print("  ⚠ No pillar scores available — using neutral 50s for radar chart")
        dna_dimensions = {
            "Shape & Occupation": 50,
            "Build-up": 50,
            "Chance Creation": 50,
            "Press & Counterpress": 50,
            "Block & Line Height": 50,
            "Transitions": 50,
            "Width & Overloads": 50,
            "Set Pieces": 50,
        }
    
    legacy_results = {
        "manager": analyzer.target_manager or manager_name,
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
    
    # Inject dna_dimensions into squad_fit_summary.json so generate_dashboard
    # can find it even without going through load_results() / aegis_analysis.json.
    # squad_fit_summary.json is written by analyzer.save() before dna_dimensions
    # is built, so it needs to be patched here.
    summary_path = Config.OUTPUT_DIR / "squad_fit_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                _summary = json.load(f)
            _summary["dna_dimensions"] = dna_dimensions
            with open(summary_path, "w") as f:
                json.dump(_summary, f, indent=2)
        except Exception:
            pass  # non-fatal; visualizer will fall back to aegis_analysis.json
    
    # ── Visualise ──
    if visualize:
        print(f"\n  Generating dashboard...")
        try:
            viz = AegisVisualizer()
            viz.load_results()
            
            # Build output filename: user override, or auto from coach + club
            if output_file:
                dashboard_filename = output_file
            else:
                safe_coach = (analyzer.target_manager or manager_name or "Unknown").replace(" ", "_").replace(".", "")
                safe_club = team_name.replace(" ", "_").replace("&", "and")
                dashboard_filename = f"{safe_coach}___{safe_club}.html"
            
            viz.generate_dashboard(filename=dashboard_filename)
            print(f"  ✓ Dashboard: {Config.OUTPUT_DIR / dashboard_filename}")
        except Exception as e:
            print(f"  ⚠ Dashboard generation: {e}")
    
    # Build results dict
    avg_fit = sum(p.fit_score for p in analyzer.squad_fit) / len(analyzer.squad_fit) if analyzer.squad_fit else 0
    
    classification_counts = {}
    for cls_name in ["Key Enabler", "Good Fit", "System Dependent", "Potentially Marginalised"]:
        classification_counts[cls_name] = sum(
            1 for p in analyzer.squad_fit if p.classification == cls_name
        )
    
    results = {
        "manager": analyzer.target_manager or manager_name,
        "club": team_name,
        "data_source": "statsbomb",
        "archetype": analyzer.target_cluster_name or "Unknown",
        "average_fit": round(avg_fit, 1),
        "classification_counts": classification_counts,
        "squad_profiles": squad_profiles,
        "manager_dna": manager_dna,
    }
    
    print(f"  ✓ {results['manager']} → {results['club']}: "
          f"Fit {results['average_fit']:.1f}, Archetype: {results['archetype']}")
    
    return results
