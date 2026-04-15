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

__version__ = "0.2.0"
__author__ = "Aegis Football Advisory Group"

from .config import Config
from .client import SportsmonksClient
from .etl import ETLPipeline
from .analysis import (
    # New ML-based classes
    ManagerDNATrainer,
    SquadFitAnalyzer,
    PlayerFit,
    ManagerProfile,
    # Constants
    MANAGER_DNA_FEATURES,
    PLAYER_FIT_FEATURES,
    FIT_THRESHOLDS,
    DEFAULT_MANAGERS,
    POSITION_GROUPS,
    # Legacy compatibility
    AegisAnalyzer,
)
from .visualizations import AegisVisualizer

__all__ = [
    # Config & Client
    "Config",
    "SportsmonksClient",
    
    # ETL
    "ETLPipeline",
    
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
    "train_manager_dna",
    "analyse_squad_fit",
]


def train_manager_dna(
    api_token: str = None,
    base_dir: str = "/content/aegis_data",
    season_id: int = 23614,
    managers: list = None,
    n_clusters: int = None
) -> dict:
    """
    Train the Manager DNA clustering model.
    
    Args:
        api_token: Sportsmonks API token (or set SPORTMONKS_API_TOKEN env var)
        base_dir: Base directory for data and outputs
        season_id: Season to analyse (default: 2024/25)
        managers: Custom list of managers (uses DEFAULT_MANAGERS if None)
        n_clusters: Number of clusters (auto-determined if None)
    
    Returns:
        Dictionary with training summary
    
    Example:
        from aegis import train_manager_dna
        
        summary = train_manager_dna(api_token="your_token")
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
    
    trainer.fetch_manager_data(managers=managers)
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
    visualize: bool = True
):
    """
    Run complete Aegis analysis pipeline (legacy mode).
    
    Note: For ML-based analysis, use train_manager_dna() then analyse_squad_fit().
    
    Args:
        coach_name: Manager name (e.g., "Thomas Frank")
        target_club: Target club name (e.g., "Tottenham")
        api_token: Sportsmonks API token
        base_dir: Base directory for data and outputs
        visualize: Generate visualization charts
    
    Returns:
        Dictionary with analysis results
    """
    import os
    
    if api_token:
        os.environ["SPORTMONKS_API_TOKEN"] = api_token
    
    Config.set_base_dir(base_dir)
    Config.setup()
    
    # Fetch
    client = SportsmonksClient()
    client.fetch_scenario(coach_name, target_club)
    
    # ETL
    etl = ETLPipeline()
    etl.run()
    
    # Analyse (legacy)
    analyzer = AegisAnalyzer()
    results = analyzer.run()
    
    # Visualize
    if visualize:
        try:
            viz = AegisVisualizer()
            viz.load_results()
            viz.plot_all()
            # Generate interactive dashboard
            viz.generate_dashboard_v2(
                manager_name=coach_name,
                target_club=target_club
            )
        except Exception as e:
            print(f"⚠ Visualization error: {e}")
    
    return results
