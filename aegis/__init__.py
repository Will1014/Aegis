"""
Aegis - Manager Tactical Fit Intelligence
==========================================

A Python package for analyzing tactical fit between football managers
and club squads.

Quick Start (Colab):
--------------------
# Install
!pip install requests matplotlib

# Setup
import os
os.environ["SPORTMONKS_API_TOKEN"] = "your_token_here"

from aegis import Config, SportsmonksClient, ETLPipeline, AegisAnalyzer, AegisVisualizer

# Configure paths (for Colab)
Config.set_base_dir("/content/aegis_data")
Config.setup()

# Fetch data
client = SportsmonksClient()
client.fetch_scenario("Thomas Frank", "Tottenham")

# Process
etl = ETLPipeline()
etl.run()

# Analyse
analyzer = AegisAnalyzer()
results = analyzer.run()

# Visualize
viz = AegisVisualizer()
viz.load_results()
viz.plot_all()
"""

__version__ = "0.1.0"
__author__ = "Aegis Football Advisory Group"

from .config import Config
from .client import SportsmonksClient
from .etl import ETLPipeline
from .analysis import AegisAnalyzer, PlayerFit
from .visualizations import AegisVisualizer

__all__ = [
    "Config",
    "SportsmonksClient",
    "ETLPipeline",
    "AegisAnalyzer",
    "AegisVisualizer",
    "PlayerFit"
]


def run_full_analysis(
    coach_name: str,
    target_club: str,
    api_token: str = None,
    base_dir: str = "/content/aegis_data",
    visualize: bool = True
):
    """
    Run complete Aegis analysis pipeline.
    
    Args:
        coach_name: Manager name (e.g., "Thomas Frank")
        target_club: Target club name (e.g., "Tottenham")
        api_token: Sportsmonks API token (or set SPORTMONKS_API_TOKEN env var)
        base_dir: Base directory for data and outputs
        visualize: Generate visualization charts
    
    Returns:
        Dictionary with analysis results
    
    Example:
        from aegis import run_full_analysis
        
        results = run_full_analysis(
            coach_name="Thomas Frank",
            target_club="Tottenham",
            api_token="your_token"
        )
    """
    import os
    
    # Set token if provided
    if api_token:
        os.environ["SPORTMONKS_API_TOKEN"] = api_token
    
    # Configure
    Config.set_base_dir(base_dir)
    Config.setup()
    
    # Fetch
    client = SportsmonksClient()
    client.fetch_scenario(coach_name, target_club)
    
    # ETL
    etl = ETLPipeline()
    etl.run()
    
    # Analyse
    analyzer = AegisAnalyzer()
    results = analyzer.run()
    
    # Visualize
    if visualize:
        try:
            viz = AegisVisualizer()
            viz.load_results()
            viz.plot_all()
        except Exception as e:
            print(f"⚠ Visualization error: {e}")
    
    return results
