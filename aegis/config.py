"""
Aegis Configuration
===================
Central configuration for paths and settings.
Supports both Sportsmonks and StatsBomb data sources.
"""

from pathlib import Path


class Config:
    """
    Configuration settings for Aegis.
    
    Usage in Colab:
        from aegis import Config
        
        # Set your base directory
        Config.set_base_dir("/content/aegis_data")
        
        # Or use defaults
        Config.setup()
    """
    
    # =========================================================================
    # SPORTSMONKS SETTINGS
    # =========================================================================
    BASE_URL = "https://api.sportmonks.com/v3"
    REQUESTS_PER_SECOND = 2.5
    
    # =========================================================================
    # STATSBOMB SETTINGS
    # =========================================================================
    STATSBOMB_BASE_URL = "https://data.statsbombservices.com"
    STATSBOMB_ALT_URL = "https://data.statsbomb.com"  # lineups, player-mapping
    STATSBOMB_REQUESTS_PER_SECOND = 3.0  # StatsBomb is generally more lenient
    
    # StatsBomb API versions
    STATSBOMB_API_VERSIONS = {
        "competitions": "v4",
        "matches": "v6",
        "lineups": "v4",
        "events": "v8",
        "360_frames": "v2",
        "player_match_stats": "v5",
        "player_season_stats": "v4",
        "team_match_stats": "v1",
        "player_mapping": "v1",
    }
    
    # =========================================================================
    # FILE PATHS
    # =========================================================================
    BASE_DIR = Path("/content/aegis_data")
    CACHE_DIR = BASE_DIR / "cache"
    DATA_DIR = BASE_DIR / "raw"
    PROCESSED_DIR = BASE_DIR / "processed"
    OUTPUT_DIR = BASE_DIR / "outputs"
    
    # =========================================================================
    # SPORTSMONKS KNOWN IDS
    # =========================================================================
    KNOWN_TEAMS = {
        "arsenal": 1,
        "tottenham": 6,
        "aston_villa": 7,
        "west_ham": 8,
        "liverpool": 9,
        "everton": 10,
        "newcastle": 11,
        "crystal_palace": 12,
        "wolves": 13,
        "man_united": 14,
        "brighton": 15,
        "fulham": 16,
        "man_city": 17,
        "chelsea": 18,
        "bournemouth": 19,
        "nottingham_forest": 21,
        "brentford": 63,
    }
    
    KNOWN_SEASONS = {
        "2024/25": 23614,
        "2023/24": 21646,
        "2022/23": 19735,
    }
    
    KNOWN_LEAGUES = {
        "premier_league": 8,
        "la_liga": 564,
        "bundesliga": 82,
        "serie_a": 384,
        "ligue_1": 301,
    }
    
    # =========================================================================
    # STATSBOMB KNOWN IDS
    # =========================================================================
    STATSBOMB_COMPETITIONS = {
        # Confirmed IDs from account (March 2026)
        "premier_league": 2,
        "championship": 3,
        "league_one": 4,
        "league_two": 5,
        "eredivisie": 6,
    }
    
    STATSBOMB_SEASONS = {
        # Premier League
        "premier_league_2024/25": {"competition_id": 2, "season_id": 317},
        "premier_league_2023/24": {"competition_id": 2, "season_id": 281},
        "premier_league_2022/23": {"competition_id": 2, "season_id": 235},
        "premier_league_2025/26": {"competition_id": 2, "season_id": 318},
        # Championship
        "championship_2024/25": {"competition_id": 3, "season_id": 317},
        "championship_2023/24": {"competition_id": 3, "season_id": 281},
        "championship_2022/23": {"competition_id": 3, "season_id": 235},
        "championship_2025/26": {"competition_id": 3, "season_id": 318},
        # League One
        "league_one_2024/25": {"competition_id": 4, "season_id": 317},
        "league_one_2023/24": {"competition_id": 4, "season_id": 281},
        "league_one_2022/23": {"competition_id": 4, "season_id": 235},
        # League Two
        "league_two_2024/25": {"competition_id": 5, "season_id": 317},
        "league_two_2023/24": {"competition_id": 5, "season_id": 281},
        "league_two_2022/23": {"competition_id": 5, "season_id": 235},
        # Eredivisie
        "eredivisie_2024/25": {"competition_id": 6, "season_id": 317},
        "eredivisie_2023/24": {"competition_id": 6, "season_id": 281},
        "eredivisie_2022/23": {"competition_id": 6, "season_id": 235},
    }
    
    # StatsBomb position ID to position name mapping (from Events API Appendix 1)
    STATSBOMB_POSITIONS = {
        1: "Goalkeeper",
        2: "Right Back",
        3: "Right Center Back",
        4: "Center Back",
        5: "Left Center Back",
        6: "Left Back",
        7: "Right Wing Back",
        8: "Left Wing Back",
        9: "Right Defensive Midfield",
        10: "Center Defensive Midfield",
        11: "Left Defensive Midfield",
        12: "Right Midfield",
        13: "Right Center Midfield",
        14: "Center Midfield",
        15: "Left Center Midfield",
        16: "Left Midfield",
        17: "Right Wing",
        18: "Right Attacking Midfield",
        19: "Center Attacking Midfield",
        20: "Left Attacking Midfield",
        21: "Left Wing",
        22: "Right Center Forward",
        23: "Striker",
        24: "Left Center Forward",
        25: "Secondary Striker",
    }
    
    @classmethod
    def set_base_dir(cls, path: str):
        """
        Update base directory and all derived paths.
        
        Args:
            path: Base directory path (e.g., "/content/aegis_data" for Colab)
        """
        cls.BASE_DIR = Path(path)
        cls.CACHE_DIR = cls.BASE_DIR / "cache"
        cls.DATA_DIR = cls.BASE_DIR / "raw"
        cls.PROCESSED_DIR = cls.BASE_DIR / "processed"
        cls.OUTPUT_DIR = cls.BASE_DIR / "outputs"
    
    @classmethod
    def setup(cls):
        """Create all required directories."""
        for dir_path in [cls.CACHE_DIR, cls.DATA_DIR, cls.PROCESSED_DIR, cls.OUTPUT_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directories created: {cls.BASE_DIR}")
        return cls
    
    @classmethod
    def info(cls):
        """Print current configuration."""
        print("Aegis Configuration")
        print("=" * 40)
        print(f"Base directory: {cls.BASE_DIR}")
        print(f"Cache:          {cls.CACHE_DIR}")
        print(f"Raw data:       {cls.DATA_DIR}")
        print(f"Processed:      {cls.PROCESSED_DIR}")
        print(f"Outputs:        {cls.OUTPUT_DIR}")
        print(f"\nSportsmonks:    {cls.BASE_URL}")
        print(f"StatsBomb:      {cls.STATSBOMB_BASE_URL}")
