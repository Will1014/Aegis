"""
Aegis Configuration
===================
Central configuration for paths and settings.
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
    
    # API Settings
    BASE_URL = "https://api.sportmonks.com/v3"
    REQUESTS_PER_SECOND = 2.5
    
    # File paths - defaults for Colab
    BASE_DIR = Path("/content/aegis_data")
    CACHE_DIR = BASE_DIR / "cache"
    DATA_DIR = BASE_DIR / "raw"
    PROCESSED_DIR = BASE_DIR / "processed"
    OUTPUT_DIR = BASE_DIR / "outputs"
    
    # Known IDs for convenience
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
