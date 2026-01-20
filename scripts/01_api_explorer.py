"""
Aegis Platform - MTFI Prototype - Sportsmonks API Explorer
Day 1: API Discovery & Testing

This script helps you:
1. Test API connectivity
2. Explore available endpoints
3. Pull sample data for a manager and team
4. Document data structure and available fields
"""

import requests
import json
from datetime import datetime, timedelta
from pathlib import Path

# ============================================
# CONFIGURATION - UPDATE THESE VALUES
# ============================================

API_TOKEN = "YOUR_API_TOKEN_HERE"  # Get from MySportmonks dashboard
BASE_URL = "https://api.sportmonks.com/v3/football"

# Target entities for prototype (update after kickoff call)
TARGET_MANAGER_NAME = "Thomas Frank"  # Example
TARGET_TEAM_NAME = "Tottenham"  # Example  
TARGET_LEAGUE_ID = 8  # Premier League

# Output paths
OUTPUT_DIR = Path("../data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# API HELPER FUNCTIONS
# ============================================

def make_request(endpoint: str, params: dict = None) -> dict:
    """Make authenticated request to Sportsmonks API"""
    url = f"{BASE_URL}/{endpoint}"
    
    if params is None:
        params = {}
    params["api_token"] = API_TOKEN
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None


def save_json(data: dict, filename: str):
    """Save response to JSON file for inspection"""
    filepath = OUTPUT_DIR / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {filepath}")


# ============================================
# 1. TEST API CONNECTIVITY
# ============================================

def test_connection():
    """Test basic API connectivity"""
    print("\n" + "="*50)
    print("1. TESTING API CONNECTION")
    print("="*50)
    
    # Test with leagues endpoint
    response = make_request("leagues", {"per_page": 5})
    
    if response and "data" in response:
        print("✅ API connection successful!")
        print(f"   Retrieved {len(response['data'])} leagues")
        return True
    else:
        print("❌ API connection failed. Check your token.")
        return False


# ============================================
# 2. EXPLORE AVAILABLE DATA
# ============================================

def explore_leagues():
    """Get available leagues in subscription"""
    print("\n" + "="*50)
    print("2. EXPLORING AVAILABLE LEAGUES")
    print("="*50)
    
    response = make_request("leagues", {
        "include": "currentSeason",
        "per_page": 50
    })
    
    if response and "data" in response:
        print(f"\nAvailable leagues ({len(response['data'])}):")
        for league in response["data"][:20]:  # Show first 20
            season = league.get("currentSeason", {})
            season_name = season.get("name", "N/A") if season else "N/A"
            print(f"  - ID {league['id']}: {league['name']} ({season_name})")
        
        save_json(response, "leagues.json")
        return response["data"]
    return []


def explore_teams(league_id: int):
    """Get teams for a specific league"""
    print("\n" + "="*50)
    print(f"3. EXPLORING TEAMS IN LEAGUE {league_id}")
    print("="*50)
    
    response = make_request(f"teams/seasons/{get_current_season_id(league_id)}", {
        "include": "coaches",
        "per_page": 50
    })
    
    if response and "data" in response:
        print(f"\nTeams found ({len(response['data'])}):")
        for team in response["data"]:
            coaches = team.get("coaches", [])
            coach_name = coaches[0].get("name", "Unknown") if coaches else "Unknown"
            print(f"  - ID {team['id']}: {team['name']} (Coach: {coach_name})")
        
        save_json(response, f"teams_league_{league_id}.json")
        return response["data"]
    return []


def get_current_season_id(league_id: int) -> int:
    """Get current season ID for a league"""
    response = make_request(f"leagues/{league_id}", {"include": "currentSeason"})
    if response and "data" in response:
        current_season = response["data"].get("currentSeason")
        if current_season:
            return current_season["id"]
    return None


# ============================================
# 3. FIXTURE DATA WITH FORMATIONS & STATS
# ============================================

def get_fixtures_with_details(team_id: int, num_matches: int = 10):
    """Get recent fixtures with formations, lineups, and statistics"""
    print("\n" + "="*50)
    print(f"4. GETTING FIXTURES FOR TEAM {team_id}")
    print("="*50)
    
    # Get fixtures from last 6 months
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    
    response = make_request(f"fixtures/between/{start_date}/{end_date}", {
        "include": "formations,statistics,scores,participants,lineups",
        "filters": f"participantIds:{team_id}",
        "per_page": num_matches
    })
    
    if response and "data" in response:
        print(f"\nFixtures retrieved: {len(response['data'])}")
        
        for fixture in response["data"][:5]:  # Show first 5
            formations = fixture.get("formations", [])
            formation_str = ", ".join([f"{f.get('formation', 'N/A')}" for f in formations])
            print(f"  - {fixture['name']}: {formation_str}")
        
        save_json(response, f"fixtures_team_{team_id}.json")
        return response["data"]
    return []


def get_fixture_statistics_types():
    """Get all available statistic types"""
    print("\n" + "="*50)
    print("5. GETTING STATISTIC TYPES")
    print("="*50)
    
    response = make_request("types", {"per_page": 200})
    
    if response and "data" in response:
        # Filter for statistic-related types
        stat_types = [t for t in response["data"] if "statistic" in t.get("model_type", "").lower()]
        
        print(f"\nStatistic types found: {len(stat_types)}")
        for t in stat_types[:30]:
            print(f"  - ID {t['id']}: {t['name']} ({t.get('code', 'N/A')})")
        
        save_json(response, "types.json")
        return response["data"]
    return []


# ============================================
# 4. COACH/MANAGER DATA
# ============================================

def search_coach(name: str):
    """Search for a coach by name"""
    print("\n" + "="*50)
    print(f"6. SEARCHING FOR COACH: {name}")
    print("="*50)
    
    response = make_request(f"coaches/search/{name}", {
        "include": "teams,statistics.details"
    })
    
    if response and "data" in response:
        print(f"\nCoaches found: {len(response['data'])}")
        
        for coach in response["data"]:
            teams = coach.get("teams", [])
            current_team = teams[0].get("name", "Unknown") if teams else "Unknown"
            print(f"  - ID {coach['id']}: {coach['name']} (Current: {current_team})")
        
        save_json(response, f"coach_search_{name.replace(' ', '_')}.json")
        return response["data"]
    return []


def get_coach_fixtures(coach_id: int):
    """Get fixtures managed by a specific coach"""
    print("\n" + "="*50)
    print(f"7. GETTING FIXTURES FOR COACH {coach_id}")
    print("="*50)
    
    response = make_request(f"coaches/{coach_id}", {
        "include": "fixtures.formations,fixtures.statistics,statistics.details"
    })
    
    if response and "data" in response:
        coach = response["data"]
        fixtures = coach.get("fixtures", [])
        print(f"\nCoach: {coach['name']}")
        print(f"Fixtures in data: {len(fixtures)}")
        
        save_json(response, f"coach_{coach_id}_details.json")
        return response["data"]
    return None


# ============================================
# 5. PLAYER DATA FOR SQUAD PROFILING
# ============================================

def get_team_squad(team_id: int):
    """Get full squad with player statistics"""
    print("\n" + "="*50)
    print(f"8. GETTING SQUAD FOR TEAM {team_id}")
    print("="*50)
    
    response = make_request(f"teams/{team_id}", {
        "include": "players.position,players.detailedPosition,players.statistics.details,players.metadata"
    })
    
    if response and "data" in response:
        team = response["data"]
        players = team.get("players", [])
        
        print(f"\nTeam: {team['name']}")
        print(f"Squad size: {len(players)}")
        
        for player in players[:10]:  # Show first 10
            pos = player.get("position", {})
            pos_name = pos.get("name", "Unknown") if pos else "Unknown"
            print(f"  - {player['name']} ({pos_name})")
        
        save_json(response, f"squad_team_{team_id}.json")
        return response["data"]
    return None


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Run all exploration functions"""
    print("\n" + "#"*60)
    print("# AEGIS PLATFORM - MTFI PROTOTYPE")
    print("# Sportsmonks API Exploration - Day 1")
    print("#"*60)
    
    # 1. Test connection
    if not test_connection():
        print("\n⚠️  Fix API token before continuing!")
        return
    
    # 2. Explore leagues
    leagues = explore_leagues()
    
    # 3. Get statistic types (important for understanding available data)
    stat_types = get_fixture_statistics_types()
    
    # 4. Search for target coach
    coaches = search_coach(TARGET_MANAGER_NAME)
    
    if coaches:
        coach_id = coaches[0]["id"]
        get_coach_fixtures(coach_id)
    
    # 5. Explore Premier League teams
    teams = explore_teams(TARGET_LEAGUE_ID)
    
    # Find target team
    target_team = next((t for t in teams if TARGET_TEAM_NAME.lower() in t["name"].lower()), None)
    
    if target_team:
        team_id = target_team["id"]
        
        # 6. Get fixtures with formations
        get_fixtures_with_details(team_id)
        
        # 7. Get squad data
        get_team_squad(team_id)
    
    print("\n" + "#"*60)
    print("# EXPLORATION COMPLETE")
    print(f"# Check {OUTPUT_DIR} for raw JSON files")
    print("#"*60)


if __name__ == "__main__":
    main()
