#!/usr/bin/env python3
"""
Aegis Live Data Fetcher
=======================
Fetches data from Sportsmonks API for a manager/club scenario.

Usage:
    export SPORTMONKS_API_TOKEN="your_token"
    python 01_fetch_live_data.py --manager "Thomas Frank" --target-club "Tottenham"
    
Or with IDs directly:
    python 01_fetch_live_data.py --coach-id 456123 --team-id 6
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sportsmonks_client import SportsmonksClient


def fetch_scenario_data(
    coach_name: str = None,
    coach_id: int = None,
    target_club_name: str = None,
    target_club_id: int = None,
    max_fixtures: int = 76,
    output_dir: str = None
):
    """
    Fetch all data needed for an Aegis analysis scenario.
    
    Args:
        coach_name: Manager name to search for
        coach_id: Or provide coach ID directly
        target_club_name: Target club name to search for
        target_club_id: Or provide team ID directly
        max_fixtures: Number of historical matches to analyse
        output_dir: Where to save JSON files (default: data/mock/)
    """
    
    # Check for API token
    token = os.environ.get("SPORTMONKS_API_TOKEN")
    if not token:
        print("ERROR: SPORTMONKS_API_TOKEN environment variable not set")
        print("\nSet it with:")
        print("  export SPORTMONKS_API_TOKEN='your_token_here'")
        sys.exit(1)
    
    # Initialize client
    client = SportsmonksClient(api_token=token)
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "mock"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("AEGIS LIVE DATA FETCHER")
    print("=" * 60)
    
    # -------------------------
    # 1. FETCH STATISTIC TYPES
    # -------------------------
    print("\n[1/5] Fetching statistic types...")
    try:
        types_data = client.get_types()
        with open(output_dir / "types.json", "w") as f:
            json.dump(types_data, f, indent=2)
        print(f"      ✓ Saved {len(types_data)} types")
    except Exception as e:
        print(f"      ✗ Error: {e}")
        print("      Using cached types if available...")
    
    # -------------------------
    # 2. FIND/FETCH COACH
    # -------------------------
    print("\n[2/5] Fetching coach data...")
    
    if coach_id:
        coach = client.get_coach(coach_id)
    elif coach_name:
        print(f"      Searching for '{coach_name}'...")
        results = client.search_coaches(coach_name)
        if not results:
            print(f"      ✗ No coach found matching '{coach_name}'")
            sys.exit(1)
        coach = results[0]
        coach_id = coach["id"]
        print(f"      Found: {coach.get('common_name', coach.get('name'))} (ID: {coach_id})")
        
        # Get full coach details
        coach = client.get_coach(coach_id)
    else:
        print("      ✗ Must provide --manager or --coach-id")
        sys.exit(1)
    
    with open(output_dir / f"coach_{coach.get('common_name', 'unknown').lower().replace(' ', '_')}.json", "w") as f:
        json.dump(coach, f, indent=2)
    print(f"      ✓ Saved coach profile")
    
    # Get manager's current team
    manager_team_id = None
    if "teams" in coach and coach["teams"]:
        # Find current team (no end date)
        for team in coach["teams"]:
            if team.get("end") is None:
                manager_team_id = team.get("team_id")
                break
        if not manager_team_id:
            manager_team_id = coach["teams"][0].get("team_id")
    
    # -------------------------
    # 3. FETCH MANAGER'S FIXTURES
    # -------------------------
    print("\n[3/5] Fetching manager's historical fixtures...")
    
    if manager_team_id:
        print(f"      Fetching fixtures for team ID {manager_team_id}...")
        fixtures = client.get_coach_fixtures(coach_id, limit=max_fixtures)
        
        if not fixtures:
            # Fallback: get team fixtures
            print("      No coach fixtures endpoint, trying team fixtures...")
            fixtures = client.get_fixtures_by_season(
                team_id=manager_team_id,
                season_id=23614,  # 2024/25 - adjust as needed
                include="formations;statistics;scores;participants"
            )
    else:
        print("      ✗ Could not determine manager's team")
        fixtures = []
    
    # Save fixtures
    fixture_filename = f"fixtures_{coach.get('common_name', 'unknown').lower().replace(' ', '_').split()[0]}.json"
    with open(output_dir / fixture_filename, "w") as f:
        json.dump(fixtures[:max_fixtures], f, indent=2)
    print(f"      ✓ Saved {len(fixtures[:max_fixtures])} fixtures")
    
    # -------------------------
    # 4. FIND/FETCH TARGET CLUB
    # -------------------------
    print("\n[4/5] Fetching target club data...")
    
    if target_club_id:
        team = client.get_team(target_club_id)
    elif target_club_name:
        print(f"      Searching for '{target_club_name}'...")
        results = client.search_teams(target_club_name)
        if not results:
            print(f"      ✗ No team found matching '{target_club_name}'")
            sys.exit(1)
        team = results[0]
        target_club_id = team["id"]
        print(f"      Found: {team.get('name')} (ID: {target_club_id})")
        
        # Get full team details
        team = client.get_team(target_club_id)
    else:
        print("      ✗ Must provide --target-club or --team-id")
        sys.exit(1)
    
    team_filename = f"team_{team.get('name', 'unknown').lower().replace(' ', '_')}.json"
    with open(output_dir / team_filename, "w") as f:
        json.dump(team, f, indent=2)
    print(f"      ✓ Saved team profile")
    
    # -------------------------
    # 5. FETCH TARGET SQUAD
    # -------------------------
    print("\n[5/5] Fetching target club squad...")
    
    squad = client.get_team_squad(target_club_id)
    
    squad_filename = f"squad_{team.get('name', 'unknown').lower().replace(' ', '_')}.json"
    with open(output_dir / squad_filename, "w") as f:
        json.dump(squad, f, indent=2)
    print(f"      ✓ Saved {len(squad)} players")
    
    # -------------------------
    # SUMMARY
    # -------------------------
    print("\n" + "=" * 60)
    print("DATA FETCH COMPLETE")
    print("=" * 60)
    print(f"\nScenario: {coach.get('common_name', 'Unknown')} → {team.get('name', 'Unknown')}")
    print(f"Fixtures analysed: {len(fixtures[:max_fixtures])}")
    print(f"Squad size: {len(squad)}")
    print(f"\nFiles saved to: {output_dir}")
    print("\nNext steps:")
    print("  python 02_etl_pipeline.py")
    print("  python 03_aegis_analysis.py")
    
    return {
        "coach": coach,
        "fixtures": fixtures[:max_fixtures],
        "team": team,
        "squad": squad
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fetch live Sportsmonks data for Aegis analysis"
    )
    parser.add_argument(
        "--manager", "-m",
        help="Manager name to search for (e.g., 'Thomas Frank')"
    )
    parser.add_argument(
        "--coach-id",
        type=int,
        help="Or provide Sportsmonks coach ID directly"
    )
    parser.add_argument(
        "--target-club", "-t",
        help="Target club name (e.g., 'Tottenham')"
    )
    parser.add_argument(
        "--team-id",
        type=int,
        help="Or provide Sportsmonks team ID directly"
    )
    parser.add_argument(
        "--fixtures", "-f",
        type=int,
        default=76,
        help="Number of historical fixtures to fetch (default: 76)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for JSON files"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.manager and not args.coach_id:
        parser.error("Must provide --manager or --coach-id")
    if not args.target_club and not args.team_id:
        parser.error("Must provide --target-club or --team-id")
    
    fetch_scenario_data(
        coach_name=args.manager,
        coach_id=args.coach_id,
        target_club_name=args.target_club,
        target_club_id=args.team_id,
        max_fixtures=args.fixtures,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
