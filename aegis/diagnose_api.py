"""
Aegis API Diagnostic
====================
Dumps raw API response structure to reveal exactly what fields and codes
the Sportsmonks API returns — before we write any parsing logic.

Run this in your Colab notebook:

    %run diagnose_api.py

Or paste into a cell and call:

    run_diagnostic(api_token="YOUR_TOKEN")

Output is saved to: aegis_api_diagnostic.txt
"""

import json
import os
import requests
import time
from pprint import pformat


# ── CONFIG ────────────────────────────────────────────────────────────────────
# One well-known Premier League team to use as the test subject.
# Man City (team_id=14) in season 2024/25 (season_id=23614).
TEST_TEAM_ID   = 14
TEST_SEASON_ID = 23614
BASE_URL       = "https://api.sportmonks.com/v3"
OUTPUT_FILE    = "aegis_api_diagnostic.txt"
# ─────────────────────────────────────────────────────────────────────────────


def _get(api_token, endpoint, params=None, include=None):
    """Raw API call — returns the full response dict."""
    time.sleep(0.5)  # gentle rate limiting
    p = params or {}
    p["api_token"] = api_token
    if include:
        p["include"] = ";".join(include) if isinstance(include, list) else include
    resp = requests.get(f"{BASE_URL}{endpoint}", params=p)
    resp.raise_for_status()
    return resp.json()


def _section(lines, title):
    lines.append("\n" + "=" * 70)
    lines.append(f"  {title}")
    lines.append("=" * 70)


def _dump_stat_detail(lines, detail, indent=4):
    """Print one statistics detail entry showing every field."""
    pad = " " * indent
    type_info = detail.get("type", detail.get("type_id", "???"))
    value     = detail.get("value", "???")
    lines.append(f"{pad}--- detail entry ---")
    lines.append(f"{pad}  type   : {json.dumps(type_info)}")
    lines.append(f"{pad}  value  : {json.dumps(value)}")
    # Show any other keys present
    other = {k: v for k, v in detail.items() if k not in ("type", "value", "type_id")}
    if other:
        lines.append(f"{pad}  other  : {json.dumps(other)}")


def run_diagnostic(api_token=None):
    token = api_token or os.environ.get("SPORTMONKS_API_TOKEN")
    if not token:
        raise ValueError(
            "Provide api_token= or set os.environ['SPORTMONKS_API_TOKEN']"
        )

    lines = []
    lines.append("AEGIS API DIAGNOSTIC")
    lines.append(f"Team ID   : {TEST_TEAM_ID}   (Man City)")
    lines.append(f"Season ID : {TEST_SEASON_ID} (2024/25)")

    # ── 1. TEAM STATS ENDPOINT ───────────────────────────────────────────────
    _section(lines, "1. TEAM STATISTICS ENDPOINT")
    lines.append("   GET /football/teams/{id}?include=statistics.details.type")
    lines.append("   This is what ManagerDNATrainer.extract_features() currently calls.\n")

    try:
        resp = _get(
            token,
            f"/football/teams/{TEST_TEAM_ID}",
            params={"filters": f"teamstatisticSeasons:{TEST_SEASON_ID}"},
            include=["statistics.details.type"]
        )
        team_data = resp.get("data", {})
        stats_list = team_data.get("statistics", [])

        lines.append(f"  statistics[] length : {len(stats_list)}")

        for i, season_stat in enumerate(stats_list):
            lines.append(f"\n  statistics[{i}]:")
            lines.append(f"    Keys present   : {sorted(season_stat.keys())}")
            lines.append(f"    season_id      : {season_stat.get('season_id')}")
            details = season_stat.get("details", [])
            lines.append(f"    details[] length: {len(details)}")

            if details:
                lines.append(f"\n    ALL detail entries:")
                for d in details:
                    _dump_stat_detail(lines, d, indent=6)
            else:
                lines.append("    ⚠  details[] is EMPTY")
                # Maybe details is nested differently
                lines.append(f"    Raw season_stat keys: {list(season_stat.keys())}")
                for k, v in season_stat.items():
                    if k != "details":
                        lines.append(f"      {k}: {json.dumps(v)[:120]}")

    except Exception as e:
        lines.append(f"  ✗ ERROR: {e}")

    # ── 2. SAME ENDPOINT WITHOUT SEASON FILTER ───────────────────────────────
    _section(lines, "2. TEAM STATISTICS — NO SEASON FILTER")
    lines.append("   Checking if filter is suppressing data.\n")

    try:
        resp2 = _get(
            token,
            f"/football/teams/{TEST_TEAM_ID}",
            include=["statistics.details.type"]
        )
        stats2 = resp2.get("data", {}).get("statistics", [])
        lines.append(f"  statistics[] length (no filter): {len(stats2)}")
        if stats2:
            d0 = stats2[0]
            lines.append(f"  First entry keys  : {sorted(d0.keys())}")
            lines.append(f"  First entry season: {d0.get('season_id')}")
            details2 = d0.get("details", [])
            lines.append(f"  details[] length  : {len(details2)}")
            if details2:
                lines.append(f"\n  First 5 detail entries (no filter):")
                for d in details2[:5]:
                    _dump_stat_detail(lines, d, indent=4)

    except Exception as e:
        lines.append(f"  ✗ ERROR: {e}")

    # ── 3. SINGLE FIXTURE — FULL STAT STRUCTURE ──────────────────────────────
    _section(lines, "3. SINGLE FIXTURE — STATISTICS STRUCTURE")
    lines.append("   First, get a list of fixtures to find a completed one.")
    lines.append("   Then dump the full statistics block from that fixture.\n")

    fixture_id = None
    try:
        fix_resp = _get(
            token,
            "/football/fixtures",
            params={
                "filters": f"fixtureSeasons:{TEST_SEASON_ID};fixtureTeams:{TEST_TEAM_ID}",
                "per_page": 5
            },
            include=["scores"]
        )
        fixtures = fix_resp.get("data", [])
        lines.append(f"  Fixtures returned: {len(fixtures)}")

        # Pick first one that has scores (i.e. completed)
        for fx in fixtures:
            if fx.get("scores"):
                fixture_id = fx["id"]
                lines.append(f"  Using fixture_id : {fixture_id}  "
                              f"({fx.get('starting_at', 'unknown date')})")
                break

        if not fixture_id and fixtures:
            fixture_id = fixtures[0]["id"]
            lines.append(f"  No scored fixture found — using first: {fixture_id}")

    except Exception as e:
        lines.append(f"  ✗ Fixture list error: {e}")

    if fixture_id:
        # Try include=statistics.type
        lines.append(f"\n  --- include=statistics.type ---")
        try:
            fx_resp = _get(
                token,
                f"/football/fixtures/{fixture_id}",
                include=["statistics.type", "scores", "participants"]
            )
            fx_data   = fx_resp.get("data", {})
            fx_stats  = fx_data.get("statistics", [])
            lines.append(f"  statistics[] length : {len(fx_stats)}")
            if fx_stats:
                lines.append(f"  First entry keys    : {sorted(fx_stats[0].keys())}")
                lines.append(f"\n  ALL fixture statistics entries:")
                for s in fx_stats:
                    _dump_stat_detail(lines, s, indent=4)
            else:
                lines.append("  ⚠  statistics[] is EMPTY with this include")

        except Exception as e:
            lines.append(f"  ✗ ERROR: {e}")

        # Try include=statistics (no sub-include) to see if codes are embedded
        lines.append(f"\n  --- include=statistics (no sub-include) ---")
        try:
            fx_resp2 = _get(
                token,
                f"/football/fixtures/{fixture_id}",
                include=["statistics", "scores"]
            )
            fx_stats2 = fx_resp2.get("data", {}).get("statistics", [])
            lines.append(f"  statistics[] length : {len(fx_stats2)}")
            if fx_stats2:
                lines.append(f"  First entry keys    : {sorted(fx_stats2[0].keys())}")
                lines.append(f"\n  First 6 entries:")
                for s in fx_stats2[:6]:
                    _dump_stat_detail(lines, s, indent=4)

        except Exception as e:
            lines.append(f"  ✗ ERROR: {e}")

    # ── 4. TEAM STATS — ALTERNATIVE INCLUDES ─────────────────────────────────
    _section(lines, "4. TEAM STATISTICS — ALTERNATIVE INCLUDES")
    lines.append("   Testing variations to see which include returns tactical data.\n")

    alt_includes = [
        ["statistics.details"],
        ["statistics"],
    ]

    for inc in alt_includes:
        inc_str = ";".join(inc)
        lines.append(f"  include={inc_str}")
        try:
            r = _get(
                token,
                f"/football/teams/{TEST_TEAM_ID}",
                params={"filters": f"teamstatisticSeasons:{TEST_SEASON_ID}"},
                include=inc
            )
            st = r.get("data", {}).get("statistics", [])
            lines.append(f"    statistics[] length: {len(st)}")
            if st:
                d = st[0].get("details", [])
                lines.append(f"    details[] length  : {len(d)}")
                if d:
                    lines.append(f"    First 3 details:")
                    for entry in d[:3]:
                        _dump_stat_detail(lines, entry, indent=6)
        except Exception as e:
            lines.append(f"    ✗ ERROR: {e}")
        lines.append("")

    # ── 5. WRITE OUTPUT ───────────────────────────────────────────────────────
    output = "\n".join(lines)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\n{'='*70}")
    print(f"  Full output saved to: {OUTPUT_FILE}")
    print(f"  Please share this file with Will.")
    print(f"{'='*70}")


# Allow running directly
if __name__ == "__main__":
    run_diagnostic()
