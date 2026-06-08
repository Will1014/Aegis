"""
aegis/dna_insights.py
=====================
Four analytical enhancements to the Manager DNA panel.

Public functions:
  compute_manager_similarity(target_name, training_dir, top_n=5)
  compute_pillar_benchmarks(target_name, target_cluster, training_dir, dna_dimensions)
  compute_pillar_confidence(result, analysis_snapshot)
  compute_formation_tendency(team_name, competition_id, season_id, sb_username, sb_password, max_matches=30)

All functions read from data already produced by the pipeline — no second
pipeline runs required. compute_formation_tendency() is the only function
that makes a new StatsBomb API call; its output also feeds Feature 4.
"""

from __future__ import annotations

import os
import collections
from pathlib import Path
from typing import Dict, List, Optional, Any


# ─────────────────────────────────────────────────────────────────────────────
# 1. MANAGER SIMILARITY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_manager_similarity(
    target_name: str,
    training_dir,
    top_n: int = 5,
) -> List[Dict]:
    """
    Return the top_n most tactically similar managers to target_name.

    Uses the fitted StandardScaler from manager_dna_model.pkl to project all
    training manager feature vectors into scaled space, then ranks by euclidean
    distance. Similarity % = (1 - dist/max_dist) * 100.

    Args:
        target_name: Manager name substring (case-insensitive match).
        training_dir: Path to directory containing manager_dna_model.pkl
                      and manager_profiles.csv.
        top_n: Number of similar managers to return.

    Returns:
        List of dicts with keys: manager, team, archetype, similarity_pct.
        Empty list if target_name is not found.
    """
    import pickle
    import numpy as np
    import pandas as pd

    training_dir = Path(training_dir)
    model_path = training_dir / "manager_dna_model.pkl"
    profiles_path = training_dir / "manager_profiles.csv"

    if not model_path.exists() or not profiles_path.exists():
        return []

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    scaler       = model["scaler"]
    feature_cols = model["feature_cols"]
    cluster_names = model.get("cluster_names", {})

    df = pd.read_csv(profiles_path)

    # Fuzzy name match — full substring first, then last-name token
    mask = df["coach_name"].str.contains(target_name, case=False, na=False)
    if not mask.any():
        tokens = target_name.split()
        if tokens:
            mask = df["coach_name"].str.contains(tokens[-1], case=False, na=False)
    if not mask.any():
        return []

    # Keep only columns present in the CSV
    available_cols = [c for c in feature_cols if c in df.columns]
    if not available_cols:
        return []

    X = scaler.transform(df[available_cols].fillna(0).values)
    target_idx = mask.idxmax()
    target_vec = X[df.index.get_loc(target_idx)]

    dists = np.linalg.norm(X - target_vec, axis=1)
    # Exclude the target manager itself
    dists[df.index.get_loc(target_idx)] = np.inf

    valid_dists = dists[dists != np.inf]
    if len(valid_dists) == 0:
        return []
    max_dist = valid_dists.max() or 1.0

    top_indices = np.argsort(dists)[:top_n]
    results = []
    for iloc in top_indices:
        row = df.iloc[iloc]
        cluster_id = int(row.get("cluster", 0))
        results.append({
            "manager":        row.get("coach_name", "Unknown"),
            "team":           row.get("team_name", "Unknown"),
            "archetype":      cluster_names.get(cluster_id, f"Cluster {cluster_id}"),
            "similarity_pct": round((1 - dists[iloc] / max_dist) * 100, 1),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. PILLAR BENCHMARKING VS ARCHETYPE CENTROID
# ─────────────────────────────────────────────────────────────────────────────

# Maps pillar_pct_* CSV column names to display pillar keys
_PILLAR_COL_MAP = {
    "pillar_pct_shape_occupation":   "shape_occupation",
    "pillar_pct_build_up":           "build_up",
    "pillar_pct_chance_creation":    "chance_creation",
    "pillar_pct_press_counterpress": "press_counterpress",
    "pillar_pct_block_line_height":  "block_line_height",
    "pillar_pct_transitions":        "transitions",
    "pillar_pct_width_overloads":    "width_overloads",
    "pillar_pct_set_pieces":         "set_pieces",
}

_PILLAR_DISPLAY = {
    "shape_occupation":   "Shape & Occupation",
    "build_up":           "Build-up",
    "chance_creation":    "Chance Creation",
    "press_counterpress": "Press & Counterpress",
    "block_line_height":  "Block & Line Height",
    "transitions":        "Transitions",
    "width_overloads":    "Width & Overloads",
    "set_pieces":         "Set Pieces",
}


def compute_pillar_benchmarks(
    target_name: str,
    target_cluster: int,
    training_dir,
    dna_dimensions: Dict[str, float],
) -> List[Dict]:
    """
    Compare each pillar score against the archetype cluster mean.

    Returns a list of dicts (one per pillar) with:
      pillar, display_name, score, archetype_mean, delta, flag

    flag is one of: 'above' (delta > +10), 'below' (delta < -10), 'typical'.
    """
    import pandas as pd

    training_dir = Path(training_dir)
    profiles_path = training_dir / "manager_profiles.csv"
    if not profiles_path.exists():
        return []

    df = pd.read_csv(profiles_path)
    pillar_cols = [c for c in df.columns if c.startswith("pillar_pct_")]
    if not pillar_cols:
        return []

    cluster_df = df[df["cluster"] == target_cluster]
    if cluster_df.empty:
        cluster_df = df   # fall back to all managers

    archetype_means = cluster_df[pillar_cols].mean()

    # Reverse-map dna_dimensions keys (may use display names like "Shape & Occupation")
    display_to_key = {v: k for k, v in _PILLAR_DISPLAY.items()}

    results = []
    for col, key in _PILLAR_COL_MAP.items():
        if col not in archetype_means:
            continue
        display = _PILLAR_DISPLAY[key]
        # Try to find score in dna_dimensions by key or display name
        score = (dna_dimensions.get(key)
                 or dna_dimensions.get(display)
                 or 50.0)
        mean  = round(float(archetype_means[col]), 1)
        delta = round(float(score) - mean, 1)
        results.append({
            "pillar":         key,
            "display_name":   display,
            "score":          round(float(score), 1),
            "archetype_mean": mean,
            "delta":          delta,
            "flag":           "above" if delta > 10 else "below" if delta < -10 else "typical",
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. PILLAR CONFIDENCE INDICATOR
# ─────────────────────────────────────────────────────────────────────────────

def compute_pillar_confidence(
    result: Dict,
    analysis_snapshot: Dict,
) -> Dict[str, Any]:
    """
    Return a confidence band based on the number of matches behind the scores.

    Bands:
      High   ≥ 20 matches
      Medium 10–19 matches
      Low    < 10 matches

    Returns dict with: matches_analysed, confidence, confidence_note, badge.
    """
    n = (
        result.get("matches_analysed")
        or result.get("total_matches")
        or analysis_snapshot.get("matches_analysed")
        or analysis_snapshot.get("total_matches")
        or 0
    )
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 0

    if n >= 20:
        label, note, badge = "high",   f"{n} matches analysed — high statistical confidence", "🟢"
    elif n >= 10:
        label, note, badge = "medium", f"{n} matches analysed — moderate confidence", "🟡"
    else:
        label, note, badge = "low",    f"{n} matches analysed — treat scores as indicative", "🔴"

    return {
        "matches_analysed":  n,
        "confidence":        label,
        "confidence_note":   note,
        "badge":             badge,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. FORMATION TENDENCY SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def compute_formation_tendency(
    team_name: str,
    competition_id: int,
    season_id: int,
    sb_username: str,
    sb_password: str,
    max_matches: int = 30,
) -> Optional[Dict]:
    """
    Fetch real formation frequency data from the StatsBomb Lineups API.

    Calls the StatsBomb licensed API (no additional cost) to retrieve lineup
    data for the target team's matches, parses the formation string per match,
    and returns a frequency summary.

    Returns:
        {
          'primary':         '4-3-3',
          'primary_pct':     68,
          'secondary':       '4-2-3-1',
          'secondary_pct':   22,
          'other_pct':       10,
          'matches_sampled': 25,
          'raw_counts':      {'433': 17, '4231': 6, ...},
        }
        None on failure or if no lineups data is available.
    """
    import requests
    from requests.auth import HTTPBasicAuth
    from .formations import normalize_formation

    if not sb_username or not sb_password:
        return None

    auth = HTTPBasicAuth(sb_username, sb_password)
    base = "https://data.statsbombservices.com/api"

    # ── 1. Fetch matches ───────────────────────────────────────────────────
    try:
        matches_url = f"{base}/v6/competitions/{competition_id}/seasons/{season_id}/matches"
        resp = requests.get(matches_url, auth=auth, timeout=30)
        resp.raise_for_status()
        all_matches = resp.json()
    except Exception as e:
        print(f"  ⚠ Formation tendency: could not fetch matches — {e}")
        return None

    # ── 2. Filter to target team ───────────────────────────────────────────
    team_matches = []
    for m in all_matches:
        home = (m.get("home_team") or {})
        away = (m.get("away_team") or {})
        h_name = home.get("home_team_name") or home.get("name") or ""
        a_name = away.get("away_team_name") or away.get("name") or ""
        if team_name.lower() in h_name.lower() or team_name.lower() in a_name.lower():
            team_matches.append(m)
        if len(team_matches) >= max_matches:
            break

    if not team_matches:
        return None

    # ── 3. Fetch lineups for each match, extract formation ─────────────────
    raw_counts: Dict[str, int] = {}

    for match in team_matches:
        match_id = match.get("match_id")
        if not match_id:
            continue
        try:
            lineup_url = f"{base}/v4/lineups/{match_id}"
            lr = requests.get(lineup_url, auth=auth, timeout=20)
            lr.raise_for_status()
            lineups = lr.json()
        except Exception:
            continue

        for team_lineup in lineups:
            t_name = (team_lineup.get("team_name") or
                      (team_lineup.get("team") or {}).get("name") or "")
            if team_name.lower() in t_name.lower():
                # StatsBomb Lineups v4: formations is an ARRAY of formation-change
                # objects (not a top-level string). Pick the "Starting XI" entry
                # from period 1 as the match's base formation.
                for fobj in (team_lineup.get("formations") or []):
                    if fobj.get("reason") == "Starting XI" and fobj.get("period", 1) == 1:
                        fmt = str(fobj.get("formation", "")).replace("-", "").strip()
                        if fmt:
                            raw_counts[fmt] = raw_counts.get(fmt, 0) + 1
                        break
                break   # only need this team's lineup per match

    if not raw_counts:
        return None

    # ── 4. Frequency analysis ──────────────────────────────────────────────
    total = sum(raw_counts.values())
    ranked = sorted(raw_counts.items(), key=lambda x: x[1], reverse=True)

    primary_raw, primary_count = ranked[0]
    primary = normalize_formation(primary_raw)
    primary_pct = round(primary_count / total * 100)

    secondary, secondary_pct = None, 0
    if len(ranked) > 1:
        sec_raw, sec_count = ranked[1]
        secondary = normalize_formation(sec_raw)
        secondary_pct = round(sec_count / total * 100)

    other_pct = max(0, 100 - primary_pct - secondary_pct)

    return {
        "primary":         primary,
        "primary_pct":     primary_pct,
        "secondary":       secondary,
        "secondary_pct":   secondary_pct,
        "other_pct":       other_pct,
        "matches_sampled": total,
        "raw_counts":      raw_counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. MANAGER HISTORICAL FORMATION
# ─────────────────────────────────────────────────────────────────────────────

def get_manager_previous_team(
    manager_name: str,
    training_dir,
) -> Optional[str]:
    """
    Look up the team a manager was analysed against in the training data.
    Returns the team_name string or None if not found.
    """
    import pandas as pd
    training_dir = Path(training_dir)
    profiles_path = training_dir / "manager_profiles.csv"
    if not profiles_path.exists():
        return None
    df = pd.read_csv(profiles_path)
    mask = df["coach_name"].str.contains(manager_name, case=False, na=False)
    if not mask.any() and " " in manager_name:
        mask = df["coach_name"].str.contains(
            manager_name.split()[-1], case=False, na=False)
    if not mask.any():
        return None
    return str(df[mask].iloc[0].get("team_name", ""))


def compute_manager_formation(
    manager_name: str,
    training_dir,
    training_league_ids: list,
    season_id: int,
    sb_username: str,
    sb_password: str,
    max_matches: int = 25,
) -> Optional[Dict]:
    """
    Derive a manager's preferred formation from their previous club's matches.

    Looks up the manager's previous team from manager_profiles.csv, then
    calls compute_formation_tendency() against that team across all provided
    training leagues until data is found.

    Returns: same dict as compute_formation_tendency(), or None.
    """
    prev_team = get_manager_previous_team(manager_name, training_dir)
    if not prev_team:
        return None

    for league_id in (training_league_ids or [2]):
        result = compute_formation_tendency(
            team_name=prev_team,
            competition_id=league_id,
            season_id=season_id,
            sb_username=sb_username,
            sb_password=sb_password,
            max_matches=max_matches,
        )
        if result:
            result["source_team"] = prev_team
            result["source_league_id"] = league_id
            return result

    return None
