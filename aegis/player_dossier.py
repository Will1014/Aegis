"""
Aegis Player Dossier Generator
==============================
Generates premium HTML scouting dossiers from StatsBomb player season stats.

Produces a multi-section interactive HTML report matching the Aegis visual standard:
  • Cover page with player bio & positional tags
  • Scouting report with profile narratives
  • Data & metrics with per-90 stat cards + percentile bars
  • Radar/pizza chart (Attacking / Distribution / Defensive)
  • Role & archetype suitability

Usage (standalone):
    from aegis.player_dossier import PlayerDossierGenerator, generate_player_dossier

    # With raw StatsBomb player season stats already fetched:
    gen = PlayerDossierGenerator(player_season_stats)
    html = gen.generate("Tae-Seok Lee", competition_name="Austrian Bundesliga")
    with open("dossier.html", "w") as f:
        f.write(html)

Usage (via convenience function that fetches data):
    from aegis.player_dossier import generate_player_dossier
    from aegis import Config, StatsBombClient

    html = generate_player_dossier(
        player_name="Tae-Seok Lee",
        competition_id=6,
        season_id=317,
    )
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# CONSTANTS
# =============================================================================

# StatsBomb field name aliases: (preferred, *fallbacks)
# Real StatsBomb Player Season Stats v4 fields are prefixed `player_season_`
_FIELD_MAP: Dict[str, Tuple[str, ...]] = {
    # Identity
    "player_name":       ("player_name", "name"),
    "team_name":         ("team_name",   "team.name",   "team"),
    "player_id":         ("player_id",   "player.id",   "id"),
    "team_id":           ("team_id",     "team.id"),
    "nationality":       ("player_nationality", "nationality", "country"),
    "position":          ("primary_position", "position", "positions"),
    "birth_date":        ("birth_date",  "dob", "date_of_birth"),
    # Season totals  ← StatsBomb uses player_season_minutes, not minutes_played
    "minutes":           ("player_season_minutes", "minutes_played", "minutes", "mins"),
    "matches":           ("player_season_appearances", "matches_played", "matches", "apps"),
    # Goals & assists — StatsBomb stores these as per-90 only, so multiply later
    # We expose season totals by deriving from *_90 * nineties in _compute_metrics
    "goals":             ("player_season_goals",    "goals"),
    "assists":           ("player_season_assists",  "assists"),
    # Per-90 attacking
    "xa_p90":            ("player_season_xa_90",            "xa_per_90", "xassists_per_90"),
    "shots_p90":         ("player_season_np_shots_90",      "shots_per_90", "total_shots_per_90"),
    "chances_p90":       ("player_season_key_passes_90",    "key_passes_per_90", "chance_created_per_90"),
    "long_balls_p90":    ("player_season_long_balls_90",    "player_season_passes_long_90",
                          "passes_long_per_90", "long_balls_per_90"),
    # Crossing
    "crosses_succ_p90":  ("player_season_successful_crosses_90", "player_season_crosses_90",
                          "successful_crosses_per_90"),
    "crosses_att_p90":   ("player_season_crosses_into_box_90", "player_season_crosses_90",
                          "crosses_per_90"),
    "cross_pct":         ("player_season_cross_accuracy",   "cross_accuracy",
                          "cross_success_rate", "crossing_accuracy"),
    # Dribbling
    "dribbles_succ_p90": ("player_season_dribbles_90",      "dribbles_completed_per_90",
                          "successful_dribbles_per_90"),
    "dribbles_att_p90":  ("player_season_dribbles_attempted_90", "dribbles_attempted_per_90"),
    "dribble_pct":       ("player_season_dribble_success_rate", "player_season_challenge_ratio",
                          "dribble_percentage", "dribble_success_rate"),
    # Duels
    "duels_won_p90":     ("player_season_ground_duels_won_90", "player_season_duels_won_90",
                          "duels_won_per_90"),
    "duels_att_p90":     ("player_season_ground_duels_90",  "player_season_duels_90",
                          "duels_per_90"),
    "duels_won_pct":     ("player_season_ground_duel_win_rate", "player_season_duel_success_rate",
                          "duels_won_percentage"),
    # Defensive
    "tackles_p90":       ("player_season_tackles_90",        "tackles_per_90"),
    "interceptions_p90": ("player_season_interceptions_90",  "interceptions_per_90"),
    "recoveries_p90":    ("player_season_ball_recoveries_90","player_season_recoveries_90",
                          "ball_recoveries_per_90", "recoveries_per_90"),
    "clearances_p90":    ("player_season_clearance_90",      "clearances_per_90"),
    "pressures_p90":     ("player_season_pressures_90",      "pressures_per_90"),
    # OBV
    "obv_p90":           ("player_season_obv_90",            "on_ball_value_per_90"),
}

# Position group mapping (StatsBomb position IDs)
_POSITION_GROUPS: Dict[str, List[int]] = {
    "Goalkeeper":  [1],
    "Defender":    [2, 3, 4, 5, 6, 7, 8],
    "Midfielder":  [9, 10, 11, 12, 13, 14, 15, 16],
    "Attacker":    [17, 18, 19, 20, 21, 22, 23, 24, 25],
}

# Human-readable position labels
_POSITION_LABELS: Dict[int, str] = {
    1: "GK", 2: "RB", 3: "RCB", 4: "CB", 5: "LCB", 6: "LB",
    7: "RWB", 8: "LWB", 9: "RDM", 10: "CDM", 11: "LDM",
    12: "RM", 13: "RCM", 14: "CM", 15: "LCM", 16: "LM",
    17: "RW", 18: "RAM", 19: "CAM", 20: "LAM", 21: "LW",
    22: "RCF", 23: "ST", 24: "LCF", 25: "SS",
}

# Metric display definitions: (key, label, category, higher_is_better)
# Categories: "attacking", "distribution", "defensive"
DOSSIER_METRICS: List[Tuple[str, str, str, bool]] = [
    ("goals",           "Goals",              "attacking",     True),
    ("assists",         "Assists",            "attacking",     True),
    ("xa_p90",          "xA / 90",            "attacking",     True),
    ("shots_p90",       "Shots / 90",         "attacking",     True),
    ("chances_p90",     "Chances / 90",       "attacking",     True),
    ("long_balls_p90",  "Long Balls / 90",    "distribution",  True),
    ("crosses_succ_p90","Succ. Crosses / 90", "distribution",  True),
    ("cross_pct",       "Cross Success %",    "distribution",  True),
    ("dribbles_succ_p90","Succ. Dribbles / 90","distribution", True),
    ("dribble_pct",     "Dribble Success %",  "distribution",  True),
    ("duels_won_p90",   "Duels Won / 90",     "defensive",     True),
    ("duels_won_pct",   "Duels Won %",        "defensive",     True),
    ("tackles_p90",     "Tackles / 90",       "defensive",     True),
    ("interceptions_p90","Interceptions / 90","defensive",     True),
    ("recoveries_p90",  "Recoveries / 90",    "defensive",     True),
]

RADAR_METRICS: List[Tuple[str, str, str]] = [
    ("goals",            "Goals",           "attacking"),
    ("assists",          "Assists",         "attacking"),
    ("xa_p90",           "xA / 90",         "attacking"),
    ("shots_p90",        "Shots / 90",      "attacking"),
    ("long_balls_p90",   "Long Balls / 90", "distribution"),
    ("crosses_succ_p90", "Succ. Crosses",   "distribution"),
    ("cross_pct",        "Cross %",         "distribution"),
    ("dribbles_succ_p90","Dribbles / 90",   "distribution"),
    ("dribble_pct",      "Dribble %",       "distribution"),
    ("duels_won_p90",    "Duels Won / 90",  "defensive"),
    ("duels_won_pct",    "Duels Won %",     "defensive"),
    ("tackles_p90",      "Tackles / 90",    "defensive"),
    ("interceptions_p90","Intercept. / 90", "defensive"),
    ("recoveries_p90",   "Recoveries / 90", "defensive"),
]

CATEGORY_COLORS = {
    "attacking":    "#7c3aed",   # purple
    "distribution": "#9ca3af",   # light grey
    "defensive":    "#d97706",   # amber
}

CATEGORY_LABELS = {
    "attacking":    "Attacking / Output",
    "distribution": "Distribution / Progression",
    "defensive":    "Defensive / Duels",
}

MIN_MINUTES = 450  # ~5 full matches

# Scout rating dimensions and their metric drivers
SCOUT_RATING_AXES: List[Tuple[str, str, List[str]]] = [
    ("Physical",         "physical",   ["duels_won_p90", "duels_won_pct", "dribbles_succ_p90"]),
    ("Technical",        "technical",  ["cross_pct", "dribble_pct", "xa_p90", "chances_p90"]),
    ("Offensive",        "offensive",  ["goals", "assists", "shots_p90", "xa_p90"]),
    ("Set Pieces",       "set_pieces", ["crosses_succ_p90", "cross_pct", "long_balls_p90"]),
    ("Against the Ball", "atb",        ["tackles_p90", "interceptions_p90", "recoveries_p90", "duels_won_p90"]),
]


# =============================================================================
# HELPERS
# =============================================================================

def _get(record: Dict, key: str, default=None):
    """Resolve a value from a record using the field-name alias map."""
    aliases = _FIELD_MAP.get(key, (key,))
    for alias in aliases:
        if alias in record and record[alias] is not None:
            return record[alias]
    return default


def _safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _percentile_rank(value: float, population: List[float], higher_is_better: bool = True) -> int:
    """Return 1–99 percentile rank of value within population."""
    if not population or len(population) < 2:
        return 50
    if higher_is_better:
        rank = sum(1 for v in population if v < value)
    else:
        rank = sum(1 for v in population if v > value)
    pct = int(round(rank / len(population) * 100))
    return max(1, min(99, pct))


def _age_from_dob(dob_str: Optional[str]) -> Optional[int]:
    if not dob_str:
        return None
    try:
        dob = datetime.strptime(dob_str[:10], "%Y-%m-%d").date()
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except Exception:
        return None


def _safe_name(name: str) -> str:
    """Safe filename-friendly version of player name."""
    return re.sub(r"[^\w\s-]", "", name).replace(" ", "_")


# =============================================================================
# CORE CLASS
# =============================================================================

class PlayerDossierGenerator:
    """
    Generate a premium HTML scouting dossier for a player from StatsBomb data.

    Args:
        player_season_stats: Raw list from StatsBombClient.get_player_season_stats()
        output_dir: Where to write HTML files (optional)
    """

    def __init__(
        self,
        player_season_stats: List[Dict],
        output_dir: Optional[Path] = None,
    ):
        self.raw_stats = player_season_stats or []
        self.output_dir = Path(output_dir) if output_dir else None

        # Cached computed data
        self._metric_cache: Dict[str, Dict[str, float]] = {}   # player_id → metrics dict
        self._peer_cache: Dict[str, List[float]] = {}           # metric_key → [peer values]

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def find_player(self, name: str) -> Optional[Dict]:
        """
        Find a player record by name (case-insensitive, partial match cascade).
        Returns the best match or None.
        """
        name_lower = name.strip().lower()

        # 1. Exact full match
        for r in self.raw_stats:
            pn = (_get(r, "player_name") or "").strip().lower()
            if pn == name_lower:
                return r

        # 2. Full substring
        for r in self.raw_stats:
            pn = (_get(r, "player_name") or "").strip().lower()
            if name_lower in pn or pn in name_lower:
                return r

        # 3. Last-name token
        last = name_lower.split()[-1]
        for r in self.raw_stats:
            pn = (_get(r, "player_name") or "").strip().lower()
            if last in pn.split():
                return r

        # 4. Any token
        tokens = name_lower.split()
        for r in self.raw_stats:
            pn = (_get(r, "player_name") or "").strip().lower()
            if any(t in pn for t in tokens):
                return r

        return None

    def list_players(self, min_minutes: int = MIN_MINUTES) -> List[str]:
        """Return sorted list of player names meeting minimum minutes."""
        names = []
        for r in self.raw_stats:
            mins = _safe_float(_get(r, "minutes"), 0)
            if mins >= min_minutes:
                n = _get(r, "player_name") or ""
                if n:
                    names.append(n)
        return sorted(set(names))

    def generate(
        self,
        player_name: str,
        competition_name: str = "",
        season_name: str = "",
        manual_overrides: Optional[Dict] = None,
    ) -> str:
        """
        Generate and return the full HTML dossier string.

        Args:
            player_name: Player to look up (fuzzy matched)
            competition_name: e.g. "Austrian Bundesliga 2025/26"
            season_name: e.g. "2025/26"
            manual_overrides: Dict of extra fields to inject (height, strong_foot, tmv, etc.)

        Returns:
            HTML string
        """
        record = self.find_player(player_name)
        if record is None:
            return self._not_found_html(player_name)

        overrides = manual_overrides or {}

        metrics = self._compute_metrics(record)
        peer_group = self._get_peer_group(record)
        percentiles = self._compute_percentiles(metrics, peer_group)
        scout_ratings = self._compute_scout_ratings(percentiles)

        player_data = self._build_player_data(record, metrics, percentiles, scout_ratings, overrides)

        html = self._render_html(player_data, competition_name, season_name)

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{_safe_name(_get(record, 'player_name') or player_name)}___Scouting_Dossier.html"
            (self.output_dir / fname).write_text(html, encoding="utf-8")
            print(f"✓ Dossier saved: {self.output_dir / fname}")

        return html

    def generate_batch(
        self,
        player_names: List[str],
        competition_name: str = "",
        season_name: str = "",
    ) -> Dict[str, str]:
        """Generate dossiers for multiple players. Returns name → html dict."""
        results = {}
        for name in player_names:
            print(f"  Generating dossier: {name}")
            results[name] = self.generate(name, competition_name, season_name)
        return results

    # -------------------------------------------------------------------------
    # METRICS & PERCENTILES
    # -------------------------------------------------------------------------

    def _compute_metrics(self, record: Dict) -> Dict[str, float]:
        pid = _get(record, "player_id")
        if pid and pid in self._metric_cache:
            return self._metric_cache[pid]

        metrics: Dict[str, float] = {}
        mins = _safe_float(_get(record, "minutes"), 0)
        nineties = mins / 90.0 if mins > 0 else 1.0

        for key, *_ in DOSSIER_METRICS:
            raw = _get(record, key)
            metrics[key] = _safe_float(raw, 0.0)

        # Goals & assists: StatsBomb only stores per-90 values, so derive totals
        # Try direct field first, then multiply per-90 by nineties
        if metrics["goals"] == 0:
            goals_90 = _safe_float(record.get("player_season_goals_90"), 0)
            metrics["goals"] = round(goals_90 * nineties)
        if metrics["assists"] == 0:
            assists_90 = _safe_float(record.get("player_season_assists_90"), 0)
            metrics["assists"] = round(assists_90 * nineties)

        # Derived percentages if raw counts available but pct missing
        if metrics["cross_pct"] == 0:
            att = metrics["crosses_att_p90"]
            succ = metrics["crosses_succ_p90"]
            if att > 0:
                metrics["cross_pct"] = round(succ / att * 100, 1)

        # Dribble % — StatsBomb stores challenge_ratio (successful / attempted)
        if metrics["dribble_pct"] == 0:
            cr = _safe_float(record.get("player_season_challenge_ratio"), 0)
            if cr > 0:
                metrics["dribble_pct"] = round(cr * 100, 1)
            else:
                att = _safe_float(_get(record, "dribbles_att_p90"), 0)
                succ = metrics["dribbles_succ_p90"]
                if att > 0:
                    metrics["dribble_pct"] = round(succ / att * 100, 1)

        # Duels won % — StatsBomb stores aerial_ratio; try ground_duel_win_rate
        if metrics["duels_won_pct"] == 0:
            win_rate = _safe_float(record.get("player_season_ground_duel_win_rate"), 0)
            if win_rate > 0:
                metrics["duels_won_pct"] = round(win_rate * 100, 1)
            else:
                att = metrics["duels_att_p90"]
                won = metrics["duels_won_p90"]
                if att > 0:
                    metrics["duels_won_pct"] = round(won / att * 100, 1)

        if pid:
            self._metric_cache[pid] = metrics
        return metrics

    def _position_group(self, record: Dict) -> str:
        pos_raw = _get(record, "position")
        if isinstance(pos_raw, int):
            for group, ids in _POSITION_GROUPS.items():
                if pos_raw in ids:
                    return group
        if isinstance(pos_raw, str):
            pos_lower = pos_raw.lower()
            if any(x in pos_lower for x in ["back", "defender", "centre-b", "center-b"]):
                return "Defender"
            if any(x in pos_lower for x in ["mid", "midfielder"]):
                return "Midfielder"
            if any(x in pos_lower for x in ["forward", "wing", "striker", "attack"]):
                return "Attacker"
            if "goal" in pos_lower:
                return "Goalkeeper"
        return "Midfielder"

    def _get_peer_group(self, record: Dict) -> List[Dict]:
        """Return all players in the same broad position group with enough minutes."""
        group = self._position_group(record)
        peers = []
        for r in self.raw_stats:
            if _safe_float(_get(r, "minutes"), 0) < MIN_MINUTES:
                continue
            if self._position_group(r) == group:
                peers.append(r)
        return peers

    def _compute_percentiles(
        self,
        metrics: Dict[str, float],
        peer_group: List[Dict],
    ) -> Dict[str, int]:
        """Compute percentile rank for each metric vs peer group."""
        percentiles: Dict[str, int] = {}

        for key, label, category, higher_is_better in DOSSIER_METRICS:
            peer_values = [_safe_float(_get(r, key), 0) for r in peer_group]
            val = metrics.get(key, 0.0)
            percentiles[key] = _percentile_rank(val, peer_values, higher_is_better)

        return percentiles

    def _compute_scout_ratings(self, percentiles: Dict[str, int]) -> Dict[str, float]:
        """Convert percentile averages to 1–10 scout rating scale."""
        ratings: Dict[str, float] = {}
        for label, key, drivers in SCOUT_RATING_AXES:
            vals = [percentiles.get(d, 50) for d in drivers if d in percentiles]
            avg_pct = sum(vals) / len(vals) if vals else 50
            # Map 0–100 percentile → 1–10 with a mid anchor at 5
            rating = round(1 + (avg_pct / 100) * 9, 1)
            ratings[key] = min(10.0, max(1.0, rating))
        return ratings

    # -------------------------------------------------------------------------
    # DATA ASSEMBLY
    # -------------------------------------------------------------------------

    def _build_player_data(
        self,
        record: Dict,
        metrics: Dict[str, float],
        percentiles: Dict[str, int],
        scout_ratings: Dict[str, float],
        overrides: Dict,
    ) -> Dict:
        name = overrides.get("player_name") or _get(record, "player_name") or "Unknown Player"
        team = overrides.get("team_name") or _get(record, "team_name") or "Unknown Club"
        dob = _get(record, "birth_date")
        age = overrides.get("age") or _age_from_dob(dob)
        nationality = overrides.get("nationality") or _get(record, "nationality") or ""
        position_group = self._position_group(record)
        mins = _safe_float(_get(record, "minutes"), 0)
        matches = int(_safe_float(_get(record, "matches"), 0)) or max(1, int(round(mins / 90)))

        # Format DOB
        dob_formatted = ""
        if dob:
            try:
                dob_formatted = datetime.strptime(dob[:10], "%Y-%m-%d").strftime("%-d %b %Y")
            except Exception:
                dob_formatted = dob[:10]

        return {
            "name":           name,
            "name_parts":     name.upper().split(),
            "team":           team,
            "nationality":    nationality,
            "dob":            dob_formatted,
            "age":            age,
            "position_group": position_group,
            "height":         overrides.get("height", ""),
            "strong_foot":    overrides.get("strong_foot", ""),
            "contract_exp":   overrides.get("contract_exp", ""),
            "tmv":            overrides.get("tmv", ""),
            "positions":      overrides.get("positions", [position_group[:2].upper()]),
            "minutes":        int(mins),
            "matches":        matches,
            "metrics":        metrics,
            "percentiles":    percentiles,
            "scout_ratings":  scout_ratings,
        }

    # -------------------------------------------------------------------------
    # HTML RENDERING
    # -------------------------------------------------------------------------

    def _render_html(self, d: Dict, competition_name: str, season_name: str) -> str:
        name = d["name"]
        name_upper = name.upper()

        # Split name for big cover display
        parts = name.split()
        first_part = " ".join(parts[:-1]) if len(parts) > 1 else name
        last_part = parts[-1] if len(parts) > 1 else ""

        positions_html = "".join(
            f'<span class="pos-tag pos-tag--active">{p}</span>'
            if i == 0 else f'<span class="pos-tag">{p}</span>'
            for i, p in enumerate(d.get("positions", ["LB"]))
        )

        stat_cards_html = self._render_stat_cards(d["metrics"], d["percentiles"])
        radar_html = self._render_radar_js(d["metrics"], d["percentiles"])
        scout_html = self._render_scout_bars(d["scout_ratings"])
        bio_rows_html = self._render_bio_rows(d)

        tmv_display = d.get("tmv") or "—"
        contract_display = d.get("contract_exp") or "—"
        height_display = d.get("height") or "—"
        strong_foot_display = d.get("strong_foot") or "—"
        age_display = f"Age {d['age']}" if d.get("age") else ""
        dob_display = d.get("dob") or ""

        bio_line = " · ".join(filter(None, [d.get("nationality"), dob_display, age_display]))
        season_label = competition_name or "Season Stats"
        season_sub = season_name or ""
        if season_sub and season_label and season_sub not in season_label:
            season_label = f"{season_label} {season_sub}".strip()

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{name} · Aegis Scouting Dossier</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800;900&family=Barlow:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:        #0c0c0c;
    --bg2:       #111111;
    --bg3:       #181818;
    --border:    #222222;
    --gold:      #c9a227;
    --gold-lt:   #f0c040;
    --purple:    #7c3aed;
    --amber:     #d97706;
    --grey:      #6b7280;
    --text:      #e8e8e8;
    --muted:     #666666;
    --label:     #444444;
    --white:     #ffffff;
    --section-gap: 48px;
  }}

  html {{ scroll-behavior: smooth; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Barlow', sans-serif;
    font-size: 14px;
    line-height: 1.5;
  }}

  /* ── LAYOUT ── */
  .page {{ max-width: 1100px; margin: 0 auto; padding: 0 24px 80px; }}

  /* ── COVER ── */
  .cover {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    min-height: 420px;
    border-bottom: 1px solid var(--border);
    margin-bottom: var(--section-gap);
    position: relative;
    overflow: hidden;
  }}
  .cover::after {{
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 70% 50%, #7c3aed18 0%, transparent 65%);
    pointer-events: none;
  }}
  .cover-left {{
    padding: 48px 0 48px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }}
  .cover-eyebrow {{
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.2em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 20px;
  }}
  .cover-name {{
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 900;
    font-size: clamp(52px, 7vw, 88px);
    line-height: 0.9;
    color: var(--white);
    text-transform: uppercase;
    letter-spacing: -1px;
    margin-bottom: 28px;
  }}
  .cover-divider {{
    width: 40px;
    height: 3px;
    background: var(--gold);
    margin-bottom: 20px;
  }}
  .cover-club {{
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 18px;
    letter-spacing: 0.08em;
    color: var(--text);
    text-transform: uppercase;
    margin-bottom: 6px;
  }}
  .cover-bio {{ color: var(--grey); font-size: 12px; margin-bottom: 32px; }}
  .pos-tags {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 40px; }}
  .pos-tag {{
    padding: 5px 14px;
    border: 1px solid var(--border);
    border-radius: 3px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.1em;
    color: var(--grey);
    cursor: default;
  }}
  .pos-tag--active {{
    background: var(--purple);
    border-color: var(--purple);
    color: #fff;
  }}
  .cover-attrs {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px 40px;
  }}
  .cover-attr-label {{
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 4px;
  }}
  .cover-attr-value {{
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 22px;
    color: var(--white);
  }}
  .cover-attr-value.gold {{ color: var(--gold); }}
  .cover-right {{
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    padding: 32px 0;
  }}
  .cover-badge {{
    width: 200px;
    height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, #1a1a2e 0%, #0c0c0c 100%);
    border: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
  }}
  .cover-badge::before {{
    content: '';
    position: absolute;
    inset: -1px;
    border-radius: 50%;
    background: conic-gradient(var(--purple) 0deg, transparent 120deg, var(--gold) 240deg, transparent 360deg);
    mask: radial-gradient(circle at center, transparent 95px, black 97px);
    -webkit-mask: radial-gradient(circle at center, transparent 95px, black 97px);
  }}
  .cover-badge-inner {{
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 900;
    font-size: 42px;
    color: var(--purple);
    opacity: 0.6;
    letter-spacing: -2px;
  }}

  /* ── SECTION HEADER ── */
  .section-header {{
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 10px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .section-header .pill {{
    background: var(--purple);
    color: #fff;
    padding: 2px 8px;
    border-radius: 2px;
    font-size: 8px;
  }}

  /* ── STAT GRID ── */
  .stat-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 2px;
    margin-bottom: var(--section-gap);
  }}
  .stat-card {{
    background: var(--bg2);
    padding: 20px 18px 16px;
    position: relative;
    overflow: hidden;
  }}
  .stat-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--cat-color, var(--border));
  }}
  .stat-card--attacking {{ --cat-color: var(--purple); }}
  .stat-card--distribution {{ --cat-color: var(--grey); }}
  .stat-card--defensive {{ --cat-color: var(--amber); }}
  .stat-label {{
    font-family: 'DM Mono', monospace;
    font-size: 8px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
  }}
  .stat-row {{ display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 10px; }}
  .stat-value {{
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 36px;
    color: var(--white);
    line-height: 1;
  }}
  .stat-rank {{
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--muted);
  }}
  .pbar-track {{
    height: 2px;
    background: var(--border);
    border-radius: 1px;
    overflow: hidden;
  }}
  .pbar-fill {{
    height: 100%;
    border-radius: 1px;
    background: var(--cat-color, var(--grey));
    transition: width 0.4s ease;
  }}

  /* ── RADAR ── */
  .radar-section {{
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 40px;
    margin-bottom: var(--section-gap);
    align-items: start;
  }}
  .radar-left {{ }}
  .radar-canvas {{ width: 100%; max-width: 460px; margin: 0 auto; display: block; }}
  .radar-legend {{
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-top: 16px;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 10px; }}
  .legend-dot {{
    width: 10px; height: 10px; border-radius: 50%;
    flex-shrink: 0;
  }}
  .legend-label {{ font-size: 11px; color: var(--grey); }}

  /* ── SCOUT RATINGS ── */
  .scout-section {{ margin-bottom: var(--section-gap); }}
  .scout-bars {{ display: flex; flex-direction: column; gap: 18px; max-width: 420px; }}
  .scout-bar-row {{ }}
  .scout-bar-header {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
  }}
  .scout-bar-name {{ font-size: 11px; color: var(--grey); text-transform: uppercase; letter-spacing: 0.1em; }}
  .scout-bar-score {{
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 16px;
    color: var(--white);
  }}
  .scout-track {{
    height: 3px;
    background: var(--border);
    border-radius: 2px;
  }}
  .scout-fill {{
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--purple), var(--gold));
  }}

  /* ── PROFILE SECTION ── */
  .profile-section {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 32px;
    margin-bottom: var(--section-gap);
  }}
  .profile-col-title {{
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
  }}
  .profile-bullet {{
    padding-left: 14px;
    position: relative;
    margin-bottom: 16px;
    font-size: 12.5px;
    color: var(--grey);
    line-height: 1.55;
  }}
  .profile-bullet::before {{
    content: '•';
    position: absolute;
    left: 0;
    color: var(--gold);
  }}
  .profile-bullet strong {{ color: var(--text); }}

  /* ── SEASON SUMMARY ── */
  .season-summary {{
    background: var(--bg2);
    border: 1px solid var(--border);
    padding: 20px 24px;
    display: flex;
    gap: 40px;
    margin-bottom: 24px;
    align-items: center;
  }}
  .summary-item {{ }}
  .summary-label {{
    font-family: 'DM Mono', monospace;
    font-size: 8px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 4px;
  }}
  .summary-value {{
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 28px;
    color: var(--white);
  }}

  /* ── SIDEBAR (left panel) ── */
  .two-col {{ display: grid; grid-template-columns: 220px 1fr; gap: 40px; }}
  .sidebar-profile {{ }}
  .sidebar-section-label {{
    font-family: 'DM Mono', monospace;
    font-size: 8px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
    margin-top: 24px;
  }}
  .sidebar-row {{
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #1a1a1a;
    font-size: 12px;
  }}
  .sidebar-row-label {{ color: var(--muted); }}
  .sidebar-row-value {{ color: var(--white); font-weight: 600; }}
  .sidebar-row-value.gold {{ color: var(--gold); }}

  /* ── AEGIS WATERMARK ── */
  .watermark {{
    text-align: center;
    padding-top: 40px;
    border-top: 1px solid var(--border);
    margin-top: 60px;
  }}
  .watermark-logo {{
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 13px;
    letter-spacing: 0.2em;
    color: var(--muted);
    text-transform: uppercase;
  }}
  .watermark-sub {{
    font-size: 10px;
    color: var(--label);
    margin-top: 4px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.1em;
  }}
</style>
</head>
<body>
<div class="page">

  <!-- ═══════════════════════════════════════════════════════ COVER -->
  <div class="cover">
    <div class="cover-left">
      <div>
        <div class="cover-eyebrow">Scouting Dossier · {season_label}</div>
        <div class="cover-name">{first_part}<br/>{last_part}</div>
        <div class="cover-divider"></div>
        <div class="cover-club">{d['team']}</div>
        <div class="cover-bio">{bio_line}</div>
        <div class="pos-tags">{positions_html}</div>
      </div>
      <div class="cover-attrs">
        <div>
          <div class="cover-attr-label">Height</div>
          <div class="cover-attr-value">{height_display}</div>
        </div>
        <div>
          <div class="cover-attr-label">Strong Foot</div>
          <div class="cover-attr-value">{strong_foot_display}</div>
        </div>
        <div>
          <div class="cover-attr-label">Contract</div>
          <div class="cover-attr-value">{contract_display}</div>
        </div>
        <div>
          <div class="cover-attr-label">TMV</div>
          <div class="cover-attr-value gold">{tmv_display}</div>
        </div>
      </div>
    </div>
    <div class="cover-right">
      <div class="cover-badge">
        <div class="cover-badge-inner">{d['positions'][0] if d['positions'] else 'FW'}</div>
      </div>
    </div>
  </div>

  <!-- ═══════════════════════════════════════════════════════ SCOUTING REPORT -->
  <div class="section-header">
    <span>Scouting Report</span>
    <span class="pill">{d['position_group'].upper()}</span>
  </div>

  <div class="two-col" style="margin-bottom: var(--section-gap);">
    <div class="sidebar-profile">
      <div class="sidebar-section-label">Player Profile</div>
      {bio_rows_html}
      <div class="sidebar-section-label" style="margin-top:28px;">Scout Rating</div>
      {scout_html}
    </div>
    <div class="profile-section" style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:24px;">
      {self._render_profile_bullets(d)}
    </div>
  </div>

  <!-- ═══════════════════════════════════════════════════════ DATA & METRICS -->
  <div class="section-header">
    <span>Data &amp; Metrics</span>
    <span style="color:var(--muted); font-size:9px;">{season_label} · Percentiles vs players in similar positions</span>
  </div>

  <div class="season-summary">
    <div class="summary-item">
      <div class="summary-label">Matches</div>
      <div class="summary-value">{d['matches']}</div>
    </div>
    <div class="summary-item">
      <div class="summary-label">Minutes</div>
      <div class="summary-value">{d['minutes']:,}</div>
    </div>
    <div class="summary-item">
      <div class="summary-label">Goals</div>
      <div class="summary-value">{int(d['metrics'].get('goals', 0))}</div>
    </div>
    <div class="summary-item">
      <div class="summary-label">Assists</div>
      <div class="summary-value">{int(d['metrics'].get('assists', 0))}</div>
    </div>
  </div>

  <div class="stat-grid">
    {stat_cards_html}
  </div>

  <!-- ═══════════════════════════════════════════════════════ RADAR -->
  <div class="section-header">
    <span>Performance Radar</span>
  </div>
  <div class="radar-section">
    <div>
      <canvas class="radar-canvas" id="radarChart" width="460" height="460"></canvas>
      <div class="radar-legend">
        <div class="legend-item">
          <div class="legend-dot" style="background: #7c3aed;"></div>
          <span class="legend-label">Attacking / Output</span>
        </div>
        <div class="legend-item">
          <div class="legend-dot" style="background: #9ca3af;"></div>
          <span class="legend-label">Distribution / Progression</span>
        </div>
        <div class="legend-item">
          <div class="legend-dot" style="background: #d97706;"></div>
          <span class="legend-label">Defensive / Duels</span>
        </div>
      </div>
    </div>
    <div style="display:flex; flex-direction:column; gap:4px; padding-top:8px;">
      {self._render_radar_labels(d['percentiles'])}
    </div>
  </div>

  <!-- ═══════════════════════════════════════════════════════ FOOTER -->
  <div class="watermark">
    <div class="watermark-logo">Aegis Football Advisory Group</div>
    <div class="watermark-sub">Data: StatsBomb · Powered by MTFI Platform · {datetime.now().strftime('%B %Y')}</div>
  </div>

</div>

<script>
{radar_html}
</script>
</body>
</html>"""
        return html

    def _render_bio_rows(self, d: Dict) -> str:
        rows = [
            ("Position",     " / ".join(d.get("positions", [d.get("position_group", "")[:2]]))),
            ("Height",       d.get("height") or "—"),
            ("Strong Foot",  d.get("strong_foot") or "—"),
            ("Club",         d.get("team") or "—"),
            ("Nationality",  d.get("nationality") or "—"),
            ("Contract Exp.", d.get("contract_exp") or "—"),
            ("TMV",          d.get("tmv") or "—"),
        ]
        html = ""
        for label, value in rows:
            is_gold = label == "TMV"
            val_class = 'gold' if is_gold else ''
            html += f"""
        <div class="sidebar-row">
          <span class="sidebar-row-label">{label}</span>
          <span class="sidebar-row-value {val_class}">{value}</span>
        </div>"""
        return html

    def _render_stat_cards(self, metrics: Dict, percentiles: Dict) -> str:
        html = ""
        for key, label, category, _ in DOSSIER_METRICS:
            val = metrics.get(key, 0.0)
            pct = percentiles.get(key, 50)
            color = CATEGORY_COLORS[category]
            suffix = "th" if pct not in (1, 21, 31, 41, 51, 61, 71, 81, 91) else ("st" if pct % 10 == 1 else "nd" if pct % 10 == 2 else "rd")
            if pct == 1: suffix = "st"
            elif pct == 2: suffix = "nd"
            elif pct == 3: suffix = "rd"

            # Format value
            if key in ("cross_pct", "dribble_pct", "duels_won_pct"):
                val_str = f"{val:.1f}%"
            elif key in ("goals", "assists"):
                val_str = str(int(val))
            else:
                val_str = f"{val:.2f}" if val < 10 else f"{val:.1f}"

            html += f"""
    <div class="stat-card stat-card--{category}" style="--cat-color:{color}">
      <div class="stat-label">{label}</div>
      <div class="stat-row">
        <div class="stat-value">{val_str}</div>
        <div class="stat-rank">{pct}{suffix}</div>
      </div>
      <div class="pbar-track">
        <div class="pbar-fill" style="width:{pct}%"></div>
      </div>
    </div>"""
        return html

    def _render_scout_bars(self, scout_ratings: Dict) -> str:
        labels = {k: l for l, k, _ in SCOUT_RATING_AXES}
        html = '<div class="scout-bars">'
        for label, key, _ in SCOUT_RATING_AXES:
            score = scout_ratings.get(key, 5.0)
            pct = (score / 10) * 100
            html += f"""
      <div class="scout-bar-row">
        <div class="scout-bar-header">
          <span class="scout-bar-name">{label}</span>
          <span class="scout-bar-score">{score:.1f}</span>
        </div>
        <div class="scout-track"><div class="scout-fill" style="width:{pct:.0f}%"></div></div>
      </div>"""
        html += "</div>"
        return html

    def _render_profile_bullets(self, d: Dict) -> str:
        m = d["metrics"]
        p = d["percentiles"]
        pos_group = d["position_group"]

        # Physical profile
        physical_bullets = [
            f"<strong>{'Athletic mover' if p.get('dribbles_succ_p90', 50) > 60 else 'Composed mover'}</strong> — dribble success rate of {m.get('dribble_pct', 0):.1f}% underpins ball-carrying contributions.",
            f"Duels won at {m.get('duels_won_pct', 0):.1f}% win rate ({p.get('duels_won_pct', 50)}th percentile), indicating <strong>{'above-average' if p.get('duels_won_pct', 50) > 50 else 'developing'} physical contest ability</strong>.",
        ]

        # Defensive profile
        def_bullets = [
            f"Records <strong>{m.get('tackles_p90', 0):.2f} tackles</strong> and <strong>{m.get('interceptions_p90', 0):.2f} interceptions</strong> per 90 minutes.",
            f"Ball recovery rate of <strong>{m.get('recoveries_p90', 0):.2f} per 90</strong> — ranked {p.get('recoveries_p90', 50)}th percentile among positional peers.",
        ]

        # Offensive profile
        off_bullets = [
            f"<strong>{m.get('chances_p90', 0):.2f} chances created per 90</strong>, supported by a crossing accuracy of {m.get('cross_pct', 0):.1f}%.",
            f"Contributes <strong>{int(m.get('goals', 0))} goals and {int(m.get('assists', 0))} assists</strong> this season — xA of {m.get('xa_p90', 0):.2f} per 90.",
        ]

        return f"""
      <div>
        <div class="profile-col-title">Physical Profile</div>
        {''.join(f'<div class="profile-bullet">{b}</div>' for b in physical_bullets)}
      </div>
      <div>
        <div class="profile-col-title">Defensive Contribution</div>
        {''.join(f'<div class="profile-bullet">{b}</div>' for b in def_bullets)}
      </div>
      <div>
        <div class="profile-col-title">Offensive Contribution</div>
        {''.join(f'<div class="profile-bullet">{b}</div>' for b in off_bullets)}
      </div>"""

    def _render_radar_labels(self, percentiles: Dict) -> str:
        html = ""
        for key, label, category in RADAR_METRICS:
            pct = percentiles.get(key, 50)
            color = CATEGORY_COLORS[category]
            suffix = "th"
            if pct % 10 == 1 and pct != 11: suffix = "st"
            elif pct % 10 == 2 and pct != 12: suffix = "nd"
            elif pct % 10 == 3 and pct != 13: suffix = "rd"
            html += f"""
      <div style="display:flex; justify-content:space-between; padding: 6px 0; border-bottom:1px solid #1a1a1a; align-items:center;">
        <span style="font-size:11px; color:#666; width:120px;">{label}</span>
        <span style="font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:16px; color:{color};">{pct}{suffix}</span>
      </div>"""
        return html

    def _render_radar_js(self, metrics: Dict, percentiles: Dict) -> str:
        """Generate the JS radar chart via Canvas API."""
        labels = [label for _, label, _ in RADAR_METRICS]
        values = [percentiles.get(key, 50) for key, _, _ in RADAR_METRICS]
        colors = [CATEGORY_COLORS[cat] for _, _, cat in RADAR_METRICS]

        labels_js = json.dumps(labels)
        values_js = json.dumps(values)
        colors_js = json.dumps(colors)

        return f"""
(function() {{
  const canvas = document.getElementById('radarChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H / 2;
  const R = Math.min(W, H) * 0.38;
  const labels = {labels_js};
  const values = {values_js};
  const colors = {colors_js};
  const n = labels.length;
  const rings = [20, 40, 60, 80, 100];

  ctx.clearRect(0, 0, W, H);

  function angleOf(i) {{ return (Math.PI * 2 * i / n) - Math.PI / 2; }}
  function point(i, val) {{
    const a = angleOf(i);
    const r = (val / 100) * R;
    return [cx + Math.cos(a) * r, cy + Math.sin(a) * r];
  }}

  // Rings
  rings.forEach(ring => {{
    ctx.beginPath();
    for (let i = 0; i < n; i++) {{
      const [x, y] = point(i, ring);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }}
    ctx.closePath();
    ctx.strokeStyle = '#1e1e1e';
    ctx.lineWidth = 1;
    ctx.stroke();
  }});

  // Spokes
  for (let i = 0; i < n; i++) {{
    const [x, y] = point(i, 100);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x, y);
    ctx.strokeStyle = '#222';
    ctx.lineWidth = 1;
    ctx.stroke();
  }}

  // Colour segments
  for (let i = 0; i < n; i++) {{
    const next = (i + 1) % n;
    const [x0, y0] = point(i, values[i]);
    const [x1, y1] = point(next, values[next]);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.closePath();
    ctx.fillStyle = colors[i] + '55';
    ctx.fill();
    ctx.strokeStyle = colors[i];
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }}

  // Dots
  for (let i = 0; i < n; i++) {{
    const [x, y] = point(i, values[i]);
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = colors[i];
    ctx.fill();
  }}

  // Labels
  ctx.font = '600 10px Barlow, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < n; i++) {{
    const a = angleOf(i);
    const labelR = R + 22;
    const x = cx + Math.cos(a) * labelR;
    const y = cy + Math.sin(a) * labelR;
    ctx.fillStyle = '#555';
    ctx.fillText(labels[i], x, y);
  }}
}})();
"""

    def _not_found_html(self, player_name: str) -> str:
        return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"/>
<style>body{{background:#0c0c0c;color:#666;font-family:monospace;display:flex;align-items:center;justify-content:center;height:80vh;flex-direction:column;gap:12px;}}</style>
</head><body>
<div style="font-size:48px;opacity:0.2">⚽</div>
<div style="font-size:14px;">Player not found: <strong style="color:#aaa">{player_name}</strong></div>
<div style="font-size:12px;color:#333;">Check spelling or ensure the player has ≥{MIN_MINUTES} minutes.</div>
</body></html>"""


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def generate_player_dossier(
    player_name: str,
    competition_id: int,
    season_id: int,
    output_dir: Optional[str] = None,
    manual_overrides: Optional[Dict] = None,
    competition_name: str = "",
    season_name: str = "",
) -> str:
    """
    Fetch StatsBomb data and generate a player dossier in one call.

    Args:
        player_name: Player to generate dossier for
        competition_id: StatsBomb competition ID
        season_id: StatsBomb season ID
        output_dir: Directory to write HTML file (optional)
        manual_overrides: Extra fields: height, strong_foot, tmv, contract_exp, positions, nationality
        competition_name: Display name for competition
        season_name: Display name for season (e.g. "2025/26")

    Returns:
        HTML string
    """
    from .client import StatsBombClient
    from .config import Config

    sb = StatsBombClient()
    print(f"Fetching player season stats for competition {competition_id}, season {season_id}...")
    player_stats = sb.get_player_season_stats(competition_id, season_id)
    print(f"  ✓ {len(player_stats)} players loaded")

    out = Path(output_dir) if output_dir else Config.OUTPUT_DIR
    gen = PlayerDossierGenerator(player_stats, output_dir=out)
    html = gen.generate(
        player_name=player_name,
        competition_name=competition_name,
        season_name=season_name,
        manual_overrides=manual_overrides,
    )
    return html
