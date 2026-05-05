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
# Source of truth: API_Player_Season_Stats_v4_0_0 spec (confirmed fields)
_FIELD_MAP: Dict[str, Tuple[str, ...]] = {
    # Identity
    "player_name":       ("player_name",),
    "team_name":         ("team_name",),
    "player_id":         ("player_id",),
    "team_id":           ("team_id",),
    "nationality":       ("player_nationality", "nationality"),
    "position":          ("primary_position", "position"),
    "birth_date":        ("birth_date",),
    "height":            ("player_height",),                          # cm, confirmed in spec
    # Season totals
    "minutes":           ("player_season_minutes",),
    "matches":           ("player_season_appearances",),
    # Per-90 — all confirmed in Player Season Stats v4 spec
    "goals_90":          ("player_season_goals_90",),
    "assists_90":        ("player_season_assists_90",),
    "xa_p90":            ("player_season_xa_90",),
    "np_xg_p90":         ("player_season_np_xg_90",),
    "shots_p90":         ("player_season_np_shots_90",),
    "chances_p90":       ("player_season_key_passes_90",),
    "long_balls_p90":    ("player_season_long_balls_90",),            # confirmed
    "crosses_p90":       ("player_season_crosses_90",),               # completed crosses, confirmed
    "cross_pct":         ("player_season_crossing_ratio",),           # confirmed
    "dribbles_p90":      ("player_season_dribbles_90",),              # successful dribbles, confirmed
    "dribble_pct":       ("player_season_dribble_ratio",),            # confirmed (NOT challenge_ratio)
    "recoveries_p90":    ("player_season_ball_recoveries_90",),       # confirmed
    "aerial_ratio":      ("player_season_aerial_ratio",),             # % aerial duels won, confirmed
    "obv_p90":           ("player_season_obv_90",),
    "pressures_p90":     ("player_season_pressures_90",),
    "tackles_p90":       ("player_season_tackles_90",),
    "interceptions_p90": ("player_season_interceptions_90",),
    "clearances_p90":    ("player_season_clearance_90",),
    "pass_acc":          ("player_season_passing_ratio",),
    "deep_prog_p90":     ("player_season_deep_progressions_90",),
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
# All confirmed present in StatsBomb Player Season Stats v4 spec
DOSSIER_METRICS: List[Tuple[str, str, str, bool]] = [
    ("goals",           "Goals",               "attacking",    True),
    ("assists",         "Assists",             "attacking",    True),
    ("xa_p90",          "xA / 90",             "attacking",    True),
    ("np_xg_p90",       "npxG / 90",           "attacking",    True),
    ("shots_p90",       "Shots / 90",          "attacking",    True),
    ("chances_p90",     "Key Passes / 90",     "distribution", True),
    ("long_balls_p90",  "Long Balls / 90",     "distribution", True),
    ("crosses_p90",     "Crosses / 90",        "distribution", True),
    ("cross_pct",       "Cross Success %",     "distribution", True),
    ("dribbles_p90",    "Dribbles / 90",       "distribution", True),
    ("dribble_pct",     "Dribble Success %",   "distribution", True),
    ("recoveries_p90",  "Recoveries / 90",     "defensive",    True),
    ("aerial_ratio",    "Aerial Win %",        "defensive",    True),
    ("tackles_p90",     "Tackles / 90",        "defensive",    True),
    ("interceptions_p90","Interceptions / 90", "defensive",    True),
]

RADAR_METRICS: List[Tuple[str, str, str]] = [
    ("goals",           "Goals",           "attacking"),
    ("assists",         "Assists",         "attacking"),
    ("xa_p90",          "xA / 90",         "attacking"),
    ("np_xg_p90",       "npxG / 90",       "attacking"),
    ("shots_p90",       "Shots / 90",      "attacking"),
    ("chances_p90",     "Key Passes",      "distribution"),
    ("long_balls_p90",  "Long Balls",      "distribution"),
    ("crosses_p90",     "Crosses / 90",    "distribution"),
    ("cross_pct",       "Cross %",         "distribution"),
    ("dribbles_p90",    "Dribbles / 90",   "distribution"),
    ("dribble_pct",     "Dribble %",       "distribution"),
    ("recoveries_p90",  "Recoveries",      "defensive"),
    ("aerial_ratio",    "Aerial %",        "defensive"),
    ("tackles_p90",     "Tackles / 90",    "defensive"),
    ("interceptions_p90","Intercept.",     "defensive"),
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
    ("Physical",         "physical",   ["aerial_ratio", "recoveries_p90", "pressures_p90"]),
    ("Technical",        "technical",  ["dribble_pct", "cross_pct", "xa_p90", "chances_p90"]),
    ("Offensive",        "offensive",  ["goals", "assists", "shots_p90", "np_xg_p90"]),
    ("Set Pieces",       "set_pieces", ["crosses_p90", "cross_pct", "long_balls_p90"]),
    ("Against the Ball", "atb",        ["tackles_p90", "interceptions_p90", "recoveries_p90", "aerial_ratio"]),
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

        mins = _safe_float(_get(record, "minutes"), 0)
        nineties = mins / 90.0 if mins > 0 else 1.0

        metrics: Dict[str, float] = {}

        # Goals & assists: StatsBomb only stores per-90; derive season totals
        metrics["goals"]   = round(_safe_float(_get(record, "goals_90"), 0) * nineties)
        metrics["assists"] = round(_safe_float(_get(record, "assists_90"), 0) * nineties)

        # All other metrics read directly; dribble_pct is a 0-1 ratio → convert to %
        for key, *_ in DOSSIER_METRICS:
            if key in ("goals", "assists"):
                continue
            raw = _get(record, key)
            val = _safe_float(raw, 0.0)
            metrics[key] = val

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

        # Height: API returns cm as integer; format as metres string
        api_height_cm = _safe_float(record.get("player_height"), 0)
        api_height_str = f"{api_height_cm / 100:.2f} m" if api_height_cm else ""

        # Strong foot: infer from player_season_left_foot_ratio (>60% = Left, <40% = Right)
        left_ratio = _safe_float(record.get("player_season_left_foot_ratio"), None)
        if left_ratio is not None:
            if left_ratio > 60:
                api_foot = "Left"
            elif left_ratio < 40:
                api_foot = "Right"
            else:
                api_foot = "Both"
        else:
            api_foot = ""

        return {
            "name":           name,
            "name_parts":     name.upper().split(),
            "team":           team,
            "nationality":    nationality,
            "dob":            dob_formatted,
            "age":            age,
            "position_group": position_group,
            "height":         overrides.get("height") or api_height_str,
            "strong_foot":    overrides.get("strong_foot") or api_foot,
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

    @staticmethod
    def _ordinal(n: int) -> str:
        """Return correct ordinal suffix for any integer."""
        if 11 <= (n % 100) <= 13:
            return "th"
        return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

    @staticmethod
    def _pct_color(pct: int) -> str:
        """Return CSS colour based on percentile tertile."""
        if pct >= 67:
            return "#34d399"   # green
        if pct >= 34:
            return "#f59e0b"   # amber
        return "#f87171"       # red

    @staticmethod
    def _fmt_val(key: str, val: float) -> str:
        """Format a metric value for display."""
        PCT_KEYS = {"cross_pct", "dribble_pct", "aerial_ratio", "pass_acc"}
        INT_KEYS = {"goals", "assists"}
        if key in INT_KEYS:
            return str(int(val))
        if key in PCT_KEYS:
            return f"{val:.1f}%"
        return f"{val:.2f}" if val < 10 else f"{val:.1f}"

    def _render_html(self, d: Dict, competition_name: str, season_name: str) -> str:
        name = d["name"]
        parts = name.split()
        first_part = " ".join(parts[:-1]) if len(parts) > 1 else name
        last_part = parts[-1] if len(parts) > 1 else ""

        positions_html = "".join(
            f'<span class="pos-tag pos-tag--active">{p}</span>'
            if i == 0 else f'<span class="pos-tag">{p}</span>'
            for i, p in enumerate(d.get("positions", ["MF"]))
        )

        stat_cards_html = self._render_stat_cards(d["metrics"], d["percentiles"])
        radar_js        = self._render_radar_js(d["metrics"], d["percentiles"])
        scout_html      = self._render_scout_ratings(d["scout_ratings"])
        bio_rows_html   = self._render_bio_rows(d)
        profile_html    = self._render_profile_bullets(d)

        height_display   = d.get("height") or "—"
        strong_foot      = d.get("strong_foot") or "—"
        contract_display = d.get("contract_exp") or "—"
        tmv_display      = d.get("tmv") or "—"
        age_display      = f"Age {d['age']}" if d.get("age") else ""
        bio_line         = " · ".join(filter(None, [d.get("nationality"), d.get("dob"), age_display]))
        season_label     = competition_name or "Season Stats"
        if season_name and season_name not in season_label:
            season_label = f"{season_label} {season_name}".strip()

        primary_pos = d["positions"][0] if d.get("positions") else d["position_group"][:2].upper()
        pos_group   = d.get("position_group", "")

        # Cover attribute tiles — only show TMV/contract if populated
        def attr_tile(label, value, gold=False):
            if value == "—":
                return ""
            cls = ' class="gold"' if gold else ''
            return f'<div><div class="attr-label">{label}</div><div class="attr-value"{cls}>{value}</div></div>'

        cover_attrs = "".join(filter(None, [
            attr_tile("Height", height_display),
            attr_tile("Strong Foot", strong_foot),
            attr_tile("Contract", contract_display),
            attr_tile("TMV", tmv_display, gold=True),
        ]))

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{name} · Aegis Scouting Dossier</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:ital,wght@0,400;0,600;0,700;0,800;0,900;1,700&family=Barlow:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root {{
  --bg:       #0b0b0b;
  --bg2:      #111111;
  --bg3:      #171717;
  --border:   #1f1f1f;
  --border2:  #2a2a2a;
  --gold:     #c9a227;
  --purple:   #7c3aed;
  --amber:    #d97706;
  --green:    #34d399;
  --red:      #f87171;
  --text:     #e8e8e8;
  --muted:    #555555;
  --white:    #ffffff;
  --gap:      52px;
}}

html {{ scroll-behavior: smooth; }}
body {{
  background: var(--bg);
  color: var(--text);
  font-family: 'Barlow', sans-serif;
  font-size: 14px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}}

.page {{ max-width: 1080px; margin: 0 auto; padding: 0 28px 100px; }}

/* ── COVER ─────────────────────────────────────────────────── */
.cover {{
  display: grid;
  grid-template-columns: 1fr 340px;
  min-height: 400px;
  border-bottom: 1px solid var(--border);
  margin-bottom: var(--gap);
  position: relative;
  overflow: hidden;
}}
.cover::before {{
  content: '';
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse 60% 80% at 80% 50%, #7c3aed14 0%, transparent 70%),
    radial-gradient(ellipse 30% 40% at 20% 80%, #c9a22709 0%, transparent 60%);
  pointer-events: none;
}}
.cover-left {{
  padding: 44px 0 44px;
  display: flex;
  flex-direction: column;
  gap: 0;
  position: relative;
  z-index: 1;
}}
.eyebrow {{
  font-family: 'DM Mono', monospace;
  font-size: 9px;
  letter-spacing: 0.22em;
  color: var(--muted);
  text-transform: uppercase;
  margin-bottom: 22px;
}}
.cover-name {{
  font-family: 'Barlow Condensed', sans-serif;
  font-weight: 900;
  font-size: clamp(56px, 6.5vw, 92px);
  line-height: 0.87;
  color: var(--white);
  text-transform: uppercase;
  letter-spacing: -1px;
  margin-bottom: 24px;
}}
.cover-rule {{
  width: 36px;
  height: 3px;
  background: var(--gold);
  margin-bottom: 18px;
  border-radius: 1px;
}}
.cover-club {{
  font-family: 'Barlow Condensed', sans-serif;
  font-weight: 700;
  font-size: 17px;
  letter-spacing: 0.1em;
  color: var(--text);
  text-transform: uppercase;
  margin-bottom: 5px;
}}
.cover-bio {{
  font-size: 12px;
  color: var(--muted);
  margin-bottom: 28px;
  letter-spacing: 0.02em;
}}
.pos-tags {{
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-bottom: var(--gap);
}}
.pos-tag {{
  padding: 4px 13px;
  border: 1px solid var(--border2);
  border-radius: 2px;
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.12em;
  color: var(--muted);
}}
.pos-tag--active {{
  background: var(--purple);
  border-color: var(--purple);
  color: #fff;
}}
.cover-attrs {{
  display: flex;
  gap: 32px;
  flex-wrap: wrap;
  margin-top: auto;
}}
.attr-label {{
  font-family: 'DM Mono', monospace;
  font-size: 8px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 5px;
}}
.attr-value {{
  font-family: 'Barlow Condensed', sans-serif;
  font-weight: 700;
  font-size: 20px;
  color: var(--white);
}}
.attr-value.gold {{ color: var(--gold); }}

.cover-right {{
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  z-index: 1;
}}
.cover-graphic {{
  position: relative;
  width: 260px;
  height: 260px;
}}
.cover-ring {{
  position: absolute;
  inset: 0;
  border-radius: 50%;
  border: 1px solid var(--border2);
}}
.cover-ring-inner {{
  position: absolute;
  inset: 24px;
  border-radius: 50%;
  border: 1px solid var(--border);
}}
.cover-pos-large {{
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 4px;
}}
.cover-pos-abbr {{
  font-family: 'Barlow Condensed', sans-serif;
  font-weight: 900;
  font-style: italic;
  font-size: 72px;
  color: var(--purple);
  opacity: 0.18;
  line-height: 1;
  letter-spacing: -4px;
}}
.cover-pos-label {{
  font-family: 'DM Mono', monospace;
  font-size: 8px;
  letter-spacing: 0.22em;
  color: var(--muted);
  text-transform: uppercase;
}}
.cover-arc {{
  position: absolute;
  inset: -1px;
  border-radius: 50%;
  background: conic-gradient(
    var(--purple) 0deg 90deg,
    transparent 90deg 180deg,
    var(--gold) 180deg 270deg,
    transparent 270deg 360deg
  );
  mask: radial-gradient(circle at center, transparent calc(50% - 2px), black 50%);
  -webkit-mask: radial-gradient(circle at center, transparent calc(50% - 2px), black 50%);
  opacity: 0.5;
}}

/* ── SECTION HEADER ─────────────────────────────────────────── */
.sh {{
  display: flex;
  align-items: center;
  gap: 14px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 11px;
  margin-bottom: 28px;
}}
.sh-label {{
  font-family: 'DM Mono', monospace;
  font-size: 9px;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  color: var(--muted);
}}
.sh-pill {{
  background: var(--purple);
  color: #fff;
  padding: 2px 9px;
  border-radius: 2px;
  font-family: 'DM Mono', monospace;
  font-size: 8px;
  letter-spacing: 0.08em;
}}
.sh-note {{
  margin-left: auto;
  font-family: 'DM Mono', monospace;
  font-size: 8px;
  letter-spacing: 0.08em;
  color: #333;
}}

/* ── SCOUTING REPORT ─────────────────────────────────────────── */
.report-grid {{
  display: grid;
  grid-template-columns: 200px 1fr;
  gap: 40px;
  margin-bottom: var(--gap);
}}
.sidebar-label {{
  font-family: 'DM Mono', monospace;
  font-size: 8px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 10px;
  margin-top: 28px;
}}
.sidebar-label:first-child {{ margin-top: 0; }}
.bio-row {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 7px 0;
  border-bottom: 1px solid #161616;
  font-size: 12px;
}}
.bio-row-key {{ color: var(--muted); }}
.bio-row-val {{ color: var(--white); font-weight: 600; font-size: 12px; }}
.bio-row-val.gold {{ color: var(--gold); }}

/* Scout star ratings */
.scout-ratings {{ display: flex; flex-direction: column; gap: 12px; }}
.sr-row {{ }}
.sr-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }}
.sr-name {{ font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }}
.sr-score {{
  font-family: 'Barlow Condensed', sans-serif;
  font-weight: 700;
  font-size: 15px;
  color: var(--white);
}}
.sr-track {{
  height: 2px;
  background: var(--border2);
  border-radius: 1px;
  overflow: hidden;
}}
.sr-fill {{ height: 100%; border-radius: 1px; }}

/* Profile bullets */
.profile-cols {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 28px;
}}
.pcol-title {{
  font-family: 'DM Mono', monospace;
  font-size: 8px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 14px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}}
.bullet {{
  position: relative;
  padding-left: 13px;
  margin-bottom: 14px;
  font-size: 12px;
  color: #888;
  line-height: 1.6;
}}
.bullet::before {{
  content: '•';
  position: absolute;
  left: 0;
  color: var(--gold);
  line-height: 1.6;
}}
.bullet strong {{ color: var(--text); font-weight: 600; }}

/* ── SEASON SUMMARY BAR ──────────────────────────────────────── */
.summary-bar {{
  display: flex;
  gap: 2px;
  margin-bottom: 20px;
}}
.summary-cell {{
  flex: 1;
  background: var(--bg2);
  padding: 16px 20px;
  border-top: 2px solid var(--border2);
}}
.summary-cell.hl {{ border-top-color: var(--purple); }}
.s-label {{
  font-family: 'DM Mono', monospace;
  font-size: 8px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 6px;
}}
.s-value {{
  font-family: 'Barlow Condensed', sans-serif;
  font-weight: 700;
  font-size: 30px;
  color: var(--white);
  line-height: 1;
}}

/* ── STAT CARDS ─────────────────────────────────────────────── */
.stat-grid {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 2px;
  margin-bottom: var(--gap);
}}
.sc {{
  background: var(--bg2);
  padding: 18px 16px 14px;
  position: relative;
  overflow: hidden;
}}
.sc::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: var(--cat-color, var(--border2));
}}
.sc-label {{
  font-family: 'DM Mono', monospace;
  font-size: 8px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 10px;
}}
.sc-main {{
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 10px;
}}
.sc-value {{
  font-family: 'Barlow Condensed', sans-serif;
  font-weight: 700;
  font-size: 34px;
  color: var(--white);
  line-height: 1;
}}
.sc-pct {{
  font-family: 'DM Mono', monospace;
  font-size: 12px;
  font-weight: 500;
}}
.sc-bar {{ height: 2px; background: var(--border); border-radius: 1px; overflow: hidden; }}
.sc-bar-fill {{ height: 100%; border-radius: 1px; background: var(--cat-color); opacity: 0.7; }}

/* ── RADAR ──────────────────────────────────────────────────── */
.radar-wrap {{
  display: grid;
  grid-template-columns: 1fr 260px;
  gap: 40px;
  margin-bottom: var(--gap);
  align-items: start;
}}
.radar-canvas-wrap {{
  display: flex;
  justify-content: center;
}}
canvas#radarChart {{
  width: 100%;
  max-width: 480px;
  height: auto;
  display: block;
}}
.radar-table {{ }}
.radar-row {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 0;
  border-bottom: 1px solid #141414;
}}
.radar-metric {{
  font-size: 11px;
  color: #555;
  flex: 1;
}}
.radar-pct {{
  font-family: 'Barlow Condensed', sans-serif;
  font-weight: 700;
  font-size: 16px;
  min-width: 40px;
  text-align: right;
}}
.radar-legend {{
  display: flex;
  gap: 20px;
  margin-top: 14px;
  flex-wrap: wrap;
}}
.rl-item {{ display: flex; align-items: center; gap: 7px; }}
.rl-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
.rl-label {{ font-size: 10px; color: var(--muted); }}

/* ── FOOTER ─────────────────────────────────────────────────── */
.footer {{
  margin-top: 64px;
  padding-top: 24px;
  border-top: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}}
.footer-logo {{
  font-family: 'Barlow Condensed', sans-serif;
  font-weight: 800;
  font-size: 12px;
  letter-spacing: 0.22em;
  color: var(--muted);
  text-transform: uppercase;
}}
.footer-sub {{
  font-family: 'DM Mono', monospace;
  font-size: 9px;
  color: #2a2a2a;
  letter-spacing: 0.08em;
}}
</style>
</head>
<body>
<div class="page">

<!-- ═══ COVER ═══════════════════════════════════════════════ -->
<div class="cover">
  <div class="cover-left">
    <div class="eyebrow">Scouting Dossier · {season_label}</div>
    <div class="cover-name">{first_part}<br/>{last_part}</div>
    <div class="cover-rule"></div>
    <div class="cover-club">{d['team']}</div>
    <div class="cover-bio">{bio_line}</div>
    <div class="pos-tags">{positions_html}</div>
    <div class="cover-attrs">{cover_attrs}</div>
  </div>
  <div class="cover-right">
    <div class="cover-graphic">
      <div class="cover-ring"></div>
      <div class="cover-ring-inner"></div>
      <div class="cover-arc"></div>
      <div class="cover-pos-large">
        <div class="cover-pos-abbr">{primary_pos}</div>
        <div class="cover-pos-label">{pos_group}</div>
      </div>
    </div>
  </div>
</div>

<!-- ═══ SCOUTING REPORT ══════════════════════════════════════ -->
<div class="sh">
  <span class="sh-label">Scouting Report</span>
  <span class="sh-pill">{pos_group.upper()}</span>
</div>

<div class="report-grid">
  <div>
    <div class="sidebar-label">Player Profile</div>
    {bio_rows_html}
    <div class="sidebar-label">Scout Rating</div>
    {scout_html}
  </div>
  <div class="profile-cols">
    {profile_html}
  </div>
</div>

<!-- ═══ DATA & METRICS ═══════════════════════════════════════ -->
<div class="sh">
  <span class="sh-label">Data &amp; Metrics</span>
  <span class="sh-note">{season_label} · Percentiles vs positional peers</span>
</div>

<div class="summary-bar">
  <div class="summary-cell hl">
    <div class="s-label">Matches</div>
    <div class="s-value">{d['matches']}</div>
  </div>
  <div class="summary-cell hl">
    <div class="s-label">Minutes</div>
    <div class="s-value">{d['minutes']:,}</div>
  </div>
  <div class="summary-cell hl">
    <div class="s-label">Goals</div>
    <div class="s-value">{int(d['metrics'].get('goals', 0))}</div>
  </div>
  <div class="summary-cell hl">
    <div class="s-label">Assists</div>
    <div class="s-value">{int(d['metrics'].get('assists', 0))}</div>
  </div>
  <div class="summary-cell hl">
    <div class="s-label">xA / 90</div>
    <div class="s-value">{d['metrics'].get('xa_p90', 0):.2f}</div>
  </div>
</div>

<div class="stat-grid">
  {stat_cards_html}
</div>

<!-- ═══ PERFORMANCE RADAR ════════════════════════════════════ -->
<div class="sh">
  <span class="sh-label">Performance Radar</span>
  <div class="radar-legend" style="margin:0; margin-left:auto;">
    <div class="rl-item"><div class="rl-dot" style="background:#7c3aed"></div><span class="rl-label">Attacking</span></div>
    <div class="rl-item"><div class="rl-dot" style="background:#9ca3af"></div><span class="rl-label">Distribution</span></div>
    <div class="rl-item"><div class="rl-dot" style="background:#d97706"></div><span class="rl-label">Defensive</span></div>
  </div>
</div>

<div class="radar-wrap">
  <div class="radar-canvas-wrap">
    <canvas id="radarChart" width="520" height="520"></canvas>
  </div>
  <div class="radar-table">
    {self._render_radar_table(d['percentiles'])}
  </div>
</div>

<!-- ═══ FOOTER ══════════════════════════════════════════════ -->
<div class="footer">
  <div class="footer-logo">Aegis Football Advisory Group</div>
  <div class="footer-sub">Data: StatsBomb · MTFI Platform · {datetime.now().strftime('%B %Y')}</div>
</div>

</div>
<script>
{radar_js}
</script>
</body>
</html>"""

    def _render_bio_rows(self, d: Dict) -> str:
        rows = [
            ("Position",     " / ".join(d.get("positions", [d.get("position_group", "")[:2]]))),
            ("Height",       d.get("height") or "—"),
            ("Strong Foot",  d.get("strong_foot") or "—"),
            ("Club",         d.get("team") or "—"),
            ("Nationality",  d.get("nationality") or "—"),
            ("Contract",     d.get("contract_exp") or "—"),
            ("TMV",          d.get("tmv") or "—"),
        ]
        html = ""
        for label, value in rows:
            gold = "gold" if label == "TMV" else ""
            html += (
                f'<div class="bio-row">'
                f'<span class="bio-row-key">{label}</span>'
                f'<span class="bio-row-val {gold}">{value}</span>'
                f'</div>'
            )
        return html

    def _render_scout_ratings(self, scout_ratings: Dict) -> str:
        # Colour scale for the fill bar based on score (1–10)
        def bar_color(score):
            if score >= 7:   return "#34d399"
            if score >= 5:   return "#f59e0b"
            return "#f87171"

        html = '<div class="scout-ratings">'
        for label, key, _ in SCOUT_RATING_AXES:
            score = scout_ratings.get(key, 5.0)
            pct   = (score / 10) * 100
            color = bar_color(score)
            html += (
                f'<div class="sr-row">'
                f'<div class="sr-header">'
                f'<span class="sr-name">{label}</span>'
                f'<span class="sr-score">{score:.1f}</span>'
                f'</div>'
                f'<div class="sr-track">'
                f'<div class="sr-fill" style="width:{pct:.0f}%;background:{color}"></div>'
                f'</div>'
                f'</div>'
            )
        html += '</div>'
        return html

    def _render_stat_cards(self, metrics: Dict, percentiles: Dict) -> str:
        html = ""
        for key, label, category, _ in DOSSIER_METRICS:
            val   = metrics.get(key, 0.0)
            pct   = percentiles.get(key, 50)
            color = CATEGORY_COLORS[category]
            pct_color = self._pct_color(pct)
            suffix = self._ordinal(pct)
            val_str = self._fmt_val(key, val)

            html += (
                f'<div class="sc" style="--cat-color:{color}">'
                f'<div class="sc-label">{label}</div>'
                f'<div class="sc-main">'
                f'<div class="sc-value">{val_str}</div>'
                f'<div class="sc-pct" style="color:{pct_color}">{pct}{suffix}</div>'
                f'</div>'
                f'<div class="sc-bar"><div class="sc-bar-fill" style="width:{pct}%"></div></div>'
                f'</div>'
            )
        return html

    def _render_radar_table(self, percentiles: Dict) -> str:
        html = ""
        for key, label, category in RADAR_METRICS:
            pct   = percentiles.get(key, 50)
            color = CATEGORY_COLORS[category]
            suffix = self._ordinal(pct)
            pct_color = self._pct_color(pct)
            html += (
                f'<div class="radar-row">'
                f'<span class="radar-metric" style="border-left:2px solid {color}33; padding-left:8px;">{label}</span>'
                f'<span class="radar-pct" style="color:{pct_color}">{pct}{suffix}</span>'
                f'</div>'
            )
        return html

    def _render_profile_bullets(self, d: Dict) -> str:
        m = d["metrics"]
        p = d["percentiles"]

        physical_bullets = [
            f"<strong>{'Strong in the air' if p.get('aerial_ratio', 50) > 60 else 'Average aerially'}</strong> — wins {m.get('aerial_ratio', 0):.1f}% of aerial duels ({p.get('aerial_ratio', 50)}{self._ordinal(p.get('aerial_ratio', 50))} percentile).",
            f"Ball recovery rate of <strong>{m.get('recoveries_p90', 0):.2f} per 90</strong> — ranked {p.get('recoveries_p90', 50)}{self._ordinal(p.get('recoveries_p90', 50))} percentile among positional peers.",
        ]
        def_bullets = [
            f"Records <strong>{m.get('tackles_p90', 0):.2f} tackles</strong> and <strong>{m.get('interceptions_p90', 0):.2f} interceptions</strong> per 90 minutes.",
            f"Dribble success rate of <strong>{m.get('dribble_pct', 0):.1f}%</strong>, completing <strong>{m.get('dribbles_p90', 0):.2f} dribbles per 90</strong>.",
        ]
        off_bullets = [
            f"<strong>{m.get('chances_p90', 0):.2f} key passes per 90</strong> with {m.get('crosses_p90', 0):.2f} completed crosses ({m.get('cross_pct', 0):.1f}% success rate).",
            f"Contributes <strong>{int(m.get('goals', 0))} goals and {int(m.get('assists', 0))} assists</strong> — xA of {m.get('xa_p90', 0):.2f} per 90, npxG of {m.get('np_xg_p90', 0):.2f} per 90.",
        ]

        def col(title, bullets):
            items = "".join(f'<div class="bullet">{b}</div>' for b in bullets)
            return f'<div><div class="pcol-title">{title}</div>{items}</div>'

        return (
            col("Physical Profile", physical_bullets)
            + col("Defensive Contribution", def_bullets)
            + col("Offensive Contribution", off_bullets)
        )

    def _render_radar_js(self, metrics: Dict, percentiles: Dict) -> str:
        labels = [label for _, label, _ in RADAR_METRICS]
        values = [percentiles.get(key, 50) for key, _, _ in RADAR_METRICS]
        colors = [CATEGORY_COLORS[cat] for _, _, cat in RADAR_METRICS]

        return f"""(function() {{
  const canvas = document.getElementById('radarChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H / 2;
  const R = Math.min(W, H) * 0.36;
  const labels = {json.dumps(labels)};
  const values = {json.dumps(values)};
  const colors = {json.dumps(colors)};
  const n = labels.length;

  function angle(i) {{ return (Math.PI * 2 * i / n) - Math.PI / 2; }}
  function pt(i, v) {{
    const a = angle(i), r = (v / 100) * R;
    return [cx + Math.cos(a) * r, cy + Math.sin(a) * r];
  }}

  // Background rings
  [20,40,60,80,100].forEach(ring => {{
    ctx.beginPath();
    for (let i = 0; i < n; i++) {{
      const [x,y] = pt(i, ring);
      i === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }}
    ctx.closePath();
    ctx.strokeStyle = ring === 100 ? '#2a2a2a' : '#1a1a1a';
    ctx.lineWidth = ring === 100 ? 1 : 0.5;
    ctx.stroke();
  }});

  // Spokes
  for (let i = 0; i < n; i++) {{
    const [x,y] = pt(i, 100);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x, y);
    ctx.strokeStyle = '#1e1e1e';
    ctx.lineWidth = 0.5;
    ctx.stroke();
  }}

  // Coloured segments
  for (let i = 0; i < n; i++) {{
    const j = (i + 1) % n;
    const [x0,y0] = pt(i, values[i]);
    const [x1,y1] = pt(j, values[j]);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.closePath();
    ctx.fillStyle = colors[i] + '44';
    ctx.fill();
    ctx.strokeStyle = colors[i];
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }}

  // Dots
  for (let i = 0; i < n; i++) {{
    const [x,y] = pt(i, values[i]);
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = colors[i];
    ctx.fill();
    ctx.strokeStyle = '#0b0b0b';
    ctx.lineWidth = 1;
    ctx.stroke();
  }}

  // Labels
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < n; i++) {{
    const a = angle(i);
    const r = R + 26;
    const x = cx + Math.cos(a) * r;
    const y = cy + Math.sin(a) * r;
    ctx.font = '500 9.5px Barlow, sans-serif';
    ctx.fillStyle = '#444';
    ctx.fillText(labels[i], x, y);
  }}
}})();"""

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
