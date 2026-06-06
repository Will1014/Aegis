"""
aegis/formations.py
===================
Single source of truth for formation data across the Aegis platform.

Provides:
  - normalize_formation(raw)        → canonical "4-3-3" display string
  - FORMATION_SLOT_CONFIGS          → per-formation slot definitions for analysis.py
  - FORMATION_PITCH_POSITIONS       → per-formation x/y pitch coordinates for visualizations.py

Supported formations:
  4-3-3  |  4-2-3-1  |  4-4-2  |  3-5-2  |  3-4-3  |  5-3-2  |  4-1-4-1

All additions here are automatically picked up by SquadFitAnalyzer and
AegisVisualizer — no other file changes required for new formations.
"""

from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# FORMATION STRING NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

_FORMATION_MAP: Dict[str, str] = {
    # 4-defender
    "433":    "4-3-3",
    "4330":   "4-3-3",
    "4231":   "4-2-3-1",
    "442":    "4-4-2",
    "4420":   "4-4-2",
    "4141":   "4-1-4-1",
    "4411":   "4-4-1-1",
    "451":    "4-5-1",
    "4510":   "4-5-1",
    "4321":   "4-3-2-1",
    "41212":  "4-1-2-1-2",
    # 3-defender
    "352":    "3-5-2",
    "3520":   "3-5-2",
    "343":    "3-4-3",
    "3430":   "3-4-3",
    "541":    "5-4-1",
    "5410":   "5-4-1",
    # 5-defender
    "532":    "5-3-2",
    "5320":   "5-3-2",
    "523":    "5-2-3",
}

_DISPLAY_TO_CANONICAL: Dict[str, str] = {
    "4-3-3":   "4-3-3",
    "4-2-3-1": "4-2-3-1",
    "4-4-2":   "4-4-2",
    "3-5-2":   "3-5-2",
    "3-4-3":   "3-4-3",
    "5-3-2":   "5-3-2",
    "4-1-4-1": "4-1-4-1",
    "4-4-1-1": "4-4-1-1",
    "4-5-1":   "4-5-1",
    "4-3-2-1": "4-3-2-1",
}


def normalize_formation(raw: str) -> str:
    """
    Convert any formation string to a canonical display format.

    Handles StatsBomb compact strings ("433", "4231") and display strings
    ("4-3-3", "4-2-3-1"). Falls back to "4-3-3" for unrecognised inputs.

    Examples:
        normalize_formation("433")    → "4-3-3"
        normalize_formation("4231")   → "4-2-3-1"
        normalize_formation("4-3-3")  → "4-3-3"
        normalize_formation("xyz")    → "4-3-3"
    """
    if not raw:
        return "4-3-3"
    s = str(raw).strip()
    # Already canonical display format
    if s in _DISPLAY_TO_CANONICAL:
        return _DISPLAY_TO_CANONICAL[s]
    # Compact numeric string (StatsBomb native)
    compact = s.replace("-", "").replace(" ", "")
    return _FORMATION_MAP.get(compact, "4-3-3")


# ─────────────────────────────────────────────────────────────────────────────
# FORMATION SLOT CONFIGS
# Used by SquadFitAnalyzer._generate_ideal_xi() in analysis.py
#
# Each entry:
#   slot_order: List[str]  — display order top-to-bottom, left-to-right
#   slots: Dict[slot_name → {group, pref}]
#     group: position group to draw from (GK / DEF / MID / ATT)
#     pref:  preferred StatsBomb detailed position labels (first-pass matching)
# ─────────────────────────────────────────────────────────────────────────────

FORMATION_SLOT_CONFIGS: Dict[str, Dict] = {

    "4-3-3": {
        "slot_order": ["GK", "LB", "CB1", "CB2", "RB", "DM", "CM", "CM2", "LW", "CF", "RW"],
        "slots": {
            "GK":  {"group": "GK",  "pref": ["Goalkeeper"]},
            "LB":  {"group": "DEF", "pref": ["Left-Back", "Left Wing-Back"]},
            "CB1": {"group": "DEF", "pref": ["Centre-Back"]},
            "CB2": {"group": "DEF", "pref": ["Centre-Back"]},
            "RB":  {"group": "DEF", "pref": ["Right-Back", "Right Wing-Back"]},
            "DM":  {"group": "MID", "pref": ["Defensive Midfield", "Defensive Midfielder"]},
            "CM":  {"group": "MID", "pref": ["Central Midfield", "Central Midfielder", "Midfielder"]},
            "CM2": {"group": "MID", "pref": ["Central Midfield", "Central Midfielder", "Midfielder"]},
            "LW":  {"group": "ATT", "pref": ["Left Winger", "Left Wing", "Left Midfield"]},
            "CF":  {"group": "ATT", "pref": ["Centre-Forward", "Forward", "Striker", "Second Striker"]},
            "RW":  {"group": "ATT", "pref": ["Right Winger", "Right Wing", "Right Midfield"]},
        },
    },

    "4-2-3-1": {
        "slot_order": ["GK", "LB", "CB1", "CB2", "RB", "DM1", "DM2", "AM_L", "AM_C", "AM_R", "CF"],
        "slots": {
            "GK":   {"group": "GK",  "pref": ["Goalkeeper"]},
            "LB":   {"group": "DEF", "pref": ["Left-Back", "Left Wing-Back"]},
            "CB1":  {"group": "DEF", "pref": ["Centre-Back"]},
            "CB2":  {"group": "DEF", "pref": ["Centre-Back"]},
            "RB":   {"group": "DEF", "pref": ["Right-Back", "Right Wing-Back"]},
            "DM1":  {"group": "MID", "pref": ["Defensive Midfield", "Defensive Midfielder"]},
            "DM2":  {"group": "MID", "pref": ["Defensive Midfield", "Central Midfield"]},
            "AM_L": {"group": "ATT", "pref": ["Left Winger", "Attacking Midfield", "Left Midfield"]},
            "AM_C": {"group": "ATT", "pref": ["Attacking Midfield", "Attacking Midfielder"]},
            "AM_R": {"group": "ATT", "pref": ["Right Winger", "Attacking Midfield", "Right Midfield"]},
            "CF":   {"group": "ATT", "pref": ["Centre-Forward", "Forward", "Striker"]},
        },
    },

    "4-4-2": {
        "slot_order": ["GK", "LB", "CB1", "CB2", "RB", "LM", "CM1", "CM2", "RM", "CF1", "CF2"],
        "slots": {
            "GK":  {"group": "GK",  "pref": ["Goalkeeper"]},
            "LB":  {"group": "DEF", "pref": ["Left-Back"]},
            "CB1": {"group": "DEF", "pref": ["Centre-Back"]},
            "CB2": {"group": "DEF", "pref": ["Centre-Back"]},
            "RB":  {"group": "DEF", "pref": ["Right-Back"]},
            "LM":  {"group": "MID", "pref": ["Left Midfield", "Left Winger", "Central Midfield"]},
            "CM1": {"group": "MID", "pref": ["Central Midfield", "Defensive Midfield", "Midfielder"]},
            "CM2": {"group": "MID", "pref": ["Central Midfield", "Midfielder"]},
            "RM":  {"group": "MID", "pref": ["Right Midfield", "Right Winger", "Central Midfield"]},
            "CF1": {"group": "ATT", "pref": ["Centre-Forward", "Forward", "Striker"]},
            "CF2": {"group": "ATT", "pref": ["Centre-Forward", "Second Striker", "Striker"]},
        },
    },

    "3-5-2": {
        "slot_order": ["GK", "CB1", "CB2", "CB3", "LWB", "CM1", "CM2", "CM3", "RWB", "CF1", "CF2"],
        "slots": {
            "GK":  {"group": "GK",  "pref": ["Goalkeeper"]},
            "CB1": {"group": "DEF", "pref": ["Centre-Back"]},
            "CB2": {"group": "DEF", "pref": ["Centre-Back"]},
            "CB3": {"group": "DEF", "pref": ["Centre-Back"]},
            "LWB": {"group": "DEF", "pref": ["Left Wing-Back", "Left-Back"]},
            "CM1": {"group": "MID", "pref": ["Defensive Midfield", "Defensive Midfielder"]},
            "CM2": {"group": "MID", "pref": ["Central Midfield", "Midfielder"]},
            "CM3": {"group": "MID", "pref": ["Central Midfield", "Attacking Midfield"]},
            "RWB": {"group": "DEF", "pref": ["Right Wing-Back", "Right-Back"]},
            "CF1": {"group": "ATT", "pref": ["Centre-Forward", "Forward", "Striker"]},
            "CF2": {"group": "ATT", "pref": ["Centre-Forward", "Second Striker", "Striker"]},
        },
    },

    "3-4-3": {
        "slot_order": ["GK", "CB1", "CB2", "CB3", "LWB", "CM1", "CM2", "RWB", "LW", "CF", "RW"],
        "slots": {
            "GK":  {"group": "GK",  "pref": ["Goalkeeper"]},
            "CB1": {"group": "DEF", "pref": ["Centre-Back"]},
            "CB2": {"group": "DEF", "pref": ["Centre-Back"]},
            "CB3": {"group": "DEF", "pref": ["Centre-Back"]},
            "LWB": {"group": "DEF", "pref": ["Left Wing-Back", "Left-Back"]},
            "CM1": {"group": "MID", "pref": ["Defensive Midfield", "Central Midfield"]},
            "CM2": {"group": "MID", "pref": ["Central Midfield", "Attacking Midfield"]},
            "RWB": {"group": "DEF", "pref": ["Right Wing-Back", "Right-Back"]},
            "LW":  {"group": "ATT", "pref": ["Left Winger", "Left Wing"]},
            "CF":  {"group": "ATT", "pref": ["Centre-Forward", "Forward", "Striker"]},
            "RW":  {"group": "ATT", "pref": ["Right Winger", "Right Wing"]},
        },
    },

    "5-3-2": {
        "slot_order": ["GK", "LWB", "CB1", "CB2", "CB3", "RWB", "CM1", "CM2", "CM3", "CF1", "CF2"],
        "slots": {
            "GK":  {"group": "GK",  "pref": ["Goalkeeper"]},
            "LWB": {"group": "DEF", "pref": ["Left Wing-Back", "Left-Back"]},
            "CB1": {"group": "DEF", "pref": ["Centre-Back"]},
            "CB2": {"group": "DEF", "pref": ["Centre-Back"]},
            "CB3": {"group": "DEF", "pref": ["Centre-Back"]},
            "RWB": {"group": "DEF", "pref": ["Right Wing-Back", "Right-Back"]},
            "CM1": {"group": "MID", "pref": ["Defensive Midfield", "Defensive Midfielder"]},
            "CM2": {"group": "MID", "pref": ["Central Midfield", "Midfielder"]},
            "CM3": {"group": "MID", "pref": ["Central Midfield", "Attacking Midfield"]},
            "CF1": {"group": "ATT", "pref": ["Centre-Forward", "Forward", "Striker"]},
            "CF2": {"group": "ATT", "pref": ["Centre-Forward", "Second Striker", "Striker"]},
        },
    },

    "4-1-4-1": {
        "slot_order": ["GK", "LB", "CB1", "CB2", "RB", "DM", "LM", "CM1", "CM2", "RM", "CF"],
        "slots": {
            "GK":  {"group": "GK",  "pref": ["Goalkeeper"]},
            "LB":  {"group": "DEF", "pref": ["Left-Back", "Left Wing-Back"]},
            "CB1": {"group": "DEF", "pref": ["Centre-Back"]},
            "CB2": {"group": "DEF", "pref": ["Centre-Back"]},
            "RB":  {"group": "DEF", "pref": ["Right-Back", "Right Wing-Back"]},
            "DM":  {"group": "MID", "pref": ["Defensive Midfield", "Defensive Midfielder"]},
            "LM":  {"group": "MID", "pref": ["Left Midfield", "Left Winger"]},
            "CM1": {"group": "MID", "pref": ["Central Midfield", "Midfielder"]},
            "CM2": {"group": "MID", "pref": ["Central Midfield", "Attacking Midfield"]},
            "RM":  {"group": "MID", "pref": ["Right Midfield", "Right Winger"]},
            "CF":  {"group": "ATT", "pref": ["Centre-Forward", "Forward", "Striker"]},
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# FORMATION PITCH POSITIONS
# Used by AegisVisualizer._generate_dashboard_v2_html() in visualizations.py
#
# Coordinates: x=50 is centre, x=0 left touchline, x=100 right touchline
#              y=90 is GK end, y=10 is opponent's goal
# ─────────────────────────────────────────────────────────────────────────────

FORMATION_PITCH_POSITIONS: Dict[str, Dict[str, Tuple[int, int]]] = {

    "4-3-3": {
        "GK":  (50, 90),
        "LB":  (14, 70), "CB1": (34, 75), "CB2": (66, 75), "RB":  (86, 70),
        "DM":  (50, 55), "CM":  (27, 42), "CM2": (73, 42),
        "LW":  (12, 22), "CF":  (50, 14), "RW":  (88, 22),
    },

    "4-2-3-1": {
        "GK":   (50, 90),
        "LB":   (14, 70), "CB1": (34, 75), "CB2": (66, 75), "RB":  (86, 70),
        "DM1":  (34, 57), "DM2": (66, 57),
        "AM_L": (18, 38), "AM_C": (50, 35), "AM_R": (82, 38),
        "CF":   (50, 14),
    },

    "4-4-2": {
        "GK":  (50, 90),
        "LB":  (14, 70), "CB1": (34, 75), "CB2": (66, 75), "RB":  (86, 70),
        "LM":  (12, 48), "CM1": (35, 50), "CM2": (65, 50), "RM":  (88, 48),
        "CF1": (34, 18), "CF2": (66, 18),
    },

    "3-5-2": {
        "GK":  (50, 90),
        "CB1": (27, 78), "CB2": (50, 80), "CB3": (73, 78),
        "LWB": (8,  55), "CM1": (28, 52), "CM2": (50, 48), "CM3": (72, 52), "RWB": (92, 55),
        "CF1": (34, 18), "CF2": (66, 18),
    },

    "3-4-3": {
        "GK":  (50, 90),
        "CB1": (27, 78), "CB2": (50, 80), "CB3": (73, 78),
        "LWB": (8,  55), "CM1": (38, 52), "CM2": (62, 52), "RWB": (92, 55),
        "LW":  (14, 24), "CF":  (50, 16), "RW":  (86, 24),
    },

    "5-3-2": {
        "GK":  (50, 90),
        "LWB": (8,  65), "CB1": (26, 75), "CB2": (50, 78), "CB3": (74, 75), "RWB": (92, 65),
        "CM1": (28, 48), "CM2": (50, 45), "CM3": (72, 48),
        "CF1": (34, 18), "CF2": (66, 18),
    },

    "4-1-4-1": {
        "GK":  (50, 90),
        "LB":  (14, 70), "CB1": (34, 75), "CB2": (66, 75), "RB":  (86, 70),
        "DM":  (50, 58),
        "LM":  (12, 44), "CM1": (36, 46), "CM2": (64, 46), "RM":  (88, 44),
        "CF":  (50, 14),
    },
}


def get_slot_config(formation: str) -> Dict:
    """Return slot config for a given formation, falling back to 4-3-3."""
    fmt = normalize_formation(formation)
    return FORMATION_SLOT_CONFIGS.get(fmt, FORMATION_SLOT_CONFIGS["4-3-3"])


def get_pitch_positions(formation: str) -> Dict[str, Tuple[int, int]]:
    """Return pitch position coordinates for a given formation, falling back to 4-3-3."""
    fmt = normalize_formation(formation)
    return FORMATION_PITCH_POSITIONS.get(fmt, FORMATION_PITCH_POSITIONS["4-3-3"])


# ─────────────────────────────────────────────────────────────────────────────
# FORMATION STRUCTURAL PROFILES
# Used by compute_formation_compatibility() to assess tactical fit between
# the club's current shape and the manager's preferred shape.
# ─────────────────────────────────────────────────────────────────────────────

FORMATION_PROFILES: Dict[str, Dict] = {
    "4-3-3":   {"backs": 4, "wing_backs": False, "mids": 3, "att": 3,  "width": "wide"},
    "4-2-3-1": {"backs": 4, "wing_backs": False, "mids": 5, "att": 1,  "width": "wide"},
    "4-4-2":   {"backs": 4, "wing_backs": False, "mids": 4, "att": 2,  "width": "wide"},
    "3-5-2":   {"backs": 3, "wing_backs": True,  "mids": 3, "att": 2,  "width": "very_wide"},
    "3-4-3":   {"backs": 3, "wing_backs": True,  "mids": 2, "att": 3,  "width": "wide"},
    "5-3-2":   {"backs": 5, "wing_backs": True,  "mids": 3, "att": 2,  "width": "narrow"},
    "4-1-4-1": {"backs": 4, "wing_backs": False, "mids": 5, "att": 1,  "width": "wide"},
    "4-4-1-1": {"backs": 4, "wing_backs": False, "mids": 4, "att": 2,  "width": "wide"},
    "4-5-1":   {"backs": 4, "wing_backs": False, "mids": 5, "att": 1,  "width": "wide"},
}

_COMPATIBILITY_LABELS = [
    (90, "Identical"),
    (70, "Compatible"),
    (45, "Adaptable"),
    (0,  "Divergent"),
]


def compute_formation_compatibility(formation_a: str, formation_b: str) -> Dict:
    """
    Score how compatible two formations are from a squad-building perspective.

    Returns:
        {
          'score':          0–100,
          'label':          'Identical' | 'Compatible' | 'Adaptable' | 'Divergent',
          'back_match':     bool,   — same number of central defenders
          'wing_back_match': bool,  — both use / don't use wing-backs
          'notes':          List[str],  — plain-English structural differences
        }
    """
    fa = normalize_formation(formation_a)
    fb = normalize_formation(formation_b)

    if fa == fb:
        return {
            "score": 100, "label": "Identical",
            "back_match": True, "wing_back_match": True,
            "notes": [f"Both use a {fa} — no structural adaptation required."],
        }

    pa = FORMATION_PROFILES.get(fa, FORMATION_PROFILES.get("4-3-3"))
    pb = FORMATION_PROFILES.get(fb, FORMATION_PROFILES.get("4-3-3"))

    score = 0
    notes: List[str] = []

    # ── Defender count (most structurally significant) ──────────────────
    back_match = pa["backs"] == pb["backs"]
    if back_match:
        score += 50
    else:
        diff = abs(pa["backs"] - pb["backs"])
        score += max(0, 30 - diff * 10)
        notes.append(
            f"Defensive structure changes from {pa['backs']}-back to "
            f"{pb['backs']}-back — requires positional conversion or recruitment."
        )

    # ── Wing-back requirement ────────────────────────────────────────────
    wb_match = pa["wing_backs"] == pb["wing_backs"]
    if wb_match:
        score += 20
    else:
        score += 5
        if pb["wing_backs"]:
            notes.append(
                "Manager's system requires wing-backs — current full-backs "
                "must have the athleticism and crossing ability for the role."
            )
        else:
            notes.append(
                "Club uses wing-backs; manager's system uses conventional "
                "full-backs — wing-backs may be redeployed as wide midfielders."
            )

    # ── Midfield density ─────────────────────────────────────────────────
    mid_diff = abs(pa["mids"] - pb["mids"])
    if mid_diff == 0:
        score += 20
    elif mid_diff == 1:
        score += 13
        notes.append("Minor midfield restructure required.")
    elif mid_diff == 2:
        score += 6
        notes.append("Significant midfield restructure — one midfielder's role changes substantially.")
    else:
        notes.append("Major midfield overhaul required.")

    # ── Attacking shape ──────────────────────────────────────────────────
    att_diff = abs(pa["att"] - pb["att"])
    if att_diff == 0:
        score += 10
    elif att_diff == 1:
        score += 5
        notes.append("Attacking personnel need minor positional adjustment.")
    else:
        notes.append("Attacking structure changes significantly — striker/winger balance shifts.")

    # ── Label ────────────────────────────────────────────────────────────
    label = "Divergent"
    for threshold, lbl in _COMPATIBILITY_LABELS:
        if score >= threshold:
            label = lbl
            break

    if not notes:
        notes.append(
            f"Both formations share similar structural characteristics "
            f"with minor positional adjustments required."
        )

    return {
        "score":           min(score, 100),
        "label":           label,
        "back_match":      back_match,
        "wing_back_match": wb_match,
        "club_formation":  fa,
        "mgr_formation":   fb,
        "notes":           notes,
    }
