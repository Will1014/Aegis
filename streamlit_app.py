"""
Aegis MTFI — Streamlit Web Application
=======================================
Interactive web interface for the Manager Tactical Fit Intelligence platform.
Supports single analysis and side-by-side scenario comparison.

Run:  streamlit run streamlit_app.py
"""

import os
import io
import sys
import json
import math
import threading
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Aegis MTFI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

    .stApp { background: #0a0e17; color: #e0e4ec; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1520 0%, #141c2b 100%);
        border-right: 1px solid #1e2a3a;
    }

    .metric-card {
        background: linear-gradient(135deg, #131b2e 0%, #0f1520 100%);
        border: 1px solid #1e2a3a;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
    }
    .metric-card .value {
        font-family: 'Space Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-card .value.gradient {
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .value.orange {
        background: linear-gradient(135deg, #fb923c, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .value.diff-positive {
        color: #34d399;
    }
    .metric-card .value.diff-negative {
        color: #f87171;
    }
    .metric-card .value.diff-neutral {
        color: #64748b;
    }
    .metric-card .label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #64748b;
        margin-top: 0.3rem;
    }

    .scenario-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        padding: 6px 14px;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 0.8rem;
    }
    .scenario-a { background: #0c4a6e33; color: #38bdf8; border: 1px solid #38bdf844; }
    .scenario-b { background: #7c2d1233; color: #fb923c; border: 1px solid #fb923c44; }

    .progress-line {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #94a3b8;
        line-height: 1.6;
    }
    .progress-line.highlight {
        color: #38bdf8;
        font-weight: 600;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #1e2a3a;
        border-radius: 10px;
        background: #0f1520;
    }

    .stProgress > div > div { background: linear-gradient(90deg, #38bdf8, #818cf8); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CONSTANTS & HELPERS
# ══════════════════════════════════════════════════════════════

COMPETITION_OPTIONS = {
    "Premier League": 2,
    "Championship": 3,
    "League One": 4,
    "League Two": 5,
    "Eredivisie": 6,
}

SEASON_OPTIONS = {
    "2025/26": 318,
    "2024/25": 317,
    "2023/24": 281,
    "2022/23": 235,
}

# All licensed league IDs — used for global team/manager discovery
ALL_LEAGUE_IDS = list(COMPETITION_OPTIONS.values())  # [2, 3, 4, 5, 6]

CLASSIFICATION_COLORS = {
    "Key Enabler": "#34d399",
    "Good Fit": "#38bdf8",
    "System Dependent": "#fbbf24",
    "Potentially Marginalised": "#f87171",
}

COLOR_A = "#38bdf8"
COLOR_B = "#fb923c"


def metric_card(value, label, color_class="gradient"):
    return (
        f'<div class="metric-card">'
        f'<div class="value {color_class}">{value}</div>'
        f'<div class="label">{label}</div>'
        f'</div>'
    )


def recruitment_display_df(rec_list):
    """
    Build a clean Recruitment Priorities display table from the raw
    recruitment list, which now also carries internal confidence-tier
    fields (cost_tier, cost_tooltip, etc. — see market_value.py
    estimate_recruitment_cost_band). Folds those into a single Data column
    with a visible flag rather than dumping them as separate raw columns.
    """
    rows = []
    for r in rec_list:
        flag = r.get("cost_flag_symbol", "")
        rows.append({
            "Position": r.get("position"),
            "Gap": r.get("gap"),
            "Urgency": r.get("urgency"),
            "Timeline": r.get("timeline"),
            "Est. Cost": f"{flag} £{r.get('cost_low')}–{r.get('cost_high')}M".strip(),
            "Data": r.get("cost_tier_label", "—"),
        })
    return pd.DataFrame(rows)


def recruitment_confidence_caption(rec_list):
    """Show a caption explaining the confidence flags, escalating to a
    warning if any row is a pure generic placeholder (not real data)."""
    if any(r.get("cost_tier") == "fallback_generic" for r in rec_list):
        st.warning(
            "⚠️ Some cost estimates below are generic placeholders, not "
            "calculated from real transfer data — the market-value model "
            "hasn't finished training yet.",
            icon="⚠️",
        )
    else:
        st.caption(
            "● Calibrated to real league club data · ◐ Thin league data or "
            "no league context · ○ Generic placeholder"
        )


def classification_bar(counts):
    """Render a horizontal classification breakdown."""
    # Minimum weight of 3 prevents columns from collapsing so narrow
    # that labels wrap vertically (e.g. count=1 vs count=14).
    weights = [max(counts.get(c, 0), 3) for c in CLASSIFICATION_COLORS]
    cols = st.columns(weights)
    for i, (cls_name, color) in enumerate(CLASSIFICATION_COLORS.items()):
        n = counts.get(cls_name, 0)
        cols[i].markdown(
            f"<div style='background:{color}22; border:1px solid {color}44; "
            f"border-radius:8px; padding:8px; text-align:center;'>"
            f"<span style='color:{color}; font-weight:700;'>{n}</span><br>"
            f"<span style='font-size:0.7rem; color:#94a3b8;'>{cls_name}</span></div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════
# LIVE PROGRESS — capture pipeline print() output
# ══════════════════════════════════════════════════════════════

class PipelineProgressCapture(io.StringIO):
    """
    Intercepts sys.stdout so pipeline print() calls are
    collected into a list we can stream to the UI.
    """

    def __init__(self, original_stdout):
        super().__init__()
        self._original = original_stdout
        self.lines: list = []
        self._lock = threading.Lock()

    def write(self, text):
        self._original.write(text)  # keep terminal output
        if text.strip():
            with self._lock:
                self.lines.append(text.strip())
        return len(text)

    def flush(self):
        self._original.flush()

    def get_new_lines(self, since: int = 0) -> list:
        with self._lock:
            return list(self.lines[since:])


def _run_with_progress(status_container, run_kwargs: dict):
    """
    Run the analysis pipeline while streaming print() output
    into a Streamlit status container as a live log.
    """
    from aegis import run_full_analysis_statsbomb

    capture = PipelineProgressCapture(sys.stdout)
    old_stdout = sys.stdout
    sys.stdout = capture

    progress_area = status_container.empty()
    seen = 0

    def _render_log():
        nonlocal seen
        new = capture.get_new_lines(seen)
        if new:
            seen += len(new)
            tail = capture.lines[-20:]
            html_lines = []
            for ln in tail:
                is_hl = any(k in ln for k in (
                    "STEP", "PHASE", "TEAM", "BATCH", "✓",
                    "Training", "Fetching", "Running ETL",
                    "Generating", "COMPLETE",
                ))
                css = "highlight" if is_hl else ""
                safe = ln.replace("<", "&lt;").replace(">", "&gt;")
                html_lines.append(
                    f'<div class="progress-line {css}">{safe}</div>'
                )
            progress_area.markdown("\n".join(html_lines),
                                   unsafe_allow_html=True)

    try:
        stop_event = threading.Event()

        def _poller():
            while not stop_event.is_set():
                _render_log()
                stop_event.wait(0.4)

        poller = threading.Thread(target=_poller, daemon=True)
        poller.start()

        result = run_full_analysis_statsbomb(**run_kwargs)

        stop_event.set()
        poller.join(timeout=2)
        _render_log()  # final flush

    finally:
        sys.stdout = old_stdout

    if isinstance(result, dict):
        result = [result]

    # Collect dashboard HTML files
    output_dir = Path(run_kwargs["base_dir"]) / "outputs"
    html_files = sorted(
        output_dir.glob("*.html"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    dashboards = {f.stem: f.read_text("utf-8") for f in html_files[:10]}

    # Also snapshot aegis_analysis.json so it survives being overwritten
    analysis_path = output_dir / "aegis_analysis.json"
    analysis_snapshot = {}
    if analysis_path.exists():
        with open(analysis_path) as f:
            analysis_snapshot = json.load(f)

    # Snapshot squad data: summary JSON + full squad CSV merged together
    squad_path = output_dir / "squad_fit_summary.json"
    squad_snapshot = None
    if squad_path.exists():
        with open(squad_path) as f:
            squad_snapshot = json.load(f)

    # Also load squad_fit_scores.csv (full squad, all players)
    csv_path = output_dir / "squad_fit_scores.csv"
    if csv_path.exists():
        import csv as _csv
        with open(csv_path, newline="") as f:
            _rows = list(_csv.DictReader(f))
        for r in _rows:
            if "fit_score" in r:
                try: r["fit_score"] = float(r["fit_score"])
                except (ValueError, TypeError): pass
        if squad_snapshot is None:
            squad_snapshot = {}
        squad_snapshot["squad_fit"] = _rows   # full squad list

    return result, dashboards, analysis_snapshot, squad_snapshot


# ══════════════════════════════════════════════════════════════
# CACHED WRAPPER — skips progress on cache hit
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600 * 2)
def _run_cached(
    league_id, season_id, team_input, coach_input,
    base_dir, train_model, training_league_ids,
    max_matches, visualize,
):
    """Cached version for instant repeat runs."""
    from aegis import run_full_analysis_statsbomb

    result = run_full_analysis_statsbomb(
        target_league_id=league_id,
        season_id=season_id,
        team_name=(list(team_input) if isinstance(team_input, tuple)
                   else team_input),
        coach_name=(list(coach_input) if isinstance(coach_input, tuple)
                    else coach_input),
        base_dir=base_dir,
        train_model=train_model,
        training_league_ids=list(training_league_ids),
        visualize=visualize,
        max_matches=max_matches,
    )
    if isinstance(result, dict):
        result = [result]

    output_dir = Path(base_dir) / "outputs"

    html_files = sorted(output_dir.glob("*.html"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    dashboards = {f.stem: f.read_text("utf-8") for f in html_files[:10]}

    analysis_path = output_dir / "aegis_analysis.json"
    analysis_snapshot = {}
    if analysis_path.exists():
        with open(analysis_path) as f:
            analysis_snapshot = json.load(f)

    squad_path = output_dir / "squad_fit_summary.json"
    squad_snapshot = None
    if squad_path.exists():
        with open(squad_path) as f:
            squad_snapshot = json.load(f)

    csv_path = output_dir / "squad_fit_scores.csv"
    if csv_path.exists():
        import csv as _csv
        with open(csv_path, newline="") as f:
            _rows = list(_csv.DictReader(f))
        for r in _rows:
            if "fit_score" in r:
                try: r["fit_score"] = float(r["fit_score"])
                except (ValueError, TypeError): pass
        if squad_snapshot is None:
            squad_snapshot = {}
        squad_snapshot["squad_fit"] = _rows

    return result, dashboards, analysis_snapshot, squad_snapshot


# ══════════════════════════════════════════════════════════════
# RADAR CHART
# ══════════════════════════════════════════════════════════════

def render_shortlist_radar(ranked):
    """Overlay DNA profiles for all shortlisted managers on one polar axis."""
    if not ranked:
        return None
    reference = next((e for e in ranked if e.dna_dimensions), None)
    if not reference:
        return None
    pillars = list(reference.dna_dimensions.keys())
    n = len(pillars)
    if n == 0:
        return None

    angles = np.linspace(0, 2 * math.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    COLORS_SL = [
        "#38bdf8", "#fb923c", "#34d399", "#a78bfa",
        "#f87171", "#fbbf24", "#60a5fa", "#f472b6",
        "#4ade80", "#e879f9",
    ]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0a0e17")
    ax.set_facecolor("#0f1520")
    ax.spines["polar"].set_color("#1e2a3a")
    ax.grid(color="#1e2a3a")

    for i, entry in enumerate(ranked):
        if not entry.dna_dimensions:
            continue
        color = COLORS_SL[i % len(COLORS_SL)]
        vals = ([entry.dna_dimensions.get(p, 50) for p in pillars]
                + [entry.dna_dimensions.get(pillars[0], 50)])
        ax.fill(angles, vals, alpha=0.08, color=color)
        ax.plot(angles, vals, color=color, linewidth=1.5,
                label=f"#{entry.rank} {entry.manager}")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(pillars, size=6, color="#94a3b8")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=5, colors="#475569")
    ax.legend(loc="upper right", fontsize=6, framealpha=0.3,
              labelcolor="#e0e4ec", facecolor="#131b2e", edgecolor="#1e2a3a")
    return fig


def _draw_pitch(ax):
    """Draw a simple football pitch on a matplotlib axes."""
    ax.set_facecolor("#2d5a27")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    lw = 0.8
    pc = "#ffffff"
    import matplotlib.patches as mpatches
    # Outer boundary
    ax.add_patch(mpatches.Rectangle((5, 3), 90, 94, fill=False,
                                     edgecolor=pc, linewidth=lw))
    # Halfway line + centre circle
    ax.plot([5, 95], [50, 50], color=pc, linewidth=lw)
    ax.add_patch(mpatches.Circle((50, 50), 8, fill=False,
                                  edgecolor=pc, linewidth=lw))
    ax.plot(50, 50, 'o', color=pc, markersize=1.5)
    # Own penalty box (top/GK end)
    ax.add_patch(mpatches.Rectangle((25, 75), 50, 22, fill=False,
                                     edgecolor=pc, linewidth=lw))
    ax.add_patch(mpatches.Rectangle((37, 86), 26, 11, fill=False,
                                     edgecolor=pc, linewidth=lw))
    # Opposition penalty box (bottom)
    ax.add_patch(mpatches.Rectangle((25, 3), 50, 22, fill=False,
                                     edgecolor=pc, linewidth=lw))
    ax.add_patch(mpatches.Rectangle((37, 3), 26, 11, fill=False,
                                     edgecolor=pc, linewidth=lw))
    ax.axis("off")


_PITCH_CLS_COLORS = {
    "Key Enabler":            "#34d399",
    "Good Fit":               "#38bdf8",
    "System Dependent":       "#fbbf24",
    "Potentially Marginalised": "#f87171",
}


def render_formation_pitch(ideal_xi, formation, title=""):
    """Render a single formation pitch diagram with colour-coded players."""
    from aegis.formations import get_pitch_positions
    import matplotlib.patches as mpatches

    positions = get_pitch_positions(formation)
    player_by_slot = {p["slot"]: p for p in ideal_xi}

    fig, ax = plt.subplots(figsize=(4, 5.5))
    fig.patch.set_facecolor("#0a0e17")
    _draw_pitch(ax)

    if title:
        ax.set_title(title, color="#e0e4ec", fontsize=8,
                     fontfamily="monospace", pad=4)

    for slot, (px, py) in positions.items():
        player = player_by_slot.get(slot)
        if player:
            color = _PITCH_CLS_COLORS.get(player["classification"], "#94a3b8")
            # Last name only to fit in circle label
            name = player["name"].split()[-1] if player.get("name") else slot
            ax.add_patch(mpatches.Circle((px, py), 4.5, color=color, zorder=5))
            ax.text(px, py - 7, name, ha="center", va="top",
                    color="#e0e4ec", fontsize=5.5, fontweight="bold", zorder=6)
        else:
            ax.add_patch(mpatches.Circle((px, py), 4.5, color="#1e2a3a",
                                          edgecolor="#475569", linewidth=0.8, zorder=5))
            ax.text(px, py, slot, ha="center", va="center",
                    color="#475569", fontsize=4.5, zorder=6)

    ax.text(50, 1, formation, ha="center", va="bottom",
            color="#94a3b8", fontsize=7, fontfamily="monospace")
    plt.tight_layout(pad=0.3)
    return fig


def render_dual_formation_pitch(club_xi, manager_xi, club_fmt, mgr_fmt,
                                 club_label="Club Shape",
                                 mgr_label="Manager Preferred"):
    """Render two formation pitches side by side."""
    from aegis.formations import get_pitch_positions
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(9, 6))
    fig.patch.set_facecolor("#0a0e17")

    for ax, ideal_xi, formation, label in [
        (axes[0], club_xi,    club_fmt, club_label),
        (axes[1], manager_xi, mgr_fmt,  mgr_label),
    ]:
        _draw_pitch(ax)
        ax.set_title(f"{label}\n{formation}", color="#e0e4ec",
                     fontsize=8, fontfamily="monospace", pad=4)
        positions = get_pitch_positions(formation)
        player_by_slot = {p["slot"]: p for p in ideal_xi}

        for slot, (px, py) in positions.items():
            player = player_by_slot.get(slot)
            if player:
                color = _PITCH_CLS_COLORS.get(player["classification"], "#94a3b8")
                name = player["name"].split()[-1] if player.get("name") else slot
                ax.add_patch(mpatches.Circle((px, py), 4.5, color=color, zorder=5))
                ax.text(px, py - 7, name, ha="center", va="top",
                        color="#e0e4ec", fontsize=5, fontweight="bold", zorder=6)
            else:
                ax.add_patch(mpatches.Circle((px, py), 4.5, color="#1e2a3a",
                                              edgecolor="#475569", linewidth=0.8, zorder=5))
                ax.text(px, py, slot, ha="center", va="center",
                        color="#475569", fontsize=4.5, zorder=6)

    # Legend
    legend_elements = [
        mpatches.Patch(color=c, label=l)
        for l, c in _PITCH_CLS_COLORS.items()
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=6.5, labelcolor="#e0e4ec",
               facecolor="#0f1520", edgecolor="#1e2a3a",
               framealpha=0.8, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(pad=0.5)
    return fig


def render_radar(dna_a: dict, label_a: str = "Scenario A",
                 dna_b: dict = None, label_b: str = "Scenario B"):
    """Render one or two DNA profiles as an overlaid radar chart."""
    pillars = list(dna_a.keys())
    n = len(pillars)
    if n == 0:
        return None

    angles = np.linspace(0, 2 * math.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    vals_a = [dna_a[p] for p in pillars] + [dna_a[pillars[0]]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0a0e17")
    ax.set_facecolor("#0f1520")

    ax.fill(angles, vals_a, alpha=0.12, color=COLOR_A)
    ax.plot(angles, vals_a, color=COLOR_A, linewidth=2, label=label_a)
    ax.scatter(angles[:-1], vals_a[:-1], color=COLOR_A, s=36, zorder=5)

    if dna_b:
        vals_b = [dna_b.get(p, 50) for p in pillars] + [dna_b.get(pillars[0], 50)]
        ax.fill(angles, vals_b, alpha=0.10, color=COLOR_B)
        ax.plot(angles, vals_b, color=COLOR_B, linewidth=2, label=label_b)
        ax.scatter(angles[:-1], vals_b[:-1], color=COLOR_B, s=36, zorder=5)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.3,
                  labelcolor="#e0e4ec", facecolor="#131b2e", edgecolor="#1e2a3a")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(pillars, size=7, color="#94a3b8")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=6, colors="#475569")
    ax.spines["polar"].set_color("#1e2a3a")
    ax.grid(color="#1e2a3a")
    return fig


# ══════════════════════════════════════════════════════════════
# SCENARIO RENDERER
# ══════════════════════════════════════════════════════════════

def render_metrics(r: dict, color_class: str = "gradient"):
    """Render headline metrics + classification bar for one scenario."""
    avg_fit = r.get("average_fit", 0)
    archetype = r.get("archetype", "—")
    counts = r.get("classification_counts", {})

    # Two headline cards only — classification bar covers the breakdown
    mc = st.columns(2)
    with mc[0]:
        st.markdown(metric_card(f"{avg_fit:.1f}", "Average Fit", color_class),
                    unsafe_allow_html=True)
    with mc[1]:
        st.markdown(metric_card(archetype, "Archetype", color_class),
                    unsafe_allow_html=True)

    st.write("")
    classification_bar(counts)
    return counts


def render_squad_table(squad_data, key_suffix=""):
    """Legacy wrapper — calls render_squad_detail_table."""
    return render_squad_detail_table(squad_data, key_suffix=key_suffix)


def render_squad_detail_table(squad_data, ideal_xi=None, key_suffix=""):
    """
    Render a rich squad fit table with classification filter.

    Shows player name, position, fit score, classification (with emoji),
    ideal XI slot (if provided), and strengths/risks where available.
    """
    if squad_data is None:
        st.caption("No squad data available.")
        return None

    # squad_fit_summary.json stores players under "squad_fit";
    # fall back to "players", "ideal_xi", or the data itself if already a list.
    if isinstance(squad_data, dict):
        players = (squad_data.get("squad_fit")
                   or squad_data.get("players")
                   or squad_data.get("ideal_xi")
                   or [])
    else:
        players = squad_data or []
    if not isinstance(players, list) or not players:
        st.caption("No player data available.")
        return None

    df = pd.DataFrame(players).copy()

    # ── Normalise name column immediately (before anything else) ──────────────
    # squad_fit_scores.csv and ideal_xi both use "name"; normalise to "player"
    if "name" in df.columns and "player" not in df.columns:
        df = df.rename(columns={"name": "player"})

    # ── Build ideal XI slot map ───────────────────────────────────────────────
    xi_slots = {}
    if ideal_xi:
        for p in (ideal_xi if isinstance(ideal_xi, list) else []):
            pname = (p.get("name") or p.get("player") or "") if isinstance(p, dict) else ""
            slot  = p.get("slot", "") if isinstance(p, dict) else ""
            if pname and slot:
                xi_slots[pname] = slot

    # ── Add derived columns ───────────────────────────────────────────────────
    CLS_EMOJI = {
        "Key Enabler":             "🟢",
        "Good Fit":                "🟡",
        "System Dependent":        "🟠",
        "Potentially Marginalised":"🔴",
    }
    if "classification" in df.columns:
        df["Status"] = df["classification"].map(
            lambda c: f"{CLS_EMOJI.get(c, '⚪')} {c}" if pd.notna(c) else "⚪ Unknown")

    if xi_slots and "player" in df.columns:
        df["Ideal XI"] = df["player"].map(lambda n: xi_slots.get(n, "—"))

    # ── Classification filter ─────────────────────────────────────────────────
    _cls_col = "classification" if "classification" in df.columns else None
    _all_cls = (sorted(df[_cls_col].dropna().unique().tolist())
                if _cls_col else [])
    _ordered = [c for c in ["Key Enabler","Good Fit","System Dependent",
                             "Potentially Marginalised"] if c in _all_cls]
    _filter_opts = ["All"] + _ordered

    _filter_key = f"squad_filter{key_suffix}"
    _selected = st.radio("Filter by classification", _filter_opts,
                         horizontal=True, key=_filter_key)

    _disp = df if _selected == "All" else df[df[_cls_col] == _selected]

    # Use detailed_position if available (e.g. "Left-Back" instead of "Defender")
    if "detailed_position" in df.columns:
        df["position"] = df["detailed_position"].where(
            df["detailed_position"].notna() & (df["detailed_position"] != ""),
            other=df.get("position", ""))

    # ── Column selection + rename ─────────────────────────────────────────────
    _col_map = {
        "player":    "Player",
        "position":  "Position",
        "fit_score": "Fit Score",
        "Status":    "Classification",
        "Ideal XI":  "Ideal XI Slot",
        "strengths": "Strengths",
        "risks":     "Risks",
    }
    _show = [c for c in _col_map if c in _disp.columns]
    _disp = (_disp[_show]
             .rename(columns=_col_map)
             .sort_values("Fit Score", ascending=False))

    st.dataframe(
        _disp,
        use_container_width=True,
        hide_index=True,
        height=min(len(_disp) * 38 + 40, 680),
        column_config={
            "Fit Score": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption(f"{len(_disp)} player{'s' if len(_disp) != 1 else ''} shown"
               + (f" · {len(df)} total" if len(_disp) != len(df) else ""))
    return df


# ── Module-level cached formation functions (must be at module level for ────
# ── Streamlit's cache to work correctly across re-runs) ─────────────────────

@st.cache_data(ttl=3600 * 4, show_spinner=False)
def _cached_formation_history(team, comp_id, s_id, sb_u, sb_p):
    """Cached formation history — module-level so cache persists across re-runs."""
    try:
        from aegis.dna_insights import compute_formation_history
        return compute_formation_history(team, comp_id, s_id, sb_u, sb_p)
    except Exception:
        return None


@st.cache_data(ttl=3600 * 4, show_spinner=False)
def _cached_formation_tendency(team, comp_id, s_id, sb_u, sb_p):
    """Cached formation tendency — module-level for reliable caching."""
    try:
        from aegis.dna_insights import compute_formation_tendency
        return compute_formation_tendency(team, comp_id, s_id, sb_u, sb_p)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════

for key, default in [
    ("results_a", None), ("results_b", None),
    ("dashboards_a", {}), ("dashboards_b", {}),
    ("analysis_a", {}), ("analysis_b", {}),
    ("squad_a", None), ("squad_b", None),
    ("authenticated", False),
    ("dossier_html", None), ("dossier_player", ""),
    ("dossier_player_list", []), ("dossier_player_positions", {}), ("d_league_last", ""),
    ("shortlist", None), ("shortlist_club", None),
    ("tc_data", None),
    ("report_sections", None),
    # Squad filter keys — must exist before any tab renders to prevent tab jump
    ("squad_filter_single", "All"),
    ("squad_filter_cmp_a", "All"),
    ("squad_filter_cmp_b", "All"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════
# CREDENTIAL HELPERS
# ══════════════════════════════════════════════════════════════

def _get_secret(key, default=""):
    """Resolve a credential from env vars or Streamlit secrets."""
    val = os.environ.get(key, "")
    if not val:
        try:
            val = st.secrets.get(key, default)
        except Exception:
            val = default
    return val


# StatsBomb creds — read silently, never shown to clients
sb_user = _get_secret("SB_USERNAME")
sb_pass = _get_secret("SB_PASSWORD")
has_creds = bool(sb_user and sb_pass)

_default_base = (
    "/content/aegis_data"
    if Path("/content").exists() and os.access("/content", os.W_OK)
    else str(Path.home() / "aegis_data")
)
base_dir = _get_secret("BASE_DIR", _default_base)


# ── Pre-trained model loader ──────────────────────────────────────────────────

def _load_pretrained() -> dict | None:
    """
    Copy the master pre-trained model bundle to {base_dir}/training/.
    The master model is trained on all leagues and all available seasons,
    giving K-means the richest possible clustering population.
    Returns metadata dict on success, None if no bundle exists yet.
    """
    try:
        from aegis.pretrain import load_pretrained
        return load_pretrained(base_dir)
    except Exception:
        return None


def _pretrained_meta() -> dict | None:
    """Return master model metadata without copying files."""
    try:
        from aegis.pretrain import master_meta
        return master_meta()
    except Exception:
        return None


def _coaches_from_model() -> list:
    """
    Load all manager names from manager_profiles.csv in the master pre-trained model.
    This covers ALL managers from ALL seasons × ALL leagues — not just the current season.
    Falls back to empty list if the model hasn't been trained yet.
    """
    try:
        from aegis.pretrain import MASTER_DIR
        import pandas as pd
        profiles_path = MASTER_DIR / "manager_profiles.csv"
        if profiles_path.exists():
            df = pd.read_csv(profiles_path)
            return sorted(df["coach_name"].dropna().unique().tolist())
    except Exception:
        pass
    return []


# ══════════════════════════════════════════════════════════════
# LOGIN SCREEN
# ══════════════════════════════════════════════════════════════

def _show_login():
    """Render a branded login screen."""
    import base64

    # Load logo from repo if available
    logo_html = ""
    logo_path = Path(__file__).parent / "assets" / "logo.png"
    if logo_path.exists():
        b64 = base64.b64encode(logo_path.read_bytes()).decode()
        logo_html = (
            f'<img src="data:image/png;base64,{b64}" '
            f'style="max-width:280px; margin-bottom:1.5rem;" />'
        )
    else:
        logo_html = (
            '<div style="font-family:\'Space Mono\',monospace; font-size:3rem;'
            ' font-weight:700; margin-bottom:0.5rem;'
            ' background:linear-gradient(135deg,#38bdf8,#818cf8);'
            ' -webkit-background-clip:text;'
            ' -webkit-text-fill-color:transparent;">'
            '⚽ Aegis MTFI</div>'
        )

    st.markdown(f"""
    <div style="display:flex; flex-direction:column; align-items:center;
                justify-content:center; min-height:60vh; text-align:center;">
        {logo_html}
        <div style="color:#64748b; font-size:0.95rem; margin-bottom:2.5rem;
                    letter-spacing:0.08em;">
            Manager Tactical Fit Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Centred narrow column for the input
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        code = st.text_input("Enter access code", type="password",
                             placeholder="Access code",
                             label_visibility="collapsed")
        if st.button("Sign in", use_container_width=True, type="primary"):
            app_password = _get_secret("APP_PASSWORD", "")
            if not app_password:
                # No password configured — allow entry (dev mode)
                st.session_state.authenticated = True
                st.rerun()
            elif code == app_password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid access code.")

        st.caption("Contact Aegis Advisory for access.")


if not st.session_state.authenticated:
    _show_login()
    st.stop()


# ══════════════════════════════════════════════════════════════
# TEAM & MANAGER DISCOVERY (cached)
# ══════════════════════════════════════════════════════════════

CURRENT_MANAGER_SENTINEL = "── Current manager ──"

@st.cache_data(show_spinner="Loading teams & managers…", ttl=3600 * 4)
def discover_teams_and_managers(
    _sb_user: str, _sb_pass: str,
    competition_ids: tuple, season_id: int,
    base_dir: str = "",
) -> tuple:
    """
    Discover all teams and their managers from StatsBomb match data.
    Returns (team_names, manager_names, team_manager_map).
    Cached for 4 hours per league/season combo.
    """
    import os
    os.environ["SB_USERNAME"] = _sb_user
    os.environ["SB_PASSWORD"] = _sb_pass

    # Ensure Config points at a writable directory BEFORE the client
    # calls Config.setup() in its constructor.
    from aegis import Config
    if base_dir:
        Config.set_base_dir(base_dir)
    elif not (Path(Config.BASE_DIR).exists()
              and os.access(str(Config.BASE_DIR), os.W_OK)):
        Config.set_base_dir(str(Path.home() / "aegis_data"))

    from aegis import StatsBombClient
    sb = StatsBombClient()

    team_manager_map = {}  # team_name → manager_name

    for comp_id in competition_ids:
        matches = sb.get_matches(comp_id, season_id)
        if not matches:
            continue
        for match in matches:
            for side_key in ("home_team", "away_team"):
                side = match.get(side_key, {})
                t_name = (
                    side.get(f"{side_key}_name")
                    or side.get("name")
                    or side.get("team_name")
                    or ""
                )
                mgrs = side.get("managers")
                if isinstance(mgrs, list) and mgrs:
                    m_name = mgrs[0].get("name") or mgrs[0].get("nickname") or "Unknown"
                elif isinstance(mgrs, dict):
                    m_name = mgrs.get("name") or mgrs.get("nickname") or "Unknown"
                else:
                    m_name = "Unknown"

                if t_name:
                    team_manager_map[t_name] = m_name

    team_names = sorted(team_manager_map.keys())
    manager_names = sorted(set(team_manager_map.values()) - {"Unknown"})
    return team_names, manager_names, team_manager_map


def _discover_all(
    sb_user: str, sb_pass: str, season_id: int, base_dir: str = ""
) -> tuple:
    """
    Build global team→league and team→manager maps across ALL licensed leagues.
    Returns (team_league_map, team_mgr_map, sorted_coach_list).

    team_league_map: {team_name: league_id}   — used to auto-detect league from team
    team_mgr_map:    {team_name: manager_name} — used to populate current-manager label
    sorted_coach_list: sorted list of all known manager names

    Builds on the cached discover_teams_and_managers() so repeated calls are cheap.
    """
    all_team_league: dict = {}
    all_team_mgr:   dict = {}
    all_coach_set:  set  = set()

    for lid in ALL_LEAGUE_IDS:
        try:
            teams, mgrs, team_mgr_map = discover_teams_and_managers(
                sb_user, sb_pass, (lid,), season_id, base_dir)
            for t in teams:
                # Earlier leagues take precedence if a team name clashes
                if t not in all_team_league:
                    all_team_league[t] = lid
                all_team_mgr[t] = team_mgr_map.get(t, "Unknown")
            all_coach_set.update(mgrs)
        except Exception:
            continue

    return all_team_league, all_team_mgr, sorted(all_coach_set)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚽ Aegis MTFI")
    st.caption("Manager Tactical Fit Intelligence")
    st.divider()

    # ── Mode toggle ──
    mode = st.radio("Mode", ["Manager Fit Report", "Compare Managers", "Shortlist",
                              "Player Dossier", "Transfer Costs"],
                    horizontal=False, label_visibility="collapsed")
    is_compare         = mode == "Compare Managers"
    is_shortlist       = mode == "Shortlist"
    is_dossier         = mode == "Player Dossier"
    is_transfer_costs  = mode == "Transfer Costs"

    st.divider()

    # ── Shared: Season + Advanced ──
    season_label = st.selectbox("Season", list(SEASON_OPTIONS.keys()))
    season_id = SEASON_OPTIONS[season_label]

    # Advanced Options — training leagues removed (master model covers all)
    with st.expander("⚙️ Advanced Options"):
        max_matches = st.slider("Max matches per team", 10, 50, 50)
        visualize = st.checkbox("Generate HTML dashboard", value=True)

    # Always use all licensed leagues for the pipeline
    training_league_ids = ALL_LEAGUE_IDS

    # Load master pre-trained model (trained on all leagues × all seasons)
    _pt_meta = _load_pretrained()
    if _pt_meta:
        _trained_at = _pt_meta.get("trained_at", "")[:10]
        _n_pts = _pt_meta.get("n_profiles", "")
        _n_str = f" · {_n_pts} data points" if _n_pts else ""
        st.sidebar.success(f"✓ Model loaded ({_trained_at}{_n_str})")
        train_model = False
    else:
        st.sidebar.caption("ℹ️ No pre-trained model yet — run `python -m aegis.pretrain` first.")
        train_model = True

    st.divider()

    # ── Discover teams from API (for team→league auto-detect) ────────────────
    all_teams_map: dict = {}   # team_name → league_id
    all_mgrs_map:  dict = {}   # team_name → current manager name

    if has_creds:
        try:
            all_teams_map, all_mgrs_map, _ = _discover_all(
                sb_user, sb_pass, season_id, base_dir)
        except Exception:
            pass

    # ── Load coaches from master model (all seasons × all leagues) ────────────
    # This means managers from 2022/23 are available even when viewing 2024/25,
    # enabling hypothetical cross-season, cross-league analysis.
    all_coach_options = _coaches_from_model()
    if not all_coach_options and has_creds:
        # Fallback: discover from API if model not yet trained
        try:
            _, _, all_coach_options = _discover_all(sb_user, sb_pass, season_id, base_dir)
        except Exception:
            all_coach_options = []

    # ── Helper: render one scenario's inputs ──
    def _scenario_inputs(label: str, key_prefix: str, css_class: str):
        """
        Render team → coach dropdowns for one scenario.
        Teams are drawn from ALL licensed competitions — no league filter.
        League is auto-detected from the selected team.
        Returns (league_id, team_name, coach_name).
        """
        st.markdown(
            f'<span class="scenario-header {css_class}">{label}</span>',
            unsafe_allow_html=True)

        # Team selector — all teams from all leagues
        team_options = sorted(all_teams_map.keys()) if all_teams_map else []
        if team_options:
            s_team = st.selectbox("Team", team_options,
                                  key=f"{key_prefix}_team")
            # Auto-detect league from the global map
            s_league_id  = all_teams_map.get(s_team, ALL_LEAGUE_IDS[0])
            current_mgr  = all_mgrs_map.get(s_team, "Unknown")
            # Show which league the team plays in as a caption
            league_name = next((k for k, v in COMPETITION_OPTIONS.items()
                                if v == s_league_id), "Unknown League")
            st.caption(f"🏟 {league_name}")
        else:
            s_team       = st.text_input("Team", key=f"{key_prefix}_team",
                                         placeholder="Enter team name")
            s_league_id  = ALL_LEAGUE_IDS[0]
            current_mgr  = None

        # Coach dropdown: current manager + all known managers from all leagues
        coach_choices = [CURRENT_MANAGER_SENTINEL] + all_coach_options
        if current_mgr and current_mgr != "Unknown":
            coach_choices[0] = f"{CURRENT_MANAGER_SENTINEL}  ({current_mgr})"

        s_coach_selection = st.selectbox(
            "Coach", coach_choices,
            key=f"{key_prefix}_coach",
            help="Pick 'Current manager' to analyse the incumbent, "
                 "or select any manager from any league for a hypothetical scenario.")

        if CURRENT_MANAGER_SENTINEL in s_coach_selection:
            s_coach = None  # pipeline auto-detects
        else:
            s_coach = s_coach_selection

        return s_league_id, s_team, s_coach

    # ── Render scenario inputs ──
    if is_dossier:
        st.markdown("**Player Scouting Dossier**")
        d_league = st.selectbox(
            "League", list(COMPETITION_OPTIONS.keys()), key="d_league")
        d_league_id = COMPETITION_OPTIONS[d_league]

        # Reset player list if league changes
        if st.session_state.get("d_league_last") != d_league:
            st.session_state.dossier_player_list = []
            st.session_state.dossier_player_positions = {}
            st.session_state["d_league_last"] = d_league

        d_position = st.selectbox(
            "Position",
            ["All", "Goalkeeper", "Defender", "Midfielder", "Attacker"],
            key="d_position",
            help="Filters the player list below by broad position group. "
                 "Click Load Players first to populate the list for this league.",
        )

        # Player list — filtered by position if a list has been loaded and
        # a specific position is selected. Positions come from the same
        # fetch as the name list (list_players_by_position), no extra call.
        player_list = st.session_state.dossier_player_list
        player_positions = st.session_state.get("dossier_player_positions", {})
        if player_list and d_position != "All":
            player_list = [n for n in player_list if player_positions.get(n) == d_position]

        if player_list:
            d_player = st.selectbox("Player", player_list, key="d_player_select")
        else:
            d_player = st.text_input("Player name", key="d_player_text",
                                     placeholder="e.g. Tae-Seok Lee")

        if has_creds and st.button("🔍  Load Players", key="d_load",
                                   use_container_width=True):
            with st.spinner("Loading player list…"):
                try:
                    os.environ["SB_USERNAME"] = sb_user
                    os.environ["SB_PASSWORD"] = sb_pass
                    from aegis import StatsBombClient, Config
                    from aegis.player_dossier import PlayerDossierGenerator, MIN_MINUTES
                    Config.set_base_dir(base_dir)
                    sb = StatsBombClient()
                    _stats = sb.get_player_season_stats(d_league_id, season_id)
                    gen = PlayerDossierGenerator(_stats)
                    st.session_state.dossier_player_list = gen.list_players(min_minutes=MIN_MINUTES)
                    st.session_state.dossier_player_positions = gen.list_players_by_position(min_minutes=MIN_MINUTES)
                    st.success(f"✓ {len(st.session_state.dossier_player_list)} players loaded")
                    # Player widget is coded above this button, so within
                    # THIS run it already read the old (empty) list — without
                    # forcing an immediate rerun it would show the success
                    # message but leave the Player field looking stale until
                    # the next unrelated interaction. Rerun now so the
                    # freshly loaded list shows up in the same click.
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not load players: {exc}")

        # Optional enrichment — only fields not auto-populated from StatsBomb
        # or the market-value model. TMV is no longer here — it's estimated
        # automatically (Transfermarkt match → comparables → model baseline,
        # see player_dossier.py._estimate_tmv) and shown with a tier flag
        # in the dossier itself.
        with st.expander("📝  Add contract info (optional)"):
            d_contract = st.text_input("Contract expiry", placeholder="e.g. Jun 2029", key="d_contract")

        dossier_clicked = st.button("⚽  Generate Dossier", use_container_width=True,
                                    type="primary", key="d_run")

    elif is_shortlist:
        st.markdown(
            '<span class="scenario-header scenario-a">Shortlist</span>',
            unsafe_allow_html=True)
        league_id_a, team_a, coach_a = _scenario_inputs(
            "Club", "sl_club", "scenario-a")
        st.caption("Select managers to rank against this squad (max 10).")
        sl_managers = st.multiselect(
            "Managers to rank",
            options=all_coach_options,
            max_selections=10,
            key="sl_managers",
        )
        league_id_b = team_b = coach_b = None

    elif is_compare:
        league_id_a, team_a, coach_a = _scenario_inputs(
            "Scenario A", "a", "scenario-a")
        st.write("")
        league_id_b, team_b, coach_b = _scenario_inputs(
            "Scenario B", "b", "scenario-b")
    elif is_transfer_costs:
        st.markdown(
            '<span class="scenario-header scenario-a">Transfer Costs</span>',
            unsafe_allow_html=True)
        st.caption("Recruitment cost bands and league market context, "
                   "built from the transfermarkt-datasets project — "
                   "independent of the StatsBomb manager-fit pipeline.")
        tc_league = st.selectbox(
            "League", list(COMPETITION_OPTIONS.keys()), key="tc_league")
        tc_force_refresh = st.checkbox(
            "Force refresh (ignore cache)", key="tc_force_refresh",
            help="Bypasses the local cache and re-fetches from Transfermarkt. "
                 "Cached tables normally refresh every 72h — use this if you "
                 "suspect the cache is stale.")
        tc_load_clicked = st.button(
            "💰  Load Market Data", key="tc_load", use_container_width=True)
        league_id_a = team_a = coach_a = None
        league_id_b = team_b = coach_b = None
    else:
        league_id_a, team_a, coach_a = _scenario_inputs(
            "Analysis", "single", "scenario-a")
        league_id_b = team_b = coach_b = None

    if not has_creds:
        st.caption("ℹ️ No data connection — check StatsBomb credentials in secrets.")

    st.divider()

    if not is_dossier and not is_transfer_costs:
        run_clicked = st.button("🚀  Run Analysis", use_container_width=True,
                                type="primary")
    else:
        run_clicked = False

    if not is_dossier and not is_transfer_costs and st.button("🗑️  Clear Cache", use_container_width=True):
        _run_cached.clear()
        discover_teams_and_managers.clear()
        st.toast("Cache cleared — next run will fetch fresh data.")

    st.divider()
    if st.button("🚪  Sign out", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()


# ══════════════════════════════════════════════════════════════
# BUILD KWARGS
# ══════════════════════════════════════════════════════════════

def _build_kwargs(scenario_league_id: int, team, coach) -> dict:
    """Build pipeline kwargs for one scenario."""
    return dict(
        target_league_id=scenario_league_id,
        season_id=season_id,
        team_name=team,
        coach_name=coach,
        base_dir=base_dir,
        train_model=train_model,
        training_league_ids=training_league_ids,
        visualize=visualize,
        max_matches=max_matches,
    )


# ══════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════

st.markdown("")  # spacer

# ══════════════════════════════════════════════════════════════
# PLAYER DOSSIER MODE
# ══════════════════════════════════════════════════════════════

if is_dossier:
    if "dossier_clicked" in dir() and dossier_clicked:  # noqa: F821
        pass  # fall through to generation below

    # Resolve player name from whichever input was used
    _d_player = (
        st.session_state.get("d_player_select", "")
        or st.session_state.get("d_player_text", "")
    )
    _d_league_id = COMPETITION_OPTIONS.get(
        st.session_state.get("d_league", list(COMPETITION_OPTIONS.keys())[0]), 2)

    # Check if generate was clicked this run
    _dossier_trigger = st.session_state.get("d_run", False)

    if _dossier_trigger and _d_player and has_creds:
        with st.spinner(f"Generating dossier for {_d_player}…"):
            try:
                os.environ["SB_USERNAME"] = sb_user
                os.environ["SB_PASSWORD"] = sb_pass
                from aegis import StatsBombClient
                from aegis.player_dossier import PlayerDossierGenerator, MIN_MINUTES
                from aegis import Config
                Config.set_base_dir(base_dir)

                sb = StatsBombClient()
                _stats = sb.get_player_season_stats(_d_league_id, season_id)

                # Build overrides — contract only now (TMV is auto-estimated
                # by PlayerDossierGenerator._estimate_tmv; height/foot auto from API)
                _overrides = {}
                _d_c = st.session_state.get("d_contract", "")
                if _d_c: _overrides["contract_exp"] = _d_c

                # Competition display name
                _comp_name = st.session_state.get("d_league", "")
                _season_name = season_label

                gen = PlayerDossierGenerator(_stats, output_dir=Path(base_dir) / "outputs")
                html = gen.generate(
                    player_name=_d_player,
                    competition_name=f"{_comp_name} {_season_name}",
                    season_name=_season_name,
                    manual_overrides=_overrides,
                )
                st.session_state.dossier_html = html
                st.session_state.dossier_player = _d_player
            except Exception as exc:
                st.error(f"Dossier generation failed: {exc}")

    elif _dossier_trigger and not _d_player:
        st.warning("Enter or select a player name first.")
    elif _dossier_trigger and not has_creds:
        st.error("StatsBomb credentials not configured.")

    # ── Render dossier ──
    dossier_html = st.session_state.get("dossier_html")
    if dossier_html:
        player_label = st.session_state.get("dossier_player", "Player")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### 📋 {player_label} — Scouting Dossier")
        with col2:
            st.download_button(
                "⬇️  Download Dossier",
                data=dossier_html,
                file_name=f"{player_label.replace(' ', '_')}___Scouting_Dossier.html",
                mime="text/html",
            )
        st.components.v1.html(dossier_html, height=1600, scrolling=True)
    elif not _dossier_trigger:
        st.markdown("""
<div style="text-align:center; padding:60px 0; color:#555;">
  <div style="font-size:40px; margin-bottom:16px; opacity:0.3">⚽</div>
  <div style="font-size:14px; margin-bottom:8px;">Select a league, load players, then generate a dossier.</div>
  <div style="font-size:12px; color:#333;">Height and strong foot are auto-detected from StatsBomb. Add TMV and contract in the expander if known.</div>
</div>
""", unsafe_allow_html=True)

    st.stop()


# ══════════════════════════════════════════════════════════════
# TRANSFER COSTS MODE
# ══════════════════════════════════════════════════════════════

if is_transfer_costs:
    st.markdown("### 💰 Transfer Cost Intelligence")
    st.caption(
        "Recruitment cost bands and league market context, built from "
        "[transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) "
        "— independent of your StatsBomb credentials."
    )

    _tc_league_id = COMPETITION_OPTIONS.get(
        st.session_state.get("tc_league", list(COMPETITION_OPTIONS.keys())[0]), 2)
    _tc_league_name = st.session_state.get("tc_league", "Premier League")

    if tc_load_clicked:
        with st.spinner("Fetching Transfermarkt data and league market context…"):
            try:
                from aegis.market_value import (
                    MarketValueClient, LEAGUE_CODE_MAP, POSITION_GROUP_TO_TM, EUR_TO_GBP,
                )
                from aegis.pretrain import load_pretrained_market_value

                client = MarketValueClient()
                bundle = load_pretrained_market_value()
                _use_cache = not st.session_state.get("tc_force_refresh", False)

                tm_code = LEAGUE_CODE_MAP.get(_tc_league_id)
                # clubs.total_market_value looks authoritative but is
                # documented (in the scraper's own source) as null whenever
                # it wasn't captured on a club's page at snapshot time —
                # confirmed empirically as unusably sparse (0 of 37 Premier
                # League clubs had it). club_squad_values() computes the
                # same thing by summing each club's players' individual
                # market values instead — the same field already verified
                # to match Transfermarkt exactly for individual players.
                _values = client.club_squad_values(tm_code, use_cache=_use_cache) if tm_code else \
                    client.club_squad_values(None, use_cache=_use_cache)
                _n_valid = len(_values)
                # n_clubs (distinct clubs matched to this league at all,
                # regardless of whether a squad value could be computed)
                # still comes from the clubs table, purely for the "X of Y"
                # display context.
                clubs = client.get_clubs(use_cache=_use_cache)
                _n_clubs_total = len(clubs[clubs["domestic_competition_id"] == tm_code]) if tm_code else 0

                st.session_state.tc_data = {
                    "league_name": _tc_league_name,
                    "league_id": _tc_league_id,
                    "tm_code": tm_code,
                    "n_clubs": _n_clubs_total,
                    "n_clubs_with_value": _n_valid,
                    "median_value_gbp_m": (
                        round(_values.median() * EUR_TO_GBP / 1_000_000, 1) if _n_valid else None
                    ),
                    "median_value_eur_m": (
                        round(_values.median() / 1_000_000, 1) if _n_valid else None
                    ),
                    "max_value_gbp_m": (
                        round(_values.max() * EUR_TO_GBP / 1_000_000, 1) if _n_valid else None
                    ),
                    "max_value_eur_m": (
                        round(_values.max() / 1_000_000, 1) if _n_valid else None
                    ),
                    "min_value_gbp_m": (
                        round(_values.min() * EUR_TO_GBP / 1_000_000, 1) if _n_valid else None
                    ),
                    "min_value_eur_m": (
                        round(_values.min() / 1_000_000, 1) if _n_valid else None
                    ),
                    "model_trained": bundle is not None,
                    "model_meta": bundle[1] if bundle else None,
                }
                st.success("✓ Market data loaded")
            except Exception as exc:
                st.error(f"Could not load market data: {exc}")
                st.session_state.tc_data = None

    tc_data = st.session_state.get("tc_data")

    if not tc_data:
        st.markdown("""
<div style="text-align:center; padding:60px 0; color:#555;">
  <div style="font-size:40px; margin-bottom:16px; opacity:0.3">💰</div>
  <div style="font-size:14px; margin-bottom:8px;">Select a league and click Load Market Data.</div>
  <div style="font-size:12px; color:#333;">Shows league squad-value context and estimated recruitment cost bands by position and urgency.</div>
</div>
""", unsafe_allow_html=True)
        st.stop()

    # Coverage caveat — Championship/League One/League Two aren't in the
    # transfermarkt-datasets scrape scope (competition_codes var), so
    # numbers for those leagues are lower-confidence. Premier League and
    # Eredivisie are well covered.
    if tc_data["league_id"] in (3, 4, 5):
        st.warning(
            f"⚠️ {tc_data['league_name']} has limited coverage in the Transfermarkt "
            "dataset — its scrape scope doesn't include English tiers below the "
            "Premier League. Treat these numbers as lower-confidence than "
            "Premier League or Eredivisie estimates.",
            icon="⚠️",
        )

    if not tc_data["model_trained"]:
        st.info(
            "ℹ️ The recruitment-cost model hasn't been trained yet (first "
            "deploy, or last night's training run failed). Showing static "
            "fallback estimates — these will become data-driven once the "
            "nightly pretrain job completes.",
            icon="ℹ️",
        )

    st.write("")
    st.markdown(f"**{tc_data['league_name']} — Squad Value Context** "
               f"({tc_data['n_clubs_with_value']} of {tc_data['n_clubs']} matched clubs have usable value data)")

    if tc_data["median_value_gbp_m"] is not None:
        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_card(f"£{tc_data['min_value_gbp_m']:.0f}M (€{tc_data['min_value_eur_m']:.0f}M)", "Lowest Squad Value"), unsafe_allow_html=True)
        c2.markdown(metric_card(f"£{tc_data['median_value_gbp_m']:.0f}M (€{tc_data['median_value_eur_m']:.0f}M)", "Median Squad Value", "orange"), unsafe_allow_html=True)
        c3.markdown(metric_card(f"£{tc_data['max_value_gbp_m']:.0f}M (€{tc_data['max_value_eur_m']:.0f}M)", "Highest Squad Value"), unsafe_allow_html=True)
    elif tc_data["n_clubs"] > 0:
        st.warning(
            f"⚠️ {tc_data['n_clubs']} clubs matched for this league, but none had "
            f"a usable market value figure (missing or non-numeric in the source "
            f"data) — squad value context unavailable.",
            icon="⚠️",
        )
    else:
        st.caption("No club market-value data matched for this league.")

    st.write("")
    st.markdown("**Estimated Recruitment Cost by Position & Urgency**")
    st.caption(
        "Critical = need a proven upgrade (top of the market-value distribution "
        "for this position/league). Medium = a squad-depth addition can suffice."
    )

    try:
        from aegis.market_value import (
            MarketValueClient as _MVC, estimate_recruitment_cost_band, EUR_TO_GBP as _EUR_TO_GBP,
        )
        from aegis.pretrain import load_pretrained_market_value as _load_mv

        _client = _MVC()
        _bundle = _load_mv()
        _rows = []
        _tier_symbols = {"league_calibrated": "●", "league_thin": "◐",
                         "no_league_context": "◐", "fallback_generic": "○"}
        _tier_labels = {"league_calibrated": "Calibrated", "league_thin": "Thin data",
                        "no_league_context": "No league", "fallback_generic": "Placeholder"}
        for _pos, _pos_label in [("GK", "Goalkeeper"), ("DEF", "Defender"),
                                  ("MID", "Midfielder"), ("ATT", "Attacker")]:
            _row = {"Position": _pos_label}
            _row_tier = None
            for _urg in ["Critical", "High", "Medium"]:
                _band = estimate_recruitment_cost_band(
                    _pos, tc_data["league_id"], _urg, _bundle, _client)
                _lo, _hi = _band["cost_low"], _band["cost_high"]
                _row_tier = _band["tier"]  # same tier across urgencies for a given position
                # estimate_recruitment_cost_band returns £ only — reverse the
                # same static rate to show € too, so this can be checked
                # directly against Transfermarkt without doing the maths by hand.
                _lo_eur = _lo / _EUR_TO_GBP
                _hi_eur = _hi / _EUR_TO_GBP
                _row[_urg] = f"£{_lo:.0f}–{_hi:.0f}M (€{_lo_eur:.0f}–{_hi_eur:.0f}M)"
            _row["Data"] = f"{_tier_symbols.get(_row_tier, '○')} {_tier_labels.get(_row_tier, 'Unknown')}"
            _rows.append(_row)

        st.dataframe(pd.DataFrame(_rows)[["Position", "Data", "Critical", "High", "Medium"]],
                    hide_index=True, use_container_width=True)
        st.caption(
            "● Calibrated to real league club data · ◐ Thin league data or no "
            "league context · ○ Model not trained — generic placeholder, not "
            "calculated from real transfer data."
        )
    except Exception as exc:
        st.error(f"Could not compute cost bands: {exc}")

    st.caption(
        "Figures are estimated from historical Transfermarkt transfer data "
        "and league squad values — not a quote. See the Recruitment "
        "Priorities table in Manager Fit Report / Compare Managers for "
        "cost bands tied to an actual squad's position gaps."
    )

    st.stop()


if run_clicked:
    if not has_creds:
        st.error("StatsBomb credentials not configured. "
                 "Set SB_USERNAME and SB_PASSWORD in your Streamlit secrets.")
        st.stop()
    if not team_a:
        st.error("Select a team for Scenario A.")
        st.stop()
    if is_compare and not team_b:
        st.error("Select a team for Scenario B.")
        st.stop()

    os.environ["SB_USERNAME"] = sb_user
    os.environ["SB_PASSWORD"] = sb_pass

    # ── Scenario A ──
    kwargs_a = _build_kwargs(league_id_a, team_a, coach_a)
    coach_label_a = coach_a or "current manager"
    label_a = f"{coach_label_a} → {team_a}"

    status_a = st.status(
        f"{'Scenario A' if is_compare else 'Analysis'}: {label_a}",
        expanded=True,
    )

    try:
        res_a, dash_a, analysis_a, squad_a = _run_with_progress(
            status_a, kwargs_a)
        st.session_state.results_a = res_a
        st.session_state.dashboards_a = dash_a
        st.session_state.analysis_a = analysis_a
        st.session_state.squad_a = squad_a

        # Auto-generate narrative report (no button required)
        try:
            from aegis.ai_reporter import generate_narrative_report
            _auto_report = generate_narrative_report(
                result     = res_a[0] if res_a else {},
                analysis   = analysis_a,
                squad      = squad_a,
                output_dir = Path(base_dir) / "outputs",
            )
            st.session_state.report_sections = _auto_report
        except Exception:
            st.session_state.report_sections = None

        status_a.update(
            label=f"✅ {'Scenario A' if is_compare else 'Analysis'} complete "
                  f"({len(res_a)} result{'s' if len(res_a) != 1 else ''})",
            state="complete", expanded=False,
        )
    except Exception as e:
        status_a.update(label="❌ Scenario A failed", state="error")
        st.error(f"```\n{e}\n```")
        st.stop()

    # ── Scenario B ──
    if is_compare:
        kwargs_b = _build_kwargs(league_id_b, team_b, coach_b)
        # If same team + league, skip re-training
        if team_b == team_a and league_id_b == league_id_a:
            kwargs_b["train_model"] = False

        coach_label_b = coach_b or "current manager"
        label_b = f"{coach_label_b} → {team_b}"
        status_b = st.status(f"Scenario B: {label_b}", expanded=True)

        try:
            res_b, dash_b, analysis_b, squad_b = _run_with_progress(
                status_b, kwargs_b)
            st.session_state.results_b = res_b
            st.session_state.dashboards_b = dash_b
            st.session_state.analysis_b = analysis_b
            st.session_state.squad_b = squad_b
            status_b.update(
                label=f"✅ Scenario B complete "
                      f"({len(res_b)} result{'s' if len(res_b) != 1 else ''})",
                state="complete", expanded=False,
            )
        except Exception as e:
            status_b.update(label="❌ Scenario B failed", state="error")
            st.error(f"```\n{e}\n```")
            st.stop()
    elif is_shortlist:
        if not team_a:
            st.error("Select a club.")
            st.stop()
        _sl_mgrs = st.session_state.get("sl_managers", [])
        if not _sl_mgrs:
            st.error("Select at least one manager.")
            st.stop()

        status_sl = st.status(
            f"Shortlist: {team_a} ({len(_sl_mgrs)} managers)",
            expanded=True,
        )
        try:
            from aegis.shortlist_ranker import run_shortlist
            ranked = run_shortlist(
                club=team_a,
                league_id=league_id_a,
                season_id=season_id,
                managers=_sl_mgrs,
                base_dir=base_dir,
                train_model=train_model,
                training_league_ids=training_league_ids,
                max_matches=max_matches,
            )
            st.session_state.shortlist      = ranked
            st.session_state.shortlist_club = team_a
            st.session_state.results_b      = None
            st.session_state.results_a      = None
            status_sl.update(
                label=f"✅ Shortlist complete — {len(ranked)} managers ranked",
                state="complete", expanded=False,
            )
        except Exception as e:
            status_sl.update(label="❌ Shortlist failed", state="error")
            st.error(f"```\n{e}\n```")
            st.stop()

    else:
        st.session_state.results_b = None
        st.session_state.dashboards_b = {}
        st.session_state.analysis_b = {}
        st.session_state.squad_b = None


# ══════════════════════════════════════════════════════════════
# DISPLAY — SINGLE MODE
# ══════════════════════════════════════════════════════════════

results_a = st.session_state.results_a
results_b = st.session_state.results_b

if results_a is None and not st.session_state.get("shortlist"):
    st.markdown("""
<div style="text-align:center; padding:80px 0;">
  <div style="font-size:52px; margin-bottom:20px; opacity:0.18">⚽</div>
  <div style="font-size:1rem; color:#64748b; margin-bottom:8px; letter-spacing:.04em;">
      Select a league, team and manager in the sidebar, then press Run Analysis.
  </div>
</div>""", unsafe_allow_html=True)
    st.stop()


if results_b is None and results_a is not None:
    r0        = results_a[0]
    mgr_name  = r0.get("manager", "Unknown")
    club_name = r0.get("club", "Unknown")
    avg_fit   = r0.get("average_fit", 0)
    archetype = r0.get("archetype", "—")
    club_fmt  = r0.get("primary_formation", "4-3-3")
    mgr_fmt   = r0.get("manager_formation", "4-3-3")
    compat    = r0.get("formation_compatibility", {})
    dual_data = r0.get("dual_ideal_xi")
    mgr_team  = r0.get("manager_prev_team", "")

    if r0.get("manager_dna_source") == "fallback_club_profile":
        st.warning(
            f"⚠️ No tactical data found for **{mgr_name}** in the training pool "
            f"(they weren't managing a licensed-league club in the selected season). "
            f"The scores below reflect **{club_name}'s own current profile**, not "
            f"{mgr_name}'s actual tactics — treat this as a placeholder, not a real "
            f"fit assessment.",
            icon="⚠️",
        )

    # ── Page header ───────────────────────────────────────────────────────────
    compat_score = compat.get("score", 0)
    compat_label = compat.get("label", "")
    compat_color = ("#34d399" if compat_score >= 70 else
                    "#fbbf24" if compat_score >= 45 else
                    "#f87171" if compat_label else "#64748b")

    st.markdown(
        f"<h2 style='margin-bottom:2px;'>{mgr_name}"
        f"<span style='color:#475569; font-weight:400'> → {club_name}</span></h2>",
        unsafe_allow_html=True)

    # Single compact info line: fit · archetype · formation compatibility
    _fmt_badge = ""
    if compat_label and compat_label not in ("Unknown", ""):
        _mgr_team_hint = f" · prev. {mgr_team}" if mgr_team else ""
        _c_disp = club_fmt or "?"
        _m_disp = mgr_fmt  or "?"
        _fmt_badge = (
            f" &nbsp;·&nbsp; "
            f"<span style='color:{compat_color}; font-weight:600;'>"
            f"{_c_disp} → {_m_disp}</span> "
            f"<span style='color:{compat_color};'>({compat_label})</span>"
            f"<span style='color:#475569; font-size:.80rem;'>"
            f"&nbsp;club shape → manager preferred{_mgr_team_hint}</span>"
        )
    elif not club_fmt or not mgr_fmt:
        _fmt_badge = (
            f" &nbsp;·&nbsp; "
            f"<span style='color:#f87171; font-size:.80rem;'>"
            f"⚠ Formation data unavailable</span>"
        )
    st.markdown(
        f"<div style='color:#64748b; font-size:.88rem; margin-bottom:1.4rem;'>"
        f"Avg Fit <strong style='color:#e0e4ec;'>{avg_fit:.1f}</strong>"
        f"&nbsp;·&nbsp; {archetype}"
        f"{_fmt_badge}</div>",
        unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_report, tab_formation, tab_dna, tab_squad, tab_dashboard = st.tabs([
        "📋 Report", "⚽ Formation", "🧬 DNA Profile", "👥 Squad", "📊 Dashboard",
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB: REPORT
    # ─────────────────────────────────────────────────────────────────────────
    with tab_report:
        _sec = st.session_state.get("report_sections")
        if _sec:
            _fmt_rpt = r0.get("primary_formation", "")
            for _title, _body in _sec.items():
                st.markdown(f"#### {_title}")
                st.markdown(_body)
                st.write("")

            st.divider()
            _dl1, _dl2, _dl3 = st.columns([1, 1, 2])
            with _dl1:
                try:
                    from aegis.ai_reporter import export_pdf
                    _pdf = export_pdf(_sec, mgr_name, club_name, _fmt_rpt)
                    st.download_button("⬇️  PDF", data=_pdf,
                        file_name=f"MTFI_{mgr_name.replace(' ','_')}.pdf",
                        mime="application/pdf", key="dl_pdf_tab")
                except ImportError:
                    st.caption("reportlab required for PDF.")
            with _dl2:
                from aegis.ai_reporter import export_html
                _html_r = export_html(_sec, mgr_name, club_name, _fmt_rpt)
                st.download_button("⬇️  HTML", data=_html_r,
                    file_name=f"MTFI_{mgr_name.replace(' ','_')}.html",
                    mime="text/html", key="dl_html_tab")
        else:
            # Fallback if auto-generation failed
            st.caption("Report generation did not complete. Try re-running the analysis.")
            if st.button("Retry report generation", type="secondary"):
                from aegis.ai_reporter import generate_narrative_report
                with st.spinner("Generating…"):
                    st.session_state.report_sections = generate_narrative_report(
                        result=r0, analysis=st.session_state.analysis_a,
                        squad=st.session_state.squad_a,
                        output_dir=Path(base_dir) / "outputs")
                st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # TAB: FORMATION
    # ─────────────────────────────────────────────────────────────────────────
    with tab_formation:
        _club_pct = r0.get("primary_formation_pct", 0)
        _mgr_pct  = r0.get("manager_formation_pct", 0)

        if compat_label and compat_label not in ("Unknown", ""):
            # ── Metric cards ─────────────────────────────────────────────────
            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                st.markdown(
                    f'<div class="metric-card"><div class="value" '
                    f'style="color:{compat_color};">{compat_score}</div>'
                    f'<div class="label">Compatibility</div></div>',
                    unsafe_allow_html=True)
            with fc2:
                st.markdown(metric_card(compat_label, "Formation Fit", "gradient"),
                            unsafe_allow_html=True)
            with fc3:
                _c_disp = club_fmt or "Unknown"
                _club_sub = f"{club_name}" + (f" · {_club_pct:.0f}% of matches" if _club_pct else "")
                st.markdown(metric_card(_c_disp, _club_sub, "gradient"),
                            unsafe_allow_html=True)
            with fc4:
                _m_disp = mgr_fmt or "Unknown"
                _mgr_sub = ("Manager preferred"
                            + (f" · {mgr_team}" if mgr_team else "")
                            + (f" · {_mgr_pct:.0f}% of matches" if _mgr_pct else ""))
                st.markdown(metric_card(_m_disp, _mgr_sub, "orange"),
                            unsafe_allow_html=True)
            st.write("")

            for _note in compat.get("notes", []):
                st.caption(f"ℹ️  {_note}")
            st.write("")

        elif not club_fmt and not mgr_fmt:
            st.warning("Formation data could not be retrieved from StatsBomb for this analysis. "
                       "Check that lineup data is available for the selected team and season.")

        # ── Formation history charts ──────────────────────────────────────────
        _hist_col1, _hist_col2 = st.columns(2)

        def _render_history_chart(col, team, comp_id, label, color_hex):
            with col:
                st.markdown(f"**{label}**")
                if not has_creds:
                    st.caption("Credentials required.")
                    return
                try:
                    from aegis.dna_insights import compute_formation_history as _cfh
                except ImportError:
                    _cfh = None
                if _cfh is None:
                    st.caption("Formation history unavailable.")
                    return

                _hist = _cached_formation_history(team, comp_id, season_id, sb_user, sb_pass)
                if not _hist:
                    st.caption(f"No formation data found for {team} in this season.")
                    return
                # Frequency bar chart
                _freq = _hist["frequency"]
                _fmts = sorted(_freq.keys(), key=lambda x: _freq[x], reverse=True)
                _vals = [_hist["frequency_pct"][f] for f in _fmts]
                fig_h, ax_h = plt.subplots(figsize=(3.5, max(1.5, len(_fmts)*0.55)))
                fig_h.patch.set_facecolor("#0a0e17")
                ax_h.set_facecolor("#0f1520")
                bars = ax_h.barh(_fmts, _vals, color=color_hex, alpha=0.85)
                ax_h.set_xlabel("% of matches", color="#94a3b8", fontsize=7)
                ax_h.tick_params(colors="#94a3b8", labelsize=7)
                ax_h.spines[["top","right","bottom","left"]].set_visible(False)
                ax_h.grid(axis="x", color="#1e2a3a", linewidth=0.5)
                for bar, val in zip(bars, _vals):
                    ax_h.text(val + 1, bar.get_y() + bar.get_height()/2,
                              f"{val}%", va="center", color="#e0e4ec", fontsize=6.5)
                plt.tight_layout(pad=0.4)
                st.pyplot(fig_h, use_container_width=True)
                plt.close(fig_h)
                st.caption(f"{_hist['matches_sampled']} matches sampled")
                return _hist   # caller uses this for text description

        _club_hist = _render_history_chart(
            _hist_col1, club_name, league_id_a,
            f"🏟 {club_name} — formation history", "#38bdf8")

        if not mgr_team:
            try:
                from aegis.dna_insights import get_manager_previous_team
                from aegis.pretrain import MASTER_DIR
                mgr_team = get_manager_previous_team(mgr_name, MASTER_DIR) or ""
            except Exception:
                mgr_team = ""

        _mgr_hist = None
        if mgr_team and has_creds:
            _mgr_comp = all_teams_map.get(mgr_team, league_id_a)
            _mgr_hist = _render_history_chart(
                _hist_col2, mgr_team, _mgr_comp,
                f"👔 {mgr_name} at {mgr_team} — formation history", "#fb923c")
        else:
            with _hist_col2:
                st.caption("Manager's previous club not identified in training data.")

        # Derive formations from chart data — single source of truth
        _chart_club_fmt = (_club_hist or {}).get("primary")
        _chart_mgr_fmt  = (_mgr_hist  or {}).get("primary")
        _chart_club_pct = (_club_hist or {}).get("primary_pct", 0)
        _chart_mgr_pct  = (_mgr_hist  or {}).get("primary_pct", 0)

        st.write("")

        # ── Pitches ──────────────────────────────────────────────────────────
        if dual_data:
            xi_club = dual_data.get("ideal_xi_club", [])
            xi_mgr  = dual_data.get("ideal_xi_manager", [])
            c_fmt   = dual_data.get("club_formation", club_fmt)
            m_fmt   = dual_data.get("manager_formation", mgr_fmt)

            if xi_club or xi_mgr:
                fig_dual = render_dual_formation_pitch(
                    xi_club, xi_mgr, c_fmt, m_fmt,
                    club_label=f"{club_name}",
                    mgr_label=f"{mgr_name} (preferred)",
                )
                st.pyplot(fig_dual, use_container_width=True)
                plt.close(fig_dual)

            # ── Transition risk table ─────────────────────────────────────────
            deltas  = dual_data.get("formation_deltas", [])
            changed = [d for d in deltas if d.get("classification_change")]
            if changed:
                st.markdown(f"**{len(changed)} player{'s' if len(changed)!=1 else ''} "
                            f"change classification between {c_fmt} and {m_fmt}**")
                st.dataframe(pd.DataFrame([{
                    "Player":             d["name"],
                    f"In {c_fmt}":        d["club_classification"],
                    f"In {m_fmt}":        d["manager_classification"],
                    "Signal":             d["risk"],
                } for d in deltas]),
                hide_index=True, use_container_width=True)
            else:
                st.caption("No players change classification between the two formations.")

        elif club_fmt and mgr_fmt and club_fmt == mgr_fmt:
            # Same formation — show single pitch with source context
            xi_single = r0.get("ideal_xi", [])
            if xi_single:
                _, _pc, _ = st.columns([1, 2, 1])
                with _pc:
                    fig_s = render_formation_pitch(xi_single, club_fmt,
                                                   title=f"{club_name} · {club_fmt}")
                    st.pyplot(fig_s, use_container_width=True)
                    plt.close(fig_s)

            # Text uses chart data so it always matches what's displayed
            _c_lbl = f"**{_chart_club_fmt}**" if _chart_club_fmt else "an unknown formation"
            _m_lbl = f"**{_chart_mgr_fmt}**"  if _chart_mgr_fmt  else "an unknown formation"
            _c_src = (f"**{club_name}** predominantly play {_c_lbl}"
                      + (f" ({_chart_club_pct:.0f}% of matches)" if _chart_club_pct else "") + ".")
            _m_src = (f"**{mgr_name}**'s preferred shape"
                      + (f" at {mgr_team}" if mgr_team else "")
                      + f" is {_m_lbl}"
                      + (f" ({_chart_mgr_pct:.0f}% of matches)" if _chart_mgr_pct else "") + ".")
            st.markdown(f"{_c_src}  {_m_src}  No structural adaptation required.")

        else:
            # Formations differ but dual XI data absent — still use chart data for text
            xi_single = r0.get("ideal_xi", [])
            _eff_club = _chart_club_fmt or club_fmt
            _eff_mgr  = _chart_mgr_fmt  or mgr_fmt
            if xi_single:
                _pc1, _pc2 = st.columns(2)
                with _pc1:
                    fig_c = render_formation_pitch(xi_single, _eff_club,
                                                   title=f"{club_name} ({_eff_club})")
                    st.pyplot(fig_c, use_container_width=True)
                    plt.close(fig_c)
                with _pc2:
                    fig_m = render_formation_pitch(xi_single, _eff_mgr,
                                                   title=f"{mgr_name} preferred ({_eff_mgr})")
                    st.pyplot(fig_m, use_container_width=True)
                    plt.close(fig_m)

        # ── Summary sentence for all formation cases ──────────────────────────
        if _chart_club_fmt or _chart_mgr_fmt:
            _c_lbl2 = f"**{_chart_club_fmt}**" if _chart_club_fmt else "unknown"
            _m_lbl2 = f"**{_chart_mgr_fmt}**"  if _chart_mgr_fmt  else "unknown"
            _c_src2 = (f"**{club_name}** predominantly play {_c_lbl2}"
                       + (f" ({_chart_club_pct:.0f}% of matches)" if _chart_club_pct else "") + ".")
            _m_src2 = (f"**{mgr_name}**'s preferred shape"
                       + (f" at {mgr_team}" if mgr_team else "")
                       + f" is {_m_lbl2}"
                       + (f" ({_chart_mgr_pct:.0f}% of matches)" if _chart_mgr_pct else "") + ".")
            if not dual_data and not (_chart_club_fmt and _chart_mgr_fmt and _chart_club_fmt == _chart_mgr_fmt):
                try:
                    from aegis.formations import compute_formation_compatibility as _cfc2
                    _c2 = _cfc2(_chart_club_fmt, _chart_mgr_fmt) if (
                        _chart_club_fmt and _chart_mgr_fmt) else {}
                    _v2 = (f"Compatibility: **{_c2['label']}** ({_c2['score']}/100)."
                           if _c2.get("label") and _c2.get("label") != "Unknown" else
                           "Structural adaptation will be required.")
                except Exception:
                    _v2 = "Structural adaptation will be required."
                st.markdown(f"{_c_src2}  {_m_src2}  {_v2}")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB: DNA PROFILE
    # ─────────────────────────────────────────────────────────────────────────
    with tab_dna:
        dna = st.session_state.analysis_a.get("dna_dimensions", {})
        if dna:
            # Radar centred
            _dc1, _dc2, _dc3 = st.columns([1, 2, 1])
            with _dc2:
                fig_r = render_radar(dna, label_a=mgr_name)
                if fig_r:
                    st.pyplot(fig_r, use_container_width=True)
                    plt.close(fig_r)

            st.write("")
            # MASTER_DIR is the source of truth for manager profiles / model
            try:
                from aegis.pretrain import MASTER_DIR as _tr_dir
            except Exception:
                _tr_dir = Path(base_dir) / "data" / "processed" / "training"
            _cluster = r0.get("cluster", 0)

            try:
                from aegis.dna_insights import (
                    compute_manager_similarity,
                    compute_pillar_benchmarks,
                    compute_pillar_confidence,
                )
                _col_sim, _col_bench = st.columns(2)

                with _col_sim:
                    st.markdown("**Tactically Similar Managers**")
                    _similar = compute_manager_similarity(mgr_name, _tr_dir)
                    if _similar:
                        st.dataframe(pd.DataFrame(_similar)[
                            ["manager", "team", "similarity_pct", "archetype"]
                        ].rename(columns={"similarity_pct": "Similarity %",
                                          "manager": "Manager", "team": "Club",
                                          "archetype": "Archetype"}),
                        hide_index=True, use_container_width=True)
                    else:
                        st.caption("No similar managers found in training data.")

                with _col_bench:
                    st.markdown("**Pillar vs Archetype**")
                    _bench = compute_pillar_benchmarks(
                        mgr_name, _cluster, _tr_dir, dna)
                    if _bench:
                        st.dataframe(pd.DataFrame(_bench)[
                            ["display_name", "score", "archetype_mean", "delta", "flag"]
                        ].rename(columns={"display_name": "Pillar",
                                          "score": "Score",
                                          "archetype_mean": "Avg",
                                          "delta": "Δ",
                                          "flag": "Signal"}),
                        hide_index=True, use_container_width=True)
                    else:
                        st.caption("Pillar benchmark data unavailable.")

                _conf = compute_pillar_confidence(r0, st.session_state.analysis_a)
                st.caption(f"{_conf['badge']}  {_conf['confidence_note']}")

            except Exception as _de:
                st.caption(f"DNA insights unavailable: {_de}")
        else:
            st.caption("DNA dimensions not available for this analysis.")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB: SQUAD
    # ─────────────────────────────────────────────────────────────────────────
    with tab_squad:
        render_metrics(r0, color_class="gradient")
        st.write("")
        render_squad_detail_table(
            st.session_state.squad_a,
            ideal_xi=r0.get("ideal_xi", []),
            key_suffix="_single",
        )
        _rec = r0.get("recruitment", []) or st.session_state.analysis_a.get("recruitment", [])
        if _rec:
            st.write("")
            st.markdown("**Recruitment Priorities**")
            st.dataframe(recruitment_display_df(_rec), use_container_width=True, hide_index=True)
            recruitment_confidence_caption(_rec)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB: DASHBOARD
    # ─────────────────────────────────────────────────────────────────────────
    with tab_dashboard:
        dashboards = st.session_state.dashboards_a
        if dashboards:
            if len(dashboards) > 1:
                selected = st.selectbox("Dashboard", list(dashboards.keys()))
            else:
                selected = list(dashboards.keys())[0]
            st.download_button("⬇️  Download Dashboard",
                               data=dashboards[selected],
                               file_name=f"{selected}.html",
                               mime="text/html", key="dl_dashboard")
            st.components.v1.html(dashboards[selected], height=920, scrolling=True)
        else:
            st.caption("No dashboard generated. Enable 'Generate HTML dashboard' "
                       "in Advanced Options and re-run.")


# ══════════════════════════════════════════════════════════════
# DISPLAY — COMPARISON MODE
# ══════════════════════════════════════════════════════════════

elif results_b is not None:
    r_a = results_a[0]
    r_b = results_b[0]

    mgr_a  = r_a.get("manager", "Unknown")
    mgr_b  = r_b.get("manager", "Unknown")
    club_a = r_a.get("club", "Unknown")
    club_b = r_b.get("club", "Unknown")
    fit_a  = r_a.get("average_fit", 0)
    fit_b  = r_b.get("average_fit", 0)
    counts_a = r_a.get("classification_counts", {})
    counts_b = r_b.get("classification_counts", {})

    _fallback_a = r_a.get("manager_dna_source") == "fallback_club_profile"
    _fallback_b = r_b.get("manager_dna_source") == "fallback_club_profile"
    if _fallback_a or _fallback_b:
        _names = ", ".join(n for n, f in [(mgr_a, _fallback_a), (mgr_b, _fallback_b)] if f)
        st.warning(
            f"⚠️ No tactical data found for **{_names}** in the training pool "
            f"(not managing a licensed-league club in the selected season). "
            f"That scenario's scores reflect the club's own current profile, "
            f"not the manager's actual tactics — treat it as a placeholder.",
            icon="⚠️",
        )

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        f"<h2 style='margin-bottom:2px;'>Comparison"
        f"<span style='color:#475569; font-weight:400; font-size:.7em;'>"
        f"  A: {mgr_a} → {club_a} &nbsp;·&nbsp; B: {mgr_b} → {club_b}"
        f"</span></h2>",
        unsafe_allow_html=True)

    # Insight line
    fit_diff   = fit_a - fit_b
    diff_sign  = "+" if fit_diff > 0 else ""
    diff_css   = ("diff-positive" if fit_diff > 0
                  else "diff-negative" if fit_diff < 0 else "diff-neutral")
    enabler_diff = counts_a.get("Key Enabler", 0) - counts_b.get("Key Enabler", 0)
    margin_diff  = counts_a.get("Potentially Marginalised", 0) - counts_b.get("Potentially Marginalised", 0)

    _insights = []
    if enabler_diff > 0:
        _insights.append(f"**{mgr_a}** unlocks **{enabler_diff} more Key "
                         f"Enabler{'s' if enabler_diff != 1 else ''}** than {mgr_b}")
    elif enabler_diff < 0:
        _insights.append(f"**{mgr_b}** unlocks **{abs(enabler_diff)} more Key "
                         f"Enabler{'s' if abs(enabler_diff) != 1 else ''}** than {mgr_a}")
    if margin_diff < 0:
        _insights.append(f"**{mgr_a}** marginalises **{abs(margin_diff)} fewer** "
                         f"player{'s' if abs(margin_diff) != 1 else ''}")
    elif margin_diff > 0:
        _insights.append(f"**{mgr_b}** marginalises **{abs(margin_diff)} fewer** "
                         f"player{'s' if abs(margin_diff) != 1 else ''}")
    if _insights:
        st.markdown(
            f"<div style='color:#64748b; font-size:.88rem; margin-bottom:1rem;'>"
            + " &nbsp;·&nbsp; ".join(_insights) + "</div>",
            unsafe_allow_html=True)

    # ── Delta summary row ─────────────────────────────────────────────────────
    dc = st.columns(4)
    with dc[0]:
        st.markdown(metric_card(f"{fit_a:.1f}", f"A: {mgr_a}", "gradient"),
                    unsafe_allow_html=True)
    with dc[1]:
        st.markdown(metric_card(f"{fit_b:.1f}", f"B: {mgr_b}", "orange"),
                    unsafe_allow_html=True)
    with dc[2]:
        st.markdown(metric_card(f"{diff_sign}{fit_diff:.1f}", "Fit Δ (A − B)", diff_css),
                    unsafe_allow_html=True)
    with dc[3]:
        winner = mgr_a if fit_a >= fit_b else mgr_b
        st.markdown(metric_card(winner, "Higher Fit",
                                "gradient" if winner == mgr_a else "orange"),
                    unsafe_allow_html=True)

    st.write("")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    ctab_report, ctab_formation, ctab_dna, ctab_squad, ctab_dashboard = st.tabs([
        "📋 Reports", "⚽ Formation", "🧬 DNA Profile", "👥 Squad", "📊 Dashboard",
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB: REPORTS
    # ─────────────────────────────────────────────────────────────────────────
    with ctab_report:
        col_ra, col_rb = st.columns(2)

        def _render_report_col(col, result, analysis_snap, squad_snap, label, color):
            mgr  = result.get("manager", "")
            club = result.get("club", "")
            fmt  = result.get("primary_formation", "")
            key  = label.replace(" ", "_").lower()
            with col:
                st.markdown(
                    f'<span class="scenario-header scenario-{color}">'
                    f'{label}: {mgr}</span>',
                    unsafe_allow_html=True)
                sec = st.session_state.get(f"report_{key}")
                if sec is None:
                    try:
                        from aegis.ai_reporter import generate_narrative_report
                        with st.spinner("Generating…"):
                            sec = generate_narrative_report(
                                result=result, analysis=analysis_snap,
                                squad=squad_snap,
                                output_dir=Path(base_dir) / "outputs")
                        st.session_state[f"report_{key}"] = sec
                    except Exception as _re:
                        st.caption(f"Report unavailable: {_re}")
                        return
                if sec:
                    for _t, _body in sec.items():
                        st.markdown(f"**{_t}**")
                        st.markdown(_body)
                        st.write("")
                    st.divider()
                    try:
                        from aegis.ai_reporter import export_pdf
                        _pdf = export_pdf(sec, mgr, club, fmt)
                        st.download_button(f"⬇️  PDF — {mgr}", data=_pdf,
                            file_name=f"MTFI_{mgr.replace(' ','_')}.pdf",
                            mime="application/pdf", key=f"dl_pdf_{key}")
                    except ImportError:
                        pass

        # Pre-populate from single-mode auto-generated report if available
        if st.session_state.get("report_sections") and not st.session_state.get("report_a"):
            st.session_state["report_a"] = st.session_state["report_sections"]

        _render_report_col(col_ra, r_a, st.session_state.analysis_a,
                           st.session_state.squad_a, "a", "a")
        _render_report_col(col_rb, r_b, st.session_state.analysis_b,
                           st.session_state.squad_b, "b", "b")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB: FORMATION
    # ─────────────────────────────────────────────────────────────────────────
    with ctab_formation:
        _fa_col, _fb_col = st.columns(2)

        try:
            from aegis.dna_insights import compute_formation_history as _cfh_cmp
        except ImportError:
            _cfh_cmp = None

        # _cmp_cached_hist defined at module level

        def _render_formation_col(col, result, analysis_snap, label, color):
            with col:
                mgr  = result.get("manager", "")
                club = result.get("club", "")
                dual = result.get("dual_ideal_xi")
                m_tm = result.get("manager_prev_team", "") or ""

                st.markdown(
                    f'<span class="scenario-header scenario-{color}">'                    f'{label}: {mgr} → {club}</span>',
                    unsafe_allow_html=True)

                # Formation history charts — source of truth
                _ch1, _ch2 = st.columns(2)
                _c_hist = _m_hist_cmp = None
                _comp_id = all_teams_map.get(club, league_id_a)

                def _mini_bar(ax_m, fg_m, freq, pct, color_hex):
                    fmts = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)
                    vals = [pct[f] for f in fmts]
                    fg_m.patch.set_facecolor("#0a0e17")
                    ax_m.set_facecolor("#0f1520")
                    ax_m.barh(fmts, vals, color=color_hex, alpha=0.85)
                    ax_m.set_xlabel("% matches", color="#94a3b8", fontsize=6)
                    ax_m.tick_params(colors="#94a3b8", labelsize=6)
                    ax_m.spines[["top","right","bottom","left"]].set_visible(False)
                    plt.tight_layout(pad=0.3)

                with _ch1:
                    st.markdown(f"**🏟 {club}**")
                    if has_creds:
                        _c_hist = _cached_formation_history(club, _comp_id, season_id, sb_user, sb_pass)
                        if _c_hist:
                            _fg, _ax = plt.subplots(figsize=(3, max(1.2, len(_c_hist["frequency"])*0.5)))
                            _mini_bar(_ax, _fg, _c_hist["frequency"], _c_hist["frequency_pct"], "#38bdf8")
                            st.pyplot(_fg, use_container_width=True)
                            plt.close(_fg)

                if not m_tm:
                    try:
                        from aegis.dna_insights import get_manager_previous_team
                        from aegis.pretrain import MASTER_DIR
                        m_tm = get_manager_previous_team(mgr, MASTER_DIR) or ""
                    except Exception:
                        m_tm = ""

                with _ch2:
                    if m_tm and has_creds:
                        st.markdown(f"**👔 {mgr} at {m_tm}**")
                        _m_comp = all_teams_map.get(m_tm, league_id_a)
                        _m_hist_cmp = _cached_formation_history(m_tm, _m_comp, season_id, sb_user, sb_pass)
                        if _m_hist_cmp:
                            _fg2, _ax2 = plt.subplots(figsize=(3, max(1.2, len(_m_hist_cmp["frequency"])*0.5)))
                            _mini_bar(_ax2, _fg2, _m_hist_cmp["frequency"], _m_hist_cmp["frequency_pct"], "#fb923c")
                            st.pyplot(_fg2, use_container_width=True)
                            plt.close(_fg2)
                    else:
                        st.caption("Previous club unknown.")

                # Text from chart data
                _chart_c = (_c_hist or {}).get("primary")
                _chart_m = (_m_hist_cmp or {}).get("primary")
                _c_pct   = (_c_hist or {}).get("primary_pct", 0)
                _m_pct   = (_m_hist_cmp or {}).get("primary_pct", 0)
                if _chart_c or _chart_m:
                    _cd = f"**{_chart_c}**" if _chart_c else "unknown"
                    _md = f"**{_chart_m}**" if _chart_m else "unknown"
                    st.caption(f"{club}: {_cd}{(f' ({_c_pct:.0f}%)'if _c_pct else '')}  ·  {mgr}: {_md}{(f' ({_m_pct:.0f}% at {m_tm})'if _m_pct and m_tm else '')}")
                st.write("")

                # Pitch diagrams
                _eff_c = _chart_c or result.get("primary_formation")
                _eff_m = _chart_m or result.get("manager_formation")
                if dual:
                    xi_c = dual.get("ideal_xi_club", [])
                    xi_m = dual.get("ideal_xi_manager", [])
                    if xi_c or xi_m:
                        _dc = dual.get("club_formation") or _eff_c or "4-3-3"
                        _dm = dual.get("manager_formation") or _eff_m or "4-3-3"
                        fig_d = render_dual_formation_pitch(xi_c, xi_m, _dc, _dm,
                                                            club_label=club, mgr_label=mgr)
                        st.pyplot(fig_d, use_container_width=True)
                        plt.close(fig_d)
                elif _eff_c:
                    xi = result.get("ideal_xi", [])
                    if xi:
                        fig_s = render_formation_pitch(xi, _eff_c, title=f"{club} ({_eff_c})")
                        st.pyplot(fig_s, use_container_width=True)
                        plt.close(fig_s)
        _render_formation_col(_fa_col, r_a, st.session_state.analysis_a, "A", "a")
        _render_formation_col(_fb_col, r_b, st.session_state.analysis_b, "B", "b")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB: DNA PROFILE
    # ─────────────────────────────────────────────────────────────────────────
    with ctab_dna:
        dna_a = st.session_state.analysis_a.get("dna_dimensions", {})
        dna_b = st.session_state.analysis_b.get("dna_dimensions", {})

        if dna_a or dna_b:
            rc1, rc2, rc3 = st.columns([1, 2, 1])
            with rc2:
                fig_r = render_radar(dna_a or dna_b, label_a=mgr_a,
                                     dna_b=dna_b if dna_a else None, label_b=mgr_b)
                if fig_r:
                    st.pyplot(fig_r, use_container_width=True)
                    plt.close(fig_r)

            if dna_a and dna_b:
                st.write("")
                st.markdown("**Pillar comparison**")
                pillar_rows = []
                for pillar in dna_a:
                    va = dna_a[pillar]
                    vb = dna_b.get(pillar, 50)
                    d  = va - vb
                    pillar_rows.append({
                        "Pillar":          pillar,
                        f"{mgr_a} (A)":    round(va, 1),
                        f"{mgr_b} (B)":    round(vb, 1),
                        "Δ (A − B)":       round(d, 1),
                    })
                st.dataframe(pd.DataFrame(pillar_rows),
                             use_container_width=True, hide_index=True)
        else:
            st.caption("DNA dimensions unavailable.")

        # Similarity and pillar benchmarks for each scenario
        if dna_a or dna_b:
            st.divider()
            try:
                from aegis.pretrain import MASTER_DIR as _cmp_tr_dir
                from aegis.dna_insights import compute_manager_similarity, compute_pillar_benchmarks
                _sim_a_col, _sim_b_col = st.columns(2)
                for _col, _mgr, _dna, _res, _snap in [
                    (_sim_a_col, mgr_a, dna_a, r_a, st.session_state.analysis_a),
                    (_sim_b_col, mgr_b, dna_b, r_b, st.session_state.analysis_b),
                ]:
                    with _col:
                        st.markdown(f"**Similar to {_mgr}**")
                        _sim = compute_manager_similarity(_mgr, _cmp_tr_dir)
                        if _sim:
                            st.dataframe(
                                pd.DataFrame(_sim)[["manager","team","similarity_pct"]].rename(
                                    columns={"manager":"Manager","team":"Club","similarity_pct":"Sim %"}),
                                hide_index=True, use_container_width=True)
                        else:
                            st.caption("No matches found.")
                        _cl = _res.get("cluster", 0)
                        _bench = compute_pillar_benchmarks(_mgr, _cl, _cmp_tr_dir, _dna or {})
                        if _bench:
                            st.markdown(f"**Pillar vs Archetype — {_mgr}**")
                            st.dataframe(
                                pd.DataFrame(_bench)[["display_name","score","archetype_mean","delta","flag"]].rename(
                                    columns={"display_name":"Pillar","archetype_mean":"Avg","flag":"Signal","delta":"Δ","score":"Score"}),
                                hide_index=True, use_container_width=True)
            except Exception as _cde:
                st.caption(f"DNA insights unavailable: {_cde}")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB: SQUAD
    # ─────────────────────────────────────────────────────────────────────────
    with ctab_squad:
        squad_a_data = st.session_state.squad_a
        squad_b_data = st.session_state.squad_b

        CLS_EMOJI_CMP = {"Key Enabler":"🟢","Good Fit":"🟡",
                         "System Dependent":"🟠","Potentially Marginalised":"🔴"}

        def _squad_to_df(data):
            p = data.get("players", data) if isinstance(data, dict) else data
            return pd.DataFrame(p) if isinstance(p, list) and p else pd.DataFrame()

        if club_a == club_b and squad_a_data and squad_b_data:
            st.markdown(f"**{club_a}** — how every player fares under each manager")

            df_a = _squad_to_df(squad_a_data)
            df_b = _squad_to_df(squad_b_data)

            if "player" in df_a.columns and "player" in df_b.columns:
                # Add emoji classification to both
                for _df in [df_a, df_b]:
                    if "classification" in _df.columns:
                        _df["cls_emoji"] = _df["classification"].map(
                            lambda c: CLS_EMOJI_CMP.get(c, "⚪"))

                merged = pd.merge(
                    df_a[["player","position","fit_score","classification","cls_emoji"]],
                    df_b[["player","fit_score","classification","cls_emoji"]],
                    on="player", how="outer",
                    suffixes=(f"_a", f"_b"))

                merged["Fit Δ"] = (merged["fit_score_a"].fillna(0)
                                   - merged["fit_score_b"].fillna(0)).round(1)
                merged = merged.sort_values("Fit Δ", ascending=False)

                display = merged.rename(columns={
                    "player": "Player", "position": "Position",
                    "fit_score_a": f"Fit ({mgr_a})",
                    "fit_score_b": f"Fit ({mgr_b})",
                    "cls_emoji_a": f"Status ({mgr_a})",
                    "cls_emoji_b": f"Status ({mgr_b})",
                    "classification_a": f"Class. ({mgr_a})",
                    "classification_b": f"Class. ({mgr_b})",
                })
                _show_cols = ["Player","Position",
                              f"Fit ({mgr_a})",f"Status ({mgr_a})",
                              f"Fit ({mgr_b})",f"Status ({mgr_b})",
                              "Fit Δ"]
                _show_cols = [c for c in _show_cols if c in display.columns]
                st.dataframe(
                    display[_show_cols],
                    use_container_width=True, hide_index=True,
                    height=min(len(display)*38+40, 700),
                    column_config={
                        f"Fit ({mgr_a})": st.column_config.NumberColumn(format="%.1f"),
                        f"Fit ({mgr_b})": st.column_config.NumberColumn(format="%.1f"),
                        "Fit Δ": st.column_config.NumberColumn(format="%.1f"),
                    })

                # Summary callouts
                winners = merged[merged["Fit Δ"] > 2]
                losers  = merged[merged["Fit Δ"] < -2]
                wl1, wl2 = st.columns(2)
                with wl1:
                    st.markdown(f"<span style='color:#34d399;font-weight:700;'>"
                                f"▲ {len(winners)} players fit better under {mgr_a}"
                                f"</span>", unsafe_allow_html=True)
                    for _, row in winners.head(5).iterrows():
                        st.caption(f"  {row['Player']}: +{row['Fit Δ']:.1f}")
                with wl2:
                    st.markdown(f"<span style='color:#f87171;font-weight:700;'>"
                                f"▼ {len(losers)} players fit better under {mgr_b}"
                                f"</span>", unsafe_allow_html=True)
                    for _, row in losers.tail(5).iterrows():
                        st.caption(f"  {row['Player']}: {row['Fit Δ']:.1f}")
        else:
            # Different clubs — full detail table for each
            sq_a, sq_b = st.columns(2)
            with sq_a:
                st.markdown(f'<span class="scenario-header scenario-a">A: {mgr_a} → {club_a}</span>',
                            unsafe_allow_html=True)
                render_squad_detail_table(
                    squad_a_data, ideal_xi=r_a.get("ideal_xi",[]), key_suffix="_cmp_a")
            with sq_b:
                st.markdown(f'<span class="scenario-header scenario-b">B: {mgr_b} → {club_b}</span>',
                            unsafe_allow_html=True)
                render_squad_detail_table(
                    squad_b_data, ideal_xi=r_b.get("ideal_xi",[]), key_suffix="_cmp_b")

        # Recruitment side by side
        rec_a = st.session_state.analysis_a.get("recruitment", [])
        rec_b = st.session_state.analysis_b.get("recruitment", [])
        if rec_a or rec_b:
            st.write("")
            st.markdown("**Recruitment Priorities**")
            rc_a, rc_b = st.columns(2)
            with rc_a:
                st.markdown(f'<span class="scenario-header scenario-a">A: {mgr_a}</span>',
                            unsafe_allow_html=True)
                if rec_a:
                    st.dataframe(recruitment_display_df(rec_a), use_container_width=True, hide_index=True)
                    recruitment_confidence_caption(rec_a)
                else:
                    st.caption("None identified.")
            with rc_b:
                st.markdown(f'<span class="scenario-header scenario-b">B: {mgr_b}</span>',
                            unsafe_allow_html=True)
                if rec_b:
                    st.dataframe(recruitment_display_df(rec_b), use_container_width=True, hide_index=True)
                    recruitment_confidence_caption(rec_b)
                else:
                    st.caption("None identified.")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB: DASHBOARD
    # ─────────────────────────────────────────────────────────────────────────
    with ctab_dashboard:
        all_dashboards = {
            **{f"A: {k}": v for k, v in st.session_state.dashboards_a.items()},
            **{f"B: {k}": v for k, v in st.session_state.dashboards_b.items()},
        }
        if all_dashboards:
            _sel = st.selectbox("Select dashboard", list(all_dashboards.keys()),
                                key="cmp_dash_sel")
            st.download_button("⬇️  Download HTML",
                               data=all_dashboards[_sel],
                               file_name=f"{_sel}.html", mime="text/html",
                               key="cmp_dl_dash")
            st.components.v1.html(all_dashboards[_sel], height=920, scrolling=True)
        else:
            st.caption("No dashboards generated. Enable 'Generate HTML dashboard' "
                       "in Advanced Options and re-run.")


# ══════════════════════════════════════════════════════════════
# DISPLAY — SHORTLIST MODE
# ══════════════════════════════════════════════════════════════

elif st.session_state.get("shortlist"):
    ranked     = st.session_state.shortlist
    _sl_club   = st.session_state.get("shortlist_club", "")

    st.markdown(f"## Manager Shortlist — {_sl_club}")

    _n_fallback = sum(1 for e in ranked if e.dna_source == "fallback_club_profile")
    if _n_fallback:
        st.warning(
            f"⚠️ {_n_fallback} of {len(ranked)} manager(s) below have no tactical "
            f"data in the training pool (they weren't managing a licensed-league "
            f"club in the selected season). Their score reflects **{_sl_club}'s own "
            f"current profile**, not their actual tactics — flagged with ⚠️ in the "
            f"table below. Treat those rows as placeholders, not real fit scores.",
            icon="⚠️",
        )

    # ── Summary table ──────────────────────────────────────────────────────
    _rows = []
    for e in ranked:
        _lo = sum(float(r.get("cost_low",  0)) for r in (e.recruitment or []))
        _hi = sum(float(r.get("cost_high", 0)) for r in (e.recruitment or []))
        _is_fallback = e.dna_source == "fallback_club_profile"
        _rows.append({
            "Rank":               e.rank,
            "Manager":            f"⚠️ {e.manager}" if _is_fallback else e.manager,
            "Data":               "⚠️ No match — using club profile" if _is_fallback else "✓ Matched",
            "Formation":          e.primary_formation,
            "Archetype":          e.archetype,
            "Avg Fit":            f"{e.average_fit:.1f}",
            "Key Enablers":       e.key_enablers,
            "Marginalised":       e.potentially_marginalised,
            "Recruit. Est. (£M)": f"{_lo:.0f}–{_hi:.0f}" if _lo > 0 else "—",
        })
    st.dataframe(pd.DataFrame(_rows), hide_index=True, use_container_width=True)

    st.write("")

    # ── Overlaid DNA radar ─────────────────────────────────────────────────
    if any(e.dna_dimensions for e in ranked):
        st.markdown("### 🧬 DNA Overlay")
        _fig_sl = render_shortlist_radar(ranked)
        if _fig_sl:
            _sc1, _sc2, _sc3 = st.columns([1, 2, 1])
            with _sc2:
                st.pyplot(_fig_sl)
            plt.close(_fig_sl)

    st.write("")

    # ── Per-manager expanders ──────────────────────────────────────────────
    st.markdown("### Manager Detail")
    for entry in ranked:
        with st.expander(
            f"#{entry.rank}  {entry.manager}  —  Avg Fit: {entry.average_fit:.1f}",
            expanded=False,
        ):
            render_metrics(entry.run_result, color_class="gradient")

            if st.button(
                f"Generate Report",
                key=f"report_sl_{entry.rank}",
                type="secondary",
            ):
                from aegis.ai_reporter import generate_narrative_report
                with st.spinner("Generating…"):
                    _sl_sections = generate_narrative_report(
                        result     = entry.run_result,
                        analysis   = {"dna_dimensions": entry.dna_dimensions,
                                      "primary_formation": entry.primary_formation},
                        squad      = None,
                        output_dir = Path(base_dir) / "outputs",
                    )
                st.session_state[f"sl_report_{entry.rank}"] = _sl_sections

            _rkey = f"sl_report_{entry.rank}"
            if st.session_state.get(_rkey):
                _sl_sec = st.session_state[_rkey]
                for _t, _c in _sl_sec.items():
                    with st.expander(_t, expanded=True):
                        st.markdown(_c)
                try:
                    from aegis.ai_reporter import export_pdf
                    _sl_pdf = export_pdf(
                        _sl_sec, entry.manager, _sl_club, entry.primary_formation)
                    st.download_button(
                        f"⬇️  Download PDF",
                        data=_sl_pdf,
                        file_name=f"MTFI_{entry.manager.replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        key=f"pdf_sl_{entry.rank}",
                    )
                except ImportError:
                    pass
