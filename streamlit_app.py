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

    # Snapshot squad_fit_summary.json
    squad_path = output_dir / "squad_fit_summary.json"
    squad_snapshot = None
    if squad_path.exists():
        with open(squad_path) as f:
            squad_snapshot = json.load(f)

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

    return result, dashboards, analysis_snapshot, squad_snapshot


# ══════════════════════════════════════════════════════════════
# RADAR CHART
# ══════════════════════════════════════════════════════════════

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
    """Render squad fit table from snapshot data."""
    if squad_data is None:
        return None
    players = (squad_data.get("players", squad_data)
               if isinstance(squad_data, dict) else squad_data)
    if not isinstance(players, list) or not players:
        return None
    df = pd.DataFrame(players)
    display_cols = [c for c in ["player", "position", "fit_score",
                                 "classification", "strengths", "risks"]
                   if c in df.columns]
    if not display_cols:
        return None
    with st.expander(f"📋 Squad Fit Details ({len(df)} players)",
                     expanded=True):
        st.dataframe(
            df[display_cols].sort_values("fit_score", ascending=False),
            use_container_width=True, hide_index=True,
            height=min(len(df) * 38 + 40, 600),
        )
    return df


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
    ("dossier_player_list", []),
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


def _get_all_managers(sb_user, sb_pass, league_ids: list, season_id: int,
                     base_dir: str = "") -> list:
    """Get a de-duplicated sorted list of managers across multiple leagues."""
    all_mgrs = set()
    for lid in league_ids:
        try:
            _, mgrs, _ = discover_teams_and_managers(
                sb_user, sb_pass, (lid,), season_id, base_dir,
            )
            all_mgrs.update(mgrs)
        except Exception:
            continue
    return sorted(all_mgrs)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚽ Aegis MTFI")
    st.caption("Manager Tactical Fit Intelligence")
    st.divider()

    # ── Mode toggle ──
    mode = st.radio("Mode", ["Single Analysis", "Compare Scenarios", "Player Dossier"],
                    horizontal=False, label_visibility="collapsed")
    is_compare = mode == "Compare Scenarios"
    is_dossier = mode == "Player Dossier"

    st.divider()

    # ── Shared: Season + Advanced ──
    season_label = st.selectbox("Season", list(SEASON_OPTIONS.keys()))
    season_id = SEASON_OPTIONS[season_label]

    with st.expander("⚙️ Advanced Options"):
        train_model = st.checkbox("Train model", value=True)
        training_leagues = st.multiselect(
            "Training leagues (manager pool)",
            list(COMPETITION_OPTIONS.keys()),
            default=["Premier League"],
            help="The model clusters managers from these leagues. "
                 "More leagues = richer context for archetypes.",
        )
        max_matches = st.slider("Max matches per team", 10, 50, 50)
        visualize = st.checkbox("Generate HTML dashboard", value=True)

    training_league_ids = [COMPETITION_OPTIONS[l] for l in training_leagues]

    st.divider()

    # ── Discover managers across all training leagues (for coach dropdown) ──
    all_coach_options = []
    if has_creds:
        try:
            all_coach_options = _get_all_managers(
                sb_user, sb_pass, training_league_ids, season_id, base_dir)
        except Exception:
            all_coach_options = []

    # ── Helper: render one scenario's inputs ──
    def _scenario_inputs(label: str, key_prefix: str, css_class: str):
        """
        Render league → team → coach dropdowns for one scenario.
        Returns (league_id, team_name, coach_name).
        """
        st.markdown(
            f'<span class="scenario-header {css_class}">{label}</span>',
            unsafe_allow_html=True)

        s_league = st.selectbox(
            "League", list(COMPETITION_OPTIONS.keys()),
            key=f"{key_prefix}_league",
            help="The league this team plays in.")
        s_league_id = COMPETITION_OPTIONS[s_league]

        # Discover teams for the selected league
        team_options = []
        team_mgr_map = {}
        if has_creds:
            try:
                teams, _, team_mgr_map = discover_teams_and_managers(
                    sb_user, sb_pass, (s_league_id,), season_id, base_dir)
                team_options = teams
            except Exception as exc:
                st.caption(f"⚠ Could not load teams: {exc}")

        if team_options:
            s_team = st.selectbox("Team", team_options,
                                  key=f"{key_prefix}_team")
            current_mgr = team_mgr_map.get(s_team, "Unknown")
        else:
            s_team = st.text_input("Team", key=f"{key_prefix}_team",
                                   placeholder="Enter team name")
            current_mgr = None

        # Coach dropdown: current manager + all known managers
        coach_choices = [CURRENT_MANAGER_SENTINEL] + all_coach_options
        if current_mgr and current_mgr != "Unknown":
            coach_label_default = (
                f"{CURRENT_MANAGER_SENTINEL}  ({current_mgr})")
            coach_choices[0] = coach_label_default

        s_coach_selection = st.selectbox(
            "Coach", coach_choices,
            key=f"{key_prefix}_coach",
            help="Pick 'Current manager' to analyse the incumbent, "
                 "or select any manager for a hypothetical scenario.")

        # Resolve selection back to a name or None
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

        # Load player list for selected league/season
        player_list = st.session_state.dossier_player_list
        if has_creds and st.button("🔍  Load Players", key="d_load",
                                   use_container_width=True):
            with st.spinner("Loading player list…"):
                try:
                    os.environ["SB_USERNAME"] = sb_user
                    os.environ["SB_PASSWORD"] = sb_pass
                    from aegis import StatsBombClient
                    from aegis.player_dossier import PlayerDossierGenerator, MIN_MINUTES
                    sb = StatsBombClient()
                    _stats = sb.get_player_season_stats(d_league_id, season_id)
                    gen = PlayerDossierGenerator(_stats)
                    player_list = gen.list_players(min_minutes=MIN_MINUTES)
                    st.session_state.dossier_player_list = player_list
                    st.success(f"✓ {len(player_list)} players loaded")
                except Exception as exc:
                    st.error(f"Could not load players: {exc}")

        if player_list:
            d_player = st.selectbox("Player", player_list, key="d_player_select")
        else:
            d_player = st.text_input("Player name", key="d_player_text",
                                     placeholder="e.g. Tae-Seok Lee")

        # Optional manual overrides
        with st.expander("📝  Manual Overrides (optional)"):
            d_height   = st.text_input("Height (e.g. 1.74 m)", key="d_height")
            d_foot     = st.selectbox("Strong foot", ["", "Left", "Right", "Both"], key="d_foot")
            d_tmv      = st.text_input("TMV (e.g. €1.5M)", key="d_tmv")
            d_contract = st.text_input("Contract expiry (e.g. Jun 2029)", key="d_contract")
            d_nat      = st.text_input("Nationality", key="d_nat")
            d_pos_raw  = st.text_input("Positions (comma-separated, e.g. LB,LWB,LM)",
                                       key="d_pos")

        dossier_clicked = st.button("⚽  Generate Dossier", use_container_width=True,
                                    type="primary", key="d_run")

    elif is_compare:
        league_id_a, team_a, coach_a = _scenario_inputs(
            "Scenario A", "a", "scenario-a")
        st.write("")
        league_id_b, team_b, coach_b = _scenario_inputs(
            "Scenario B", "b", "scenario-b")
    else:
        league_id_a, team_a, coach_a = _scenario_inputs(
            "Analysis", "single", "scenario-a")
        league_id_b = team_b = coach_b = None

    if not has_creds:
        st.caption("ℹ️ No data connection — check StatsBomb credentials in secrets.")

    st.divider()

    if not is_dossier:
        run_clicked = st.button("🚀  Run Analysis", use_container_width=True,
                                type="primary")
    else:
        run_clicked = False

    if not is_dossier and st.button("🗑️  Clear Cache", use_container_width=True):
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

                # Build overrides from sidebar inputs
                _overrides = {}
                _d_h = st.session_state.get("d_height", "")
                _d_f = st.session_state.get("d_foot", "")
                _d_t = st.session_state.get("d_tmv", "")
                _d_c = st.session_state.get("d_contract", "")
                _d_n = st.session_state.get("d_nat", "")
                _d_p = st.session_state.get("d_pos", "")
                if _d_h: _overrides["height"] = _d_h
                if _d_f: _overrides["strong_foot"] = _d_f
                if _d_t: _overrides["tmv"] = _d_t
                if _d_c: _overrides["contract_exp"] = _d_c
                if _d_n: _overrides["nationality"] = _d_n
                if _d_p:
                    _overrides["positions"] = [p.strip() for p in _d_p.split(",") if p.strip()]

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
        st.components.v1.html(dossier_html, height=1400, scrolling=True)
    elif not _dossier_trigger:
        st.info("Select a league and season, load players, then choose a player and press **Generate Dossier**.")

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

if results_a is None:
    st.info("Configure the analysis in the sidebar and press **Run Analysis**.")
    st.stop()


if results_b is None:
    # Single mode — the interactive dashboard has everything
    for idx, r in enumerate(results_a):
        manager = r.get("manager", "Unknown")
        club = r.get("club", "Unknown")

        if len(results_a) > 1:
            st.divider()
            st.markdown(f"### {manager} → {club}")

    dashboards = st.session_state.dashboards_a
    if dashboards:
        if len(dashboards) > 1:
            selected = st.selectbox("Dashboard", list(dashboards.keys()))
        else:
            selected = list(dashboards.keys())[0]
        st.components.v1.html(dashboards[selected], height=900,
                              scrolling=True)
        st.download_button("⬇️  Download Dashboard",
                           data=dashboards[selected],
                           file_name=f"{selected}.html", mime="text/html")
    else:
        st.warning("Analysis complete but no dashboard was generated. "
                   "Enable 'Generate HTML dashboard' in Advanced Options.")


# ══════════════════════════════════════════════════════════════
# DISPLAY — COMPARISON MODE
# ══════════════════════════════════════════════════════════════

elif results_b is not None:
    r_a = results_a[0]
    r_b = results_b[0]

    mgr_a = r_a.get("manager", "Unknown")
    mgr_b = r_b.get("manager", "Unknown")
    club_a = r_a.get("club", "Unknown")
    club_b = r_b.get("club", "Unknown")
    fit_a = r_a.get("average_fit", 0)
    fit_b = r_b.get("average_fit", 0)
    counts_a = r_a.get("classification_counts", {})
    counts_b = r_b.get("classification_counts", {})

    # ── Headline ──
    st.markdown("## Side-by-Side Comparison")

    fit_diff = fit_a - fit_b
    diff_sign = "+" if fit_diff > 0 else ""
    diff_css = ("diff-positive" if fit_diff > 0
                else "diff-negative" if fit_diff < 0
                else "diff-neutral")

    enabler_a = counts_a.get("Key Enabler", 0)
    enabler_b = counts_b.get("Key Enabler", 0)
    enabler_diff = enabler_a - enabler_b
    margin_a = counts_a.get("Potentially Marginalised", 0)
    margin_b = counts_b.get("Potentially Marginalised", 0)
    margin_diff = margin_a - margin_b

    insights = []
    if enabler_diff > 0:
        insights.append(
            f"**{mgr_a}** unlocks **{enabler_diff} more Key "
            f"Enabler{'s' if enabler_diff != 1 else ''}** than {mgr_b}")
    elif enabler_diff < 0:
        insights.append(
            f"**{mgr_b}** unlocks **{abs(enabler_diff)} more Key "
            f"Enabler{'s' if abs(enabler_diff) != 1 else ''}** than {mgr_a}")
    if margin_diff > 0:
        insights.append(
            f"**{mgr_b}** marginalises **{abs(margin_diff)} fewer** "
            f"player{'s' if abs(margin_diff) != 1 else ''}")
    elif margin_diff < 0:
        insights.append(
            f"**{mgr_a}** marginalises **{abs(margin_diff)} fewer** "
            f"player{'s' if abs(margin_diff) != 1 else ''}")

    if insights:
        st.markdown(" · ".join(insights))

    st.write("")

    # ── Delta summary row ──
    dc = st.columns(4)
    with dc[0]:
        st.markdown(metric_card(f"{fit_a:.1f}", f"A: {mgr_a}", "gradient"),
                    unsafe_allow_html=True)
    with dc[1]:
        st.markdown(metric_card(f"{fit_b:.1f}", f"B: {mgr_b}", "orange"),
                    unsafe_allow_html=True)
    with dc[2]:
        st.markdown(metric_card(f"{diff_sign}{fit_diff:.1f}",
                                "Fit Δ (A − B)", diff_css),
                    unsafe_allow_html=True)
    with dc[3]:
        winner = mgr_a if fit_a >= fit_b else mgr_b
        st.markdown(metric_card(winner, "Higher Fit",
                                "gradient" if winner == mgr_a else "orange"),
                    unsafe_allow_html=True)

    st.write("")

    # ── Side-by-side metrics ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            f'<span class="scenario-header scenario-a">'
            f'A: {mgr_a} → {club_a}</span>',
            unsafe_allow_html=True)
        render_metrics(r_a, color_class="gradient")

    with col_b:
        st.markdown(
            f'<span class="scenario-header scenario-b">'
            f'B: {mgr_b} → {club_b}</span>',
            unsafe_allow_html=True)
        render_metrics(r_b, color_class="orange")

    st.write("")

    # ── Overlaid radar ──
    dna_a = st.session_state.analysis_a.get("dna_dimensions", {})
    dna_b = st.session_state.analysis_b.get("dna_dimensions", {})

    if dna_a or dna_b:
        st.markdown("### 🧬 DNA Radar — Overlay")
        fig = render_radar(
            dna_a or dna_b,
            label_a=mgr_a,
            dna_b=dna_b if dna_a else None,
            label_b=mgr_b,
        )
        if fig:
            rc1, rc2, rc3 = st.columns([1, 2, 1])
            with rc2:
                st.pyplot(fig)
            plt.close(fig)

            # Pillar-by-pillar diff table
            if dna_a and dna_b:
                pillar_rows = []
                for pillar in dna_a:
                    va = dna_a[pillar]
                    vb = dna_b.get(pillar, 50)
                    d = va - vb
                    pillar_rows.append({
                        "Pillar": pillar,
                        f"{mgr_a} (A)": round(va, 1),
                        f"{mgr_b} (B)": round(vb, 1),
                        "Δ (A − B)": round(d, 1),
                    })
                st.dataframe(pd.DataFrame(pillar_rows),
                             use_container_width=True, hide_index=True)

    st.write("")

    # ── Squad tables side by side ──
    squad_a_data = st.session_state.squad_a
    squad_b_data = st.session_state.squad_b

    if squad_a_data or squad_b_data:
        st.markdown("### 📋 Squad Fit Comparison")

        # If same club, build a merged diff table
        if club_a == club_b and squad_a_data and squad_b_data:
            st.caption(f"Same squad ({club_a}) — showing fit score change "
                       f"under each manager")

            def _to_df(data):
                players = (data.get("players", data)
                           if isinstance(data, dict) else data)
                if isinstance(players, list) and players:
                    return pd.DataFrame(players)
                return pd.DataFrame()

            df_a = _to_df(squad_a_data)
            df_b = _to_df(squad_b_data)

            if "player" in df_a.columns and "player" in df_b.columns:
                merged = pd.merge(
                    df_a[["player", "position", "fit_score", "classification"]],
                    df_b[["player", "fit_score", "classification"]],
                    on="player", how="outer",
                    suffixes=(f" ({mgr_a})", f" ({mgr_b})"),
                )

                # Compute delta
                col_fit_a = f"fit_score ({mgr_a})"
                col_fit_b = f"fit_score ({mgr_b})"
                if col_fit_a in merged.columns and col_fit_b in merged.columns:
                    merged["Fit Δ"] = (
                        merged[col_fit_a].fillna(0)
                        - merged[col_fit_b].fillna(0)
                    ).round(1)
                    merged = merged.sort_values("Fit Δ", ascending=False)

                st.dataframe(merged, use_container_width=True,
                             hide_index=True,
                             height=min(len(merged) * 38 + 40, 700))

                # Winners / losers summary
                if "Fit Δ" in merged.columns:
                    winners = merged[merged["Fit Δ"] > 2]
                    losers = merged[merged["Fit Δ"] < -2]

                    wl_cols = st.columns(2)
                    with wl_cols[0]:
                        st.markdown(
                            f"<span style='color:#34d399; font-weight:700;'>"
                            f"▲ {len(winners)} players gain ≥2 fit under "
                            f"{mgr_a}</span>",
                            unsafe_allow_html=True)
                        if not winners.empty:
                            top = winners.head(5)
                            for _, row in top.iterrows():
                                st.caption(
                                    f"  {row['player']}: "
                                    f"+{row['Fit Δ']:.1f}")

                    with wl_cols[1]:
                        st.markdown(
                            f"<span style='color:#f87171; font-weight:700;'>"
                            f"▼ {len(losers)} players lose ≥2 fit under "
                            f"{mgr_a}</span>",
                            unsafe_allow_html=True)
                        if not losers.empty:
                            bot = losers.tail(5)
                            for _, row in bot.iterrows():
                                st.caption(
                                    f"  {row['player']}: "
                                    f"{row['Fit Δ']:.1f}")
        else:
            # Different clubs — show tables side by side
            sq_a, sq_b = st.columns(2)
            with sq_a:
                st.markdown(
                    f'<span class="scenario-header scenario-a">A: {club_a}'
                    f'</span>', unsafe_allow_html=True)
                render_squad_table(squad_a_data, key_suffix="_cmp_a")
            with sq_b:
                st.markdown(
                    f'<span class="scenario-header scenario-b">B: {club_b}'
                    f'</span>', unsafe_allow_html=True)
                render_squad_table(squad_b_data, key_suffix="_cmp_b")

    # ── Recruitment comparison ──
    rec_a = st.session_state.analysis_a.get("recruitment", [])
    rec_b = st.session_state.analysis_b.get("recruitment", [])
    if rec_a or rec_b:
        st.markdown("### 🎯 Recruitment Priorities")
        rc_a, rc_b = st.columns(2)
        with rc_a:
            st.markdown(
                f'<span class="scenario-header scenario-a">A: {mgr_a}'
                f'</span>', unsafe_allow_html=True)
            if rec_a:
                st.dataframe(pd.DataFrame(rec_a),
                             use_container_width=True, hide_index=True)
            else:
                st.caption("No recruitment gaps identified.")
        with rc_b:
            st.markdown(
                f'<span class="scenario-header scenario-b">B: {mgr_b}'
                f'</span>', unsafe_allow_html=True)
            if rec_b:
                st.dataframe(pd.DataFrame(rec_b),
                             use_container_width=True, hide_index=True)
            else:
                st.caption("No recruitment gaps identified.")

    # ── Dashboards ──
    all_dashboards = {
        **{f"A: {k}": v for k, v in st.session_state.dashboards_a.items()},
        **{f"B: {k}": v for k, v in st.session_state.dashboards_b.items()},
    }
    if all_dashboards:
        st.divider()
        st.markdown("### 📊 Interactive Dashboards")
        selected = st.selectbox("Dashboard", list(all_dashboards.keys()))
        with st.expander("Open full dashboard", expanded=False):
            st.components.v1.html(all_dashboards[selected], height=900,
                                  scrolling=True)
        st.download_button("⬇️  Download HTML",
                           data=all_dashboards[selected],
                           file_name=f"{selected}.html", mime="text/html")
