"""
streamlit_app_additions.py
==========================
All additions required for streamlit_app.py to support Features 1–4.

HOW TO APPLY
------------
This file is NOT a standalone script. It contains labelled code blocks,
each marked with the exact location in streamlit_app.py where they must
be inserted. Search for the anchor comment in each block.

Additions summary:
  A) Mode selector — add 'Shortlist' option
  B) Shortlist sidebar inputs
  C) Shortlist run block inside run_clicked guard
  D) Single Mode: Narrative Report section (after dashboard iframe)
  E) Single Mode: DNA Insights expander (after Report section)
  F) Shortlist results display (new elif block)
  G) render_shortlist_radar() helper function
  H) requirements.txt: add reportlab>=4.0.0
"""

# ══════════════════════════════════════════════════════════════════════════════
# A) MODE SELECTOR
# Location: find the existing radio that defines is_compare.
# Typically looks like:
#   mode = st.radio("Mode", ["Single Analysis", "Compare Scenarios"], horizontal=True)
#   is_compare = mode == "Compare Scenarios"
#
# REPLACE with:
# ══════════════════════════════════════════════════════════════════════════════

MODE_SELECTOR = """
mode = st.radio(
    "Mode",
    ["Single Analysis", "Compare Scenarios", "Shortlist"],
    horizontal=True,
)
is_compare   = (mode == "Compare Scenarios")
is_shortlist = (mode == "Shortlist")
"""


# ══════════════════════════════════════════════════════════════════════════════
# B) SHORTLIST SIDEBAR INPUTS
# Location: inside the sidebar, after the existing scenario A/B input block.
# Wrap in:  if is_shortlist:
# ══════════════════════════════════════════════════════════════════════════════

SHORTLIST_SIDEBAR = """
# ── Shortlist mode inputs ─────────────────────────────────────────────────────
if is_shortlist:
    sl_league_id, sl_team, _ = _scenario_inputs("Club", "sl", "scenario-a")
    st.caption("Select managers to rank against this squad (max 10).")
    sl_managers = st.multiselect(
        "Managers to rank",
        options=all_manager_names,
        max_selections=10,
        key="sl_managers",
    )
"""


# ══════════════════════════════════════════════════════════════════════════════
# C) SHORTLIST RUN BLOCK
# Location: inside the `if run_clicked:` guard, AFTER the existing
#           Scenario B block (the `if is_compare:` / `else:` section).
# Insert as a new `elif is_shortlist:` branch.
# ══════════════════════════════════════════════════════════════════════════════

SHORTLIST_RUN_BLOCK = """
    # ── Shortlist ──
    elif is_shortlist:
        if not sl_team:
            st.error("Select a club.")
            st.stop()
        if not sl_managers:
            st.error("Select at least one manager.")
            st.stop()

        status_sl = st.status(
            f"Shortlist: {sl_team} ({len(sl_managers)} managers)",
            expanded=True,
        )
        try:
            from aegis.shortlist_ranker import run_shortlist
            ranked = run_shortlist(
                club                = sl_team,
                league_id           = sl_league_id,
                season_id           = season_id,
                managers            = sl_managers,
                base_dir            = base_dir,
                train_model         = train_model,
                training_league_ids = training_league_ids,
                max_matches         = max_matches,
            )
            st.session_state.shortlist      = ranked
            st.session_state.shortlist_club = sl_team
            status_sl.update(
                label=f"✅ Shortlist complete — {len(ranked)} managers ranked",
                state="complete", expanded=False,
            )
        except Exception as e:
            status_sl.update(label="❌ Shortlist failed", state="error")
            st.error(f"```\\n{e}\\n```")
            st.stop()
    else:
        st.session_state.shortlist      = None
        st.session_state.shortlist_club = None
"""


# ══════════════════════════════════════════════════════════════════════════════
# D) NARRATIVE REPORT — Single Mode
# Location: in the `if results_b is None:` block, AFTER the existing
#           dashboard iframe and download button.
# ══════════════════════════════════════════════════════════════════════════════

SINGLE_MODE_REPORT = """
        # ── Narrative Report ─────────────────────────────────────────────
        st.divider()
        st.markdown("### 📄 Narrative Report")

        if st.button("Generate Report", type="secondary", key="gen_report"):
            from aegis.ai_reporter import generate_narrative_report
            with st.spinner("Generating report…"):
                _sections = generate_narrative_report(
                    result     = results_a[0],
                    analysis   = st.session_state.analysis_a,
                    squad      = st.session_state.squad_a,
                    output_dir = Path(base_dir) / "outputs",
                )
            st.session_state.report_sections = _sections

        if st.session_state.get("report_sections"):
            _sec   = st.session_state.report_sections
            _mgr   = results_a[0].get("manager", "")
            _club  = results_a[0].get("club", "")
            _fmt   = st.session_state.analysis_a.get("primary_formation", "")

            for _title, _content in _sec.items():
                with st.expander(_title, expanded=True):
                    st.markdown(_content)

            _dl1, _dl2 = st.columns(2)
            with _dl1:
                try:
                    from aegis.ai_reporter import export_pdf
                    _pdf = export_pdf(_sec, _mgr, _club, _fmt)
                    st.download_button(
                        "⬇️  Download PDF", data=_pdf,
                        file_name=f"MTFI_Report_{_mgr.replace(' ','_')}.pdf",
                        mime="application/pdf",
                    )
                except ImportError:
                    st.caption("Install reportlab>=4.0.0 to enable PDF export.")
            with _dl2:
                from aegis.ai_reporter import export_html
                _html = export_html(_sec, _mgr, _club, _fmt)
                st.download_button(
                    "⬇️  Download HTML", data=_html,
                    file_name=f"MTFI_Report_{_mgr.replace(' ','_')}.html",
                    mime="text/html",
                )
"""


# ══════════════════════════════════════════════════════════════════════════════
# E) DNA INSIGHTS EXPANDER — Single Mode
# Location: AFTER the Narrative Report block (D), still inside
#           the `if results_b is None:` block.
# ══════════════════════════════════════════════════════════════════════════════

SINGLE_MODE_DNA_INSIGHTS = """
        # ── DNA Insights ─────────────────────────────────────────────────
        if st.session_state.analysis_a.get("dna_dimensions"):
            with st.expander("🔬 DNA Insights", expanded=False):
                from aegis.dna_insights import (
                    compute_manager_similarity,
                    compute_pillar_benchmarks,
                    compute_pillar_confidence,
                    compute_formation_tendency,
                )
                _r        = results_a[0]
                _dna      = st.session_state.analysis_a.get("dna_dimensions", {})
                _tr_dir   = Path(base_dir) / "training"
                _cluster  = _r.get("cluster", 0)

                # ── Similarity ────────────────────────────────────────────
                st.markdown("#### Tactically Similar Managers")
                _similar = compute_manager_similarity(_r.get("manager", ""), _tr_dir)
                if _similar:
                    st.dataframe(
                        pd.DataFrame(_similar),
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.caption("Similarity data unavailable — training data required.")

                # ── Pillar Benchmarks ─────────────────────────────────────
                st.markdown("#### Pillar vs Archetype Benchmark")
                _benchmarks = compute_pillar_benchmarks(
                    _r.get("manager", ""), _cluster, _tr_dir, _dna)
                if _benchmarks:
                    _bdf = pd.DataFrame(_benchmarks)[
                        ["display_name", "score", "archetype_mean", "delta", "flag"]
                    ].rename(columns={
                        "display_name":   "Pillar",
                        "score":          "Score",
                        "archetype_mean": "Archetype Mean",
                        "delta":          "Delta",
                        "flag":           "Signal",
                    })
                    st.dataframe(_bdf, hide_index=True, use_container_width=True)

                # ── Confidence ────────────────────────────────────────────
                _conf = compute_pillar_confidence(_r, st.session_state.analysis_a)
                st.caption(
                    f"{_conf['badge']}  {_conf['confidence_note']}"
                )

                # ── Formation Tendency ────────────────────────────────────
                st.markdown("#### Formation Tendency")
                if has_creds:
                    @st.cache_data(ttl=3600 * 4, show_spinner=False)
                    def _cached_formation(team, comp_id, s_id, u, p):
                        return compute_formation_tendency(team, comp_id, s_id, u, p)

                    _ft = _cached_formation(
                        _r.get("club", ""), league_id_a, season_id, sb_user, sb_pass)
                    if _ft:
                        _pri = f"**{_ft['primary']}** ({_ft['primary_pct']}%)"
                        _sec = (f"  ·  Secondary: **{_ft['secondary']}** "
                                f"({_ft['secondary_pct']}%)"
                                if _ft.get("secondary") else "")
                        st.markdown(f"Primary: {_pri}{_sec}"
                                    f"  ·  {_ft['matches_sampled']} matches sampled")
                    else:
                        st.caption("Formation data unavailable.")
                else:
                    st.caption("StatsBomb credentials required for formation data.")
"""


# ══════════════════════════════════════════════════════════════════════════════
# F) SHORTLIST RESULTS DISPLAY
# Location: AFTER the existing `elif results_b is not None:` block
#           (the comparison mode display). Add as a new `elif` branch:
#   elif st.session_state.get("shortlist"):
# ══════════════════════════════════════════════════════════════════════════════

SHORTLIST_DISPLAY = """
elif st.session_state.get("shortlist"):
    ranked     = st.session_state.shortlist
    _sl_club   = st.session_state.get("shortlist_club", "")

    st.markdown(f"## Manager Shortlist — {_sl_club}")

    # ── Summary table ──────────────────────────────────────────────────────
    _rows = []
    for e in ranked:
        _lo = sum(float(r.get("cost_low",  0)) for r in e.recruitment)
        _hi = sum(float(r.get("cost_high", 0)) for r in e.recruitment)
        _rows.append({
            "Rank":                  e.rank,
            "Manager":               e.manager,
            "Formation":             e.primary_formation,
            "Archetype":             e.archetype,
            "Avg Fit":               f"{e.average_fit:.1f}",
            "Key Enablers":          e.key_enablers,
            "Marginalised":          e.potentially_marginalised,
            "Recruit. Est. (£M)":    f"{_lo:.0f}–{_hi:.0f}" if _lo > 0 else "—",
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

            # Inline report + PDF download per manager
            if st.button(
                f"Generate Report — {entry.manager}",
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

            _key = f"sl_report_{entry.rank}"
            if st.session_state.get(_key):
                _sl_sec = st.session_state[_key]
                for _t, _c in _sl_sec.items():
                    with st.expander(_t, expanded=True):
                        st.markdown(_c)
                try:
                    from aegis.ai_reporter import export_pdf
                    _sl_pdf = export_pdf(
                        _sl_sec, entry.manager, _sl_club, entry.primary_formation)
                    st.download_button(
                        f"⬇️  PDF — {entry.manager}",
                        data=_sl_pdf,
                        file_name=f"MTFI_{entry.manager.replace(' ','_')}.pdf",
                        mime="application/pdf",
                        key=f"pdf_sl_{entry.rank}",
                    )
                except ImportError:
                    pass
"""


# ══════════════════════════════════════════════════════════════════════════════
# G) render_shortlist_radar() HELPER
# Location: add alongside the existing render_radar() function in
#           streamlit_app.py (search for "def render_radar").
# ══════════════════════════════════════════════════════════════════════════════

SHORTLIST_RADAR_HELPER = """
def render_shortlist_radar(ranked):
    \"\"\"Overlay DNA profiles for all shortlisted managers on one polar axis.\"\"\"
    if not ranked:
        return None

    # Use first entry with dna_dimensions to define pillar order
    reference = next((e for e in ranked if e.dna_dimensions), None)
    if not reference:
        return None

    pillars = list(reference.dna_dimensions.keys())
    n = len(pillars)
    if n == 0:
        return None

    angles = np.linspace(0, 2 * math.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    # Colour cycle for up to 10 managers
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
        vals = [entry.dna_dimensions.get(p, 50) for p in pillars] + \
               [entry.dna_dimensions.get(pillars[0], 50)]
        ax.fill(angles, vals, alpha=0.08, color=color)
        ax.plot(angles, vals, color=color, linewidth=1.5,
                label=f"#{entry.rank} {entry.manager}")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(pillars, size=6, color="#94a3b8")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=5, colors="#475569")
    ax.legend(
        loc="upper right", fontsize=6, framealpha=0.3,
        labelcolor="#e0e4ec", facecolor="#131b2e", edgecolor="#1e2a3a",
    )
    return fig
"""


# ══════════════════════════════════════════════════════════════════════════════
# H) REQUIREMENTS.TXT
# Add this line:
#   reportlab>=4.0.0
# ══════════════════════════════════════════════════════════════════════════════

REQUIREMENTS_ADDITION = "reportlab>=4.0.0"


# ══════════════════════════════════════════════════════════════════════════════
# INITIALISE SESSION STATE KEYS
# Location: alongside the existing session_state initialisation block
#           (search for "if 'results_a' not in st.session_state")
# ADD these lines:
# ══════════════════════════════════════════════════════════════════════════════

SESSION_STATE_ADDITIONS = """
if "shortlist"        not in st.session_state: st.session_state.shortlist        = None
if "shortlist_club"   not in st.session_state: st.session_state.shortlist_club   = None
if "report_sections"  not in st.session_state: st.session_state.report_sections  = None
"""
