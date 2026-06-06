"""
aegis/ai_reporter.py
====================
Template-based AI Narrative Report Generator.

No external API calls. Produces a structured five-section briefing from
MTFI pipeline outputs using deterministic Python template logic.

Public API:
    generate_narrative_report(result, analysis, squad, output_dir) -> Dict[str, str]
    export_pdf(sections, manager, club, formation=None) -> bytes
    export_html(sections, manager, club, formation=None) -> str
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PILLAR_LABELS: Dict[str, tuple] = {
    "shape_occupation":   ("Shape & Occupation",   "positional structure and rest defence"),
    "build_up":           ("Build-up",              "possession patience and GK distribution"),
    "chance_creation":    ("Chance Creation",        "shot quality and creative carry threat"),
    "press_counterpress": ("Press & Counterpress",   "pressing intensity and ball regain rate"),
    "block_line_height":  ("Block & Line Height",    "defensive line depth and box protection"),
    "transitions":        ("Transitions",            "counter speed and defensive transition"),
    "width_overloads":    ("Width & Overloads",      "crossing volume and take-on success"),
    "set_pieces":         ("Set Pieces",             "set-piece xG share and routine efficiency"),
}

# Map display names back to pillar keys (dna_dimensions may use either)
_DISPLAY_TO_KEY = {v[0]: k for k, v in PILLAR_LABELS.items()}
_DISPLAY_TO_KEY.update({
    "Shape & Occupation": "shape_occupation",
    "Build-up": "build_up",
    "Chance Creation": "chance_creation",
    "Press & Counterpress": "press_counterpress",
    "Block & Line Height": "block_line_height",
    "Transitions": "transitions",
    "Width & Overloads": "width_overloads",
    "Set Pieces": "set_pieces",
})

_FIT_ADJECTIVES = {
    (80, 101): ("excellent", "outstanding"),
    (65, 80):  ("strong",    "very positive"),
    (50, 65):  ("moderate",  "workable"),
    (0,  50):  ("weak",      "challenging"),
}

def _fit_adj(score: float) -> tuple:
    for (lo, hi), (adj, adv) in _FIT_ADJECTIVES.items():
        if lo <= score < hi:
            return adj, adv
    return "moderate", "workable"


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def generate_narrative_report(
    result: Dict,
    analysis: Dict,
    squad,
    output_dir,
) -> Dict[str, str]:
    """
    Generate a five-section narrative report from MTFI outputs.

    Args:
        result:      results_a[0] from st.session_state — the main results dict.
        analysis:    analysis_a snapshot — contains dna_dimensions.
        squad:       squad_a snapshot — squad_fit_summary.json dict.
        output_dir:  Path to outputs directory (for recruitment_priorities.csv).

    Returns:
        Ordered dict of {section_title: markdown_content}.
    """
    ctx = _extract_context(result, analysis, squad, output_dir)
    return _write_sections(ctx)


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_context(result, analysis, squad, output_dir) -> Dict:
    """Pull all required values from existing session data."""
    import csv

    # Core result values
    manager   = result.get("manager", "Unknown Manager")
    club      = result.get("club", "Unknown Club")
    archetype = result.get("archetype", "Unknown")
    avg_fit   = float(result.get("average_fit", 0))
    formation = result.get("primary_formation") or analysis.get("primary_formation", "4-3-3")
    matches   = result.get("matches_analysed") or analysis.get("matches_analysed", 0)

    counts = result.get("classification_counts", {})
    key_enablers   = int(counts.get("Key Enabler", 0))
    good_fit       = int(counts.get("Good Fit", 0))
    sys_dependent  = int(counts.get("System Dependent", 0))
    marginalised   = int(counts.get("Potentially Marginalised", 0))
    total_players  = key_enablers + good_fit + sys_dependent + marginalised

    # DNA pillars — normalise keys
    raw_dna = analysis.get("dna_dimensions", {})
    dna: Dict[str, float] = {}
    for k, v in raw_dna.items():
        canonical = _DISPLAY_TO_KEY.get(k, k)
        dna[canonical] = float(v)

    # Sort pillars high → low
    sorted_pillars = sorted(dna.items(), key=lambda x: x[1], reverse=True)
    top_pillars    = sorted_pillars[:2]
    bottom_pillars = sorted_pillars[-2:]

    # Player name lists
    enabler_names = []
    marginalised_names = []
    if squad and isinstance(squad, dict):
        ideal = squad.get("ideal_xi", [])
        for p in ideal:
            cls = p.get("classification", "")
            name = p.get("name", "")
            if cls == "Key Enabler" and name:
                enabler_names.append(f"{name} ({p.get('position', p.get('slot', ''))})")
        # Also scan squad_fit list inside squad
    sq_fit = result.get("squad_fit") or (squad.get("ideal_xi") if squad else [])
    if isinstance(sq_fit, list):
        for p in sq_fit:
            cls = p.get("classification", "")
            name = p.get("name", "")
            pos  = p.get("position", p.get("detailed_position", ""))
            if cls == "Key Enabler" and name and name not in str(enabler_names):
                enabler_names.append(f"{name} ({pos})")
            elif cls == "Potentially Marginalised" and name:
                marginalised_names.append(f"{name} ({pos})")

    enabler_names    = enabler_names[:5]
    marginalised_names = marginalised_names[:4]

    # Recruitment priorities
    recruitment = []
    try:
        rec_path = Path(output_dir) / "recruitment_priorities.csv"
        if rec_path.exists():
            with open(rec_path, newline="") as f:
                for row in csv.DictReader(f):
                    recruitment.append(row)
    except Exception:
        pass
    recruitment = recruitment[:4]

    total_cost_low  = sum(float(r.get("cost_low",  0)) for r in recruitment)
    total_cost_high = sum(float(r.get("cost_high", 0)) for r in recruitment)

    # Transition disruption estimate
    at_risk_pct = ((sys_dependent + marginalised) / max(total_players, 1)) * 100
    if at_risk_pct > 50:
        transition_window = "12–18 months"
        disruption = "significant"
    elif at_risk_pct > 30:
        transition_window = "6–12 months"
        disruption = "moderate"
    else:
        transition_window = "3–6 months"
        disruption = "low"

    return dict(
        manager=manager, club=club, archetype=archetype, avg_fit=avg_fit,
        formation=formation, matches=matches,
        key_enablers=key_enablers, good_fit=good_fit,
        sys_dependent=sys_dependent, marginalised=marginalised,
        total_players=total_players,
        dna=dna, sorted_pillars=sorted_pillars,
        top_pillars=top_pillars, bottom_pillars=bottom_pillars,
        enabler_names=enabler_names, marginalised_names=marginalised_names,
        recruitment=recruitment,
        total_cost_low=total_cost_low, total_cost_high=total_cost_high,
        at_risk_pct=at_risk_pct, transition_window=transition_window,
        disruption=disruption,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION WRITERS
# ─────────────────────────────────────────────────────────────────────────────

def _write_sections(ctx: Dict) -> Dict[str, str]:
    """Return ordered dict of section_title → markdown text."""
    return {
        "1. Executive Summary":          _section_executive(ctx),
        "2. Manager DNA Profile":        _section_dna(ctx),
        "3. Squad Impact Assessment":    _section_squad(ctx),
        "4. Recruitment Brief":          _section_recruitment(ctx),
        "5. Risk & Transition Timeline": _section_transition(ctx),
    }


def _section_executive(c: Dict) -> str:
    adj, adv = _fit_adj(c["avg_fit"])
    action = (
        "warrants serious consideration"     if c["avg_fit"] >= 65 else
        "offers a viable but partial solution" if c["avg_fit"] >= 50 else
        "presents significant adaptation challenges"
    )
    enabler_line = (
        f"The squad contains **{c['key_enablers']} Key Enablers** "
        f"whose profiles align closely with his demands"
        if c["key_enablers"] > 0
        else "No players currently qualify as Key Enablers under this profile"
    )
    margin_line = (
        f" and **{c['marginalised']} players are at risk of marginalisation**"
        if c["marginalised"] > 0
        else ""
    )
    return (
        f"A tactical fit assessment of **{c['manager']}** against the "
        f"**{c['club']}** squad produces an average fit score of "
        f"**{c['avg_fit']:.1f}/100** — a **{adj}** result that {action}. "
        f"His tactical archetype is classified as **{c['archetype']}**, "
        f"typically deploying a **{c['formation']}** shape. "
        f"{enabler_line}{margin_line}. "
        f"Based on {c['matches']} matches of StatsBomb data, this profile "
        f"{'presents a strong platform' if c['avg_fit'] >= 65 else 'requires targeted investment'} "
        f"for a successful appointment."
    )


def _section_dna(c: Dict) -> str:
    if not c["dna"]:
        return "DNA pillar data unavailable for this analysis."

    lines = ["The 8-pillar tactical DNA profile reveals the following characteristics:\n"]

    for key, score in c["sorted_pillars"]:
        label, football_desc = PILLAR_LABELS.get(key, (key, key))
        if score >= 70:
            qualifier = f"a high score of **{score:.0f}**, indicating strong emphasis on {football_desc}"
        elif score >= 45:
            qualifier = f"a mid-range score of **{score:.0f}**, suggesting a balanced approach to {football_desc}"
        else:
            qualifier = f"a low score of **{score:.0f}**, indicating limited focus on {football_desc}"
        lines.append(f"- **{label}**: {qualifier}.")

    top_names    = " and ".join(PILLAR_LABELS.get(k, (k,))[0] for k, _ in c["top_pillars"])
    bottom_names = " and ".join(PILLAR_LABELS.get(k, (k,))[0] for k, _ in c["bottom_pillars"])
    lines.append(
        f"\nThe standout strengths are **{top_names}**, which will define the "
        f"tempo and structure of play. The relative weaknesses in **{bottom_names}** "
        f"may require specific recruitment or tactical adjustments."
    )
    return "\n".join(lines)


def _section_squad(c: Dict) -> str:
    total = c["total_players"]
    lines = [
        f"Analysis of the **{c['club']}** squad against {c['manager']}'s "
        f"tactical profile produces the following classification breakdown "
        f"across **{total} players**:\n",
        f"| Classification | Count | % of Squad |",
        f"|---|---|---|",
        f"| 🟢 Key Enabler | {c['key_enablers']} | {c['key_enablers']/max(total,1)*100:.0f}% |",
        f"| 🟡 Good Fit | {c['good_fit']} | {c['good_fit']/max(total,1)*100:.0f}% |",
        f"| 🟠 System Dependent | {c['sys_dependent']} | {c['sys_dependent']/max(total,1)*100:.0f}% |",
        f"| 🔴 Potentially Marginalised | {c['marginalised']} | {c['marginalised']/max(total,1)*100:.0f}% |",
    ]
    if c["enabler_names"]:
        lines.append(f"\n**Key Enablers** — players whose profiles align most closely with the manager's demands:")
        for n in c["enabler_names"]:
            lines.append(f"- {n}")
    if c["marginalised_names"]:
        lines.append(f"\n**Potentially Marginalised** — players at risk of reduced minutes or poor form under this system:")
        for n in c["marginalised_names"]:
            lines.append(f"- {n}")
    if not c["enabler_names"] and not c["marginalised_names"]:
        lines.append("\nPlayer-level detail unavailable — run with full squad data to populate this section.")
    return "\n".join(lines)


def _section_recruitment(c: Dict) -> str:
    if not c["recruitment"]:
        return (
            "No specific recruitment priorities were identified, suggesting the existing "
            "squad provides adequate coverage across all positional groups under this profile."
        )
    lines = [
        f"The following positional gaps require addressing to fully implement "
        f"{c['manager']}'s system:\n",
        f"| Position | Urgency | Timeline | Est. Cost |",
        f"|---|---|---|---|",
    ]
    for r in c["recruitment"]:
        pos      = r.get("position", "Unknown")
        urgency  = r.get("urgency", "Medium")
        timeline = r.get("timeline", "Summer")
        lo       = r.get("cost_low", 0)
        hi       = r.get("cost_high", 0)
        lines.append(f"| {pos} | {urgency} | {timeline} | £{lo}M – £{hi}M |")

    if c["total_cost_low"] > 0:
        lines.append(
            f"\n**Total estimated investment: £{c['total_cost_low']:.0f}M – "
            f"£{c['total_cost_high']:.0f}M** across {len(c['recruitment'])} priority signings."
        )
    return "\n".join(lines)


def _section_transition(c: Dict) -> str:
    pct = c["at_risk_pct"]
    window = c["transition_window"]
    disruption = c["disruption"]
    lines = [
        f"**{pct:.0f}%** of the squad ({c['sys_dependent'] + c['marginalised']} players) "
        f"are classified as System Dependent or Potentially Marginalised, indicating "
        f"**{disruption} disruption** during the adaptation period.\n",
        f"**Estimated transition window: {window}**\n",
    ]
    if pct > 50:
        lines.append(
            "The scale of change warrants phased implementation: preserving defensive "
            "structure in year one while gradually introducing pressing and positional "
            "demands. Immediate loan or sale of marginalised players is recommended to "
            "fund priority signings."
        )
    elif pct > 30:
        lines.append(
            "A targeted pre-season programme focused on pressing triggers and positional "
            "responsibilities should mitigate most adaptation risk. One or two key signings "
            "in the highest-urgency positions will accelerate the transition."
        )
    else:
        lines.append(
            "The squad is well-placed to adapt quickly. Transition risk is low, "
            "and the manager's system should be broadly operational within the first "
            "competitive month."
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PDF EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_pdf(
    sections: Dict[str, str],
    manager: str,
    club: str,
    formation: str = "",
) -> bytes:
    """
    Convert report sections to a formatted A4 PDF using reportlab.

    Returns raw PDF bytes suitable for st.download_button().
    Requires: reportlab>=4.0.0
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=15*mm, bottomMargin=15*mm,
        title=f"MTFI Report — {manager}",
    )

    # ── Colour palette ────────────────────────────────────────────────────
    AEGIS_BLUE  = colors.HexColor("#1A3A5C")
    ACCENT      = colors.HexColor("#2196F3")
    LIGHT_GREY  = colors.HexColor("#F5F7FA")
    MED_GREY    = colors.HexColor("#64748B")
    TEXT_DARK   = colors.HexColor("#1A1A2E")
    WHITE       = colors.white

    # ── Styles ────────────────────────────────────────────────────────────
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("ReportTitle", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=22, textColor=WHITE,
            spaceAfter=2, alignment=TA_CENTER),
        "subtitle": ParagraphStyle("ReportSubtitle", parent=base["Normal"],
            fontName="Helvetica", fontSize=11, textColor=colors.HexColor("#B0C4DE"),
            spaceAfter=0, alignment=TA_CENTER),
        "section": ParagraphStyle("SectionHead", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=13, textColor=AEGIS_BLUE,
            spaceBefore=14, spaceAfter=6,
            borderPad=4),
        "body": ParagraphStyle("Body", parent=base["Normal"],
            fontName="Helvetica", fontSize=10, textColor=TEXT_DARK,
            spaceAfter=6, leading=15),
        "footer": ParagraphStyle("Footer", parent=base["Normal"],
            fontName="Helvetica", fontSize=8, textColor=MED_GREY,
            alignment=TA_CENTER),
    }

    story = []

    # ── Header block ──────────────────────────────────────────────────────
    header_data = [[
        Paragraph("AEGIS MTFI", styles["title"]),
    ]]
    header_table = Table(header_data, colWidths=[170*mm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), AEGIS_BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(header_table)

    sub_data = [[
        Paragraph(f"Manager Tactical Fit Intelligence", styles["subtitle"]),
    ]]
    sub_table = Table(sub_data, colWidths=[170*mm])
    sub_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), AEGIS_BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(sub_table)
    story.append(Spacer(1, 8*mm))

    # ── Meta row ─────────────────────────────────────────────────────────
    meta_line = f"<b>{manager}</b>  ›  {club}"
    if formation:
        meta_line += f"  ·  {formation}"
    story.append(Paragraph(meta_line, styles["body"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT, spaceAfter=8))

    # ── Sections ─────────────────────────────────────────────────────────
    for title, content in sections.items():
        story.append(Paragraph(title, styles["section"]))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#C0D0E8"), spaceAfter=4))
        # Convert markdown-style content to reportlab-safe paragraphs
        for line in _md_to_paragraphs(content, styles["body"]):
            story.append(line)
        story.append(Spacer(1, 4*mm))

    # ── Footer ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#C0D0E8"), spaceAfter=4))
    story.append(Paragraph(
        "Produced by Aegis Football Advisory Group  ·  Confidential",
        styles["footer"]))

    doc.build(story)
    return buf.getvalue()


def _md_to_paragraphs(text: str, style) -> list:
    """
    Convert simple markdown to a list of reportlab Paragraph / Spacer objects.
    Handles: **bold**, bullet lists (- item), markdown tables (|col|), blank lines.
    """
    from reportlab.platypus import Paragraph, Spacer
    from reportlab.lib.units import mm

    items = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            items.append(Spacer(1, 2*mm))
            i += 1
            continue

        # Markdown table — collect all rows
        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            items.append(_md_table_to_rl(table_lines, style))
            continue

        # Bullet
        if stripped.startswith("- ") or stripped.startswith("* "):
            content = stripped[2:].strip()
            content = _md_inline(content)
            items.append(Paragraph(f"&nbsp;&nbsp;&nbsp;• {content}", style))
            i += 1
            continue

        # Normal paragraph
        items.append(Paragraph(_md_inline(stripped), style))
        i += 1

    return items


def _md_inline(text: str) -> str:
    """Convert **bold** to <b>bold</b> for reportlab."""
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Escape remaining & < > not inside tags
    # (reportlab uses XML-style markup)
    return text


def _md_table_to_rl(lines: list, body_style):
    """Convert markdown table lines to a reportlab Table."""
    from reportlab.platypus import Table, Paragraph
    from reportlab.lib import colors
    from reportlab.platypus.tables import TableStyle

    rows = []
    for line in lines:
        if set(line.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
            continue  # separator row
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)

    if not rows:
        return Paragraph("", body_style)

    n_cols = max(len(r) for r in rows)
    # Pad short rows
    rows = [r + [""] * (n_cols - len(r)) for r in rows]

    from reportlab.lib.units import mm
    col_w = 165 * mm / n_cols

    rl_rows = []
    for r_idx, row in enumerate(rows):
        rl_row = []
        for cell in row:
            p_style = body_style
            txt = _md_inline(cell)
            if r_idx == 0:
                from reportlab.lib.styles import ParagraphStyle
                from reportlab.lib.enums import TA_LEFT
                p_style = ParagraphStyle("TH", parent=body_style,
                    fontName="Helvetica-Bold", fontSize=9)
            rl_row.append(Paragraph(txt, p_style))
        rl_rows.append(rl_row)

    t = Table(rl_rows, colWidths=[col_w] * n_cols, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1A3A5C")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#F7FAFF"), colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


# ─────────────────────────────────────────────────────────────────────────────
# HTML EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_html(
    sections: Dict[str, str],
    manager: str,
    club: str,
    formation: str = "",
) -> str:
    """Return a self-contained branded HTML string of the report."""
    import re

    def md_to_html(text: str) -> str:
        """Convert basic markdown to HTML."""
        lines, out = text.split("\n"), []
        in_table = False
        for line in lines:
            s = line.strip()
            if s.startswith("|"):
                if not in_table:
                    out.append("<table>")
                    in_table = True
                if set(s.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
                    continue
                cells = [c.strip() for c in s.strip("|").split("|")]
                tag = "th" if not any(o.startswith("<td") for o in out[-10:]) else "td"
                out.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
            else:
                if in_table:
                    out.append("</table>")
                    in_table = False
                if s.startswith("- ") or s.startswith("* "):
                    out.append(f"<li>{re.sub(r'\\*\\*(.+?)\\*\\*', r'<b>\\1</b>', s[2:])}</li>")
                elif s:
                    out.append(f"<p>{re.sub(r'\\*\\*(.+?)\\*\\*', r'<b>\\1</b>', s)}</p>")
                else:
                    out.append("")
        if in_table:
            out.append("</table>")
        return "\n".join(out)

    meta = f"{manager}  ›  {club}" + (f"  ·  {formation}" if formation else "")

    sections_html = ""
    for title, content in sections.items():
        sections_html += f"""
        <div class="section">
          <h2>{title}</h2>
          <div class="content">{md_to_html(content)}</div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MTFI Report — {manager}</title>
<style>
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background:#f0f4f8;
          color:#1a1a2e; margin:0; padding:0; }}
  .wrapper {{ max-width:860px; margin:40px auto; padding:0 20px 60px; }}
  header {{ background:#1A3A5C; color:white; padding:28px 32px 20px;
            border-radius:10px 10px 0 0; }}
  header h1 {{ margin:0 0 4px; font-size:2rem; letter-spacing:.04em; }}
  header p  {{ margin:0; color:#a0bcd8; font-size:.9rem; }}
  .section {{ background:white; border:1px solid #dde6f0;
              margin-top:16px; border-radius:8px; padding:24px 28px; }}
  .section h2 {{ margin:0 0 10px; font-size:1.05rem; color:#1A3A5C;
                 border-bottom:2px solid #2196F3; padding-bottom:6px; }}
  .content p  {{ margin:.4em 0; line-height:1.65; }}
  .content li {{ margin:.3em 0 .3em 1.2em; line-height:1.6; }}
  table {{ border-collapse:collapse; width:100%; margin:12px 0; font-size:.88rem; }}
  th {{ background:#1A3A5C; color:white; padding:7px 10px; text-align:left; }}
  td {{ padding:6px 10px; border-bottom:1px solid #eee; }}
  tr:nth-child(even) td {{ background:#f7faff; }}
  footer {{ text-align:center; margin-top:32px; font-size:.8rem; color:#64748b; }}
</style>
</head>
<body>
<div class="wrapper">
  <header>
    <h1>AEGIS MTFI</h1>
    <p>Manager Tactical Fit Intelligence &nbsp;·&nbsp; {meta}</p>
  </header>
  {sections_html}
  <footer>Produced by Aegis Football Advisory Group &nbsp;·&nbsp; Confidential</footer>
</div>
</body>
</html>"""
