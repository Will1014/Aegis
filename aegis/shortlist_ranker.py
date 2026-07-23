"""
aegis/shortlist_ranker.py
=========================
Manager Shortlist Ranker — runs MTFI analysis for multiple managers against
a single squad and returns a ranked list ordered by average fit score.

Public API:
    run_shortlist(club, league_id, season_id, managers, base_dir, ...) -> List[ShortlistEntry]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ShortlistEntry:
    """Results for one manager in a shortlist run."""
    rank:                     int
    manager:                  str
    archetype:                str
    average_fit:              float
    key_enablers:             int
    good_fit:                 int
    system_dependent:         int
    potentially_marginalised: int
    dna_dimensions:           Dict[str, float]   = field(default_factory=dict)
    squad_fit:                List[Dict]         = field(default_factory=list)
    ideal_xi:                 List[Dict]         = field(default_factory=list)
    recruitment:              List[Dict]         = field(default_factory=list)
    primary_formation:        str                = "4-3-3"
    dna_source:               str                = "unknown"
    run_result:               Dict               = field(default_factory=dict)


def run_shortlist(
    club: str,
    league_id: int,
    season_id: int,
    managers: List[str],
    base_dir: str,
    train_model: bool = True,
    training_league_ids: Optional[List[int]] = None,
    max_matches: int = 20,
) -> List[ShortlistEntry]:
    """
    Run MTFI analysis for each manager against a single club's squad.

    The K-means model is trained once on the first iteration (or reused when
    train_model=False), keeping runtime proportional to the number of managers
    rather than quadratic.  HTML dashboard generation is suppressed for batch
    runs — use the per-entry run_result to generate on demand.

    Args:
        club:               Target club name.
        league_id:          StatsBomb competition ID.
        season_id:          StatsBomb season ID.
        managers:           List of manager name strings (max 10 recommended).
        base_dir:           Base directory for data and outputs.
        train_model:        Whether to train the model on the first run.
        training_league_ids: League IDs for the training pool.
        max_matches:        Max matches to fetch per manager.

    Returns:
        List[ShortlistEntry] sorted descending by average_fit.
    """
    from aegis import run_full_analysis_statsbomb

    if not managers:
        return []

    entries: List[ShortlistEntry] = []

    for i, manager in enumerate(managers):
        print(f"\n  ── Shortlist {i+1}/{len(managers)}: {manager} → {club} ──")
        try:
            result = run_full_analysis_statsbomb(
                target_league_id    = league_id,
                season_id           = season_id,
                team_name           = club,
                coach_name          = manager,
                base_dir            = base_dir,
                train_model         = (train_model and i == 0),
                training_league_ids = training_league_ids or [league_id],
                visualize           = False,   # suppress HTML in batch
                max_matches         = max_matches,
            )
            if isinstance(result, list):
                result = result[0] if result else {}
            if result:
                entries.append(_build_entry(i + 1, manager, result))
        except Exception as e:
            print(f"  ⚠ {manager}: {e}")
            continue

    # Sort by fit score descending; re-assign ranks
    entries.sort(key=lambda e: e.average_fit, reverse=True)
    for rank, entry in enumerate(entries, start=1):
        entry.rank = rank

    return entries


def _build_entry(provisional_rank: int, manager: str, result: Dict) -> ShortlistEntry:
    """Construct a ShortlistEntry from a pipeline result dict."""
    counts = result.get("classification_counts", {})

    # dna_dimensions is at the top level of legacy_results
    dna_dims: Dict[str, float] = {}
    _raw_dna = result.get("dna_dimensions") or {}
    if isinstance(_raw_dna, dict):
        dna_dims = {k: float(v) for k, v in _raw_dna.items() if v is not None}

    squad_fit = result.get("squad_fit", [])
    if isinstance(squad_fit, list):
        squad_fit = [
            {k: v for k, v in (p.items() if hasattr(p, "items") else vars(p).items())}
            for p in squad_fit
        ]

    return ShortlistEntry(
        rank                     = provisional_rank,
        manager                  = result.get("manager", manager),
        archetype                = result.get("archetype", "Unknown"),
        average_fit              = float(result.get("average_fit", 0)),
        key_enablers             = int(counts.get("Key Enabler", 0)),
        good_fit                 = int(counts.get("Good Fit", 0)),
        system_dependent         = int(counts.get("System Dependent", 0)),
        potentially_marginalised = int(counts.get("Potentially Marginalised", 0)),
        dna_dimensions           = dna_dims,
        squad_fit                = squad_fit,
        ideal_xi                 = result.get("ideal_xi", []),
        recruitment              = result.get("recruitment", []),
        primary_formation        = result.get("primary_formation", "4-3-3"),
        dna_source                = result.get("manager_dna_source", "unknown"),
        run_result               = result,
    )
