"""
aegis/pretrain.py
=================
Trains ONE master Manager DNA model on all available StatsBomb data
(all five leagues, all four available seasons) and saves it to pretrained/master/.

The master model gives K-means the richest possible population to cluster against.
Squad-specific analysis (ETL, fit scoring) still uses the user's selected league
and season — only the clustering context is global.

After the master model saves, this also runs train_position_demands.py's
derivation step (position-specific pillar demand weights, empirically
estimated from the same training population - see that file's docstring
for details). That step is non-fatal: if it errors, the master pillar
model above is still valid and this job still counts as a successful run.
No separate GitHub Actions step is needed for it - one `python -m
aegis.pretrain` call does both.

Usage:
    python -m aegis.pretrain         # train master model + derive demand weights
    python -m aegis.pretrain --dry-run  # check connectivity without training

Scheduled via .github/workflows/pretrain.yml — runs daily at 5am UTC.
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import argparse
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

# All leagues and seasons available under the StatsBomb licence
ALL_LEAGUES = [2, 3, 4, 5, 6]   # PL, Championship, L1, L2, Eredivisie
ALL_SEASONS = [235, 281, 317, 318]  # 2022/23, 2023/24, 2024/25, 2025/26

REPO_ROOT      = Path(__file__).resolve().parent.parent
PRETRAINED_DIR = REPO_ROOT / "pretrained"
MASTER_DIR     = PRETRAINED_DIR / "master"
MANIFEST_PATH  = PRETRAINED_DIR / "manifest.json"

REQUIRED_FILES = [
    "manager_dna_model.pkl",
    "manager_profiles.csv",
    "cluster_centroids.csv",
]


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_master(base_dir: str, verbose: bool = True) -> bool:
    """
    Train one master model across all leagues and seasons.

    Strategy: call fetch_manager_data_statsbomb once per season to populate
    manager_features, accumulate them all manually, then fit() once on the
    combined dataset. This gives K-means the full population without requiring
    any changes to ManagerDNATrainer.

    Returns True on success.
    """
    from aegis import Config, StatsBombClient
    from aegis.analysis import ManagerDNATrainer

    sb_user = os.environ.get("SB_USERNAME", "")
    sb_pass = os.environ.get("SB_PASSWORD", "")
    if not sb_user or not sb_pass:
        print("  ✗ SB_USERNAME / SB_PASSWORD not set")
        return False

    os.environ["SB_USERNAME"] = sb_user
    os.environ["SB_PASSWORD"] = sb_pass

    print(f"\n{'=' * 60}")
    print("  Master model — all leagues × all seasons")
    print(f"  Leagues:  {ALL_LEAGUES}")
    print(f"  Seasons:  {ALL_SEASONS}")
    print(f"{'=' * 60}\n")

    with tempfile.TemporaryDirectory() as tmp:
        Config.set_base_dir(tmp)
        sb_client = StatsBombClient()

        combined_features: List[dict] = []
        feature_names: Optional[List[str]] = None

        for season_id in ALL_SEASONS:
            print(f"\n  ── Season {season_id} ──")
            try:
                trainer = ManagerDNATrainer(
                    training_dir=Path(tmp) / "training",
                    season_id=season_id,
                )
                trainer.fetch_manager_data_statsbomb(
                    sb_client=sb_client,
                    competition_id=ALL_LEAGUES[0],   # primary for ETL ordering
                    season_id=season_id,
                    competition_ids=ALL_LEAGUES,
                )
                n = len(trainer.manager_features)
                print(f"  ✓ Season {season_id}: {n} manager profiles collected")

                if n > 0:
                    combined_features.extend(trainer.manager_features)
                    if feature_names is None and trainer.feature_names:
                        feature_names = trainer.feature_names

            except Exception as e:
                print(f"  ⚠ Season {season_id} failed — skipping: {e}")
                continue

        if not combined_features:
            print("  ✗ No data collected across any season — aborting")
            return False

        print(f"\n  Combined: {len(combined_features)} data points "
              f"across {len(ALL_SEASONS)} seasons")

        # ── Fit master model on combined data ─────────────────────────────
        MASTER_DIR.mkdir(parents=True, exist_ok=True)
        try:
            master = ManagerDNATrainer(training_dir=MASTER_DIR)
            master.manager_features = combined_features
            master.feature_names    = feature_names   # None → falls back to default
            # Fix n_clusters=6 — silhouette auto-selection collapses to K=2 on
            # large cross-league datasets. 6 archetypes gives tactically meaningful
            # separation: High-Press, Possession-Based, Counter-Attack,
            # Defensive, Attacking, Balanced.
            master.fit(verbose=verbose, n_clusters=6)
            master.save(verbose=True)
        except Exception as e:
            print(f"  ✗ fit/save failed: {e}")
            import traceback; traceback.print_exc()
            return False

        # ── Verify output files ───────────────────────────────────────────
        for fname in REQUIRED_FILES:
            if not (MASTER_DIR / fname).exists():
                print(f"  ✗ Missing {fname} after training")
                return False

        # ── Write metadata ────────────────────────────────────────────────
        meta = {
            "key":        "master",
            "description": "All leagues × all seasons",
            "league_ids": ALL_LEAGUES,
            "season_ids": ALL_SEASONS,
            "n_profiles": len(combined_features),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        (MASTER_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"\n  ✓ Master model saved to pretrained/master/")

        # ── Derive position-specific pillar demand weights ─────────────────
        # Runs against the manager_profiles.csv just written above - reads
        # its own pillar_pct_* + team_id/season_id columns, fetches
        # player_season_stats for the same team-seasons, and regresses
        # player features against team pillar scores (shrunk toward the
        # hand-authored PILLAR_PLAYER_DEMANDS table). Writes
        # pretrained/master/pillar_player_demands.json, which analysis.py
        # picks up automatically at runtime if present - no code change
        # needed beyond this file to get the improved weights live.
        #
        # Deliberately non-fatal: this step is an enhancement on top of the
        # core pillar model, not a dependency of it. If it errors (network
        # blip, a season temporarily unavailable, etc.) the master model
        # trained above is still valid and should still be treated as a
        # successful pretrain run - the existing hand-authored demand table
        # remains in effect as the fallback either way.
        try:
            from aegis.train_position_demands import run as run_position_demands
            ok = run_position_demands(verbose=verbose)
            if not ok:
                print("  ⚠ Position demand derivation did not complete - "
                      "hand-authored PILLAR_PLAYER_DEMANDS remains in effect. "
                      "Master pillar model above is unaffected.")
        except Exception as e:
            print(f"  ⚠ Position demand derivation failed - skipping: {e}")
            print("  Hand-authored PILLAR_PLAYER_DEMANDS remains in effect. "
                  "Master pillar model above is unaffected.")

        return True


# ─────────────────────────────────────────────────────────────────────────────
# MANIFEST
# ─────────────────────────────────────────────────────────────────────────────

def update_manifest():
    meta_path = MASTER_DIR / "meta.json"
    if meta_path.exists():
        models = [json.loads(meta_path.read_text())]
    else:
        models = []
    manifest = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "models":     models,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"✓ Manifest updated")


# ─────────────────────────────────────────────────────────────────────────────
# LOADING  (used by streamlit_app.py at startup)
# ─────────────────────────────────────────────────────────────────────────────

def load_pretrained(base_dir: str) -> dict | None:
    """
    Copy the master pre-trained bundle into {base_dir}/data/processed/training/.
    Returns metadata dict on success, None if the bundle doesn't exist yet.
    Called by streamlit_app.py immediately after login.
    """
    if not all((MASTER_DIR / f).exists() for f in REQUIRED_FILES):
        return None

    from aegis import Config
    Config.set_base_dir(base_dir)
    target = Path(base_dir) / "data" / "processed" / "training"
    target.mkdir(parents=True, exist_ok=True)

    for fname in REQUIRED_FILES:
        shutil.copy2(MASTER_DIR / fname, target / fname)

    meta_path = MASTER_DIR / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {"key": "master", "trained_at": "unknown"}


def master_meta() -> dict | None:
    """Return master model metadata, or None if not trained yet."""
    meta_path = MASTER_DIR / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pre-train Aegis MTFI master Manager DNA model")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check credentials and connectivity only — don't train")
    parser.add_argument("--base-dir",
                        default=str(Path.home() / "aegis_pretrain"),
                        help="Temporary directory for intermediate data")
    args = parser.parse_args()

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("── Dry run — checking credentials ──")
        u = os.environ.get("SB_USERNAME", "")
        p = os.environ.get("SB_PASSWORD", "")
        print(f"  SB_USERNAME: {'✓ set' if u else '✗ missing'}")
        print(f"  SB_PASSWORD: {'✓ set' if p else '✗ missing'}")
        return

    ok = train_master(args.base_dir)
    if ok:
        update_manifest()
        print("\n✓ Pre-training complete")
    else:
        print("\n✗ Pre-training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
