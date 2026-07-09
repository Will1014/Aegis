"""
aegis/pretrain.py
=================
Trains ONE master Manager DNA model on all available StatsBomb data
(all five leagues, all four available seasons) and saves it to pretrained/master/.

The master model gives K-means the richest possible population to cluster against.
Squad-specific analysis (ETL, fit scoring) still uses the user's selected league
and season — only the clustering context is global.

Also trains the recruitment-cost market-value model (aegis/market_value.py)
on transfermarkt-datasets and saves it to pretrained/market_value/. This is a
separate, non-fatal step — if it fails, the DNA model training still succeeds
and recruitment costs just fall back to static estimates until the next run.

Usage:
    python -m aegis.pretrain         # train master model
    python -m aegis.pretrain --dry-run  # check connectivity without training

Scheduled via .github/workflows/pretrain.yml — runs daily at 5am UTC.
No workflow changes needed for the market-value step — it runs inside this
same script.
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import argparse
import tempfile
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

# All leagues and seasons available under the StatsBomb licence
ALL_LEAGUES = [2, 3, 4, 5, 6]   # PL, Championship, L1, L2, Eredivisie
ALL_SEASONS = [235, 281, 317, 318]  # 2022/23, 2023/24, 2024/25, 2025/26

REPO_ROOT        = Path(__file__).resolve().parent.parent
PRETRAINED_DIR   = REPO_ROOT / "pretrained"
MASTER_DIR       = PRETRAINED_DIR / "master"
MARKET_VALUE_DIR = PRETRAINED_DIR / "market_value"
MANIFEST_PATH    = PRETRAINED_DIR / "manifest.json"

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
            print(traceback.format_exc())
            sys.stdout.flush()
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

        # ── Market-value / recruitment-cost model (non-fatal) ──────────────
        # Trained on transfermarkt-datasets (github.com/dcaribou/transfermarkt-datasets),
        # a separate public data source from StatsBomb — failure here should
        # never fail the whole pretrain run, same as any other optional step.
        try:
            print("\n  Training market-rate model (transfermarkt-datasets)...")
            from aegis.market_value import (
                MarketValueClient, train_market_value_model, save_pretrained_market_model,
            )
            mv_model, mv_meta = train_market_value_model(MarketValueClient())
            save_pretrained_market_model(mv_model, mv_meta, MARKET_VALUE_DIR)
            print(f"  ✓ Market value model saved to pretrained/market_value/ "
                  f"(MAE €{mv_meta['mae_eur']:,}, n_train={mv_meta['n_train']})")
        except Exception as e:
            print(f"  ⚠ Market value training failed — skipping (recruitment costs "
                  f"will fall back to static estimates): {e}")
            print("  Full traceback (for diagnosis — training still continues non-fatally):")
            # traceback.print_exc() writes to stderr, while every other line in
            # this file uses print() (stdout). In CI, stdout/stderr are often
            # buffered and flushed independently, so mixing streams here is
            # exactly what silently dropped the traceback last run — it
            # printed the header (stdout) but the actual trace (stderr)
            # landed somewhere the captured log didn't preserve in order.
            # format_exc() + print() keeps everything on one stream.
            print(traceback.format_exc())
            sys.stdout.flush()

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

    mv_meta_path = MARKET_VALUE_DIR / "metadata.json"
    market_value_status = (
        json.loads(mv_meta_path.read_text()) if mv_meta_path.exists() else None
    )

    manifest = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "models":     models,
        "market_value_model": market_value_status,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"✓ Manifest updated")


# ─────────────────────────────────────────────────────────────────────────────
# LOADING  (used by streamlit_app.py at startup)
# ─────────────────────────────────────────────────────────────────────────────

def load_pretrained(base_dir: str) -> dict | None:
    """
    Copy the master pre-trained bundle into Config.PROCESSED_DIR / "training"
    (i.e. {base_dir}/processed/training/) — the exact path SquadFitAnalyzer
    and ManagerDNATrainer read from by default. Returns metadata dict on
    success, None if the bundle doesn't exist yet.

    IMPORTANT: build the target path from Config.PROCESSED_DIR itself, not
    by independently reconstructing it from base_dir. A previous version of
    this function hardcoded {base_dir}/data/processed/training — an extra
    "data" segment that doesn't match Config.PROCESSED_DIR's actual
    resolution ({base_dir}/processed). The copy silently succeeded into a
    location nothing ever read from, so load_model() kept failing with
    "Model not found" even though this function reported success. Deriving
    the path from Config.PROCESSED_DIR instead of duplicating the join
    logic makes it structurally impossible for these two to drift apart.

    Called by streamlit_app.py immediately after login.
    """
    if not all((MASTER_DIR / f).exists() for f in REQUIRED_FILES):
        return None

    from aegis import Config
    Config.set_base_dir(base_dir)
    target = Config.PROCESSED_DIR / "training"
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


def load_pretrained_market_value():
    """
    Return (model, metadata) for the recruitment-cost market-value model,
    or None if it hasn't been trained yet (first deploy, or a nightly run
    failed non-fatally). Callers should treat None as "use the static
    fallback cost_map" rather than erroring — see market_value.py's
    estimate_recruitment_cost_band(), which already does this.
    """
    from aegis.market_value import load_pretrained_market_model
    return load_pretrained_market_model(MARKET_VALUE_DIR)


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
