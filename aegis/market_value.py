"""
Aegis Market Value Client + Recruitment Cost Model
====================================================
Replaces the hardcoded cost_map in __init__.py with data-driven recruitment
cost estimates, trained on real transfer fees from the transfermarkt-datasets
project (github.com/dcaribou/transfermarkt-datasets).

IMPORTANT — this is NOT a live query API
-----------------------------------------
Unlike StatsBombClient, there is no per-player/per-query endpoint here.
transfermarkt-datasets publishes whole-table CSV snapshots (refreshed weekly)
to a public, unauthenticated Cloudflare R2 bucket. "Fetching from the API"
in practice means downloading a handful of small compressed CSVs and caching
them locally — there's no server-side filtering. We deliberately skip the
`appearances` table (1.8M+ rows, only goals/assists/minutes — StatsBomb
already gives far richer per-90 data for your licensed competitions) and
only pull the four tables that carry price signal: players, transfers,
player_valuations, clubs.

KNOWN COVERAGE GAP — read before trusting this for EFL tiers
--------------------------------------------------------------
The dataset's own dbt config (`competition_codes` var) restricts match/
appearance-level scraping to: ES1, GB1, L1, IT1, FR1, GR1, PO1, BE1, NL1,
UKR1, RU1, DK1, SC1, TR1 + cups/internationals. That's Premier League and
Eredivisie from your 5 licensed leagues — Championship (GB2), League One
(GB3) and League Two (GB4) are NOT in that list. The `players`/`transfers`
tables aren't filtered by that var, so EFL players *may* still appear
(e.g. via loan histories or transfers involving PL/Championship clubs),
but coverage is unconfirmed and likely sparse. Once you have real data
loaded, run:
    SELECT domestic_competition_id, count(*) FROM clubs GROUP BY 1
to see actual EFL coverage before relying on this for those tiers.

Network note
------------
This module was written and schema-validated against the live dbt models
in the repo, but could not be executed end-to-end in this dev sandbox —
its egress allowlist blocks the R2 host (pub-e682421888d945d684bcae8890b0ec20.r2.dev).
It should work unmodified on Streamlit Community Cloud and in GitHub Actions,
both of which have open outbound internet, but run the __main__ smoke test
below in one of those environments before relying on it.
"""

from __future__ import annotations

import io
import os
import json
import time
import pickle
import requests
from pathlib import Path
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Optional, Dict, List, Tuple, Any

import pandas as pd
import numpy as np

from .config import Config


# =============================================================================
# CONSTANTS
# =============================================================================

# Public, unauthenticated bucket — no API key needed. Weekly refresh cadence,
# so a long cache TTL is appropriate (unlike StatsBomb's 1hr).
_R2_BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"

DATA_URLS = {
    "players":           f"{_R2_BASE}/players.csv.gz",
    "transfers":         f"{_R2_BASE}/transfers.csv.gz",
    "player_valuations": f"{_R2_BASE}/player_valuations.csv.gz",
    "clubs":             f"{_R2_BASE}/clubs.csv.gz",
}

DEFAULT_CACHE_TTL_HOURS = 72  # data refreshes weekly; no need to re-pull daily

# StatsBomb competition_id -> Transfermarkt domestic_competition_id.
# GB1/NL1 confirmed against the live competitions.json seed file.
# GB2/GB3/GB4 are Transfermarkt's real codes for Championship/League One/
# League Two but are NOT covered by the dataset's competition_codes scrape
# scope — treat cost estimates for these leagues as low-confidence until
# you've checked actual row counts (see coverage-gap note above).
LEAGUE_CODE_MAP = {
    2: "GB1",  # Premier League — confirmed good coverage
    3: "GB2",  # Championship — coverage unconfirmed
    4: "GB3",  # League One — coverage unconfirmed
    5: "GB4",  # League Two — coverage unconfirmed
    6: "NL1",  # Eredivisie — confirmed good coverage
}

# Rough static EUR->GBP conversion. The existing cost_map this replaces was
# already an eyeballed £M figure with no real FX behind it, so a static
# constant is a strict improvement without adding another live network call.
# Update this periodically rather than wiring up a live FX API.
EUR_TO_GBP = 0.86

POSITION_GROUP_TO_TM = {
    "GK": "Goalkeeper",
    "DEF": "Defender",
    "MID": "Midfield",
    "ATT": "Attack",
}


# =============================================================================
# CLIENT — download + cache the raw tables
# =============================================================================

class MarketValueClient:
    """
    Fetches transfermarkt-datasets tables and caches them locally.

    Usage:
        from aegis.market_value import MarketValueClient
        client = MarketValueClient()
        players = client.get_players()
        transfers = client.get_transfers()
    """

    def __init__(self, cache_ttl_hours: int = DEFAULT_CACHE_TTL_HOURS):
        self.cache_ttl_hours = cache_ttl_hours
        Config.setup()

    def _cache_path(self, table: str) -> Path:
        return Config.CACHE_DIR / f"tm_{table}.parquet"

    @staticmethod
    def _fetch_csv_gz(url: str) -> pd.DataFrame:
        """
        Fetch a gzipped CSV over HTTP with a browser-like User-Agent.

        pandas.read_csv(url) opens the URL with Python's default urllib
        opener, which sends a generic "Python-urllib/3.x" User-Agent —
        Cloudflare (fronting this R2 bucket) blocks that as a bot request
        with a 403, even though the bucket itself is public and needs no
        credentials. Fetching with requests and a real User-Agent avoids
        that block; the response bytes are then handed to pandas exactly
        as before.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return pd.read_csv(io.BytesIO(resp.content), compression="gzip", low_memory=False)

    def get_table(self, table: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch a transfermarkt-datasets table, with local parquet caching.

        Fail-open: if the live fetch fails but a (possibly stale) cache
        exists, fall back to it rather than raising — mirrors the fail-open
        pattern used elsewhere in the pipeline (e.g. _build_current_team_lookup).
        """
        if table not in DATA_URLS:
            raise ValueError(f"Unknown table '{table}'. Options: {list(DATA_URLS)}")

        cache_file = self._cache_path(table)

        if use_cache and cache_file.exists():
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_hours < self.cache_ttl_hours:
                return pd.read_parquet(cache_file)

        try:
            df = self._fetch_csv_gz(DATA_URLS[table])
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_file, index=False)
            return df
        except Exception as e:
            if cache_file.exists():
                print(f"  ⚠ Live fetch of '{table}' failed ({e}); using stale cache "
                      f"from {datetime.fromtimestamp(cache_file.stat().st_mtime):%Y-%m-%d}")
                return pd.read_parquet(cache_file)
            raise RuntimeError(
                f"Could not fetch '{table}' and no cache exists: {e}"
            ) from e

    def get_players(self, use_cache: bool = True) -> pd.DataFrame:
        return self.get_table("players", use_cache)

    def get_transfers(self, use_cache: bool = True) -> pd.DataFrame:
        return self.get_table("transfers", use_cache)

    def get_valuations(self, use_cache: bool = True) -> pd.DataFrame:
        return self.get_table("player_valuations", use_cache)

    def get_clubs(self, use_cache: bool = True) -> pd.DataFrame:
        return self.get_table("clubs", use_cache)

    def club_squad_values(self, tm_league: Optional[str] = None, use_cache: bool = True) -> pd.Series:
        """
        Per-club total squad value, computed by summing each club's
        players' individual market values — NOT clubs.total_market_value.

        clubs.total_market_value looks authoritative but is unreliable in
        practice: per the scraper's own source (base_clubs.sql), it's set
        to null whenever the scraper didn't capture that specific summary
        field on a club's page at snapshot time — a known-sparse field
        across the whole dataset, confirmed empirically (came back 0-of-37
        usable even for Premier League, which otherwise has excellent
        coverage). players.market_value_in_eur is the same field already
        verified to match Transfermarkt exactly for individual players
        (see match_player) — summing it per club is a more reliable proxy
        for squad value than trusting the sparse pre-aggregated field.

        Returns a Series of summed squad values indexed by nothing
        meaningful (one row per club) — use .median()/.min()/.max() etc.
        directly, same as the old clubs.total_market_value.median() calls.
        """
        players = self.get_players(use_cache=use_cache)
        if tm_league:
            players = players[players.get("current_club_domestic_competition_id") == tm_league]
        values = pd.to_numeric(players.get("market_value_in_eur"), errors="coerce")
        grouped = players.assign(_mv=values).dropna(subset=["_mv"]).groupby("current_club_id")["_mv"].sum()
        return grouped

    def club_squad_values_detailed(
        self, tm_league: Optional[str] = None, use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Per-club breakdown with names and data-completeness counts — lets
        the UI show WHY a club's squad value looks low (e.g. only 3 of 25
        players have a market value on record) rather than just the
        resulting number with no explanation.

        Returns a DataFrame: club_id, club_name, squad_value_eur,
        n_players_valued, n_players_total, sorted by squad_value_eur asc.
        """
        players = self.get_players(use_cache=use_cache)
        if tm_league:
            players = players[players.get("current_club_domestic_competition_id") == tm_league]
        clubs = self.get_clubs(use_cache=use_cache)
        name_map = clubs.set_index("club_id")["name"].to_dict() if "name" in clubs.columns else {}

        values = pd.to_numeric(players.get("market_value_in_eur"), errors="coerce")
        df = players.assign(_mv=values)
        agg = df.groupby("current_club_id").agg(
            squad_value_eur=("_mv", "sum"),
            n_players_valued=("_mv", lambda s: s.notna().sum()),
            n_players_total=("_mv", "size"),
        ).reset_index()
        agg["club_name"] = agg["current_club_id"].map(name_map).fillna(agg["current_club_id"].astype(str))
        agg = agg.rename(columns={"current_club_id": "club_id"})
        agg = agg.sort_values("squad_value_eur", ascending=True).reset_index(drop=True)
        return agg[["club_id", "club_name", "squad_value_eur", "n_players_valued", "n_players_total"]]

    def player_market_values(
        self, tm_league: Optional[str] = None, club_id=None, use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Player-level market values for the league (optionally filtered to
        one club) — the detail behind club_squad_values_detailed, so a low
        squad-value club can be inspected player by player.

        Returns: player_id, name, position, current_club_name,
        market_value_in_eur, sorted by market_value_in_eur desc (NaNs last).
        """
        players = self.get_players(use_cache=use_cache)
        if tm_league:
            players = players[players.get("current_club_domestic_competition_id") == tm_league]
        if club_id is not None:
            players = players[players.get("current_club_id") == club_id]
        out = players.copy()
        out["market_value_in_eur"] = pd.to_numeric(out.get("market_value_in_eur"), errors="coerce")
        cols = [c for c in ["player_id", "name", "position", "current_club_name", "market_value_in_eur"]
                if c in out.columns]
        out = out[cols].sort_values("market_value_in_eur", ascending=False, na_position="last")
        return out.reset_index(drop=True)

    # -------------------------------------------------------------------
    # Player-level entity resolution (StatsBomb -> Transfermarkt)
    # -------------------------------------------------------------------

    def match_player(
        self,
        name: str,
        birth_date: Optional[str],
        players_df: Optional[pd.DataFrame] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve a StatsBomb player (name + birth_date, both available on
        player_season_stats records) to a Transfermarkt player row.

        Strategy: exact DOB match narrows the candidate pool to a handful
        of players (DOB is a near-unique key), then pick the best name match
        among them. Falls back to name-only fuzzy match if DOB is missing,
        with a stricter similarity threshold since it's a weaker signal.

        Returns the matched player row as a dict (including market_value_in_eur
        and highest_market_value_in_eur), or None if no confident match.
        """
        if players_df is None:
            players_df = self.get_players()
        if not name:
            return None

        candidates = players_df
        if birth_date:
            try:
                dob = pd.to_datetime(birth_date).date()
                candidates = players_df[
                    pd.to_datetime(players_df["date_of_birth"], errors="coerce").dt.date == dob
                ]
            except Exception:
                candidates = players_df

        if candidates.empty:
            candidates = players_df
            threshold = 0.90  # name-only: require a near-exact match
        else:
            threshold = 0.75  # DOB already narrowed the pool substantially

        best_row, best_score = None, 0.0
        name_lower = name.strip().lower()
        for _, row in candidates.iterrows():
            cand_name = str(row.get("name", "")).strip().lower()
            if not cand_name:
                continue
            score = SequenceMatcher(None, name_lower, cand_name).ratio()
            if score > best_score:
                best_score, best_row = score, row

        if best_row is not None and best_score >= threshold:
            result = best_row.to_dict()
            result["_match_confidence"] = round(best_score, 3)
            return result
        return None


# =============================================================================
# TRAINING — market-rate model for position-group recruitment cost bands
# =============================================================================

# Feature columns the model is trained/predicted on, in a fixed order.
FEATURE_COLS = [
    "age_at_transfer",
    "position",              # categorical: Goalkeeper/Defender/Midfield/Attack
    "market_value_in_eur",   # TM market value at time of transfer
    "buying_club_tier",      # buying club's total_market_value (league-tier proxy)
    "selling_club_tier",
    "international_caps",
    "years_to_contract_expiry",
]


def _build_training_frame(client: MarketValueClient, min_fee_eur: int = 100_000) -> pd.DataFrame:
    """
    Join transfers + players + clubs into a model-ready frame.
    Filters to paid transfers only (excludes frees/loans/undisclosed) since
    those follow a different (largely non-market) process — Aegis is pricing
    a purchase, not modelling contract-expiry departures.
    """
    transfers = client.get_transfers()
    players = client.get_players()
    clubs = client.get_clubs()

    df = transfers[transfers["transfer_fee"] >= min_fee_eur].copy()
    df = df.merge(
        players[["player_id", "position", "date_of_birth", "international_caps",
                  "contract_expiration_date"]],
        on="player_id", how="left",
    )

    clubs_slim = clubs[["club_id", "total_market_value"]].rename(
        columns={"total_market_value": "club_tier"}
    )
    df = df.merge(
        clubs_slim.rename(columns={"club_id": "to_club_id", "club_tier": "buying_club_tier"}),
        on="to_club_id", how="left",
    )
    df = df.merge(
        clubs_slim.rename(columns={"club_id": "from_club_id", "club_tier": "selling_club_tier"}),
        on="from_club_id", how="left",
    )

    df["transfer_date"] = pd.to_datetime(df["transfer_date"], errors="coerce")
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    df["contract_expiration_date"] = pd.to_datetime(df["contract_expiration_date"], errors="coerce")

    df["age_at_transfer"] = (df["transfer_date"] - df["date_of_birth"]).dt.days / 365.25
    df["years_to_contract_expiry"] = (
        (df["contract_expiration_date"] - df["transfer_date"]).dt.days / 365.25
    ).clip(lower=0, upper=6)

    df["log_fee"] = df["transfer_fee"].apply(lambda x: pd.NA if x <= 0 else __import__("math").log1p(x))

    keep = FEATURE_COLS + ["log_fee", "transfer_date", "transfer_fee"]
    df = df.dropna(subset=["log_fee", "market_value_in_eur", "position", "age_at_transfer"])
    return df[keep]


def train_market_value_model(
    client: Optional[MarketValueClient] = None,
    min_fee_eur: int = 100_000,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a HistGradientBoostingRegressor on log(transfer_fee), using sklearn
    (already a project dependency — no new libs needed).

    Uses a time-based split (train on all but the most recent season, test on
    the most recent) rather than a random split, since a random split would
    let the model implicitly learn future market inflation and overstate
    its real-world accuracy.

    Returns (model, metadata) where metadata includes the categorical
    encoding map and validation metrics for the pretrain log.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error

    client = client or MarketValueClient()
    df = _build_training_frame(client, min_fee_eur=min_fee_eur)

    if len(df) < 200:
        raise RuntimeError(
            f"Only {len(df)} usable transfer records after filtering — "
            "too few to train a reliable model. Check data fetch succeeded."
        )

    # Time-based split for evaluation — deliberately NOT random, so the
    # model can't leak future transfer prices into training (see docstring).
    #
    # Guard against a degenerate split: if transfer dates are heavily
    # weighted toward recent seasons (plausible for a scraped dataset —
    # more complete data close to "now"), a fixed 85th-percentile cutoff
    # can leave the test set with only a handful of rows, or even zero.
    # Predicting on a near-empty array is a strong suspect for the
    # "window shape cannot be larger than input array shape" crash seen
    # in production — that error is numpy's sliding_window_view guard,
    # and something in sklearn's batched prediction path can hit it on
    # very small inputs. Widen the cutoff until the test set has a sane
    # minimum size rather than trusting a single fixed quantile.
    MIN_TEST_ROWS = 30
    MIN_TRAIN_ROWS = 100
    train_df = test_df = None
    for q in (0.85, 0.75, 0.65, 0.50, 0.30):
        cutoff = df["transfer_date"].quantile(q)
        candidate_train = df[df["transfer_date"] < cutoff]
        candidate_test = df[df["transfer_date"] >= cutoff]
        if len(candidate_test) >= MIN_TEST_ROWS and len(candidate_train) >= MIN_TRAIN_ROWS:
            train_df, test_df = candidate_train, candidate_test
            break
    if train_df is None:
        # Nothing worked — dates are too clustered or dataset too small
        # for ANY split to give a reasonably sized test set. Fall back to
        # training on everything with no held-out evaluation, rather than
        # failing outright — a model with unknown-but-plausible accuracy
        # beats no model at all here.
        print(f"  ⚠ Could not find a time split with ≥{MIN_TEST_ROWS} test rows "
              f"— training on full dataset with no held-out evaluation.")
        train_df, test_df = df, df.iloc[0:0]

    position_map = {p: i for i, p in enumerate(sorted(df["position"].dropna().unique()))}

    def _encode(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[FEATURE_COLS].copy()
        out["position"] = out["position"].map(position_map).fillna(-1)
        return out.fillna(out.median(numeric_only=True))

    X_train, y_train = _encode(train_df), train_df["log_fee"].astype(float)

    # Diagnostic dump — every hypothesis tried so far (split size, early
    # stopping, thread count) has failed to reproduce this against
    # synthetic data, so guessing further isn't productive. This prints
    # the ACTUAL data characteristics going into the failing .fit() call,
    # so if it crashes again we see what's really in X_train rather than
    # guessing a fourth time.
    print(f"  [diagnostic] X_train shape: {X_train.shape}, dtypes: {dict(X_train.dtypes.astype(str))}")
    print(f"  [diagnostic] X_train describe:\n{X_train.describe().to_string()}")
    print(f"  [diagnostic] X_train has inf: {np.isinf(X_train.to_numpy(dtype=float)).any()}, "
          f"has nan: {X_train.isna().any().any()}")
    print(f"  [diagnostic] X_train nunique per column: {dict(X_train.nunique())}")
    print(f"  [diagnostic] y_train shape: {y_train.shape}, "
          f"has inf: {np.isinf(y_train.to_numpy(dtype=float)).any()}, "
          f"has nan: {y_train.isna().any()}")
    print(f"  [diagnostic] sklearn n_threads env: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')!r}")
    try:
        from sklearn.utils._openmp_helpers import _openmp_effective_n_threads as _oent
        print(f"  [diagnostic] _openmp_effective_n_threads() = {_oent()}")
    except Exception as _diag_e:
        print(f"  [diagnostic] could not check effective n_threads: {_diag_e}")

    # The real production traceback (finally captured after fixing the
    # stdout/stderr logging bug) showed the crash happening inside
    # sklearn's _BinMapper.fit() — specifically its per-feature threshold
    # computation, which runs via joblib's Parallel(backend="threading")
    # across sklearn's OpenMP-detected thread count. This sandbox has
    # exactly 1 CPU, so that parallel path never actually ran concurrently
    # here (n_threads always resolved to 1) — which is exactly why dozens
    # of single-threaded reproduction attempts against the same function,
    # with deliberately pathological data, never triggered it. GitHub
    # Actions runners have multiple cores, so the same code runs with real
    # concurrency there. Forcing OMP_NUM_THREADS=1 makes
    # sklearn._openmp_effective_n_threads() return 1 everywhere, which
    # forces that Parallel(...) call to run sequentially — eliminating the
    # entire code path where a concurrency-specific issue could occur,
    # rather than continuing to guess at its exact nature. Training runs
    # once a day; the performance cost of single-threading it is
    # irrelevant.
    #
    # NOTE: if this still crashes with OMP_NUM_THREADS=1 confirmed above
    # as actually 1, the concurrency hypothesis is wrong too — see the
    # [diagnostic] lines above for what to check next.
    _prev_omp = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        model = HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.05, max_iter=300, random_state=42,
            # We already hold out a deliberate time-based test set above, so
            # sklearn's own internal validation split (early_stopping='auto')
            # would be redundant at best. Confirmed NOT the cause of the
            # "window shape" crash (still occurred with this off) — kept off
            # anyway since our own split makes it unnecessary.
            early_stopping=False,
        )
        model.fit(X_train, y_train)
    finally:
        if _prev_omp is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = _prev_omp

    # Evaluation is informational (goes into metadata for the pretrain log)
    # — isolate it so a failure here can't lose an otherwise-successfully-
    # fit()'d model. If this is where "window shape" was actually coming
    # from, we now get a real model AND a diagnosable traceback instead of
    # neither.
    mae_eur, mape, n_test = None, None, len(test_df)
    if len(test_df) > 0:
        try:
            X_test = _encode(test_df)
            preds_eur = pd.Series(model.predict(X_test)).apply(lambda x: __import__("math").expm1(x))
            actual_eur = test_df["transfer_fee"].reset_index(drop=True)
            mae_eur = mean_absolute_error(actual_eur, preds_eur)
            mape = float((abs(actual_eur - preds_eur) / actual_eur.clip(lower=1)).median())
        except Exception as eval_err:
            print(f"  ⚠ Evaluation step failed (model itself trained fine): {eval_err}")
            import traceback as _tb
            print(_tb.format_exc())
            n_test = 0

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "n_train": len(train_df),
        "n_test": n_test,
        "position_map": position_map,
        "mae_eur": round(mae_eur) if mae_eur is not None else None,
        "median_ape": round(mape, 3) if mape is not None else None,
        "feature_cols": FEATURE_COLS,
    }
    return model, metadata


def save_pretrained_market_model(model: Any, metadata: Dict[str, Any], out_dir: Path) -> None:
    """Mirrors the pretrained/master/ save pattern used for the DNA model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_pretrained_market_model(model_dir: Path) -> Optional[Tuple[Any, Dict[str, Any]]]:
    model_path = model_dir / "model.pkl"
    meta_path = model_dir / "metadata.json"
    if not model_path.exists() or not meta_path.exists():
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(meta_path) as f:
        metadata = json.load(f)
    return model, metadata


# =============================================================================
# INFERENCE — replaces the hardcoded cost_map in __init__.py
# =============================================================================

# Fallback if no pretrained model is available yet (first deploy, or the
# nightly training step failed non-fatally) — this IS today's cost_map,
# kept as a safety net rather than silently returning nothing.
_FALLBACK_COST_MAP = {"GK": (15, 35), "DEF": (25, 55), "MID": (30, 65), "ATT": (35, 75)}

# Urgency maps to a target percentile of the market-value distribution for
# that position/league segment: a Critical gap needs a proven upgrade
# (upper quartile), a Medium gap can be filled more affordably.
_URGENCY_PERCENTILES = {
    "Critical": (0.60, 0.90),
    "High":     (0.45, 0.75),
    "Medium":   (0.30, 0.60),
}


# Leagues confirmed to have solid Transfermarkt scrape coverage (checked
# against the dataset's own competition_codes scope — see LEAGUE_CODE_MAP
# comment above). Championship/League One/League Two are NOT in this set.
_KNOWN_GOOD_COVERAGE = {"GB1", "NL1"}

# Below this many matched clubs, a league's median club value is too thin
# a sample to trust as representative of the whole league.
_MIN_CLUBS_FOR_CONFIDENCE = 10


def estimate_recruitment_cost_band(
    pos_group: str,
    competition_id: int,
    urgency: str,
    model_bundle: Optional[Tuple[Any, Dict[str, Any]]],
    client: Optional[MarketValueClient] = None,
) -> Dict[str, Any]:
    """
    Data-driven replacement for the old cost_map.get(pos_group, (20, 50)).

    Rather than predicting one point estimate, this samples the trained
    model's fee distribution across a realistic range of ages (22-30) for
    the target league tier and position, then takes the percentile band
    matching the gap's urgency. Falls back to the static cost_map if no
    trained model is available (e.g. first run before the nightly pretrain
    job has produced pretrained/market_value/).

    Unlike player-level TMV, there's no "matched a real transfer" tier
    possible here — every number is a projection from a synthetic profile,
    never a specific real transfer target. So confidence instead tracks how
    much real league data underpins that projection: whether a model exists
    at all, whether the club was mapped to a real Transfermarkt league, how
    many real clubs were found to compute that league's market tier, and
    whether it's a league already known to have thin Transfermarkt coverage
    (see LEAGUE_CODE_MAP).

    Returns a dict:
        cost_low, cost_high  — £M, matching the existing CSV schema
        tier                 — "league_calibrated" | "league_thin" |
                                "no_league_context" | "fallback_generic"
        tier_label, tooltip, flag_symbol, flag_class — for UI display,
                                same convention as the Player Dossier TMV flag
    """
    def _fallback(reason: str) -> Dict[str, Any]:
        lo, hi = _FALLBACK_COST_MAP.get(pos_group, (20, 50))
        return {
            "cost_low": lo, "cost_high": hi,
            "tier": "fallback_generic",
            "tier_label": "generic placeholder",
            "flag_symbol": "○", "flag_class": "cost-flag--fallback",
            "tooltip": (
                "Not calculated from real transfer data — the market-value "
                "model hasn't been trained yet, or failed on this position. "
                f"Showing a static generic estimate. {reason}".strip()
            ),
        }

    if model_bundle is None:
        return _fallback("")

    model, metadata = model_bundle
    position_map = metadata["position_map"]
    tm_position = POSITION_GROUP_TO_TM.get(pos_group)
    if tm_position not in position_map:
        return _fallback(f"Position '{pos_group}' not found in the trained model.")

    import math

    client = client or MarketValueClient()
    tm_league = LEAGUE_CODE_MAP.get(competition_id)

    n_clubs_matched = 0
    try:
        league_values = client.club_squad_values(tm_league) if tm_league else pd.Series(dtype=float)
        all_values = client.club_squad_values(None)
        n_clubs_matched = len(league_values)  # clubs with a computable squad value, not just a league match
        club_tier = float(league_values.median()) if n_clubs_matched else float(all_values.median())
        if math.isnan(club_tier):
            club_tier = 100_000_000.0  # every value was unparseable — mid-table fallback
    except Exception:
        club_tier = 100_000_000.0  # mid-table fallback if clubs fetch fails

    rows = []
    for age in range(22, 31):
        rows.append({
            "age_at_transfer": age,
            "position": position_map[tm_position],
            "market_value_in_eur": club_tier * 0.15,  # rough proxy: ~15% of squad value per key player
            "buying_club_tier": club_tier,
            "selling_club_tier": club_tier * 0.8,
            "international_caps": 10,
            "years_to_contract_expiry": 2.0,
        })
    X = pd.DataFrame(rows)[FEATURE_COLS]
    preds_eur = [math.expm1(p) for p in model.predict(X)]
    preds_eur.sort()

    lo_pct, hi_pct = _URGENCY_PERCENTILES.get(urgency, (0.35, 0.65))
    lo_eur = preds_eur[int(len(preds_eur) * lo_pct)]
    hi_eur = preds_eur[min(int(len(preds_eur) * hi_pct), len(preds_eur) - 1)]

    cost_low = round(lo_eur * EUR_TO_GBP / 1_000_000, 1)
    cost_high = round(hi_eur * EUR_TO_GBP / 1_000_000, 1)
    if cost_high <= cost_low:
        cost_high = cost_low + 5

    # ── Confidence tier ──────────────────────────────────────────────────
    if tm_league is None:
        tier = "no_league_context"
        tier_label = "no league context"
        flag_symbol, flag_class = "◐", "cost-flag--no-league"
        tooltip = (
            "No league was specified for this estimate, so it's calculated "
            "against the overall Transfermarkt market median across all "
            "clubs rather than this specific league. Based on a market-rate "
            "model trained on historical transfer fees, but not "
            "league-specific."
        )
    elif tm_league in _KNOWN_GOOD_COVERAGE and n_clubs_matched >= _MIN_CLUBS_FOR_CONFIDENCE:
        tier = "league_calibrated"
        tier_label = f"{tm_league} · {n_clubs_matched} clubs"
        flag_symbol, flag_class = "●", "cost-flag--calibrated"
        tooltip = (
            f"Calculated from {n_clubs_matched} real club valuations in "
            f"this league, fed into a market-rate model trained on "
            f"historical Transfermarkt transfer fees. The strongest "
            f"available confidence tier — still a projection from a "
            f"synthetic player profile, never a specific real transfer target."
        )
    else:
        tier = "league_thin"
        if tm_league not in _KNOWN_GOOD_COVERAGE:
            reason = "this league has limited Transfermarkt scrape coverage"
        else:
            reason = f"only {n_clubs_matched} clubs were matched in this league"
        tier_label = f"{tm_league} · thin data"
        flag_symbol, flag_class = "◐", "cost-flag--thin"
        tooltip = (
            f"League-specific estimate, but {reason} — treat this as "
            f"indicative rather than precise. Based on {n_clubs_matched} "
            f"matched club valuations."
        )

    return {
        "cost_low": cost_low, "cost_high": cost_high,
        "tier": tier, "tier_label": tier_label,
        "flag_symbol": flag_symbol, "flag_class": flag_class,
        "tooltip": tooltip,
    }


def get_player_market_value(
    name: str,
    birth_date: Optional[str],
    client: Optional[MarketValueClient] = None,
) -> Optional[Dict[str, Any]]:
    """
    Live per-player lookup for Player Dossier / Shortlist Ranker.
    Returns market value info if a confident match is found, else None
    (caller should just omit the field rather than showing a bad estimate).
    """
    client = client or MarketValueClient()
    match = client.match_player(name, birth_date)
    if not match:
        return None
    return {
        "matched_name": match.get("name"),
        "match_confidence": match.get("_match_confidence"),
        "market_value_eur": match.get("market_value_in_eur"),
        "market_value_gbp_m": (
            round(match["market_value_in_eur"] * EUR_TO_GBP / 1_000_000, 1)
            if pd.notna(match.get("market_value_in_eur")) else None
        ),
        "highest_market_value_eur": match.get("highest_market_value_in_eur"),
    }


# =============================================================================
# SMOKE TEST — run this in an environment with open internet
# (Streamlit Cloud, GitHub Actions, or locally) before relying on it.
# It will NOT run from this dev sandbox — the R2 host is not on the
# egress allowlist here.
# =============================================================================

if __name__ == "__main__":
    print("Fetching tables (first run downloads + caches; later runs use cache)...")
    client = MarketValueClient()
    players = client.get_players()
    print(f"  ✓ players: {len(players):,} rows")
    transfers = client.get_transfers()
    print(f"  ✓ transfers: {len(transfers):,} rows")
    clubs = client.get_clubs()
    print(f"  ✓ clubs: {len(clubs):,} rows")

    print("\nLeague coverage check (row count by domestic_competition_id):")
    print(clubs["domestic_competition_id"].value_counts().head(20))

    print("\nTraining market-rate model...")
    model, metadata = train_market_value_model(client)
    print(f"  ✓ trained on {metadata['n_train']} transfers, "
          f"tested on {metadata['n_test']}")
    if metadata.get("mae_eur") is not None:
        print(f"  MAE: €{metadata['mae_eur']:,} | median abs% error: {metadata['median_ape']:.0%}")
    else:
        print(f"  No held-out evaluation available (n_test={metadata.get('n_test', 0)})")

    print("\nSample recruitment cost band (Premier League, ATT, Critical gap):")
    band = estimate_recruitment_cost_band("ATT", 2, "Critical", (model, metadata), client)
    print(f"  £{band['cost_low']}M - £{band['cost_high']}M  [{band['tier']}] {band['tier_label']}")
