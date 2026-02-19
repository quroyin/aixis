"""
Phase 2 — Stage 0: Preselection Audit
======================================
Leakage detection and redundancy pre-filtering BEFORE feature selection.

Pipeline position:
    Phase 1 (Cleaned OHLCV) → [Stage 0: Preselection Audit] → Stage 1 (Individual Evaluation)

Version: 1.0.6
Changelog:
    - v1.0.6: Fixed pandas_ta FutureWarning — now suppressed BEFORE library execution
    - v1.0.5: Fixed version mismatch, removed unused imports, moved hash to module level
    - v1.0.4: Fixed pandas FutureWarning, magic numbers, redundant checks
    - v1.0.3: Fixed "daemonic processes" crash — sequential indicator generation
    - v1.0.2: cwd-independent path resolution
    - v1.0.1: Fixed cascading ImportError
    - v1.0.0: Initial implementation
"""

import hashlib
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# SUPPRESS PANDAS_TA UPSTREAM FUTUREWARNING (BEFORE IMPORT)
# ═══════════════════════════════════════════════════════════════
# pandas_ta_classic's MFI indicator has a dtype compatibility issue
# with pandas 2.0+. This is an upstream bug, not our code.
# Suppress it globally before importing the library.

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Setting an item of incompatible dtype is deprecated",
    module="pandas_ta_classic",
)

import pandas_ta_classic as ta  # noqa: E402 (import after filter is intentional)
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════
# PATH SETUP — Must be BEFORE all project imports
# ═══════════════════════════════════════════════════════════════

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ═══════════════════════════════════════════════════════════════
# PROJECT IMPORTS
# ═══════════════════════════════════════════════════════════════

from core.audit import Auditor
from core.io import read_parquet, write_parquet, write_json, write_csv
from phases.phase2.config import Phase2Config


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Columns excluded from feature set (identifiers, raw OHLCV, target)
_NON_FEATURE_COLS_LOWER: frozenset = frozenset({
    "ticker", "date", "open", "high", "low", "close", "volume",
    "adj_close", "target_log_return",
})

# Phase 2's input contract: minimum columns required from Phase 1
_PHASE2_REQUIRED_COLS: frozenset = frozenset({
    "ticker", "date", "open", "high", "low", "close", "volume",
})

# Audit manifest sampling (full dataframe may be 600K+ rows)
_MANIFEST_SAMPLE_ROWS: int = 1000

_DEFAULT_ANOMALY_THRESHOLD: float = 0.3
_DEFAULT_REDUNDANCY_THRESHOLD: float = 0.85


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (module-level)
# ═══════════════════════════════════════════════════════════════

def compute_file_hash(path: Path) -> str:
    """
    Compute SHA-256 hash of a file for reproducibility tracking.

    Reads in 64KB chunks for memory efficiency on large parquet files.

    Args:
        path: File path to hash.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def greedy_correlation_clustering(
    corr_matrix: pd.DataFrame,
    threshold: float = _DEFAULT_REDUNDANCY_THRESHOLD,
) -> List[List[str]]:
    """
    Greedy correlation clustering for redundancy detection.

    Algorithm:
        1. Sort features alphabetically (deterministic seed order)
        2. Pick first unassigned feature as cluster seed
        3. Add all features with |correlation| > threshold
        4. Repeat until all features assigned

    Determinism guarantee:
        - Features sorted alphabetically before clustering
        - Within clusters, members added in alphabetical order
        - Same input → same clusters, always

    Args:
        corr_matrix: Square correlation matrix (features × features).
        threshold: Absolute correlation threshold for grouping.

    Returns:
        List of clusters, where each cluster is a list of feature names.
        Single-member clusters represent non-redundant features.
    """
    features = sorted(corr_matrix.columns.tolist())
    unassigned: Set[str] = set(features)
    clusters: List[List[str]] = []

    for seed_candidate in features:
        if seed_candidate not in unassigned:
            continue

        unassigned.remove(seed_candidate)
        cluster = [seed_candidate]

        for feat in sorted(unassigned):
            if abs(corr_matrix.loc[seed_candidate, feat]) > threshold:
                cluster.append(feat)

        for feat in cluster[1:]:
            unassigned.discard(feat)

        clusters.append(cluster)

    return clusters


# ═══════════════════════════════════════════════════════════════
# MAIN CLASS: PreselectionAuditor
# ═══════════════════════════════════════════════════════════════

class PreselectionAuditor:
    """
    Phase 2 Stage 0: Preselection audit for leakage and redundancy.

    Generates technical indicators from Phase 1 cleaned OHLCV data,
    detects potential look-ahead bias via correlation analysis, and
    pre-filters redundant features via greedy clustering.

    All paths resolved via config.get_resolved_*() methods — works
    regardless of the current working directory.

    Output artifacts:
        - preselection_report.json:   Full audit results + config snapshot
        - correlation_matrix.parquet: Feature cross-correlation matrix
        - candidate_features.csv:    Vetted feature list for Stage 1
    """

    def __init__(self, config: Phase2Config) -> None:
        """
        Initialize the PreselectionAuditor.

        Args:
            config: Validated Phase2Config instance (v1.0.5+).
        """
        self.config = config
        self.auditor = Auditor(
            phase=config.phase,
            output_dir=str(config.get_resolved_output_dir()),
            version=config.version,
        )

        self._leakage_flagged: List[str] = []
        self._redundancy_dropped: List[str] = []
        self._generation_errors: List[str] = []

    # ── INDICATOR GENERATION ────────────────────────────────────

    def generate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply curated pandas_ta strategy to each ticker SEQUENTIALLY.

        Why not multiprocessing.Pool:
            pandas_ta_classic internally spawns child processes for
            certain indicators. Python's multiprocessing.Pool uses
            daemon workers, and daemon processes cannot have children.
            This causes: "daemonic processes are not allowed to have children"

            Sequential per-ticker processing avoids this entirely.
            pandas_ta's own internal parallelism still utilizes multiple
            cores where the library supports it.

        Temporal integrity:
            - Each ticker processed independently (no cross-ticker leakage)
            - All indicators are backward-looking only
            - Tickers with < min_rows_per_ticker are skipped
        """
        tickers = df["ticker"].unique()
        ticker_groups = []
        skipped_tickers = []

        for ticker in sorted(tickers):  # Deterministic ordering
            ticker_df = df[df["ticker"] == ticker].copy()

            if len(ticker_df) < self.config.min_rows_per_ticker:
                skipped_tickers.append(
                    f"{ticker} ({len(ticker_df)} rows < "
                    f"{self.config.min_rows_per_ticker} min)"
                )
                continue

            ticker_groups.append((ticker, ticker_df))

        if skipped_tickers:
            print(f"[Stage 0] Skipped {len(skipped_tickers)} tickers "
                  f"(insufficient history):")
            for msg in skipped_tickers[:10]:
                print(f"  ⊘ {msg}")
            if len(skipped_tickers) > 10:
                print(f"  ... and {len(skipped_tickers) - 10} more")

        if not ticker_groups:
            raise ValueError(
                f"No tickers have >= {self.config.min_rows_per_ticker} rows. "
                f"Cannot generate indicators."
            )

        # Build the pandas_ta strategy once (reused for every ticker)
        strategy = ta.Strategy(
            name="curated_phase2",
            ta=self.config.curated_indicators,
        )

        print(f"[Stage 0] Generating indicators for {len(ticker_groups)} "
              f"tickers (sequential — pandas_ta handles internal parallelism)...")

        valid_results: List[pd.DataFrame] = []

        for ticker, ticker_df in tqdm(
            ticker_groups,
            desc="Indicator generation",
            unit="ticker",
        ):
            try:
                # pandas_ta .strategy() appends columns in-place
                # FutureWarning suppressed globally at module import
                ticker_df.ta.strategy(strategy)

                # Remove candlestick pattern columns if configured
                if self.config.exclude_candlestick:
                    candlestick_cols = [
                        col for col in ticker_df.columns
                        if col.startswith("CDL_")
                    ]
                    ticker_df = ticker_df.drop(
                        columns=candlestick_cols, errors="ignore"
                    )

                valid_results.append(ticker_df)

            except Exception as e:
                self._generation_errors.append(ticker)
                # Only warn for first 5 failures to avoid log spam
                if len(self._generation_errors) <= 5:
                    warnings.warn(
                        f"[Stage 0] pandas_ta failed for '{ticker}': {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        if not valid_results:
            raise RuntimeError(
                "All tickers failed indicator generation. "
                "Check pandas_ta_classic installation and input data quality."
            )

        if self._generation_errors:
            print(f"[Stage 0] ⚠ {len(self._generation_errors)} tickers "
                  f"failed: {self._generation_errors[:10]}")

        combined = pd.concat(valid_results, axis=0, ignore_index=False)

        # Sort for deterministic output ordering
        sort_cols = []
        if "ticker" in combined.columns:
            sort_cols.append("ticker")
        if "date" in combined.columns:
            sort_cols.append("date")
        if sort_cols:
            combined = combined.sort_values(sort_cols).reset_index(drop=True)

        n_indicators = len(self._get_feature_columns(combined))
        n_success = len(valid_results)
        n_failed = len(self._generation_errors)
        print(f"[Stage 0] Generated {n_indicators} indicator columns "
              f"across {n_success} tickers "
              f"({n_failed} failed)")

        return combined

    # ── TARGET COMPUTATION ──────────────────────────────────────

    def compute_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute forward log return target per-ticker.

        Formula: y_t = ln(P_{t+h} / P_t)
        Computed per-ticker to prevent cross-ticker contamination.
        Last h rows per ticker → NaN (never filled, dropped at analysis).
        """
        df = df.copy()
        horizon = self.config.target_horizon
        price_col = self.config.price_col

        if price_col not in df.columns:
            raise ValueError(
                f"Price column '{price_col}' not found. "
                f"Available: {sorted(df.columns.tolist())}"
            )

        def _compute_ticker_target(group: pd.DataFrame) -> pd.Series:
            prices = group[price_col]

            if (prices <= 0).any():
                non_pos = prices[prices <= 0]
                raise ValueError(
                    f"Non-positive prices in ticker "
                    f"'{group['ticker'].iloc[0]}': {non_pos.head().to_dict()}"
                )

            future_price = prices.shift(-horizon)
            log_ret = np.log(future_price / prices)

            # Temporal integrity assertion
            assert log_ret.iloc[-horizon:].isna().all(), (
                f"TEMPORAL INTEGRITY VIOLATION in ticker "
                f"'{group['ticker'].iloc[0]}': last {horizon} rows must be NaN"
            )
            return log_ret

        # Fixed: Added include_groups=False to suppress pandas FutureWarning
        df["target_log_return"] = df.groupby(
            "ticker", group_keys=False
        ).apply(_compute_ticker_target, include_groups=False)

        n_valid = df["target_log_return"].notna().sum()
        n_nan = df["target_log_return"].isna().sum()
        print(f"[Stage 0] Target computed: {n_valid} valid, "
              f"{n_nan} NaN (temporal boundary)")

        return df

    # ── LEAKAGE DETECTION ───────────────────────────────────────

    def detect_leakage_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect features with suspiciously high Spearman correlation to target.

        |ρ| > anomaly_threshold flags a feature as a leakage suspect.
        Known leaky indicators (e.g. DPO) are pre-flagged regardless.
        """
        feature_cols = self._get_feature_columns(df)
        threshold = self.config.anomaly_threshold

        analysis_df = df.dropna(subset=["target_log_return"])

        if len(analysis_df) == 0:
            raise ValueError("No rows with valid target after dropping NaN.")

        target = analysis_df["target_log_return"]

        correlations: Dict[str, float] = {}
        flagged_features: List[str] = []
        computation_failures: List[str] = []

        for col in sorted(feature_cols):
            col_data = analysis_df[col]

            if col_data.isna().all() or col_data.nunique() < 2:
                computation_failures.append(col)
                continue

            valid_mask = col_data.notna() & target.notna()
            if valid_mask.sum() < 30:
                computation_failures.append(col)
                continue

            spearman_corr = col_data[valid_mask].corr(
                target[valid_mask], method="spearman"
            )

            if np.isnan(spearman_corr):
                computation_failures.append(col)
                continue

            correlations[col] = float(spearman_corr)

            if abs(spearman_corr) > threshold:
                flagged_features.append(col)

        # Add known leaky indicators
        known_leaky_found: List[str] = []
        for leaky_name in self.config.exclude_known_leaky:
            leaky_cols = [
                col for col in feature_cols
                if leaky_name.upper() in col.upper()
            ]
            for col in leaky_cols:
                if col not in flagged_features:
                    flagged_features.append(col)
                known_leaky_found.append(col)

        # Sort by absolute correlation (most suspicious first)
        flagged_features = sorted(
            flagged_features,
            key=lambda f: abs(correlations.get(f, 0.0)),
            reverse=True,
        )

        self._leakage_flagged = flagged_features

        print(f"\n[Stage 0] Leakage Detection Report:")
        print(f"  Features analyzed     : {len(correlations)}")
        print(f"  Computation failures  : {len(computation_failures)}")
        print(f"  Anomaly threshold     : |ρ| > {threshold}")
        print(f"  Flagged (suspicious)  : {len(flagged_features)}")
        print(f"  Known leaky found     : {len(known_leaky_found)}")

        if flagged_features:
            print(f"  Top flagged features:")
            for feat in flagged_features[:10]:
                corr_val = correlations.get(feat, float("nan"))
                source = " [KNOWN LEAKY]" if feat in known_leaky_found else ""
                print(f"    ⚠ {feat:<40s}  ρ = {corr_val:+.4f}{source}")

        return {
            "flagged_features": flagged_features,
            "correlations": correlations,
            "known_leaky_found": known_leaky_found,
            "computation_failures": computation_failures,
            "anomaly_threshold": threshold,
        }

    # ── REDUNDANCY CLUSTERING ───────────────────────────────────

    def cluster_redundant_features(
        self,
        df: pd.DataFrame,
        threshold: float = _DEFAULT_REDUNDANCY_THRESHOLD,
    ) -> Dict[str, Any]:
        """Group redundant features via greedy correlation clustering."""
        feature_cols = sorted(self._get_feature_columns(df))

        analysis_df = df[feature_cols].dropna(how="all")

        if len(analysis_df) == 0:
            raise ValueError("No valid rows for correlation computation.")

        print(f"\n[Stage 0] Computing {len(feature_cols)}×{len(feature_cols)} "
              f"correlation matrix...")

        corr_matrix = analysis_df[feature_cols].corr(method="spearman")
        corr_matrix = corr_matrix.fillna(0.0)

        clusters = greedy_correlation_clustering(corr_matrix, threshold)

        representatives: List[str] = []
        dropped_as_redundant: List[str] = []

        for cluster in clusters:
            representatives.append(cluster[0])
            for feat in cluster[1:]:
                dropped_as_redundant.append(feat)

        self._redundancy_dropped = dropped_as_redundant

        n_multi = sum(1 for c in clusters if len(c) > 1)
        print(f"[Stage 0] Redundancy Clustering Report:")
        print(f"  Total features       : {len(feature_cols)}")
        print(f"  Correlation threshold : |r| > {threshold}")
        print(f"  Clusters formed      : {len(clusters)}")
        print(f"  Multi-member clusters: {n_multi}")
        print(f"  Representatives kept : {len(representatives)}")
        print(f"  Dropped as redundant : {len(dropped_as_redundant)}")

        if n_multi > 0:
            print(f"  Largest clusters:")
            sorted_clusters = sorted(clusters, key=len, reverse=True)
            for cluster in sorted_clusters[:5]:
                if len(cluster) > 1:
                    print(f"    [{len(cluster)} members] "
                          f"rep='{cluster[0]}' ← {cluster[1:]}")

        return {
            "clusters": clusters,
            "representatives": representatives,
            "dropped_as_redundant": dropped_as_redundant,
            "correlation_matrix": corr_matrix,
            "threshold": threshold,
        }

    # ── MAIN ENTRY POINT ────────────────────────────────────────

    def run_audit(self) -> Dict[str, Any]:
        """Execute the full preselection audit pipeline."""
        t_start = time.time()
        output_dir = self.config.get_resolved_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        self.auditor.start()

        print("=" * 70)
        print("Phase 2 — Stage 0: Preselection Audit")
        print("=" * 70)
        print(f"  Config version : {self.config.version}")
        print(f"  PROJECT_ROOT   : {PROJECT_ROOT}")
        print(f"  Input          : {self.config.get_resolved_input_path()}")
        print(f"  Output dir     : {output_dir}")
        print(f"  Target horizon : {self.config.target_horizon} days")
        print(f"  Indicators     : {self.config.get_indicator_count()} curated")

        # ── 1. Load and validate Phase 1 data ───────────────────
        print(f"\n[Stage 0] Loading Phase 1 data...")
        input_path = self.config.get_resolved_input_path()
        input_hash = compute_file_hash(input_path)

        raw_df = read_parquet(input_path)

        # Validate Phase 2's input contract
        missing = _PHASE2_REQUIRED_COLS - set(raw_df.columns)
        if missing:
            raise ValueError(
                f"Phase 1 output missing columns required by Phase 2: "
                f"{sorted(missing)}\n"
                f"  Available: {sorted(raw_df.columns.tolist())}\n"
                f"  File: {input_path}"
            )

        self.auditor.record_input(str(input_path), raw_df)

        n_tickers = raw_df["ticker"].unique()
        print(f"  Loaded: {len(raw_df)} rows, {n_tickers} tickers")
        print(f"  Input SHA-256: {input_hash[:16]}...")

        # ── 2. Generate indicators ──────────────────────────────
        print(f"\n{'─' * 50}")
        print(f"[Stage 0] Step 1/4: Indicator Generation")
        print(f"{'─' * 50}")

        indicator_df = self.generate_indicators(raw_df)

        # ── 3. Compute target ───────────────────────────────────
        print(f"\n{'─' * 50}")
        print(f"[Stage 0] Step 2/4: Target Computation")
        print(f"{'─' * 50}")

        target_df = self.compute_target(indicator_df)

        # ── 4. Leakage detection ────────────────────────────────
        print(f"\n{'─' * 50}")
        print(f"[Stage 0] Step 3/4: Leakage Detection")
        print(f"{'─' * 50}")

        leakage_report = self.detect_leakage_anomalies(target_df)

        # ── 5. Redundancy clustering ────────────────────────────
        print(f"\n{'─' * 50}")
        print(f"[Stage 0] Step 4/4: Redundancy Clustering")
        print(f"{'─' * 50}")

        redundancy_report = self.cluster_redundant_features(target_df)
        corr_matrix = redundancy_report.pop("correlation_matrix")

        # ── 6. Compile candidate feature list ───────────────────
        representatives = set(redundancy_report["representatives"])
        flagged = set(leakage_report["flagged_features"])

        candidates = sorted(representatives - flagged)

        removed_by_leakage = sorted(representatives & flagged)
        removed_by_redundancy = sorted(
            set(self._redundancy_dropped) - flagged
        )

        print(f"\n{'═' * 70}")
        print(f"[Stage 0] CANDIDATE FEATURE SUMMARY")
        print(f"{'═' * 70}")

        all_features = sorted(self._get_feature_columns(target_df))
        print(f"  Total generated features    : {len(all_features)}")
        print(f"  Removed (leakage flagged)   : {len(leakage_report['flagged_features'])}")
        print(f"  Removed (redundancy)        : {len(self._redundancy_dropped)}")
        print(f"  Removed (overlap leak+red)  : {len(removed_by_leakage)}")
        print(f"  ─────────────────────────────")
        print(f"  Candidates for Stage 1      : {len(candidates)}")

        # ── 7. Save output artifacts ────────────────────────────
        print(f"\n[Stage 0] Saving artifacts...")

        candidate_path = output_dir / "candidate_features.csv"
        candidate_df = pd.DataFrame({
            "feature": candidates,
            "status": "candidate",
            "source": "preselection_audit_v" + self.config.version,
        })
        write_csv(candidate_df, candidate_path)
        print(f"  ✓ {candidate_path} ({len(candidates)} features)")

        corr_path = output_dir / "correlation_matrix.parquet"
        write_parquet(corr_matrix, corr_path, compression=self.config.compression)
        print(f"  ✓ {corr_path} ({corr_matrix.shape[0]}×{corr_matrix.shape[1]})")

        t_elapsed = time.time() - t_start

        report = {
            "stage": "0_preselection_audit",
            "version": self.config.version,
            "config_snapshot": self.config.to_snapshot(),
            "input": {
                "path": str(input_path),
                "sha256": input_hash,
                "rows": len(raw_df),
                "tickers": n_tickers,
            },
            "indicator_generation": {
                "total_indicators_generated": len(all_features),
                "tickers_processed": len(raw_df["ticker"].unique()) - len(self._generation_errors),
                "tickers_failed": self._generation_errors,
                "curated_strategy_used": self.config.use_curated_strategy,
                "indicator_definitions": self.config.get_indicator_count(),
            },
            "leakage_detection": {
                "anomaly_threshold": leakage_report["anomaly_threshold"],
                "features_analyzed": len(leakage_report["correlations"]),
                "features_flagged": leakage_report["flagged_features"],
                "known_leaky_found": leakage_report["known_leaky_found"],
                "computation_failures": leakage_report["computation_failures"],
                "top_correlations": dict(sorted(
                    leakage_report["correlations"].items(),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:20]),
            },
            "redundancy_clustering": {
                "threshold": redundancy_report["threshold"],
                "total_clusters": len(redundancy_report["clusters"]),
                "multi_member_clusters": sum(
                    1 for c in redundancy_report["clusters"] if len(c) > 1
                ),
                "representatives": redundancy_report["representatives"],
                "dropped_as_redundant": redundancy_report["dropped_as_redundant"],
                "cluster_details": [
                    {"representative": c[0], "members": c, "size": len(c)}
                    for c in redundancy_report["clusters"]
                    if len(c) > 1
                ],
            },
            "output": {
                "candidates": candidates,
                "candidate_count": len(candidates),
                "removed_by_leakage": removed_by_leakage,
                "removed_by_redundancy": removed_by_redundancy,
                "total_removed": len(all_features) - len(candidates),
            },
            "runtime_seconds": round(t_elapsed, 2),
        }

        report_path = output_dir / "preselection_report.json"
        write_json(report, report_path)
        print(f"  ✓ {report_path}")

        # ── 8. Complete audit lifecycle ─────────────────────────
        # Sample for audit manifest — full dataset may be 600K+ rows
        output_df = target_df.head(_MANIFEST_SAMPLE_ROWS)
        self.auditor.record_output(output_df, self.config.to_snapshot())
        self.auditor.success()

        print(f"\n{'═' * 70}")
        print(f"[Stage 0] Preselection audit COMPLETE in {t_elapsed:.1f}s")
        print(f"  → {len(candidates)} candidate features ready for Stage 1")
        print(f"{'═' * 70}")

        return report

    # ── PRIVATE HELPERS ───────────────────────────────���─────────

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Extract feature column names (excludes identifiers, OHLCV, target).
        
        Fixed: Removed redundant check — lowercase comparison suffices.
        """
        return sorted([
            col for col in df.columns
            if col.lower() not in _NON_FEATURE_COLS_LOWER
        ])


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from phases.phase2.config import get_phase2_config

    config = get_phase2_config()

    auditor = PreselectionAuditor(config)
    report = auditor.run_audit()

    print(f"\nCandidate features ({report['output']['candidate_count']}):")
    for feat in report["output"]["candidates"][:20]:
        print(f"  ✓ {feat}")
    if report['output']['candidate_count'] > 20:
        print(f"  ... and {report['output']['candidate_count'] - 20} more")

    if report["leakage_detection"]["features_flagged"]:
        print(f"\n⚠ Leakage-flagged features "
              f"({len(report['leakage_detection']['features_flagged'])}):")
        for feat in report["leakage_detection"]["features_flagged"][:10]:
            corr = report["leakage_detection"]["top_correlations"].get(feat, "N/A")
            print(f"  ✗ {feat}  (ρ = {corr})")
