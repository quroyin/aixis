# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Phase 2 · Stage 4 — Walk-Forward Stability Validation                     ║
║                                                                            ║
║  Pipeline position:                                                        ║
║    Stage 0 (Preselection) → Stage 1 (Univariate) → Stage 2 (SFFS)         ║
║      → Stage 3 (Interactions) → [Stage 4: Stability Validation]            ║
║                                                                            ║
║  Purpose:                                                                  ║
║    Verify that the final 7-feature model performs CONSISTENTLY over time,   ║
║    not just on average. A model with high average IC but wild variance      ║
║    across time periods is unreliable for live trading. Walk-forward         ║
║    analysis simulates how the model would have been retrained and           ║
║    deployed sequentially through history.                                   ║
║                                                                            ║
║  Method: Walk-Forward Validation (Rolling Window)                          ║
║    1. Load final 7 features from interaction_report.json.                  ║
║    2. Regenerate 5 base TA features, then create 2 interaction columns.    ║
║    3. Flatten the panel (all tickers stacked, sorted by date then ticker). ║
║    4. Roll a sliding window through time:                                  ║
║       ┌────────────────────────┬───────────┐                               ║
║       │   Train (500 days)     │Test (125d) │                              ║
║       └────────────────────────┴───────────┘                               ║
║                                 ──step (125d)──►                           ║
║       ┌───────────────���────────┬───────────┐                               ║
║       │   Train (500 days)     │Test (125d) │                              ║
║       └────────────────────────┴───────────┘                               ║
║    5. For each window: fit Ridge(α=1.0) on train, predict on test,         ║
║       compute Spearman IC on test predictions.                             ║
║    6. Aggregate: Mean IC, Std IC, Hit Rate, Min/Max, Decay Check.          ║
║                                                                            ║
║  Walk-Forward vs TimeSeriesSplit:                                           ║
║    Stages 1-3 used TimeSeriesSplit (sklearn) — a single static split       ║
║    of the full dataset. Walk-forward is different: it simulates actual     ║
║    deployment by SLIDING through time, so each test window contains        ║
║    data the model has never seen AND that comes strictly after training.   ║
║    This is the gold standard for temporal validation in quant finance.     ║
║                                                                            ║
║  Inputs  (from disk — fully decoupled):                                    ║
║    • artifacts/phase_2_features/interaction_report.json       (Stage 3)    ║
║    • artifacts/phase_1_data/merged_data.parquet                (Phase 1)   ║
║                                                                            ║
║  Outputs:                                                                  ║
║    • artifacts/phase_2_features/stability_report.json                      ║
║    • artifacts/phase_2_features/manifest.json  (via Auditor)               ║
║                                                                            ║
║  Quality Gates:                                                            ║
║    ✓ Decoupled — reads only disk artifacts from prior stages               ║
║    ✓ Deterministic — Ridge has no random state; windows are date-ordered   ║
║    ✓ Atomic — writes to .tmp then renames                                  ║
║    ✓ Efficient — TA regenerated ONCE; walk-forward uses column slicing     ║
║    ✓ Traceable — per-window stats with exact date boundaries               ║
║    ✓ Temporal — strict train < test ordering; no future leakage            ║
║    ✓ Graceful — last window may be smaller; handled explicitly             ║
║                                                                            ║
║  TA Generation Note:                                                       ║
║    Sequential per-ticker (pandas_ta daemon constraint — see Stage 0).      ║
║    Only indicator specs producing the 5 base columns are generated.        ║
║    The 2 interaction columns are computed via element-wise product          ║
║    AFTER TA generation (they are not TA indicators).                        ║
║                                                                            ║
║  Version: 1.0.0                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─── Imports ────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# SUPPRESS PANDAS_TA UPSTREAM FUTUREWARNING (BEFORE IMPORT)
# ═══════════════════════════════════════════════════════════════
# Pattern from 1_preselection_audit.py lines 37-42.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Setting an item of incompatible dtype is deprecated",
    module="pandas_ta_classic",
)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from tqdm import tqdm

# ─── Project path bootstrap ────────────────────────────────────────────────
# Pattern from 1_preselection_audit.py lines 51-55.
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas_ta_classic as ta                         # noqa: E402
from core.audit import Auditor                         # noqa: E402
from core.io import write_json                         # noqa: E402
from phases.phase2.config import Phase2Config          # noqa: E402


# ─── Constants ──────────────────────────────────────────────────────────────
VERSION = "1.0.0"
STAGE_NAME = "4_stability_validation"

PHASE1_ARTIFACT = PROJECT_ROOT / "artifacts" / "phase_1_data" / "merged_data.parquet"
PHASE2_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "phase_2_features"
INTERACTION_REPORT_JSON = PHASE2_ARTIFACT_DIR / "interaction_report.json"
OUTPUT_JSON = PHASE2_ARTIFACT_DIR / "stability_report.json"

REQUIRED_COLUMNS = {"ticker", "date", "open", "high", "low", "close", "volume"}

# Matched exactly to Stage 1/2/3 hyperparameters
RIDGE_ALPHA = 1.0
FORWARD_DAYS = 5

# Walk-forward window parameters (in UNIQUE trading days)
TRAIN_DAYS = 500     # ~2 years of trading days
TEST_DAYS = 125      # ~6 months of trading days
STEP_DAYS = 125      # ~6 months step (non-overlapping test windows)

# Interaction column prefix — must match Stage 3 naming convention
_IX_PREFIX = "IX_"
_IX_SEPARATOR = "_x_"


# ─── Indicator-to-column mapping ───────────────────────────────────────────
# Reused from Stage 1/2/3.
_KIND_TO_PREFIX: Dict[str, List[str]] = {
    "adx": ["ADX_", "DMP_", "DMN_"],
    "chop": ["CHOP_"],
    "aroon": ["AROOND_", "AROONU_", "AROONOSC_"],
    "supertrend": ["SUPERT_", "SUPERTd_", "SUPERTl_", "SUPERTs_"],
    "rsi": ["RSI_"],
    "roc": ["ROC_"],
    "macd": ["MACD_", "MACDh_", "MACDs_"],
    "mom": ["MOM_"],
    "willr": ["WILLR_"],
    "natr": ["NATR_"],
    "bbands": ["BBL_", "BBM_", "BBU_", "BBB_", "BBP_"],
    "atr": ["ATRr_"],
    "rvi": ["RVI_"],
    "cmf": ["CMF_"],
    "mfi": ["MFI_"],
    "obv": ["OBV"],
    "sma": ["SMA_"],
    "ema": ["EMA_"],
    "zscore": ["ZS_"],
    "skew": ["SKEW_"],
    "stdev": ["STDEV_"],
    "log_return": ["LOGRET_"],
    "percent_return": ["PCTRET_"],
}


# ─── Main Class ─────────────────────────────────────────────────────────────
class StabilityValidator:
    """Walk-forward stability validation for the final feature set.

    Slides a train/test window through the date axis to verify that the
    7-feature model performs consistently across all time periods, not
    just on average via a single CV split.
    """

    def __init__(self, config: Phase2Config) -> None:
        self.config = config

        # Auditor: exact contract from 1_preselection_audit.py lines 189-193
        self.auditor = Auditor(
            phase=config.phase,
            output_dir=str(config.get_resolved_output_dir()),
            version=config.version,
        )

        self._final_features: Optional[List[str]] = None
        self._base_features: Optional[List[str]] = None
        self._interaction_features: Optional[List[str]] = None
        self._stage3_score: Optional[float] = None
        self._filtered_specs: Optional[List[Dict[str, Any]]] = None

    # ── 1. Load Stage 3 (Interaction) results ───────────────────────────────
    def load_stage3_results(self) -> List[str]:
        """Load interaction_report.json and extract the final 7 features.

        Separates features into base (TA indicators) and interactions
        (product columns) for correct regeneration.

        Returns:
            List of all 7 final feature names.

        Raises:
            FileNotFoundError: If Stage 3 output is missing.
            AssertionError: If report is malformed.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 4] Loading Stage 3 (Interaction) Results...")
        print(f"{'='*72}")

        if not INTERACTION_REPORT_JSON.exists():
            raise FileNotFoundError(
                f"[FATAL] Stage 3 output not found: {INTERACTION_REPORT_JSON}\n"
                f"       Run 4_interaction_terms.py before this script."
            )

        with open(INTERACTION_REPORT_JSON, "r", encoding="utf-8") as f:
            stage3_data = json.load(f)

        assert "final_feature_set" in stage3_data, (
            f"[SCHEMA VIOLATION] interaction_report.json missing "
            f"'final_feature_set' key.\n"
            f"  Found keys: {list(stage3_data.keys())}"
        )

        final_features = stage3_data["final_feature_set"]
        assert isinstance(final_features, list) and len(final_features) > 0, (
            f"[SCHEMA VIOLATION] final_feature_set must be non-empty list, "
            f"got: {final_features}"
        )

        # Separate base features (TA indicators) from interactions (IX_ prefix)
        base_features = [f for f in final_features if not f.startswith(_IX_PREFIX)]
        interaction_features = [f for f in final_features if f.startswith(_IX_PREFIX)]

        self._final_features = final_features
        self._base_features = base_features
        self._interaction_features = interaction_features
        self._stage3_score = stage3_data.get("final_score_mean_ic")

        print(f"  ✓ Final feature set ({len(final_features)} features):")
        print(f"    Base features ({len(base_features)}):")
        for i, feat in enumerate(base_features, 1):
            print(f"      {i}. {feat}")
        if interaction_features:
            print(f"    Interaction features ({len(interaction_features)}):")
            for i, ix in enumerate(interaction_features, 1):
                parts = ix.replace(_IX_PREFIX, "").split(_IX_SEPARATOR)
                desc = f"{parts[0]} × {parts[1]}" if len(parts) == 2 else ix
                print(f"      +{i}. {ix}  ({desc})")
        if self._stage3_score is not None:
            print(f"  ✓ Stage 3 reported Mean IC: {self._stage3_score:.6f}")

        return final_features

    # ── 2. Load Phase 1 data ────────────────────────────────────────────────
    def load_phase1_data(self) -> pd.DataFrame:
        """Load merged OHLCV and validate schema.

        Returns:
            pd.DataFrame: Raw OHLCV sorted by (ticker, date).
        """
        print(f"\n{'='*72}")
        print(f"[Stage 4] Loading Phase 1 Data...")
        print(f"{'='*72}")

        if not PHASE1_ARTIFACT.exists():
            raise FileNotFoundError(
                f"[FATAL] Phase 1 artifact not found: {PHASE1_ARTIFACT}\n"
                f"       Run Phase 1 before Phase 2."
            )

        t0 = perf_counter()
        df = pd.read_parquet(PHASE1_ARTIFACT)
        t_load = perf_counter() - t0

        df.columns = df.columns.str.lower().str.strip()
        print(f"  ✓ Loaded merged_data.parquet: {df.shape[0]:,} rows × "
              f"{df.shape[1]} cols  ({t_load:.2f}s)")

        missing = REQUIRED_COLUMNS - set(df.columns)
        assert len(missing) == 0, (
            f"[SCHEMA VIOLATION] Missing required columns: {missing}\n"
            f"  Present columns: {sorted(df.columns.tolist())}"
        )
        print(f"  ✓ Schema validated: {sorted(REQUIRED_COLUMNS)} all present")

        # Auditor lifecycle: start + record_input
        self.auditor.start()
        self.auditor.record_input(str(PHASE1_ARTIFACT), df)

        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        return df

    # ── 3. Compute target ───────────────────────────────────────────────────
    @staticmethod
    def compute_target(df: pd.DataFrame) -> pd.Series:
        """Compute 5-day forward log return per ticker.

        Identical formula to Stage 1/2/3:  target_t = ln(close_{t+5} / close_t)

        Args:
            df: DataFrame sorted by (ticker, date) with 'close' column.

        Returns:
            pd.Series: Named 'target_log_return', index-aligned to df.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 4] Computing Target: {FORWARD_DAYS}-day forward log return")
        print(f"{'='*72}")

        t0 = perf_counter()

        future_close = df.groupby("ticker")["close"].shift(-FORWARD_DAYS)
        target = np.log(future_close / df["close"])
        target.name = "target_log_return"

        n_valid = target.notna().sum()
        n_nan = target.isna().sum()
        t_elapsed = perf_counter() - t0
        print(f"  ✓ Target computed: {n_valid:,} valid / {n_nan:,} NaN  "
              f"({t_elapsed:.3f}s)")

        return target

    # ── 4. Regenerate base features + create interactions ───────────────────
    def regenerate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Regenerate the 5 base TA features, then create 2 interaction columns.

        Two-phase feature construction:
          Phase A: pandas_ta generates base TA indicators (ADX_14, CMF_20, etc.)
          Phase B: Interaction columns computed as element-wise products of
                   base features (IX_ADX_14_x_CMF_20 = ADX_14 * CMF_20).

        Uses _KIND_TO_PREFIX filtering. Sequential per-ticker.

        Args:
            df: Raw OHLCV DataFrame with target column.

        Returns:
            pd.DataFrame: Input df augmented with all 7 feature columns.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 4] Regenerating Features")
        print(f"{'='*72}")

        # ═══════════════════════════════════════════════════════════════════
        # Phase A: Regenerate 5 base TA features
        # ═══════════════════════════════════════════════════════════════════
        t0 = perf_counter()

        all_specs = self.config.curated_indicators
        base_set = set(self._base_features)

        filtered_specs: List[Dict[str, Any]] = []
        for spec in all_specs:
            kind = spec.get("kind", "")
            prefixes = _KIND_TO_PREFIX.get(kind, [])
            produces_needed = any(
                any(feat.startswith(prefix) for feat in base_set)
                for prefix in prefixes
            )
            if produces_needed:
                filtered_specs.append(spec)

        self._filtered_specs = filtered_specs
        print(f"  ℹ Phase A: Filtered {len(all_specs)} curated specs → "
              f"{len(filtered_specs)} needed for "
              f"{len(self._base_features)} base features")

        tickers = sorted(df["ticker"].unique())
        print(f"  ℹ Processing {len(tickers)} tickers with "
              f"{len(filtered_specs)} indicator specs (sequential)")

        ta_strategy = ta.Strategy(
            name="Phase2_Stability_Base",
            ta=filtered_specs,
        )

        frames: List[pd.DataFrame] = []
        initial_cols = set(df.columns)
        generation_errors: List[str] = []

        for ticker_id in tqdm(
            tickers, desc="  [TA Generation]", unit="ticker"
        ):
            ticker_df = df[df["ticker"] == ticker_id].copy()

            try:
                ticker_df.ta.strategy(ta_strategy, verbose=False)
                frames.append(ticker_df)
            except Exception as e:
                generation_errors.append(str(ticker_id))
                if len(generation_errors) <= 5:
                    warnings.warn(
                        f"[Stage 4] pandas_ta failed for '{ticker_id}': {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        if not frames:
            raise RuntimeError(
                "All tickers failed indicator generation. "
                "Check pandas_ta_classic installation and input data quality."
            )

        if generation_errors:
            print(f"  ⚠ {len(generation_errors)} tickers failed: "
                  f"{generation_errors[:10]}")

        result_df = pd.concat(frames, axis=0, ignore_index=True)
        new_cols = set(result_df.columns) - initial_cols
        print(f"  ✓ TA strategy generated {len(new_cols)} new columns")

        # Verify base columns
        generated_cols = set(result_df.columns)
        matched_base = [c for c in self._base_features if c in generated_cols]
        missing_base = [c for c in self._base_features if c not in generated_cols]

        # Case-insensitive fallback
        if missing_base:
            col_map = {c.upper(): c for c in generated_cols}
            for m in missing_base[:]:
                if m.upper() in col_map:
                    actual = col_map[m.upper()]
                    result_df.rename(columns={actual: m}, inplace=True)
                    matched_base.append(m)
                    missing_base.remove(m)

        if missing_base:
            raise RuntimeError(
                f"[FATAL] Cannot regenerate base features: {missing_base}\n"
                f"  Generated columns: {sorted(new_cols)}"
            )

        print(f"  ✓ Matched {len(matched_base)}/{len(self._base_features)} "
              f"base features")

        t_ta = perf_counter() - t0

        # ═══════════════════════════════════════════════════════════════════
        # Phase B: Create interaction columns from base features
        # ═══════════════════════════════════════════════════════════════════
        t_ix = perf_counter()

        for ix_name in self._interaction_features:
            # Parse IX_ADX_14_x_CMF_20 → ("ADX_14", "CMF_20")
            inner = ix_name[len(_IX_PREFIX):]         # "ADX_14_x_CMF_20"
            parts = inner.split(_IX_SEPARATOR)         # ["ADX_14", "CMF_20"]

            if len(parts) != 2:
                raise ValueError(
                    f"[FATAL] Cannot parse interaction column '{ix_name}'.\n"
                    f"  Expected format: IX_<featureA>_x_<featureB>\n"
                    f"  Got parts: {parts}"
                )

            feat_a, feat_b = parts

            if feat_a not in result_df.columns:
                raise RuntimeError(
                    f"[FATAL] Interaction '{ix_name}' requires base feature "
                    f"'{feat_a}' which is not present."
                )
            if feat_b not in result_df.columns:
                raise RuntimeError(
                    f"[FATAL] Interaction '{ix_name}' requires base feature "
                    f"'{feat_b}' which is not present."
                )

            result_df[ix_name] = result_df[feat_a] * result_df[feat_b]

        t_ix_elapsed = perf_counter() - t_ix

        if self._interaction_features:
            print(f"  ✓ Phase B: Created {len(self._interaction_features)} "
                  f"interaction columns ({t_ix_elapsed:.3f}s)")
        else:
            print(f"  ℹ Phase B: No interaction columns to create")

        # Final verification: all 7 features present
        all_present = [
            f for f in self._final_features
            if f in result_df.columns
        ]
        all_missing = [
            f for f in self._final_features
            if f not in result_df.columns
        ]

        if all_missing:
            raise RuntimeError(
                f"[FATAL] Final feature set incomplete. Missing: {all_missing}"
            )

        print(f"  ✓ All {len(self._final_features)} features verified present")
        print(f"  ✓ Feature regeneration complete ({perf_counter() - t0:.1f}s)")

        return result_df

    # ── 5. Walk-Forward Validation ──────────────────────────────────────────
    def run_walk_forward(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute walk-forward validation with rolling train/test windows.

        Window mechanics (using UNIQUE trading dates across all tickers):
          - Extract the sorted unique date array from the dataset.
          - Slide a window: train on dates[i:i+500], test on dates[i+500:i+625].
          - Step forward by 125 dates.
          - For each window, select ALL rows (all tickers) within those dates.

        This is a PANEL walk-forward: at each step, we train on ALL tickers
        in the train period and test on ALL tickers in the test period.
        This matches the cross-sectional Ridge approach used in Stages 1-3.

        Temporal integrity:
          - Train dates are STRICTLY before test dates (no overlap).
          - Each ticker's data within a window is continuous (no gaps
            possible since we filter by date range).
          - The target (forward log return) may have NaN at ticker
            boundaries within each window — these rows are dropped.

        Graceful end-of-data:
          - The last window's test period may be shorter than TEST_DAYS.
          - Windows with < 50 test rows are skipped (not enough for IC).

        Args:
            df: DataFrame with all 7 features, target, ticker, date columns.
                Must be sorted by (ticker, date).

        Returns:
            Dict with per-window results and aggregate stability metrics.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 4] Running Walk-Forward Validation")
        print(f"{'='*72}")
        print(f"  ℹ Features        : {len(self._final_features)}")
        print(f"  ℹ Model           : Ridge(alpha={RIDGE_ALPHA})")
        print(f"  ℹ Train window    : {TRAIN_DAYS} trading days (~2 years)")
        print(f"  ℹ Test window     : {TEST_DAYS} trading days (~6 months)")
        print(f"  ℹ Step size       : {STEP_DAYS} trading days (~6 months)")

        t0 = perf_counter()

        target_col = "target_log_return"
        feature_cols = list(self._final_features)

        assert target_col in df.columns, (
            f"[FATAL] Target column '{target_col}' not in DataFrame."
        )

        # ── Build date index ────────────────────────────────────────────────
        # Parse dates if necessary
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

        unique_dates = np.sort(df["date"].unique())
        n_dates = len(unique_dates)
        print(f"  ℹ Unique trading dates: {n_dates}")
        print(f"  ℹ Date range: {unique_dates[0]} to {unique_dates[-1]}")

        min_window = TRAIN_DAYS + 50  # Need at least 50 test rows
        if n_dates < min_window:
            raise ValueError(
                f"[FATAL] Insufficient data for walk-forward validation.\n"
                f"  Need at least {min_window} unique dates, have {n_dates}.\n"
                f"  Train={TRAIN_DAYS} + min_test=50 = {min_window}"
            )

        # ── Enumerate windows ───────────────────────────────────────────────
        windows: List[Dict[str, Any]] = []
        ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)

        window_idx = 0
        train_start_idx = 0

        while train_start_idx + TRAIN_DAYS < n_dates:
            window_idx += 1

            # Define date boundaries via index into unique_dates
            train_end_idx = train_start_idx + TRAIN_DAYS    # exclusive
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + TEST_DAYS, n_dates)  # graceful

            # Actual dates
            train_date_start = unique_dates[train_start_idx]
            train_date_end = unique_dates[train_end_idx - 1]  # inclusive
            test_date_start = unique_dates[test_start_idx]
            test_date_end = unique_dates[test_end_idx - 1]    # inclusive

            actual_test_days = test_end_idx - test_start_idx

            # ── Extract train and test DataFrames ───────────────────────────
            train_dates_set = set(unique_dates[train_start_idx:train_end_idx])
            test_dates_set = set(unique_dates[test_start_idx:test_end_idx])

            train_mask = df["date"].isin(train_dates_set)
            test_mask = df["date"].isin(test_dates_set)

            train_df = df.loc[train_mask, feature_cols + [target_col]].dropna()
            test_df = df.loc[test_mask, feature_cols + [target_col]].dropna()

            n_train = len(train_df)
            n_test = len(test_df)

            # Skip windows with insufficient test data
            if n_test < 50:
                print(f"  ⊘ Window {window_idx}: Skipped "
                      f"(only {n_test} test rows < 50 minimum)")
                windows.append({
                    "window": window_idx,
                    "train_start": str(train_date_start)[:10],
                    "train_end": str(train_date_end)[:10],
                    "test_start": str(test_date_start)[:10],
                    "test_end": str(test_date_end)[:10],
                    "train_days": TRAIN_DAYS,
                    "test_days": actual_test_days,
                    "train_rows": n_train,
                    "test_rows": n_test,
                    "ic": None,
                    "status": "SKIPPED_INSUFFICIENT_TEST_DATA",
                })
                train_start_idx += STEP_DAYS
                continue

            # ── Fit Ridge on train, predict on test ─────────────────────────
            X_train = train_df[feature_cols].values
            y_train = train_df[target_col].values
            X_test = test_df[feature_cols].values
            y_test = test_df[target_col].values

            ridge.fit(X_train, y_train)
            y_pred = ridge.predict(X_test)

            # ── Compute Spearman IC on test window ──────────────────────────
            if np.std(y_pred) < 1e-15 or np.std(y_test) < 1e-15:
                ic = 0.0
            else:
                ic, p_value = spearmanr(y_pred, y_test)
                if np.isnan(ic):
                    ic = 0.0
                p_value = float(p_value) if not np.isnan(p_value) else 1.0

            ic = round(float(ic), 6)

            window_result = {
                "window": window_idx,
                "train_start": str(train_date_start)[:10],
                "train_end": str(train_date_end)[:10],
                "test_start": str(test_date_start)[:10],
                "test_end": str(test_date_end)[:10],
                "train_days": TRAIN_DAYS,
                "test_days": actual_test_days,
                "train_rows": n_train,
                "test_rows": n_test,
                "ic": ic,
                "status": "OK",
            }
            windows.append(window_result)

            ic_str = f"{ic:+.6f}"
            indicator = "✓" if ic > 0 else "✗"
            print(f"  {indicator} Window {window_idx}: "
                  f"train [{str(train_date_start)[:10]} → "
                  f"{str(train_date_end)[:10]}] "
                  f"test [{str(test_date_start)[:10]} → "
                  f"{str(test_date_end)[:10]}] "
                  f"IC = {ic_str}  "
                  f"(train={n_train:,} test={n_test:,})")

            # Step forward
            train_start_idx += STEP_DAYS

        total_time = perf_counter() - t0

        # ── Aggregate stability metrics ─────────────────────────────────────
        scored_windows = [w for w in windows if w["ic"] is not None]
        ics = [w["ic"] for w in scored_windows]

        if not ics:
            raise RuntimeError(
                "[FATAL] No valid walk-forward windows produced. "
                "Check data coverage."
            )

        mean_ic = float(np.mean(ics))
        std_ic = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
        median_ic = float(np.median(ics))
        min_ic = float(np.min(ics))
        max_ic = float(np.max(ics))
        hit_rate = sum(1 for ic in ics if ic > 0) / len(ics)

        # Information Ratio across windows
        if std_ic > 1e-10:
            ir = mean_ic / std_ic
        else:
            ir = float("inf") if mean_ic > 0 else 0.0

        # Decay check: compare first half vs second half
        n_windows = len(ics)
        half = n_windows // 2
        if half > 0:
            first_half_ic = float(np.mean(ics[:half]))
            second_half_ic = float(np.mean(ics[half:]))
            decay_delta = second_half_ic - first_half_ic
            # Positive = signal strengthening; Negative = signal decaying
        else:
            first_half_ic = mean_ic
            second_half_ic = mean_ic
            decay_delta = 0.0

        # First vs last window comparison
        first_window_ic = ics[0]
        last_window_ic = ics[-1]

        # Worst drawdown: largest single-window IC drop
        worst_window_idx = int(np.argmin(ics))
        worst_window = scored_windows[worst_window_idx]

        stability_metrics = {
            "n_windows_total": len(windows),
            "n_windows_scored": len(scored_windows),
            "n_windows_skipped": len(windows) - len(scored_windows),
            "mean_ic": round(mean_ic, 6),
            "std_ic": round(std_ic, 6),
            "median_ic": round(median_ic, 6),
            "min_ic": round(min_ic, 6),
            "max_ic": round(max_ic, 6),
            "ir": round(ir, 4) if np.isfinite(ir) else ir,
            "hit_rate": round(hit_rate, 4),
            "hit_rate_pct": round(hit_rate * 100, 1),
            "first_window_ic": round(first_window_ic, 6),
            "last_window_ic": round(last_window_ic, 6),
            "first_half_mean_ic": round(first_half_ic, 6),
            "second_half_mean_ic": round(second_half_ic, 6),
            "decay_delta": round(decay_delta, 6),
            "decay_assessment": (
                "STABLE" if abs(decay_delta) < 0.005
                else ("IMPROVING" if decay_delta > 0 else "DECAYING")
            ),
            "worst_window": {
                "window": worst_window["window"],
                "test_period": (
                    f"{worst_window['test_start']} → "
                    f"{worst_window['test_end']}"
                ),
                "ic": worst_window["ic"],
            },
        }

        # ── Assemble output ─────────────────────────────────────────────────
        output = {
            "stage": STAGE_NAME,
            "version": VERSION,
            "algorithm": "Walk-Forward Validation (Rolling Window)",
            "config_snapshot": {
                "ridge_alpha": RIDGE_ALPHA,
                "forward_days": FORWARD_DAYS,
                "train_days": TRAIN_DAYS,
                "test_days": TEST_DAYS,
                "step_days": STEP_DAYS,
                "features": list(self._final_features),
                "feature_count": len(self._final_features),
                "base_feature_count": len(self._base_features),
                "interaction_count": len(self._interaction_features),
                "curated_specs_used": len(self._filtered_specs)
                    if self._filtered_specs else 0,
            },
            "stage3_reference": {
                "reported_mean_ic": self._stage3_score,
                "note": (
                    "Stage 3 IC was computed via TimeSeriesSplit (static). "
                    "Walk-forward IC below may differ due to rolling windows."
                ),
            },
            "windows": windows,
            "stability_metrics": stability_metrics,
            "runtime_seconds": round(total_time, 2),
        }

        # ── Print summary ───────────────────────────────────────────────────
        print(f"\n  {'═'*60}")
        print(f"  WALK-FORWARD STABILITY REPORT:")
        print(f"  {'═'*60}")
        print(f"  Windows scored    : {len(scored_windows)}")
        print(f"  Mean IC           : {mean_ic:.6f}")
        print(f"  Std IC            : {std_ic:.6f}")
        print(f"  Median IC         : {median_ic:.6f}")
        print(f"  IR (Mean/Std)     : {ir:.4f}" if np.isfinite(ir) else
              f"  IR (Mean/Std)     : ∞")
        print(f"  Hit Rate          : {hit_rate*100:.1f}% "
              f"({sum(1 for ic in ics if ic > 0)}/{len(ics)} windows IC > 0)")
        print(f"  Min IC            : {min_ic:.6f}")
        print(f"  Max IC            : {max_ic:.6f}")
        print(f"  ──────────────────────────────────────────────")
        print(f"  1st half Mean IC  : {first_half_ic:.6f}")
        print(f"  2nd half Mean IC  : {second_half_ic:.6f}")
        print(f"  Decay Δ           : {decay_delta:+.6f}")
        print(f"  Assessment        : {stability_metrics['decay_assessment']}")
        print(f"  ──────────────────────────────────────────────")
        print(f"  Worst window      : #{worst_window['window']} "
              f"({worst_window['test_start']} → {worst_window['test_end']}) "
              f"IC = {worst_window['ic']:.6f}")
        print(f"  Stage 3 ref IC    : "
              f"{self._stage3_score:.6f}" if self._stage3_score else "N/A")
        print(f"  Runtime           : {total_time:.2f}s")
        print(f"  {'═'*60}")

        return output

    # ── 6. Save results atomically ──────────────────────────────────────────
    def save_results(self, results: Dict[str, Any]) -> None:
        """Write stability_report.json atomically (.tmp → rename).

        Also completes the Auditor lifecycle.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 4] Saving Results")
        print(f"{'='*72}")

        PHASE2_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

        # ── Atomic write ────────────────────────────────────────────────────
        tmp_path = OUTPUT_JSON.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(
                    results, f, indent=2, ensure_ascii=False, default=str
                )
            tmp_path.replace(OUTPUT_JSON)
            print(f"  ✓ {OUTPUT_JSON.name} "
                  f"({OUTPUT_JSON.stat().st_size:,} bytes)")
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(
                f"[FATAL] Failed to write results: {e}"
            ) from e

        # ── Auditor lifecycle completion ────────────────────────────────────
        try:
            metrics = results.get("stability_metrics", {})
            summary_df = pd.DataFrame([{
                "mean_ic": metrics.get("mean_ic"),
                "std_ic": metrics.get("std_ic"),
                "hit_rate": metrics.get("hit_rate"),
                "n_windows": metrics.get("n_windows_scored"),
                "decay_assessment": metrics.get("decay_assessment"),
            }])
            self.auditor.record_output(summary_df, self.config.to_snapshot())
            self.auditor.success()
            print(f"  ✓ manifest.json (via Auditor)")
        except Exception as e:
            print(f"  ⚠ Auditor manifest write failed (non-fatal): {e}")

    # ── Orchestrator ────────────────────────────────────────────────────────
    def run(self) -> None:
        """Full pipeline: Stage3 → Data → Target → Features → Walk-Forward → Save."""
        pipeline_t0 = perf_counter()

        # Step 1: Load Stage 3 results → extract final 7 features
        self.load_stage3_results()

        # Step 2: Load Phase 1 raw OHLCV
        df = self.load_phase1_data()

        # Step 3: Compute target
        target = self.compute_target(df)
        df["target_log_return"] = target

        # Step 4: Regenerate 5 base TA features + 2 interaction columns
        df = self.regenerate_features(df)

        # Step 5: Walk-forward validation
        results = self.run_walk_forward(df)

        # Step 6: Save atomically
        self.save_results(results)

        total_time = perf_counter() - pipeline_t0
        print(f"\n{'='*72}")
        print(f"[Stage 4] COMPLETE — Total runtime: {total_time:.1f}s")
        print(f"{'='*72}\n")


# ─── Main Execution ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Phase 2 · Stage 4 — Walk-Forward Stability Validation     ║")
    print(f"║  Version: {VERSION:<49}║")
    print("╚══════════════════════════════════════════════════════════════╝")

    try:
        config = Phase2Config()

        validator = StabilityValidator(config=config)
        validator.run()

        sys.exit(0)

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except AssertionError as e:
        print(f"\n[ASSERTION FAILED] {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Walk-forward validation aborted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n[UNHANDLED ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(99)
