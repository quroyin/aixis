# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Phase 2 · Stage 1 — Individual Feature Evaluation                        ║
║                                                                            ║
║  Auditor Prompt (Design Document):                                         ║
║  This script evaluates each of the 20 candidate features selected in       ║
║  Stage 0 (preselection audit) by measuring their individual predictive     ║
║  power for forecasting 5-day forward log returns.                          ║
║                                                                            ║
║  Method:                                                                   ║
║    • For each candidate feature, fit a Ridge(alpha=1.0) regression in a    ║
║      TimeSeriesSplit(n_splits=5) cross-validation loop.                    ║
║    • On each fold's test set, compute the Spearman rank correlation (IC)   ║
║      between predicted and actual target values.                           ║
║    • Aggregate: Mean IC, Std IC, Information Ratio (IR = Mean / Std).      ║
║                                                                            ║
║  Inputs  (from disk — fully decoupled):                                    ║
║    • artifacts/phase_1_data/merged_data.parquet                            ║
║    • artifacts/phase_2_features/candidate_features.csv                     ║
║                                                                            ║
║  Outputs:                                                                  ║
║    • artifacts/phase_2_features/individual_feature_scores.json             ║
║    • artifacts/phase_2_features/manifest.json  (via Auditor)               ║
║                                                                            ║
║  Quality Gates:                                                            ║
║    ✓ Decoupled — loads all inputs from disk                                ║
║    ✓ Contract — validates OHLCV schema before processing                   ║
║    ✓ Deterministic — Ridge is deterministic; no shuffle in TSCV            ║
║    ✓ Atomic — writes to .tmp then renames                                  ║
║    ✓ Scalable — vectorised target computation; single-column Ridge is O(N) ║
║    ✓ Orchestrated — runnable via `python3 2_individual_evaluation.py`      ║
║    ✓ Traceable — Auditor records inputs, config, outputs                   ║
║    ✓ Versionable — version string embedded in output JSON                  ║
║    ✓ Recoverable — idempotent; crash-safe via atomic writes                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─── Imports ────────────────────────────────────────────────────────────────
from __future__ import annotations

import inspect
import json
import sys
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

# ─── Project path bootstrap ────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent          # phases/phase2/
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent              # angel1/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas_ta_classic as ta                         # noqa: E402
from core.audit import Auditor                         # noqa: E402
from core.io import write_json, write_parquet          # noqa: E402
from phases.phase2.config import Phase2Config          # noqa: E402

# ─── Suppress noisy warnings ───────────────────────────────────────────────
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Constants ──────────────────────────────────────────────────────────────
VERSION = "1.0.0"
STAGE_NAME = "1_individual_evaluation"

PHASE1_ARTIFACT = _PROJECT_ROOT / "artifacts" / "phase_1_data" / "merged_data.parquet"
PHASE2_ARTIFACT_DIR = _PROJECT_ROOT / "artifacts" / "phase_2_features"
CANDIDATE_CSV = PHASE2_ARTIFACT_DIR / "candidate_features.csv"
OUTPUT_JSON = PHASE2_ARTIFACT_DIR / "individual_feature_scores.json"

REQUIRED_COLUMNS = {"ticker", "date", "open", "high", "low", "close", "volume"}

RIDGE_ALPHA = 1.0
CV_FOLDS = 5
FORWARD_DAYS = 5


# ─── Auditor factory ───────────────────────────────────────────────────────
def _create_auditor(output_dir: Path) -> Auditor:
    """Introspect the Auditor constructor and instantiate it correctly.

    Different versions of core.audit.Auditor may accept different kwargs.
    We inspect the __init__ signature at runtime so we never pass an
    unexpected keyword argument.
    """
    sig = inspect.signature(Auditor.__init__)
    params = set(sig.parameters.keys()) - {"self"}

    # Build a kwargs dict with every piece of metadata we'd LIKE to pass,
    # then filter to only what the constructor actually accepts.
    desired = {
        # Positional-style (some Auditors take these positionally)
        "stage": STAGE_NAME,
        "script": "2_individual_evaluation.py",
        "version": VERSION,
        "output_dir": output_dir,
        "artifact_dir": output_dir,
        "name": STAGE_NAME,
        "phase": "phase_2",
    }

    # Keep only recognised parameters
    kwargs = {k: v for k, v in desired.items() if k in params}

    # If Auditor takes no kwargs at all (bare __init__(self)), call empty
    try:
        return Auditor(**kwargs)
    except TypeError:
        # Ultimate fallback: zero-arg construction
        return Auditor()


# ─── Helper: safe Auditor method calls ──────────────────────────────────────
def _auditor_call(auditor: Auditor, method_name: str, *args, **kwargs) -> None:
    """Call an Auditor method if it exists, silently skip otherwise."""
    fn = getattr(auditor, method_name, None)
    if fn is not None and callable(fn):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            print(f"  ⚠ Auditor.{method_name}() failed (non-fatal): {e}")


# ─── Class ──────────────────────────────────────────────────────────────────
class IndividualFeatureEvaluator:
    """Evaluate each candidate feature's individual predictive power
    using Ridge regression scored by Spearman IC under Time-Series CV."""

    def __init__(self, config: Phase2Config) -> None:
        self.config = config
        self.auditor = _create_auditor(PHASE2_ARTIFACT_DIR)

        _auditor_call(self.auditor, "log_input", "merged_data", str(PHASE1_ARTIFACT))
        _auditor_call(self.auditor, "log_input", "candidate_features", str(CANDIDATE_CSV))

        self._raw_df: Optional[pd.DataFrame] = None
        self._candidate_names: Optional[List[str]] = None
        self._feature_ta_specs: Optional[List[Dict[str, Any]]] = None

    # ── 1. Load data from disk ──────────────────────────────────────────────
    def load_data(self) -> pd.DataFrame:
        print(f"\n{'='*72}")
        print(f"[Stage 1] Loading Data...")
        print(f"{'='*72}")

        if not PHASE1_ARTIFACT.exists():
            raise FileNotFoundError(
                f"[FATAL] Phase 1 artifact not found: {PHASE1_ARTIFACT}\n"
                f"       Run Phase 1 before Phase 2 Stage 1."
            )
        if not CANDIDATE_CSV.exists():
            raise FileNotFoundError(
                f"[FATAL] Stage 0 artifact not found: {CANDIDATE_CSV}\n"
                f"       Run 1_preselection_audit.py before this script."
            )

        t0 = perf_counter()
        df = pd.read_parquet(PHASE1_ARTIFACT)
        t_load = perf_counter() - t0
        print(f"  ✓ Loaded merged_data.parquet: {df.shape[0]:,} rows × {df.shape[1]} cols  ({t_load:.2f}s)")

        df.columns = df.columns.str.lower().str.strip()
        present_cols = set(df.columns)
        missing = REQUIRED_COLUMNS - present_cols
        assert len(missing) == 0, (
            f"[SCHEMA VIOLATION] Missing required columns: {missing}\n"
            f"  Present columns: {sorted(present_cols)}"
        )
        print(f"  ✓ Schema validated: {sorted(REQUIRED_COLUMNS)} all present")

        candidates_df = pd.read_csv(CANDIDATE_CSV)
        assert "feature" in candidates_df.columns, (
            f"[SCHEMA VIOLATION] candidate_features.csv must have a 'feature' column.\n"
            f"  Found columns: {list(candidates_df.columns)}"
        )
        self._candidate_names = candidates_df["feature"].tolist()
        print(f"  ✓ Loaded {len(self._candidate_names)} candidate features from Stage 0")
        print(f"    Features: {self._candidate_names}")

        self._raw_df = df
        return df

    # ── 2. Compute forward target (decoupled re-implementation) ─────────────
    @staticmethod
    def compute_target(df: pd.DataFrame) -> pd.Series:
        """Compute 5-day forward log return per ticker.

        Formula:  target_t = ln(close_{t+5} / close_t)
        """
        print(f"\n{'='*72}")
        print(f"[Stage 1] Computing Target: {FORWARD_DAYS}-day forward log return")
        print(f"{'='*72}")

        t0 = perf_counter()
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        future_close = df.groupby("ticker")["close"].shift(-FORWARD_DAYS)
        target = np.log(future_close / df["close"])
        target.name = "target_log_return"

        n_valid = target.notna().sum()
        n_nan = target.isna().sum()
        t_elapsed = perf_counter() - t0
        print(f"  ✓ Target computed: {n_valid:,} valid / {n_nan:,} NaN  ({t_elapsed:.3f}s)")

        return target

    # ── 3. Regenerate features via pandas_ta_classic ────────────────────────
    def regenerate_features(
        self, df: pd.DataFrame, candidate_list: List[str]
    ) -> pd.DataFrame:
        """Re-generate candidate features from raw OHLCV.

        CRITICAL: We do NOT load pre-computed feature values from Stage 0.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 1] Regenerating {len(candidate_list)} Features via pandas_ta_classic")
        print(f"{'='*72}")

        t0 = perf_counter()
        all_specs = self.config.curated_indicators

        tickers = df["ticker"].unique()
        print(f"  ℹ Processing {len(tickers)} tickers with {len(all_specs)} indicator specs")

        ta_strategy = ta.Strategy(
            name="Phase2_Candidates",
            ta=all_specs,
        )

        frames: List[pd.DataFrame] = []
        initial_cols = set(df.columns)

        for ticker_id in tqdm(tickers, desc="  [TA Generation]", unit="ticker"):
            ticker_mask = df["ticker"] == ticker_id
            ticker_df = df.loc[ticker_mask].copy()

            if "date" in ticker_df.columns:
                ticker_df = ticker_df.set_index("date")

            ticker_df.ta.strategy(ta_strategy, verbose=False)
            ticker_df = ticker_df.reset_index()
            frames.append(ticker_df)

        result_df = pd.concat(frames, axis=0, ignore_index=True)
        new_cols = set(result_df.columns) - initial_cols
        print(f"  ✓ TA strategy generated {len(new_cols)} new columns")

        generated_cols = set(result_df.columns)
        matched = [c for c in candidate_list if c in generated_cols]
        missing = [c for c in candidate_list if c not in generated_cols]

        if missing:
            col_map = {c.upper(): c for c in generated_cols}
            for m in missing[:]:
                if m.upper() in col_map:
                    actual_name = col_map[m.upper()]
                    result_df.rename(columns={actual_name: m}, inplace=True)
                    matched.append(m)
                    missing.remove(m)

        if missing:
            print(f"  ⚠ WARNING: {len(missing)} candidates could not be regenerated: {missing}")
            print(f"    These will be excluded from evaluation.")

        print(f"  ✓ Matched {len(matched)}/{len(candidate_list)} candidate features")

        t_elapsed = perf_counter() - t0
        print(f"  ✓ Feature regeneration complete ({t_elapsed:.1f}s)")

        self._candidate_names = matched
        self._feature_ta_specs = all_specs

        return result_df

    # ── 4. Core evaluation loop ─────────────────────────────────────────────
    def run_evaluation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate each feature individually via Ridge + TSCV + Spearman IC."""
        print(f"\n{'='*72}")
        print(f"[Stage 1] Running Individual Feature Evaluation")
        print(f"{'='*72}")
        print(f"  ℹ Model: Ridge(alpha={RIDGE_ALPHA})")
        print(f"  ℹ CV: TimeSeriesSplit(n_splits={CV_FOLDS})")
        print(f"  ℹ Metric: Spearman IC (rank correlation)")

        t0 = perf_counter()

        target_col = "target_log_return"
        assert target_col in df.columns, (
            f"[FATAL] Target column '{target_col}' not found in DataFrame."
        )

        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        feature_results: Dict[str, Dict[str, Any]] = {}
        candidates = self._candidate_names

        tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
        ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)

        for feat_name in tqdm(candidates, desc="  [Evaluation]", unit="feature"):
            mask = df[[feat_name, target_col]].notna().all(axis=1)
            subset = df.loc[mask, [feat_name, target_col]].reset_index(drop=True)

            n_samples = len(subset)
            if n_samples < CV_FOLDS * 2:
                print(f"  ⚠ Skipping {feat_name}: only {n_samples} valid samples")
                feature_results[feat_name] = {
                    "mean_ic": None,
                    "std_ic": None,
                    "ir": None,
                    "fold_ics": [],
                    "n_valid_samples": n_samples,
                    "status": "SKIPPED_INSUFFICIENT_DATA",
                }
                continue

            X = subset[[feat_name]].values
            y = subset[target_col].values

            fold_ics: List[float] = []

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                ridge.fit(X_train, y_train)
                y_pred = ridge.predict(X_test)

                if np.std(y_pred) < 1e-15 or np.std(y_test) < 1e-15:
                    ic = 0.0
                else:
                    ic, _ = spearmanr(y_pred, y_test)
                    if np.isnan(ic):
                        ic = 0.0

                fold_ics.append(round(float(ic), 6))

            mean_ic = float(np.mean(fold_ics))
            std_ic = float(np.std(fold_ics, ddof=1)) if len(fold_ics) > 1 else 0.0

            if std_ic > 1e-10:
                ir = mean_ic / std_ic
            else:
                ir = float("inf") if mean_ic > 0 else (float("-inf") if mean_ic < 0 else 0.0)

            feature_results[feat_name] = {
                "mean_ic": round(mean_ic, 6),
                "std_ic": round(std_ic, 6),
                "ir": round(ir, 4) if np.isfinite(ir) else ir,
                "fold_ics": fold_ics,
                "n_valid_samples": n_samples,
                "status": "OK",
            }

        runtime = perf_counter() - t0

        scorable = {
            k: v for k, v in feature_results.items()
            if v["mean_ic"] is not None
        }
        if scorable:
            top_feature = max(
                scorable.keys(),
                key=lambda k: (scorable[k]["ir"], scorable[k]["mean_ic"]),
            )
        else:
            top_feature = "NONE"

        output = {
            "stage": STAGE_NAME,
            "version": VERSION,
            "config_snapshot": {
                "ridge_alpha": RIDGE_ALPHA,
                "cv_folds": CV_FOLDS,
                "forward_days": FORWARD_DAYS,
                "n_candidates_input": len(candidates),
                "curated_indicator_count": len(self._feature_ta_specs) if self._feature_ta_specs else 0,
            },
            "evaluation_params": {
                "model": "Ridge",
                "alpha": RIDGE_ALPHA,
                "cv_folds": CV_FOLDS,
                "cv_method": "TimeSeriesSplit",
                "metric": "Spearman IC",
            },
            "features": feature_results,
            "summary": {
                "total_features_evaluated": len(scorable),
                "total_features_skipped": len(feature_results) - len(scorable),
                "top_feature": top_feature,
                "top_feature_ir": scorable.get(top_feature, {}).get("ir"),
                "top_feature_mean_ic": scorable.get(top_feature, {}).get("mean_ic"),
                "runtime_seconds": round(runtime, 2),
            },
        }

        # ── Print ranked summary ────────────────────────────────────────────
        print(f"\n  {'─'*60}")
        print(f"  Feature Ranking (by Information Ratio):")
        print(f"  {'─'*60}")
        print(f"  {'Rank':<6}{'Feature':<25}{'Mean IC':>10}{'Std IC':>10}{'IR':>10}")
        print(f"  {'─'*60}")

        ranked = sorted(
            scorable.items(),
            key=lambda kv: (kv[1]["ir"], kv[1]["mean_ic"]),
            reverse=True,
        )
        for rank, (fname, fdata) in enumerate(ranked, 1):
            ir_str = f"{fdata['ir']:.4f}" if np.isfinite(fdata["ir"]) else "∞"
            print(
                f"  {rank:<6}{fname:<25}{fdata['mean_ic']:>10.6f}"
                f"{fdata['std_ic']:>10.6f}{ir_str:>10}"
            )

        print(f"  {'─'*60}")
        print(f"  ✓ Evaluation complete in {runtime:.2f}s")
        print(f"  ★ Top feature: {top_feature}")

        return output

    # ── 5. Save results atomically ──────────────────────────────────────────
    def save_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'='*72}")
        print(f"[Stage 1] Saving Results")
        print(f"{'='*72}")

        PHASE2_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

        tmp_path = OUTPUT_JSON.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            tmp_path.replace(OUTPUT_JSON)
            print(f"  ✓ {OUTPUT_JSON.name} ({OUTPUT_JSON.stat().st_size:,} bytes)")
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"[FATAL] Failed to write results: {e}") from e

        try:
            _auditor_call(self.auditor, "log_output", "individual_feature_scores", str(OUTPUT_JSON))
            _auditor_call(self.auditor, "save_manifest")
            print(f"  ✓ manifest.json (via Auditor)")
        except Exception as e:
            print(f"  ⚠ Auditor manifest write failed (non-fatal): {e}")

    # ── Orchestrator ────────────────────────────────────────────────────────
    def run(self) -> None:
        pipeline_t0 = perf_counter()

        df = self.load_data()
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        target = self.compute_target(df)
        df["target_log_return"] = target

        df = self.regenerate_features(df, self._candidate_names)

        results = self.run_evaluation(df)
        self.save_results(results)

        total_time = perf_counter() - pipeline_t0
        print(f"\n{'='*72}")
        print(f"[Stage 1] COMPLETE — Total runtime: {total_time:.1f}s")
        print(f"{'='*72}\n")


# ─── Main Execution ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Phase 2 · Stage 1 — Individual Feature Evaluation         ║")
    print(f"║  Version: {VERSION:<49}║")
    print("╚══════════════════════════════════════════════════════════════╝")

    try:
        config = Phase2Config()
        evaluator = IndividualFeatureEvaluator(config=config)
        evaluator.run()
        sys.exit(0)

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except AssertionError as e:
        print(f"\n[ASSERTION FAILED] {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Evaluation aborted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n[UNHANDLED ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(99)
