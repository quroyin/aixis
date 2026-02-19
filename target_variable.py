"""
Phase 2: Target Variable Construction
======================================
Reads ONLY from Phase 1 output (cleaned OHLCV prices).
Writes a single output artifact consumed by Phase 3 (feature-target join).

Principles enforced:
  - Temporal integrity: target at index t uses only P[t] and P[t+h]
  - No look-ahead bias: shift is strictly negative (future prices shifted back)
  - Determinism: no random state, no hidden parameters
  - Strict data boundary: reads only phase_1_output.parquet

Changelog:
  - v2: Removed quantile discretization (look-ahead bias; must be fit in Phase 4)
  - v2: Added input validation (positive prices, no NaNs, horizon bounds)
  - v2: Added SHA-256 input hash to audit log for reproducibility verification
"""

import hashlib
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Literal


# ── CONFIGURATION ───────────────────────────────────────────────
@dataclass(frozen=True)
class TargetConfig:
    """Immutable configuration — zero hidden state."""
    horizon: int                                          # forward bars
    price_col: str = "close"                              # price field to use
    method: Literal["log_return", "sign", "quantile"] = "log_return"

    def __post_init__(self):
        if self.horizon < 1:
            raise ValueError(f"Horizon must be >= 1, got {self.horizon}")

        if self.method == "quantile":
            warnings.warn(
                "DEPRECATED: method='quantile' computes bin boundaries on ALL "
                "data including the test period, introducing look-ahead bias. "
                "This option exists ONLY for debugging/EDA. For modeling, use "
                "method='log_return' in Phase 2 and apply quantile "
                "discretization in Phase 4 (train/test split) using ONLY "
                "training-set boundaries. This option will be removed in v3.",
                FutureWarning,
                stacklevel=2,
            )


# ── INPUT VALIDATION ────────────────────────────────────────────
def validate_phase1_output(df: pd.DataFrame, config: TargetConfig) -> None:
    """
    Enforce the Phase 1 → Phase 2 data contract.

    These are hard assertions, not warnings. If Phase 1 produced
    invalid data, we must fail loudly rather than propagate errors.
    """
    # Column existence
    if config.price_col not in df.columns:
        raise ValueError(
            f"Price column '{config.price_col}' not in Phase 1 output. "
            f"Available: {list(df.columns)}"
        )

    prices = df[config.price_col]

    # Temporal ordering
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "TEMPORAL INTEGRITY VIOLATION: index must be sorted ascending by time"
        )

    # No NaN prices — Phase 1 contract requires clean data
    nan_count = prices.isna().sum()
    if nan_count > 0:
        nan_indices = prices[prices.isna()].index.tolist()
        raise ValueError(
            f"PHASE 1 CONTRACT VIOLATION: {nan_count} NaN price(s) found. "
            f"Phase 1 must deliver clean, gap-free prices. "
            f"First NaN indices: {nan_indices[:5]}"
        )

    # All prices must be strictly positive (log is undefined at 0 and negative)
    non_positive = prices[prices <= 0]
    if len(non_positive) > 0:
        raise ValueError(
            f"MATHEMATICAL VIOLATION: {len(non_positive)} non-positive price(s) "
            f"found. log-return requires strictly positive prices. "
            f"First offenders: {non_positive.head().to_dict()}"
        )

    # Horizon must be strictly less than series length
    # (otherwise every single target row would be NaN — a silent no-op)
    if config.horizon >= len(prices):
        raise ValueError(
            f"CONFIGURATION ERROR: horizon ({config.horizon}) >= number of "
            f"price observations ({len(prices)}). This would produce zero "
            f"valid target rows."
        )


# ── INPUT HASHING ───────────────────────────────────────────────
def compute_file_sha256(path: Path) -> str:
    """
    Compute SHA-256 of the input file for reproducibility verification.

    Reads in 64KB chunks to handle large files without excessive memory.
    Deterministic: same file content → same hash, always.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# ── CORE COMPUTATION ────────────────────────────────────────────
def compute_forward_log_return(
    prices: pd.Series,
    horizon: int,
) -> pd.Series:
    """
    Core target computation.

    y_t = ln(P_{t+h} / P_t)

    Temporal integrity proof:
      - prices.shift(-horizon) aligns P_{t+h} with index t
      - At index t, we reference P_t (known) and P_{t+h} (future, label only)
      - The last `horizon` rows become NaN — they have no valid future price
      - We NEVER fill these NaNs; they are dropped downstream

    Why log-returns and not simple returns?
      1. Additive over time:  ln(P_T/P_0) = Σ ln(P_{t+1}/P_t)
      2. Symmetric: +10% and -10% are equidistant from zero
      3. Better statistical properties (closer to Gaussian for modeling)
      4. Numerically stable for compounding over multi-period horizons
    """
    future_price = prices.shift(-horizon)  # look FORWARD by h bars
    log_ret = np.log(future_price / prices)

    # Paranoia check: ensure the tail is NaN (no accidental fill)
    assert log_ret.iloc[-horizon:].isna().all(), (
        "TEMPORAL INTEGRITY VIOLATION: last `horizon` rows must be NaN"
    )
    return log_ret


# ── DISCRETIZATION ──────────────────────────────────────────────
def discretize_target_sign(continuous_target: pd.Series) -> pd.Series:
    """
    Convert continuous log-return to directional label: {-1, 0, +1}.

    This is temporally safe: sign(y_t) depends only on y_t itself,
    not on the distribution of other y values. No look-ahead bias.
    """
    return np.sign(continuous_target).astype(int)


def _discretize_target_quantile_DEPRECATED(
    continuous_target: pd.Series,
    quantile_bins: int = 3,
) -> pd.Series:
    """
    DEPRECATED — FOR DEBUGGING / EDA ONLY.

    Computes quantile bin boundaries on the ENTIRE series, which
    includes future test-period data → LOOK-AHEAD BIAS.

    For modeling, apply quantile discretization in Phase 4 using
    ONLY training-set boundaries:

        boundaries = np.quantile(y_train, np.linspace(0, 1, q+1))
        y_train_q  = np.digitize(y_train, boundaries[1:-1])
        y_test_q   = np.digitize(y_test,  boundaries[1:-1])  # same boundaries

    This function will be removed in v3.
    """
    warnings.warn(
        "DEPRECATED: quantile discretization in Phase 2 introduces "
        "look-ahead bias. Move to Phase 4 (train/test split).",
        FutureWarning,
        stacklevel=2,
    )
    return pd.qcut(
        continuous_target, q=quantile_bins, labels=False, duplicates="drop"
    )


# ── PHASE 2 PIPELINE STEP ──────────────────────────────────────
def build_target(
    phase_1_output_path: Path,
    config: TargetConfig,
    output_path: Path,
) -> pd.DataFrame:
    """
    Full Phase 2 pipeline step.

    Reads  → phase_1_output.parquet  (cleaned OHLCV)
    Writes → phase_2_target.parquet  (index-aligned target series)
    """
    # ── STRICT DATA BOUNDARY: read only Phase 1 output ──────────
    phase_1_output_path = Path(phase_1_output_path)
    output_path = Path(output_path)

    input_hash = compute_file_sha256(phase_1_output_path)
    df = pd.read_parquet(phase_1_output_path)

    # ── VALIDATE PHASE 1 CONTRACT ───────────────────────────────
    validate_phase1_output(df, config)

    prices = df[config.price_col]

    # ── CORE COMPUTATION ────────────────────────────────────────
    log_ret = compute_forward_log_return(prices, config.horizon)

    result = pd.DataFrame({
        "target_log_return": log_ret,               # always include continuous
    }, index=df.index)

    if config.method == "sign":
        result["target_class"] = discretize_target_sign(log_ret.dropna())
    elif config.method == "quantile":
        result["target_class"] = _discretize_target_quantile_DEPRECATED(
            log_ret.dropna(),
        )

    # ── DROP ROWS WITH NO VALID TARGET (temporal boundary) ──────
    valid = result.dropna(subset=["target_log_return"])

    # ── WRITE PHASE 2 OUTPUT ────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid.to_parquet(output_path, engine="pyarrow")

    output_hash = compute_file_sha256(output_path)

    # ── AUDIT LOG (deterministic, no side effects) ──────────────
    print(f"[Phase 2] Target variable constructed")
    print(f"  Config           : {config}")
    print(f"  Input file       : {phase_1_output_path}")
    print(f"  Input SHA-256    : {input_hash}")
    print(f"  Input rows       : {len(df)}")
    print(f"  Valid targets    : {len(valid)}")
    print(f"  Dropped (tail)   : {len(df) - len(valid)}")
    print(f"  Target stats     :\n{valid['target_log_return'].describe()}")
    print(f"  Output file      : {output_path}")
    print(f"  Output SHA-256   : {output_hash}")

    return valid


# ── ENTRY POINT ─────────────────────────────────────────────────
if __name__ == "__main__":
    config = TargetConfig(
        horizon=5,                # 5-day forward return
        price_col="close",
        method="log_return",      # keep continuous for regression
    )

    build_target(
        phase_1_output_path=Path("artifacts/phase_1/phase_1_output.parquet"),
        config=config,
        output_path=Path("artifacts/phase_2/phase_2_target.parquet"),
    )