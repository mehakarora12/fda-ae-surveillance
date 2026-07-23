"""
Phase 3 — Forecasting & Anomaly Detection
============================================
Takes the daily serious-AE counts per drug (Phase 1a's `counts` output —
the full population, not the sampled reports) and flags days where the
count is anomalously high relative to what a time-series model expected.

Two independent, complementary detectors (this ensemble design — not
relying on a single method — is worth explaining in interviews):

1. Holt-Winters exponential smoothing (trend + weekly seasonality) gives
   an expected count per day. Residual = actual - expected, converted to
   a rolling z-score. Flags days that break the *univariate time pattern*
   for that specific drug.
   (We use Holt-Winters instead of Prophet: same trend/seasonality
   decomposition concept, but installs as a pure-Python wheel with no
   C++/Stan compilation step — avoids a whole class of Windows install
   failures for very little methodological cost.)

2. Isolation Forest on engineered features (rolling mean/std, day-of-week)
   catches multivariate anomalies a single-series model might miss —
   e.g. a moderate but sustained shift that a smoothing model absorbs
   into its own trend estimate rather than flagging.

A day flagged by BOTH methods is a high-confidence anomaly — that's your
Phase 4 RAG trigger and your Phase 5 ground-truth comparison target.

Usage
-----
    python src/forecasting.py run
    python src/forecasting.py run --plot     # also saves PNGs per drug
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("forecasting")

COUNTS_PATH = config.PROCESSED_DIR / "daily_counts.csv"
ANOMALIES_PATH = config.PROCESSED_DIR / "anomalies.csv"
FIGURES_DIR = config.ROOT_DIR / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ROLLING_WINDOW = 30       # days, for rolling z-score of residuals
Z_THRESHOLD = 2.5         # |z| above this = flagged by the smoothing detector
ISO_CONTAMINATION = 0.03  # expected anomaly fraction for Isolation Forest
MIN_SERIES_LENGTH = 60    # Holt-Winters needs enough data for weekly seasonality
WARMUP_DAYS = 45          # exclude flags in this initial window — the rolling
                           # z-score baseline and HW's level/trend initialization
                           # haven't stabilized yet, so early flags are unreliable


# --------------------------------------------------------------------------
# Per-drug pipeline
# --------------------------------------------------------------------------
def _fill_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """openFDA only returns days WITH reports — missing days are implicitly
    zero. Time-series models need a continuous index, so fill the gaps."""
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_range, fill_value=0)
    df.index.name = "date"
    return df.reset_index()


def _holt_winters_zscore(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Returns (yhat, zscore) aligned to `series`. Falls back to a rolling-
    mean baseline if the series is too short/degenerate for seasonal HW."""
    if len(series) >= MIN_SERIES_LENGTH and series.std() > 0:
        try:
            model = ExponentialSmoothing(
                series, trend="add", seasonal="add", seasonal_periods=7,
                initialization_method="estimated",
            ).fit()
            yhat = pd.Series(model.fittedvalues, index=series.index)
        except Exception as e:  # noqa: BLE001
            log.warning(f"Holt-Winters failed ({e}), falling back to rolling mean.")
            yhat = series.rolling(7, min_periods=1).mean()
    else:
        yhat = series.rolling(7, min_periods=1).mean()

    residual = series - yhat
    roll_mean = residual.rolling(ROLLING_WINDOW, min_periods=7).mean()
    roll_std = residual.rolling(ROLLING_WINDOW, min_periods=7).std().replace(0, np.nan)
    zscore = (residual - roll_mean) / roll_std
    return yhat, zscore.fillna(0)


def _isolation_forest_flags(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Returns (iforest_score, is_anomaly) using rolling stats + day-of-week
    as features — catches patterns a single-series model can miss."""
    feats = pd.DataFrame({
        "count": df["count"],
        "roll_mean_7": df["count"].rolling(7, min_periods=1).mean(),
        "roll_std_7": df["count"].rolling(7, min_periods=1).std().fillna(0),
        "dow_sin": np.sin(2 * np.pi * df["date"].dt.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * df["date"].dt.dayofweek / 7),
    })
    model = IsolationForest(contamination=ISO_CONTAMINATION, random_state=42)
    preds = model.fit_predict(feats)          # -1 = anomaly, 1 = normal
    scores = -model.decision_function(feats)  # higher = more anomalous
    return pd.Series(scores, index=df.index), pd.Series(preds == -1, index=df.index)


def process_drug(drug_df: pd.DataFrame, drug: str) -> pd.DataFrame:
    drug_df = _fill_missing_dates(drug_df[["date", "count"]].sort_values("date"))
    yhat, zscore = _holt_winters_zscore(drug_df["count"])
    iforest_score, iforest_flag = _isolation_forest_flags(drug_df)

    result = drug_df.copy()
    result["drug"] = drug
    result["expected_count"] = yhat.round(2)
    result["residual"] = (result["count"] - result["expected_count"]).round(2)
    result["zscore"] = zscore.round(2)
    result["is_anomaly_zscore"] = zscore.abs() > Z_THRESHOLD
    result["isolation_forest_score"] = iforest_score.round(3)
    result["is_anomaly_isoforest"] = iforest_flag

    # Don't trust flags before the models have stabilized (see WARMUP_DAYS note above)
    warmup_mask = result.index < WARMUP_DAYS
    result.loc[warmup_mask, ["is_anomaly_zscore", "is_anomaly_isoforest"]] = False

    result["is_anomaly_high_confidence"] = result["is_anomaly_zscore"] & result["is_anomaly_isoforest"]
    return result


# --------------------------------------------------------------------------
# Plotting (optional, for portfolio/demo visuals)
# --------------------------------------------------------------------------
def _plot_drug(result: pd.DataFrame, drug: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(result["date"], result["count"], label="Actual", linewidth=1, color="#4C72B0")
    ax.plot(result["date"], result["expected_count"], label="Expected (Holt-Winters)", linewidth=1, linestyle="--", color="#55A868")

    flagged = result[result["is_anomaly_high_confidence"]]
    ax.scatter(flagged["date"], flagged["count"], color="#C44E52", zorder=5, label="High-confidence anomaly", s=30)

    ax.set_title(f"Daily serious AE reports — {drug}")
    ax.set_ylabel("Report count")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path = FIGURES_DIR / f"{drug}_anomalies.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info(f"Saved plot -> {out_path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def run(make_plots: bool) -> None:
    if not COUNTS_PATH.exists():
        raise FileNotFoundError(f"{COUNTS_PATH} not found. Run Phase 1a `counts` first.")

    df = pd.read_csv(COUNTS_PATH, parse_dates=["date"])
    all_results = []

    for drug, drug_df in df.groupby("drug"):
        log.info(f"Processing {drug} ({len(drug_df)} days of data)...")
        result = process_drug(drug_df, drug)
        all_results.append(result)

        n_anomalies = int(result["is_anomaly_high_confidence"].sum())
        log.info(f"  {drug}: {n_anomalies} high-confidence anomalies flagged")

        if make_plots:
            _plot_drug(result, drug)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(ANOMALIES_PATH, index=False)
    log.info(f"Saved {len(combined)} rows -> {ANOMALIES_PATH}")

    total_anomalies = int(combined["is_anomaly_high_confidence"].sum())
    log.info(f"TOTAL high-confidence anomalies across all drugs: {total_anomalies}")
    log.info("This anomalies.csv is your Phase 4 RAG trigger input and Phase 5 ground-truth comparison target.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 — forecasting & anomaly detection")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run forecasting + anomaly detection for all drugs")
    p_run.add_argument("--plot", action="store_true", help="Save a PNG chart per drug to outputs/figures/")

    args = parser.parse_args()
    if args.command == "run":
        run(args.plot)


if __name__ == "__main__":
    main()