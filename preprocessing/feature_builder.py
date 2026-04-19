# preprocessing/feature_builder.py

from __future__ import annotations
import numpy as np
import pandas as pd


def make_binary_label(label: str) -> int | None:
    """
    Maps labels to binary:
    1 = stress
    0 = non-stress
    """
    if label is None:
        return None

    label = str(label).strip().lower()

    if label in {"stress", "stressed", "anxious", "overloaded"}:
        return 1

    if label in {"calm", "neutral", "happy", "relaxed", "non_stress", "non-stress"}:
        return 0

    return None


def safe_std(series: pd.Series) -> float:
    val = series.std()
    return 0.0 if pd.isna(val) else float(val)


def safe_mode(series: pd.Series):
    m = series.mode()
    return m.iloc[0] if not m.empty else None


def compute_rmssd(rr_intervals: list[float]) -> float:
    """
    RMSSD from RR intervals in seconds.
    """
    if rr_intervals is None or len(rr_intervals) < 2:
        return 0.0

    rr = np.array(rr_intervals, dtype=float)
    diffs = np.diff(rr)
    return float(np.sqrt(np.mean(diffs ** 2)))


def build_window_features(
    df: pd.DataFrame,
    window_size: int = 20,
    step: int = 10,
) -> pd.DataFrame:
    """
    Builds window-level features from time-series data.

    Expected possible columns:
    - timestamp
    - hr
    - rr_intervals (optional; list or stringified list)
    - stress_conf
    - calm_conf
    - happy_conf
    - neutral_conf
    - label
    - session_id
    """
    required = {"hr", "label", "session_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []

    df = df.reset_index(drop=True).copy()

    for start in range(0, len(df) - window_size + 1, step):
        w = df.iloc[start:start + window_size].copy()

        row = {
            "hr_mean": float(w["hr"].mean()),
            "hr_std": safe_std(w["hr"]),
            "hr_min": float(w["hr"].min()),
            "hr_max": float(w["hr"].max()),
            "hr_range": float(w["hr"].max() - w["hr"].min()),
            "label": safe_mode(w["label"]),
            "session_id": str(w["session_id"].iloc[0]),
        }

        # Optional facial confidence features
        for col in ["stress_conf", "calm_conf", "happy_conf", "neutral_conf"]:
            if col in w.columns:
                row[f"{col}_mean"] = float(w[col].mean())
                row[f"{col}_std"] = safe_std(w[col])

        # Optional RR / HRV feature
        if "rr_intervals" in w.columns:
            rr_all = []
            for item in w["rr_intervals"].dropna():
                if isinstance(item, list):
                    rr_all.extend(item)
                elif isinstance(item, str):
                    try:
                        parsed = eval(item)
                        if isinstance(parsed, list):
                            rr_all.extend(parsed)
                    except Exception:
                        pass
            row["rmssd"] = compute_rmssd(rr_all)

        rows.append(row)

    out = pd.DataFrame(rows)
    out["label"] = out["label"].apply(make_binary_label)
    out = out.dropna(subset=["label"]).reset_index(drop=True)
    out["label"] = out["label"].astype(int)
    return out