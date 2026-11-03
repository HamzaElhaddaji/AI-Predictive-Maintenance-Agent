"""Train-fitted, causal cleaning. Identifiers are rejected rather than imputed."""

import numpy as np
import pandas as pd

from src.data.preprocessing import COLUMNS

VALUES = COLUMNS[2:]


def validate(frame):
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("No observations supplied")
    out = frame[COLUMNS].apply(pd.to_numeric, errors="coerce")
    keys = out[["unit_id", "cycle"]]
    if not np.isfinite(keys).all().all() or (keys <= 0).any().any() or (keys % 1 != 0).any().any():
        raise ValueError("unit_id and cycle must be positive finite integers")
    out[["unit_id", "cycle"]] = keys.astype(int)
    return out.replace([np.inf, -np.inf], np.nan)


def read_data(path):
    """Accept a named CSV or the original 26-column whitespace format."""
    if str(getattr(path, "name", path)).lower().endswith(".csv"):
        return validate(pd.read_csv(path))
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != len(COLUMNS):
        raise ValueError(f"Expected 26 columns, received {frame.shape[1]}")
    frame.columns = COLUMNS
    return validate(frame)


class Cleaner:
    def fit(self, frame):
        x = validate(frame).drop_duplicates(["unit_id", "cycle"])[VALUES]
        self.medians = x.median().fillna(0)
        q1, q3 = x.quantile(0.01), x.quantile(0.99)
        spread = (q3 - q1).clip(lower=1e-6)
        self.lower, self.upper = q1 - 5 * spread, q3 + 5 * spread
        return self

    def transform(self, frame):
        out = validate(frame)
        duplicate_count = int(out.duplicated(["unit_id", "cycle"]).sum())
        out = out.drop_duplicates(["unit_id", "cycle"], keep="first")
        out = out.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
        values = out[VALUES]
        outliers = values.lt(self.lower) | values.gt(self.upper)
        missing = int(values.isna().sum().sum())
        out[VALUES] = values.mask(outliers)
        out[VALUES] = out.groupby("unit_id")[VALUES].ffill().fillna(self.medians)
        report = {
            "input_rows": len(frame),
            "output_rows": len(out),
            "duplicates_removed": duplicate_count,
            "missing_values": missing,
            "outliers_replaced": int(outliers.sum().sum()),
        }
        report["repair_fraction"] = (missing + report["outliers_replaced"]) / max(
            1, len(out) * len(VALUES)
        )
        return out, report


def inject_mess(frame, seed=42, missing=0.02, duplicates=0.02, outliers=0.005):
    for rate in (missing, duplicates, outliers):
        if not 0 <= rate <= 1:
            raise ValueError("Corruption rates must be between zero and one")
    rng = np.random.default_rng(seed)
    out = validate(frame).copy()
    values = out[VALUES].to_numpy(copy=True)
    values[rng.random(values.shape) < missing] = np.nan
    mask = rng.random(values.shape) < outliers
    values[mask] = values[mask] * 100 + 1e6
    out[VALUES] = values
    return pd.concat([out, out.sample(frac=duplicates, random_state=seed)], ignore_index=True)
