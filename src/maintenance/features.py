import numpy as np
import pandas as pd

from src.data.preprocessing import select_sensors

SENSORS = select_sensors()
FEATURES = [f"{sensor}_{stat}" for stat in ("mean", "std", "delta") for sensor in SENSORS]


def features(frame, window=30):
    """One causal feature vector per observation; no label or engine ID features."""
    ordered = frame.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    pieces = []
    for _, g in ordered.groupby("unit_id", sort=False):
        values = g[SENSORS]
        rolling = values.rolling(window, min_periods=1)
        means = rolling.mean().to_numpy()
        std = rolling.std(ddof=0).fillna(0).to_numpy()
        first = values.to_numpy()[np.maximum(np.arange(len(g)) - window + 1, 0)]
        delta = values.to_numpy() - first
        pieces.append(
            pd.DataFrame(
                np.concatenate([means, std, delta], axis=1), columns=FEATURES, index=g.index
            )
        )
    return pd.concat(pieces).sort_index(), ordered
