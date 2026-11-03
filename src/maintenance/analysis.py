import numpy as np

from src.maintenance.features import SENSORS


def analyze_machine(frame, unit_id):
    g = frame[frame.unit_id == unit_id].sort_values("cycle").tail(30)
    if g.empty:
        raise ValueError(f"Machine {unit_id} not found")
    trends = []
    for sensor in SENSORS:
        values = g[sensor].to_numpy()
        slope = float(np.polyfit(g.cycle.to_numpy(), values, 1)[0]) if len(g) > 1 else 0.0
        trends.append(
            {
                "sensor": sensor,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "slope_per_cycle": slope,
                "relative_change": float((values[-1] - values[0]) / max(abs(values.mean()), 1e-8)),
            }
        )
    trends.sort(key=lambda row: abs(row["relative_change"]), reverse=True)
    corr = g[SENSORS].corr().replace([np.inf, -np.inf], np.nan)
    pairs = [
        {"sensors": [a, b], "correlation": float(corr.loc[a, b])}
        for i, a in enumerate(SENSORS)
        for b in SENSORS[i + 1 :]
        if np.isfinite(corr.loc[a, b])
    ]
    pairs.sort(key=lambda row: abs(row["correlation"]), reverse=True)
    return {
        "unit_id": int(unit_id),
        "window_rows": len(g),
        "trends": trends,
        "correlations": pairs[:5],
        "interpretation": "Observed associations; operating settings can confound sensor trends.",
    }


def similar_machines(x, bundle, limit=3):
    reference = bundle["reference"]
    scaled = bundle["scaler"].transform(reference)
    query = bundle["scaler"].transform(x)
    distance = np.linalg.norm(scaled - query[0], axis=1) / np.sqrt(scaled.shape[1])
    candidates = bundle["reference_meta"].copy()
    candidates["distance"] = distance
    candidates = candidates.sort_values("distance").drop_duplicates("unit_id").head(limit)
    return candidates.to_dict(orient="records")
