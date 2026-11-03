"""Engine-disjoint training and deployable model bundle."""

import hashlib
import json
from pathlib import Path

import joblib
import sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.maintenance.cleaner import Cleaner, read_data
from src.maintenance.evaluation import classification_metrics, tune_threshold
from src.maintenance.features import FEATURES, features

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "models" / "maintenance.joblib"


def train(path, output=DEFAULT_BUNDLE, horizon=30, seed=42, trees=160):
    if horizon < 1 or trees < 1:
        raise ValueError("horizon and trees must be positive")
    raw = read_data(path)
    ids = sorted(raw.unit_id.unique())
    train_ids, rest = train_test_split(ids, test_size=0.4, random_state=seed)
    val_ids, test_ids = train_test_split(rest, test_size=0.5, random_state=seed)
    cleaner = Cleaner().fit(raw[raw.unit_id.isin(train_ids)])
    partitions = []
    for selected in (train_ids, val_ids, test_ids):
        cleaned, _ = cleaner.transform(raw[raw.unit_id.isin(selected)])
        x, ordered = features(cleaned)
        rul = ordered.groupby("unit_id").cycle.transform("max") - ordered.cycle
        y = (rul <= horizon).astype(int)
        if y.nunique() != 2:
            raise ValueError("Each partition must contain healthy and near-failure observations")
        partitions.append((x, y, ordered, rul))
    x, y, ordered, rul = partitions[0]
    classifier = RandomForestClassifier(
        n_estimators=trees,
        max_depth=16,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    ).fit(x, y)
    scaler = StandardScaler().fit(x)
    scaled = scaler.transform(x)
    # Unsupervised second model; fit only the healthy training observations.
    anomaly = IsolationForest(
        n_estimators=120, contamination=0.03, random_state=seed, n_jobs=-1
    ).fit(scaled[y.to_numpy() == 0])
    val_x, val_y, _, _ = partitions[1]
    tuned = tune_threshold(val_y, classifier.predict_proba(val_x)[:, 1])
    threshold = tuned["threshold"]
    test_x, test_y, _, _ = partitions[2]
    test_probability = classifier.predict_proba(test_x)[:, 1]
    metrics = {
        "validation": tuned,
        "test": classification_metrics(test_y, test_probability, threshold),
        "anomaly_test": classification_metrics(
            test_y, -anomaly.decision_function(scaler.transform(test_x)), threshold=0
        ),
        "anomaly_evaluation_note": "Failure-horizon labels are a proxy, not ground-truth anomaly labels.",
        "split_ids": {
            k: [int(i) for i in v]
            for k, v in zip(("train", "validation", "test"), (train_ids, val_ids, test_ids))
        },
        "horizon_cycles": horizon,
        "false_negative_cost": 10,
        "false_positive_cost": 1,
        "seed": seed,
        "sklearn_version": sklearn.__version__,
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "feature_importance": dict(zip(FEATURES, classifier.feature_importances_.tolist())),
    }
    # Store a bounded reference library, with approximately ten snapshots per training engine.
    indexes = (
        ordered.groupby("unit_id")
        .apply(lambda g: list(g.index[:: max(1, len(g) // 10)]), include_groups=False)
        .explode()
        .astype(int)
    )
    reference = x.loc[indexes].reset_index(drop=True)
    reference_meta = ordered.loc[indexes, ["unit_id", "cycle"]].copy()
    reference_meta["observed_rul"] = rul.loc[indexes].to_numpy()
    bundle = {
        "version": 1,
        "cleaner": cleaner,
        "classifier": classifier,
        "scaler": scaler,
        "anomaly": anomaly,
        "threshold": threshold,
        "horizon": horizon,
        "metrics": metrics,
        "feature_min": x.min(),
        "feature_max": x.max(),
        "reference": reference,
        "reference_meta": reference_meta.reset_index(drop=True),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return bundle


def load_bundle(path=DEFAULT_BUNDLE):
    if not Path(path).is_file():
        raise FileNotFoundError("Train first: python -m src.maintenance.cli train")
    return joblib.load(path)  # Only load trusted, locally trained artifacts.


def predict_machine_failure(frame, bundle):
    cleaned, quality = bundle["cleaner"].transform(frame)
    x, ordered = features(cleaned)
    latest = ordered.groupby("unit_id").tail(1).index
    x = x.loc[latest]
    scaled = bundle["scaler"].transform(x)
    spread = (bundle["feature_max"] - bundle["feature_min"]).clip(lower=1e-6)
    unsupported = (
        (x.lt(bundle["feature_min"] - 0.1 * spread) | x.gt(bundle["feature_max"] + 0.1 * spread))
        .mean(axis=1)
        .to_numpy()
    )
    probabilities = bundle["classifier"].predict_proba(x)[:, 1]
    anomaly_score = -bundle["anomaly"].decision_function(scaled)
    machines = [
        {
            "unit_id": int(ordered.loc[i, "unit_id"]),
            "cycle": int(ordered.loc[i, "cycle"]),
            "failure_probability": float(p),
            "alert": bool(p >= bundle["threshold"]),
            "anomaly_score": float(a),
            "anomaly": bool(a > 0),
            "outside_training_range_fraction": float(unsupported[list(latest).index(i)]),
            "history_cycles": int((ordered.unit_id == ordered.loc[i, "unit_id"]).sum()),
        }
        for i, p, a in zip(latest, probabilities, anomaly_score)
    ]
    return (
        {
            "machines": machines,
            "quality": quality,
            "threshold": bundle["threshold"],
            "horizon_cycles": bundle["horizon"],
        },
        cleaned,
        x.reset_index(drop=True),
    )
