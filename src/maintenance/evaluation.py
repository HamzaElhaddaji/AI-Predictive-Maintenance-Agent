import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y, probability, threshold=0.5, false_negative_cost=10, false_positive_cost=1
):
    y, probability = np.asarray(y), np.asarray(probability)
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)) if len(set(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, probability)) if sum(y) else None,
        "cost": int(fn * false_negative_cost + fp * false_positive_cost),
        "observations": len(y),
    }


def tune_threshold(y, probability):
    results = [classification_metrics(y, probability, t) for t in np.linspace(0.05, 0.95, 37)]
    return min(results, key=lambda row: (row["cost"], -row["recall"]))
