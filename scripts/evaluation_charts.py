"""Render held-out classifier diagnostics from the saved, seeded split."""

import os
import sys
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".runtime/matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.maintenance.cleaner import read_data
from src.maintenance.evaluation import classification_metrics
from src.maintenance.features import features
from src.maintenance.model import ROOT, load_bundle

bundle = load_bundle()
raw = read_data(ROOT / "data/train.txt")
raw = raw[raw.unit_id.isin(bundle["metrics"]["split_ids"]["test"])]
cleaned, _ = bundle["cleaner"].transform(raw)
x, ordered = features(cleaned)
y = (ordered.groupby("unit_id").cycle.transform("max") - ordered.cycle <= bundle["horizon"]).astype(
    int
)
probability = bundle["classifier"].predict_proba(x)[:, 1]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
ConfusionMatrixDisplay.from_predictions(
    y,
    probability >= bundle["threshold"],
    display_labels=["Healthy", "Within horizon"],
    ax=axes[0, 0],
    colorbar=False,
)
axes[0, 0].set_title("Held-out engines: confusion matrix")
RocCurveDisplay.from_predictions(y, probability, ax=axes[0, 1])
PrecisionRecallDisplay.from_predictions(y, probability, ax=axes[1, 0])
thresholds = np.linspace(0.05, 0.95, 37)
costs = [classification_metrics(y, probability, t)["cost"] for t in thresholds]
axes[1, 1].plot(thresholds, costs)
axes[1, 1].axvline(
    bundle["threshold"], color="red", linestyle="--", label="Validation-selected threshold"
)
axes[1, 1].set(
    xlabel="Alert threshold",
    ylabel="10 Ã— FN + FP",
    title="Test-set cost diagnostic (not used to tune)",
)
axes[1, 1].legend()
fig.tight_layout()
folder = ROOT / "reports"
folder.mkdir(exist_ok=True)
fig.savefig(folder / "classifier-evaluation.png", dpi=150)
print(folder / "classifier-evaluation.png")
