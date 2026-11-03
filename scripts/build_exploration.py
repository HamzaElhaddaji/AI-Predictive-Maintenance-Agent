"""Create and execute the reproducible Phase 1 exploration notebook."""

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(root / ".runtime/matplotlib")
os.environ["IPYTHONDIR"] = str(root / ".runtime/ipython")
os.environ["JUPYTER_RUNTIME_DIR"] = str(root / ".runtime/jupyter")
cells = [
    nbformat.v4.new_markdown_cell("""# PHM08 turbofan exploration
This notebook uses the bundled run-to-failure training trajectories. The truncated
test trajectories have no released RUL labels and are never labeled using their final cycle.
Sensor numbers follow the repository's 26-column schema; sensor names alone do not
establish a physical failure cause."""),
    nbformat.v4.new_code_cell("""from pathlib import Path
import sys
ROOT = Path.cwd()
if not (ROOT / 'src').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.maintenance.cleaner import read_data, Cleaner, inject_mess
from src.data.preprocessing import SENSORS, OP_SETTINGS
df = read_data(ROOT / 'data/train.txt')
display(pd.DataFrame({'observations': [len(df)], 'engines': [df.unit_id.nunique()],
                     'sensors': [len(SENSORS)], 'missing': [int(df.isna().sum().sum())]}))
display(df.describe().T)
"""),
    nbformat.v4.new_markdown_cell("## Engine lifetimes and operating conditions"),
    nbformat.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df.groupby('unit_id').cycle.max().hist(bins=25, ax=axes[0])
axes[0].set(title='Engine run-to-failure lifetimes', xlabel='Cycles', ylabel='Engines')
axes[1].scatter(df.op_setting_1, df.op_setting_3, s=1, alpha=.15)
axes[1].set(title='Operating regimes', xlabel='Setting 1', ylabel='Setting 3')
plt.tight_layout()
plt.show()
display(df[SENSORS].std().sort_values().to_frame('standard_deviation'))
"""),
    nbformat.v4.new_markdown_cell(
        "## Sensor distributions and correlations\nOperating regimes can confound these associations."
    ),
    nbformat.v4.new_code_cell("""df[['sensor_2','sensor_3','sensor_4','sensor_11','sensor_12','sensor_15']].hist(
    bins=40, figsize=(12, 6))
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(df[SENSORS].corr(), vmin=-1, vmax=1, cmap='coolwarm')
ax.set_xticks(range(len(SENSORS)), SENSORS, rotation=90)
ax.set_yticks(range(len(SENSORS)), SENSORS)
fig.colorbar(im, ax=ax, label='Pearson correlation')
ax.set_title('Raw sensor correlations across operating regimes')
plt.tight_layout()
plt.show()
"""),
    nbformat.v4.new_markdown_cell(
        "## Degradation patterns\nRolling means use past observations only. Variations also reflect operating conditions."
    ),
    nbformat.v4.new_code_cell("""fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for uid in [1, 2, 3]:
    g = df[df.unit_id == uid]
    for ax, sensor in zip(axes, ['sensor_4', 'sensor_11']):
        ax.plot(g.cycle, g[sensor].rolling(30, min_periods=1).mean(), label=f'Engine {uid}')
        ax.set_ylabel(sensor + ' (30-row mean)')
for ax in axes:
    ax.legend()
axes[-1].set_xlabel('Observed cycle')
plt.tight_layout()
plt.show()
"""),
    nbformat.v4.new_markdown_cell("## Reproducible corruption and causal cleaning"),
    nbformat.v4.new_code_cell("""engine_ids = sorted(df.unit_id.unique())
fit_ids = engine_ids[:int(len(engine_ids)*.6)]
cleaner = Cleaner().fit(df[df.unit_id.isin(fit_ids)])
held_out = df[~df.unit_id.isin(fit_ids)]
dirty = inject_mess(held_out, seed=42)
cleaned, quality = cleaner.transform(dirty)
display(pd.Series(quality, name='cleaning_report'))
assert not cleaned.isna().any().any()
assert not cleaned.duplicated(['unit_id', 'cycle']).any()
"""),
    nbformat.v4.new_markdown_cell("""## Modeling decisions
Use engine-disjoint training, validation, and test partitions. Fit cleaning statistics only on
training engines. Build causal window means, standard deviations, and changes, then classify
whether observed end of life is within 30 cycles. Tune the alert threshold only on validation
engines using a configurable false-negative penalty (10 versus 1 by default).
Report precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, and cost on held-out engines.
The notebook's exploration split is illustrative; the canonical seeded split is saved by the trainer.
"""),
]
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Maintenance Python",
    "language": "python",
    "name": "maintenance",
}
NotebookClient(
    nb, timeout=180, kernel_name="maintenance", resources={"metadata": {"path": str(root)}}
).execute()
target = root / "notebooks/01_dataset_exploration.ipynb"
nbformat.write(nb, target)
print(target)
