"""Rebuild the legacy RUL model and export a stable XGBoost model format."""

import joblib
from xgboost import XGBRegressor

from src.data.preprocessing import (
    add_regime,
    add_rul,
    fit_regime_stats,
    load_raw,
    normalize,
    select_sensors,
)
from src.data.windowing import make_windows
from src.maintenance.model import ROOT


def main():
    sensors = select_sensors()
    frame = add_regime(load_raw(ROOT / "data/train.txt"))
    stats = fit_regime_stats(frame, sensors)
    frame = add_rul(normalize(frame, stats, sensors))
    x, y, _ = make_windows(frame, sensors)
    model = XGBRegressor(
        n_estimators=160,
        max_depth=5,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    ).fit(x, y)
    folder = ROOT / "models"
    folder.mkdir(exist_ok=True)
    model.save_model(folder / "rul_model.ubj")
    joblib.dump(stats, folder / "regime_stats.pkl")
    print("Legacy RUL model rebuilt; use classifier held-out metrics for current evaluation.")


if __name__ == "__main__":
    main()
