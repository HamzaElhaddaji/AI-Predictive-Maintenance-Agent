import pandas as pd
from src.data.preprocessing import (add_regime, add_rul, fit_regime_stats,
                                    normalize, select_sensors)


def test_rul_last_cycle_is_zero():
    df = pd.DataFrame({"unit_id": [1, 1, 1], "cycle": [1, 2, 3]})
    assert add_rul(df)["RUL"].tolist() == [2, 1, 0]


def test_rul_is_capped():
    df = pd.DataFrame({"unit_id": [1] * 200, "cycle": range(1, 201)})
    assert add_rul(df, cap=125)["RUL_cap"].max() == 125


def test_useless_sensors_removed():
    s = select_sensors()
    assert "sensor_1" not in s and "sensor_10" not in s
    assert len(s) == 16


def test_regime_ignores_setting_2():
    df = pd.DataFrame({"op_setting_1": [10.0, 10.0], "op_setting_2": [0.2, 0.3],
                       "op_setting_3": [20.0, 20.0]})
    assert add_regime(df)["regime"].nunique() == 1


def test_constant_sensor_gives_zero_not_nan():
    df = pd.DataFrame({"op_setting_1": [0.0] * 4, "op_setting_2": [0.0] * 4,
                       "op_setting_3": [100.0] * 4, "sensor_a": [5.0] * 4})
    df = add_regime(df)
    stats = fit_regime_stats(df, ["sensor_a"])
    out = normalize(df, stats, ["sensor_a"])
    assert out["sensor_a"].eq(0).all()
