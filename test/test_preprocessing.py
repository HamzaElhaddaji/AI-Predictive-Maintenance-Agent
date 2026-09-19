import pandas as pd
from src.data.preprocessing import add_rul, select_sensors


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
