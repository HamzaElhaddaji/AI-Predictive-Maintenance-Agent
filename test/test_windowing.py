import pandas as pd
from src.data.windowing import make_windows, last_window


def _toy_df():
    return pd.DataFrame({
        "unit_id": [1] * 5 + [2] * 5,
        "cycle": list(range(1, 6)) * 2,
        "sensor_a": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
        "RUL_cap": [4, 3, 2, 1, 0, 4, 3, 2, 1, 0],
    })


def test_make_windows_counts_rows():
    df = _toy_df()
    X, y, groups = make_windows(df, ["sensor_a"], window=3)
    # machine 1: positions 3,4,5 -> 3 fenetres ; machine 2: idem -> 3 fenetres
    assert len(X) == 6
    assert len(y) == 6
    assert set(groups.unique()) == {1, 2}


def test_make_windows_label_is_last_cycle_rul():
    df = _toy_df()
    X, y, groups = make_windows(df, ["sensor_a"], window=3)
    assert y.iloc[0] == 2  # fin de la 1ere fenetre de la machine 1 -> cycle 3 -> RUL_cap=2


def test_last_window_one_row_per_machine():
    df = _toy_df()
    X, ids = last_window(df, ["sensor_a"], window=3)
    assert len(X) == 2
    assert list(ids) == [1, 2]
