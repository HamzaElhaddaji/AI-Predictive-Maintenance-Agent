import numpy as np
import pandas as pd

WINDOW = 30


def make_windows(df, sensors, window=WINDOW, label_col="RUL_cap"):
    """Une ligne = un resume (moyenne, std, pente) des `window` derniers cycles
    se terminant a chaque position, pour chaque moteur. Retourne X, y, groups."""
    rows, labels, groups = [], [], []
    for uid, g in df.groupby("unit_id"):
        g = g.sort_values("cycle")
        vals = g[sensors].to_numpy()
        lab = g[label_col].to_numpy()
        n = len(g)
        for end in range(window, n + 1):
            chunk = vals[end - window:end]
            mean = chunk.mean(axis=0)
            std = chunk.std(axis=0)
            slope = chunk[-1] - chunk[0]
            rows.append(np.concatenate([mean, std, slope]))
            labels.append(lab[end - 1])
            groups.append(uid)
    feat_names = ([f"{s}_mean" for s in sensors]
                  + [f"{s}_std" for s in sensors]
                  + [f"{s}_slope" for s in sensors])
    X = pd.DataFrame(rows, columns=feat_names)
    y = pd.Series(labels, name="RUL")
    groups = pd.Series(groups, name="unit_id")
    return X, y, groups


def last_window(df, sensors, window=WINDOW):
    """Pour chaque moteur : un seul resume, celui des derniers cycles disponibles.
    Utilise sur le test, ou un moteur n a peut-etre pas `window` cycles."""
    rows, ids = [], []
    for uid, g in df.groupby("unit_id"):
        g = g.sort_values("cycle")
        vals = g[sensors].to_numpy()
        chunk = vals[-window:] if len(g) >= window else vals
        mean = chunk.mean(axis=0)
        std = chunk.std(axis=0)
        slope = chunk[-1] - chunk[0]
        rows.append(np.concatenate([mean, std, slope]))
        ids.append(uid)
    feat_names = ([f"{s}_mean" for s in sensors]
                  + [f"{s}_std" for s in sensors]
                  + [f"{s}_slope" for s in sensors])
    X = pd.DataFrame(rows, columns=feat_names)
    ids = pd.Series(ids, name="unit_id")
    return X, ids
