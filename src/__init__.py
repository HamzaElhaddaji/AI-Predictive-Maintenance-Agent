import pandas as pd

OP_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSORS = [f"sensor_{i}" for i in range(1, 22)]
COLUMNS = ["unit_id", "cycle"] + OP_SETTINGS + SENSORS

CONSTANT_SENSORS = ["sensor_1", "sensor_5", "sensor_18", "sensor_19"]
WEAK_SENSORS = ["sensor_10"]
RUL_CAP = 125


def select_sensors():
    dropped = CONSTANT_SENSORS + WEAK_SENSORS
    return [s for s in SENSORS if s not in dropped]


def load_raw(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def add_regime(df):
    df = df.copy()
    df["regime"] = df[OP_SETTINGS].round(1).astype(str).agg("|".join, axis=1)
    return df


def fit_regime_stats(train, sensors):
    # mean and std of each sensor per flight regime, computed on train only
    g = train.groupby("regime")[sensors]
    return g.mean(), g.std()


def normalize(df, stats, sensors):
    mean, std = stats
    df = df.copy()
    m = mean.loc[df["regime"]].to_numpy()
    s = std.loc[df["regime"]].to_numpy()
    df[sensors] = (df[sensors].to_numpy() - m) / s
    return df


def add_rul(df, cap=RUL_CAP):
    df = df.copy()
    df["RUL"] = df.groupby("unit_id")["cycle"].transform("max") - df["cycle"]
    df["RUL_cap"] = df["RUL"].clip(upper=cap)
    return df


def prepare_train(path):
    sensors = select_sensors()
    df = add_regime(load_raw(path))
    stats = fit_regime_stats(df, sensors)
    df = normalize(df, stats, sensors)
    df = add_rul(df)
    return df, stats