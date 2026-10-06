"""
data_tools.py — transforme des données brutes de capteurs en features
prêtes pour le modèle, en réutilisant le pipeline existant
(src/data/preprocessing.py + src/data/windowing.py).
"""

import os
import joblib

from src.data.preprocessing import load_raw, add_regime, normalize, select_sensors
from src.data.windowing import last_window

STATS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "regime_stats.pkl")


def prepare_features(path, stats_path=STATS_PATH):
    """
    Lit des données brutes (même format 26 colonnes que train.txt) et
    retourne (X, unit_ids) prêts pour le modèle :
      - X : une ligne par moteur, résumé (mean/std/slope) de ses 30
            derniers cycles
      - unit_ids : l'identifiant du moteur pour chaque ligne de X

    `stats_path` doit pointer vers les statistiques de régime
    (moyenne/écart-type) calculées sur le jeu d'entraînement.
    """
    stats = joblib.load(stats_path)
    sensors = select_sensors()

    df = add_regime(load_raw(path))
    df = normalize(df, stats, sensors)

    X, unit_ids = last_window(df, sensors)
    return X, unit_ids
