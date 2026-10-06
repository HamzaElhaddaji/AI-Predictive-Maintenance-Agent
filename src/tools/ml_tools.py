"""
ml_tools.py — charge le modèle entraîné et transforme des features en
une prédiction de risque de panne structurée (dict/JSON), utilisable
par un agent.
"""

import os
import joblib
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "xgb_rul_model.pkl")
RUL_CAP = 125

RISK_THRESHOLDS = {
    "critique": 20,
    "eleve": 50,
    "modere": 90,
}


def _rul_to_risk_level(rul):
    if rul <= RISK_THRESHOLDS["critique"]:
        return "critique"
    if rul <= RISK_THRESHOLDS["eleve"]:
        return "eleve"
    if rul <= RISK_THRESHOLDS["modere"]:
        return "modere"
    return "faible"


def load_model(model_path=MODEL_PATH):
    return joblib.load(model_path)


def predict_risk(X, unit_ids, model=None, model_path=MODEL_PATH):
    """
    X : DataFrame de features (une ligne par moteur), comme retourné
        par data_tools.prepare_features.
    unit_ids : l'identifiant de moteur correspondant à chaque ligne de X.
    model : modèle déjà chargé (optionnel) ; si None, chargé depuis
            `model_path`.

    Retourne une liste de dicts, un par moteur :
        {"unit_id": 12, "rul_predit": 43.2, "niveau_risque": "eleve"}
    """
    if model is None:
        model = load_model(model_path)

    raw_pred = model.predict(X)
    clipped_pred = np.clip(raw_pred, 0, RUL_CAP)

    results = []
    for uid, rul in zip(unit_ids, clipped_pred):
        results.append({
            "unit_id": int(uid),
            "rul_predit": round(float(rul), 1),
            "niveau_risque": _rul_to_risk_level(rul),
        })
    return results
