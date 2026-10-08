"""
Registre des tools de l'agent.

Role : relier les fonctions de la Phase 5 (data_tools / ml_tools) a l'agent.
 - fonctions "tool" : entrees simples, sortie = dict JSON-serialisable
 - TOOL_SCHEMAS    : description JSON de chaque tool (ce que le LLM voit)
 - execute_tool()  : point d'entree unique, ne leve jamais d'exception
"""

from functools import lru_cache

DEFAULT_DATA_PATH = "data/test.txt"
RISK_ORDER = ["critique", "eleve", "modere", "faible"]


@lru_cache(maxsize=4)
def _predict_fleet(data_path):
    """Prepare les donnees et predit pour TOUS les moteurs (mis en cache)."""
    from src.tools.data_tools import prepare_features
    from src.tools.ml_tools import predict_risk, load_model

    X, unit_ids = prepare_features(data_path)
    return tuple(predict_risk(X, unit_ids, model=load_model()))


# ---------------------------------------------------------------- tools
def get_machine_risk(unit_id, data_path=DEFAULT_DATA_PATH):
    """RUL predit et niveau de risque d'UN moteur."""
    for r in _predict_fleet(data_path):
        if r["unit_id"] == int(unit_id):
            return dict(r)
    return {"error": f"Machine {unit_id} introuvable dans {data_path}."}


def list_high_risk_machines(max_level="eleve", data_path=DEFAULT_DATA_PATH):
    """Liste les moteurs dont le risque est <= max_level, du plus urgent au moins urgent."""
    if max_level not in RISK_ORDER:
        return {"error": f"max_level doit etre dans {RISK_ORDER}."}
    allowed = RISK_ORDER[: RISK_ORDER.index(max_level) + 1]
    machines = sorted(
        (dict(r) for r in _predict_fleet(data_path) if r["niveau_risque"] in allowed),
        key=lambda r: r["rul_predit"],
    )
    return {"niveau_max": max_level, "count": len(machines), "machines": machines}


def get_fleet_summary(data_path=DEFAULT_DATA_PATH):
    """Resume global de la flotte : repartition par niveau de risque, RUL moyen/min."""
    fleet = _predict_fleet(data_path)
    if not fleet:
        return {"error": "Aucune machine trouvee."}
    repartition = {lvl: 0 for lvl in RISK_ORDER}
    for r in fleet:
        repartition[r["niveau_risque"]] += 1
    ruls = [r["rul_predit"] for r in fleet]
    worst = min(fleet, key=lambda r: r["rul_predit"])
    return {
        "n_machines": len(fleet),
        "repartition": repartition,
        "rul_moyen": round(sum(ruls) / len(ruls), 1),
        "rul_min": min(ruls),
        "machine_la_plus_critique": dict(worst),
    }


TOOLS = {
    "get_machine_risk": get_machine_risk,
    "list_high_risk_machines": list_high_risk_machines,
    "get_fleet_summary": get_fleet_summary,
}

# ---------------------------------------------------------------- schemas
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_machine_risk",
            "description": (
                "Retourne la duree de vie restante predite (RUL, en cycles) et le niveau "
                "de risque d'une machine precise. A utiliser quand l'utilisateur parle "
                "d'un numero de machine."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "unit_id": {"type": "integer", "description": "Numero de la machine."},
                },
                "required": ["unit_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_high_risk_machines",
            "description": (
                "Liste les machines a risque, triees de la plus urgente a la moins urgente. "
                "A utiliser pour 'quelles machines sont en danger / a maintenir en priorite'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_level": {
                        "type": "string",
                        "enum": RISK_ORDER,
                        "description": "Niveau de risque maximum inclus (defaut: eleve = critique + eleve).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fleet_summary",
            "description": (
                "Resume global de toutes les machines : repartition par niveau de risque, "
                "RUL moyen et machine la plus critique."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def execute_tool(name, args=None):
    """Execute un tool par son nom. Retourne toujours un dict (jamais d'exception)."""
    func = TOOLS.get(name)
    if func is None:
        return {"error": f"Tool inconnu : {name}."}
    try:
        return func(**(args or {}))
    except FileNotFoundError as e:
        return {"error": f"Fichier manquant : {e}"}
    except Exception as e:  # l'agent doit pouvoir repondre proprement
        return {"error": f"{type(e).__name__}: {e}"}
