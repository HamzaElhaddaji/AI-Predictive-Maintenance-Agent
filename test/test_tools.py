from src.tools.data_tools import prepare_features
from src.tools.ml_tools import predict_risk

def test_pipeline_runs():
    X, unit_ids = prepare_features("data/test.txt")
    results = predict_risk(X, unit_ids)
    assert len(results) == len(unit_ids)
    assert all("rul_predit" in r for r in results)
