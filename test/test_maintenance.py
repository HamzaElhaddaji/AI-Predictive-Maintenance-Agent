import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data.preprocessing import COLUMNS
from src.maintenance.cleaner import Cleaner, inject_mess, read_data
from src.maintenance.evaluation import classification_metrics
from src.maintenance.features import features
from src.maintenance.knowledge import KnowledgeBase
from src.maintenance.model import predict_machine_failure, train
from src.maintenance.workflow import MaintenanceAgent, rule_plan


@pytest.fixture(scope="module")
def fleet():
    rng = np.random.default_rng(10)
    rows = []
    for uid in range(1, 31):
        for cycle in range(1, 31):
            rows.append(
                [uid, cycle, 0.0, 0.0, 100.0] + (rng.normal(0, 0.05, 21) + cycle / 30).tolist()
            )
    return pd.DataFrame(rows, columns=COLUMNS)


@pytest.fixture(scope="module")
def trained(fleet, tmp_path_factory):
    folder = tmp_path_factory.mktemp("model")
    path = folder / "train.csv"
    fleet.to_csv(path, index=False)
    return train(path, folder / "model.joblib", horizon=5, trees=12)


def test_cleaner_is_reproducible_and_preserves_keys(fleet):
    dirty = inject_mess(fleet)
    pd.testing.assert_frame_equal(dirty, inject_mess(fleet))
    repaired, report = Cleaner().fit(fleet).transform(dirty)
    assert not repaired.isna().any().any()
    assert report["duplicates_removed"] > 0
    assert report["outliers_replaced"] > 0
    assert not repaired.duplicated(["unit_id", "cycle"]).any()
    invalid = fleet.copy()
    invalid.loc[0, "unit_id"] = np.nan
    with pytest.raises(ValueError, match="positive finite integers"):
        Cleaner().fit(fleet).transform(invalid)


def test_imputation_and_features_never_see_future(fleet):
    fit = Cleaner().fit(fleet)
    one = fleet[fleet.unit_id == 1].copy()
    one.loc[one.index[0], "sensor_2"] = np.nan
    before, _ = fit.transform(one.iloc[:10])
    after, _ = fit.transform(one)
    pd.testing.assert_frame_equal(before, after.iloc[:10])
    x_before, _ = features(before)
    x_after, _ = features(after)
    pd.testing.assert_frame_equal(x_before, x_after.iloc[:10])
    assert x_before.iloc[0].filter(like="_delta").eq(0).all()


def test_splits_disjoint_and_inference_unlabeled(fleet, trained):
    splits = trained["metrics"]["split_ids"]
    assert not set(splits["train"]) & set(splits["validation"])
    assert not set(splits["train"]) & set(splits["test"])
    assert not set(splits["validation"]) & set(splits["test"])
    result, _, _ = predict_machine_failure(fleet, trained)
    assert len(result["machines"]) == 30
    assert all(0 <= m["failure_probability"] <= 1 for m in result["machines"])
    json.dumps(result, allow_nan=False)


def test_cost_sensitive_metrics():
    result = classification_metrics([0, 0, 1, 1], [0.1, 0.8, 0.3, 0.9], 0.5)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["cost"] == 11


def test_citable_retrieval_and_reingestion(tmp_path):
    (tmp_path / "manual.txt").write_text(
        "Bearing vibration inspection. Check bearing vibration trends."
    )
    kb = KnowledgeBase()
    try:
        assert kb.ingest(tmp_path) == 1
        assert kb.ingest(tmp_path) == 1
        hits = kb.search("bearing vibration inspection")
        assert len(hits) == 1 and hits[0]["source"] == "manual.txt"
        assert hits[0]["page"] == 1 and hits[0]["text"]
        assert kb.search("zyxqwvunrelated") == []
        (tmp_path / "manual.txt").write_text("Temperature sensor inspection.")
        kb.ingest(tmp_path)
        assert not kb.search("bearing vibration")
    finally:
        kb.close()


def test_agent_routing_and_error_handling(fleet, trained):
    agent = MaintenanceAgent(trained)
    assert rule_plan("compare similar engines") == ["similar"]
    result = agent.run(fleet, "Full report for machine 1")
    assert len(result["machines"]) == 1
    assert "similar" in result["plan"]
    assert result["verification"]["predictions_checked"] == 1
    assert "No supporting maintenance documentation retrieved." in result["warnings"]
    assert len(result["evidence"]["1"]["similar_machines"]) == 3
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="not found"):
        agent.run(fleet, "machine 999")


def test_bad_llm_plan_falls_back(fleet, trained):
    def create(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content='{"tools":["execute_shell"]}'))
            ],
            usage=None,
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = MaintenanceAgent(trained, client=client, model="fake").run(fleet, "sensor trends")
    assert "analyze" in result["plan"]
    assert any("local routing" in warning for warning in result["warnings"])


def test_api_validation_and_reports(fleet, trained, monkeypatch):
    from src.maintenance import api

    monkeypatch.setattr(api, "bundle", lambda: trained)
    with TestClient(api.app) as client:
        bad = client.post("/predict", files={"file": ("bad.csv", b"foo\nbar")})
        assert bad.status_code == 422
        response = client.post(
            "/report",
            files={"file": ("fleet.csv", fleet.to_csv(index=False).encode())},
            data={"question": "Full report", "unit_id": "1"},
        )
        assert response.status_code == 200, response.text
        assert len(response.json()["machines"]) == 1
        assert (
            client.post(
                "/report",
                files={"file": ("fleet.csv", fleet.to_csv(index=False).encode())},
                data={"question": "machine 999"},
            ).status_code
            == 422
        )


def test_empty_and_wrong_shape_rejected():
    with pytest.raises(ValueError):
        read_data(io.StringIO("1 2 3"))


def test_verification_rejects_unsupported_or_forged_evidence(fleet, trained):
    from src.maintenance.verification import verify

    result, _, _ = predict_machine_failure(fleet, trained)
    machines = result["machines"]
    forged = [{"source": "manual.txt", "page": 1, "text": "Fabricated", "content_sha256": "wrong"}]
    assert not verify(result, machines, forged, True)["passed"]
    assert not verify(result, machines, [], True)["passed"]
    assert verify(result, machines, [], False)["predictions_valid"]
    unsupported = [{**m, "outside_training_range_fraction": 1.0} for m in machines]
    assert not verify(result, unsupported, [], False)["passed"]



def test_dashboard_renders_report(fleet, trained, monkeypatch):
    import httpx
    from streamlit.testing.v1 import AppTest

    report = MaintenanceAgent(trained).run(fleet, "Full maintenance report")

    def post(*args, **kwargs):
        return httpx.Response(200, json=report, request=httpx.Request("POST", "http://test/report"))

    monkeypatch.setattr(httpx, "post", post)
    dashboard = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "src/maintenance/ui.py", default_timeout=60
    ).run()
    dashboard.sidebar.button[0].click().run()
    assert not dashboard.exception
    assert not dashboard.error
    assert len(dashboard.metric) == 3


def test_anomaly_detects_out_of_distribution_sensor_shift(fleet, trained):
    from src.maintenance.features import SENSORS

    healthy = fleet[fleet.cycle <= 10].copy()
    shifted = healthy.copy()
    shifted[SENSORS] = shifted[SENSORS] + 2
    normal, _, _ = predict_machine_failure(healthy, trained)
    anomalous, _, _ = predict_machine_failure(shifted, trained)
    assert sum(m["anomaly"] for m in anomalous["machines"]) > sum(
        m["anomaly"] for m in normal["machines"]
    )
    assert all(m["outside_training_range_fraction"] > 0.25 for m in anomalous["machines"])
