import json
from types import SimpleNamespace

import pytest

from src.agent import agent as agent_mod
from src.agent import tool_registry as reg

FLEET = (
    {"unit_id": 1, "rul_predit": 118.2, "niveau_risque": "faible"},
    {"unit_id": 2, "rul_predit": 12.0, "niveau_risque": "critique"},
    {"unit_id": 3, "rul_predit": 45.7, "niveau_risque": "eleve"},
)


@pytest.fixture(autouse=True)
def fake_fleet(monkeypatch):
    monkeypatch.setattr(reg, "_predict_fleet", lambda path: FLEET)


def test_get_machine_risk():
    assert reg.get_machine_risk(3)["niveau_risque"] == "eleve"


def test_unknown_machine_returns_error():
    assert "error" in reg.get_machine_risk(999)


def test_list_high_risk_sorted_by_urgency():
    r = reg.list_high_risk_machines("eleve")
    assert [m["unit_id"] for m in r["machines"]] == [2, 3]


def test_execute_tool_never_raises():
    assert "error" in reg.execute_tool("nope")
    assert "error" in reg.execute_tool("get_machine_risk", {"wrong_arg": 1})


def test_manual_agent_routes_to_right_tool():
    a = agent_mod.ManualAgent(verbose=False)
    assert "Machine 2" in a.run("Analyse la machine 2")
    assert "2 machine(s)" in a.run("Quelles machines sont critiques ?")
    assert "3 machines" in a.run("Donne-moi un resume")


def _tool_call_message():
    call = SimpleNamespace(
        id="c1",
        function=SimpleNamespace(name="get_machine_risk", arguments=json.dumps({"unit_id": 2})),
    )
    return SimpleNamespace(
        content=None,
        tool_calls=[call],
        model_dump=lambda exclude_none=True: {"role": "assistant", "tool_calls": []},
    )


def test_llm_agent_loop_calls_tool_then_answers():
    final = SimpleNamespace(content="La machine 2 est critique.", tool_calls=None)
    replies = iter([_tool_call_message(), final])

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return SimpleNamespace(choices=[SimpleNamespace(message=next(replies))])

    a = agent_mod.LLMAgent(client=FakeClient, model="fake", verbose=False)
    assert a.run("Analyse la machine 2") == "La machine 2 est critique."
