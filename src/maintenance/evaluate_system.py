"""Small, separate agent and retrieval contract evaluation. No external LLM required."""

import json
import tempfile
from pathlib import Path

from src.maintenance.knowledge import KnowledgeBase
from src.maintenance.model import ROOT
from src.maintenance.workflow import rule_plan


def evaluate():
    routing_cases = [
        ("show sensor trends", ["analyze"]),
        ("compare historical machines", ["similar"]),
        ("maintenance manual inspection", ["retrieve"]),
        ("Full report", ["analyze", "similar", "retrieve"]),
        ("fleet risk", []),
    ]
    correct = sum(rule_plan(question) == expected for question, expected in routing_cases)
    queries = [
        ("bearing vibration", "bearing.txt"),
        ("temperature coolant", "cooling.txt"),
        ("pressure leak", "pressure.txt"),
    ]
    reciprocal_ranks, recalls = [], []
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        for text, name in queries:
            (folder / name).write_text(text + " inspection guidance and maintenance checks.")
        kb = KnowledgeBase()
        try:
            kb.ingest(folder)
            for query, source in queries:
                hits = kb.search(query)
                rank = next((i + 1 for i, hit in enumerate(hits) if hit["source"] == source), None)
                recalls.append(int(rank is not None))
                reciprocal_ranks.append(1 / rank if rank else 0)
            unsupported = kb.search("zyxwvuts")
        finally:
            kb.close()
    result = {
        "agent": {"routing_accuracy": correct / len(routing_cases), "cases": len(routing_cases)},
        "rag": {
            "recall_at_3": sum(recalls) / len(recalls),
            "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
            "unsupported_query_abstained": not unsupported,
            "cases": len(queries),
        },
        "scope": "Synthetic contract cases, not a real manual or live LLM quality benchmark.",
    }
    if correct != len(routing_cases) or not all(recalls) or unsupported:
        raise AssertionError(result)
    output = ROOT / "reports/system-evaluation.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
