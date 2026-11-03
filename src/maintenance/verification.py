"""Checks run after tools and before the final report is rendered."""

import hashlib
import math


def verify(result, machines, citations, documentation_requested):
    quality_ok = result["quality"]["repair_fraction"] <= 0.1
    history_ok = all(m["history_cycles"] >= 30 for m in machines)
    domain_ok = all(m.get("outside_training_range_fraction", 0) <= 0.25 for m in machines)
    predictions_ok = all(
        math.isfinite(m["failure_probability"])
        and 0 <= m["failure_probability"] <= 1
        and math.isfinite(m["anomaly_score"])
        and m["alert"] == (m["failure_probability"] >= result["threshold"])
        and m["anomaly"] == (m["anomaly_score"] > 0)
        for m in machines
    )
    citations_ok = all(
        c.get("source")
        and c.get("page", 0) > 0
        and c.get("text")
        and c.get("content_sha256") == hashlib.sha256(c["text"].encode()).hexdigest()
        for c in citations
    )
    documentation_ok = bool(citations) and citations_ok if documentation_requested else None
    return {
        "passed": bool(
            quality_ok
            and history_ok
            and domain_ok
            and predictions_ok
            and citations_ok
            and documentation_ok is not False
        ),
        "data_quality_passed": quality_ok,
        "history_support_passed": history_ok,
        "predictions_valid": predictions_ok,
        "feature_support_passed": domain_ok,
        "documentation_supported": documentation_ok,
        "predictions_checked": len(machines),
        "citations_checked": len(citations),
        "recommendation_policy": "General engineering review actions; no generated repair procedures.",
    }
