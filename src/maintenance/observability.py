import json
import logging
import time
import uuid

logger = logging.getLogger("maintenance.tools")


class Trace:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.events = []

    def call(self, name, function, *args, **kwargs):
        start = time.perf_counter()
        event = {"trace_id": self.id, "tool": name, "status": "ok"}
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            event.update(status="error", error_type=type(exc).__name__)
            raise
        finally:
            event["duration_ms"] = round((time.perf_counter() - start) * 1000, 2)
            self.events.append(event)
            logger.info(json.dumps(event))

    def record_tokens(self, usage):
        self.events[-1]["tokens"] = usage
        logger.info(json.dumps({"trace_id": self.id, "event": "token_usage", "tokens": usage}))
