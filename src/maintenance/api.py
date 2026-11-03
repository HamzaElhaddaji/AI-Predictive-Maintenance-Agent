"""HTTP boundary. Uploaded content is parsed as data and never deserialized as a model."""

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.maintenance.cleaner import read_data
from src.maintenance.features import features
from src.maintenance.knowledge import KnowledgeBase
from src.maintenance.model import ROOT, load_bundle, predict_machine_failure
from src.maintenance.workflow import build_maintenance_agent

logging.basicConfig(level=logging.INFO, format="%(message)s")
MAX_BYTES = 25 * 1024 * 1024


@lru_cache(maxsize=1)
def bundle():
    return load_bundle(os.environ.get("MODEL_PATH", ROOT / "models/maintenance.joblib"))


@asynccontextmanager
async def lifespan(app):
    app.state.knowledge = None
    app.state.knowledge_error = None
    try:
        for attempt in range(5):
            try:
                kb = KnowledgeBase(url=os.environ.get("QDRANT_URL"))
                kb.ingest(os.environ.get("KNOWLEDGE_DIR", ROOT / "knowledge"))
                app.state.knowledge = kb
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(1)
    except Exception as exc:
        app.state.knowledge_error = type(exc).__name__
        logging.getLogger(__name__).exception("Knowledge store unavailable")
    yield
    if app.state.knowledge:
        app.state.knowledge.close()


app = FastAPI(title="Predictive Maintenance Agent", version="1.0.0", lifespan=lifespan)


def parse_upload(file):
    if file is None:
        return read_data(ROOT / "data/test.txt")
    name = file.filename or ""
    if not name.lower().endswith((".csv", ".txt")):
        raise HTTPException(422, "Upload a .csv or .txt file")
    content = file.file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(413, "Maximum upload size is 25 MiB")
    stream = io.BytesIO(content)
    stream.name = name
    try:
        return read_data(stream)
    except (ValueError, TypeError, UnicodeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/health")
def health():
    ready = os.path.isfile(os.environ.get("MODEL_PATH", ROOT / "models/maintenance.joblib"))
    return {
        "status": "ok" if ready else "model_missing",
        "model_ready": ready,
        "knowledge_ready": app.state.knowledge is not None,
    }


@app.post("/predict")
def predict(file: UploadFile | None = File(None)):
    frame = parse_upload(file)
    try:
        return predict_machine_failure(frame, bundle())[0]
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def remote_predictor(frame, model):
    cleaned, _ = model["cleaner"].transform(frame)
    x, ordered = features(cleaned)
    latest = ordered.groupby("unit_id").tail(1).index
    content = frame.to_csv(index=False).encode()
    response = httpx.post(
        os.environ["ML_SERVICE_URL"].rstrip("/") + "/predict",
        files={"file": ("observations.csv", content, "text/csv")},
        timeout=90,
    )
    response.raise_for_status()
    return response.json(), cleaned, x.loc[latest].reset_index(drop=True)


@app.post("/report")
def report(
    file: UploadFile | None = File(None),
    question: str = Form("Full maintenance report"),
    unit_id: int | None = Form(None),
):
    frame = parse_upload(file)
    try:
        agent = build_maintenance_agent(bundle(), app.state.knowledge)
        result = agent.run(
            frame,
            question,
            unit_id,
            predictor=remote_predictor if os.environ.get("ML_SERVICE_URL") else None,
        )
        if app.state.knowledge_error:
            result["warnings"].append(
                "Knowledge store unavailable; document support could not be checked."
            )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(503, "ML service unavailable") from exc
