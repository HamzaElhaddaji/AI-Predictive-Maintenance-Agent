# AI Predictive Maintenance Agent

An executable PHM08 turbofan maintenance project covering the 19 development phases:
reproducible cleaning and failure classification, independent anomaly detection,
conditional agent workflows, Qdrant retrieval with PDF citations, historical comparison,
verification, JSON tool traces, a FastAPI service, and a Streamlit upload dashboard.

The original RUL notebook and agent remain available. The new failure classifier
estimates whether a machine is within **30 cycles of observed end of life**.
It does not infer physical repair procedures.

## Run locally

Use Python 3.12. From this repository:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.lock
.venv/Scripts/python.exe -m src.maintenance.cli train
.venv/Scripts/python.exe -m uvicorn src.maintenance.api:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
.venv/Scripts/python.exe -m streamlit run src/maintenance/ui.py --server.address=127.0.0.1
```

Open http://localhost:8501. API documentation: http://localhost:8000/docs.
Windows convenience launcher: `./scripts/start.ps1`.
On Linux/macOS use `.venv/bin/python` and `.venv/bin/streamlit` instead.
The delivery archive includes the trained classifier; source checkouts train it using the command above.

No LLM key or external vector service is needed locally. An in-memory Qdrant index
loads the files in `knowledge/` at API startup. For persistent local ingestion:

```sh
python -m src.maintenance.cli ingest --docs knowledge --store .qdrant
python -m src.maintenance.cli report --question "Full report for machine 5"
python -m src.maintenance.cli mess
```

The report command writes `reports/fleet.json`; the corruption command writes reproducible
messy and cleaned CSVs. Add approved PDF, Markdown, or text documents to `knowledge/`,
then restart the API. PDFs are extracted page by page; image-only scans require OCR
before ingestion. Source filenames, page numbers, chunk numbers, content hashes,
and similarity scores accompany retrieved text.

## Docker

```sh
docker compose up --build
```

The training job populates a shared model volume, the ML service serves predictions,
the agent service calls ML over HTTP and Qdrant for retrieval, and the UI calls the agent.
Only localhost ports 8000 and 8501 are published. The containers run as a non-root user.
Training runs on a fresh Compose deployment. To retrain an existing deployment,
stop the services, rerun `docker compose run --rm train`, then start services again.
This is a local research deployment; authentication, HTTPS, and access control must
be added before exposing it publicly.

## Input format

Upload a whitespace-delimited `.txt` file with the original **26 columns**, or a `.csv`
with these headers:

```text
unit_id,cycle,op_setting_1,op_setting_2,op_setting_3,sensor_1,...,sensor_21
```

Each row is one observation. IDs and cycles must be positive integers.
Missing measurements, infinities, duplicate engine/cycle pairs, and gross outliers
are repaired using train-fitted statistics and past observations within each engine.
Conflicting duplicates retain the first row. Outliers beyond broad train-fitted
bounds are flagged and imputed. Repairs are reported, not silently hidden.
Histories shorter than 30 observations are accepted with an explicit support warning.
Windows contain the latest 30 observed rows; missing cycles are not fabricated.

The original PHM08 `data/test.txt` and `data/final_test.txt` have no published RUL
labels in this repository. Their last observed cycles are **not** assumed to be failure.
Training labels come only from `data/train.txt` run-to-failure trajectories.

## Evaluation

Fixed seed 42; 130 training engines, 44 validation engines, 44 held-out test engines.
Cleaner statistics and model fitting use training engines only. Threshold selection
uses validation engines only, minimizing `10 × false negatives + false positives`.
These costs are demonstration assumptions and should be adapted to operations.

| Held-out classifier metric | Result |
| --- | ---: |
| Recall | 0.9333 |
| Precision | 0.5630 |
| F1 | 0.7023 |
| ROC-AUC | 0.9622 |
| PR-AUC (average precision) | 0.8405 |
| Alert threshold | 0.225 |
| TN / FP / FN / TP | 7127 / 988 / 91 / 1273 |
| Weighted cost | 1898 |

Metrics are per observed cycle, with the engine groups disjoint across partitions.
Repeated cycles within an engine are correlated; these are not independent confidence
intervals or external deployment validation. Forest scores are not calibrated guarantees.
Regime-specific performance, probability calibration, and real equipment validation
remain deployment work. Isolation Forest is fitted independently on healthy training
windows; its comparison against failure labels is explicitly a proxy evaluation,
because there are no ground-truth anomaly labels. Its failure-horizon ROC-AUC is
0.5242: it is not a second reliable failure predictor. Synthetic sensor-shift tests
check anomaly sensitivity independently; real anomaly validation remains necessary.

`models/maintenance.metrics.json` records metrics, split engine IDs, seed,
training-data SHA-256, library version, and feature importance.

```sh
pytest -q
ruff check src/maintenance test/test_maintenance.py scripts
python -m src.maintenance.evaluate_system
python scripts/evaluation_charts.py
```

The agent/RAG evaluation uses five routing cases and three synthetic document cases.
It measures routing accuracy, retrieval recall@3, MRR, and unsupported-query abstention.
These small contract cases do not establish real manual retrieval quality or live LLM
reliability. Pytest additionally checks malformed inputs, causality, engine isolation,
duplicate handling, API uploads, missing machines, invalid LLM plans, and citations.

Rebuild the exploration notebook:

```sh
python -m ipykernel install --prefix .venv --name maintenance
python scripts/build_exploration.py
```

## Agent behavior and verification

The agent always obtains predictions through a callable tool. It selects analysis,
historical comparison, and retrieval based on the question, then adds investigation
for observed anomalies and document retrieval for risk alerts. Deep evidence is
bounded to the five most urgent machines in a fleet request; all risk rows remain
visible. Asking about one machine scopes the report to that engine.

Verification checks finite probabilities and anomaly scores, threshold consistency,
history length, repair fraction, feature-range support, and citation content hashes. Missing documentation
or weak data support prevents a full verification pass. Recommendations are restricted
to engineering review and investigation; retrieved documents appear as source excerpts.
Untrusted document content is never sent to a tool-executing LLM.

Optional LLM planning uses the original OpenAI-compatible provider configuration:
`LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`. Set these environment variables in
the API process; `.env.example` is a template and is not loaded automatically.
The LLM chooses from an allowlist and never supplies the report's measurements.
Malformed responses or provider failures fall back to local routing. Local routing
is the tested default; no live provider credentials were used in validation.

Tool logs contain trace IDs, names, status, durations, and errors. LLM planner usage
is recorded when the provider returns it. Report JSON includes its trace for inspection.
Sensor uploads and secret API keys are not logged.

## Knowledge sources and limitations

The corpus contains project-authored demonstration guidance and NASA's technical paper
[Damage Propagation Modeling for Aircraft Engine Prognostics](https://ntrs.nasa.gov/citations/20090029214).
The paper explains simulated degradation and is not a manufacturer maintenance manual.
Provenance is recorded in `knowledge/sources.json`. Add approved equipment manuals
before using document retrieval for real maintenance decisions.

Embeddings are deterministic, normalized lexical hashing vectors. They require no model
download or paid API, but do not provide multilingual semantic understanding. Qdrant
stores and ranks the vectors. The client and Docker server use the same 1.13 minor version,
following [Qdrant's compatibility guidance](https://qdrant.tech/documentation/faq/qdrant-fundamentals/).
Deleted source files require a fresh index to remove their old chunks; re-ingesting
an existing source replaces its prior chunks.

The saved legacy XGBoost pickle was incompatible with the installed runtime. A fresh
RUL model is supplied in the portable `models/rul_model.ubj` format, preferred by
the legacy loader. Rebuild it using `python -m src.maintenance.train_legacy`.
That compatibility model uses all run-to-failure data and has no new held-out performance
claim. The historical numbers in `docs/model_evaluation.md` are retained as prior work.

## Repository map

- `src/maintenance/`: cleaner, features, training, evaluation, analysis, retrieval, agent, verification, traces, API, and UI.
- `src/data/`, `src/tools/`, `src/agent/`: original RUL workflow.
- `notebooks/01_dataset_exploration.ipynb`: executed exploration.
- `test/`: original and new integration/contract tests.
- `scripts/`: exploration, charts, and Windows launcher.
- `knowledge/`: source corpus and provenance.
- `docs/roadmap.md`: phase-to-implementation map.
- `.github/workflows/ci.yml`: lint, tests, agent/RAG evaluation, Compose validation.

Do not load untrusted joblib/pickle files. Uploaded data cannot select or replace model artifacts.
