# Phase implementation map

| Phase | Implementation |
| --- | --- |
| 0 · Setup | pyproject.toml, requirements.lock, .gitignore, isolated virtual environment, pytest, CI and codex branch |
| 1 · Exploration | Executed 01_dataset_exploration.ipynb: distributions, correlations, regimes, lifetimes, degradation |
| 2 · Messy data | cleaner.py, seeded inject_mess, causal imputation, repair report, CLI mess |
| 3 · Classifier | features.py and model.py; Random Forest with engine-disjoint partitions |
| 4 · ML evaluation | ROC/PR AUC, confusion matrix, precision/recall/F1, cost, validation-only threshold tuning |
| 5 · Tools | predict_machine_failure and analysis/retrieval/history callable functions |
| 6 · Agent | MaintenanceAgent, local planner, optional bounded LLM planner; original tool loop retained |
| 7 · Data analysis | Per-machine descriptive statistics, slopes, changes, strongest correlations |
| 8 · Anomalies | Independent healthy-training Isolation Forest and separately reported proxy evaluation |
| 9 · Documents | NASA technical PDF, source manifest, project guidance, page-by-page extraction |
| 10 · RAG | Overlapping chunks, normalized lexical hash embeddings, Qdrant, scored page citations |
| 11 · Integration | Report combines risks, anomalies, analysis, retrieved evidence and review actions |
| 12 · Planning | Conditional question routing plus branches triggered by risk and anomaly results |
| 13 · Verification | Data repair, history support, numeric consistency, citation hashes and abstention |
| 14 · Comparison | Scaled nearest historical training snapshots; distinct reference engines |
| 15 · Evaluation | Separate ML metrics, agent routing cases, RAG recall/MRR/abstention, integration tests |
| 16 · Observability | JSON logging with trace IDs, tool timing/status/errors and optional planner token usage |
| 17 · Containers | Non-root Docker image, training job, ML API, agent API, Qdrant and UI services |
| 18 · Interface | Streamlit upload, question, focus engine, chart/table, evidence, quality and JSON download |

## Validation scope

This completes a runnable research implementation of the roadmap. It does not establish
operational fitness for real aircraft. Manufacturer manuals, equipment-specific acceptance
criteria, live LLM evaluations, a larger real-document retrieval benchmark, probability
calibration, access controls, and deployment validation are external deployment requirements.

Docker Compose syntax was validated locally. Container execution could not be checked
because the Docker daemon was stopped. FastAPI and the Streamlit workflow were exercised
outside Docker. See README for reproducible commands and measured classifier results.
