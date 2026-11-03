# Development workflow

Use Python 3.12 and install requirements.lock in an isolated virtual environment.
Create a feature branch from main, keep generated CSV reports and model bundles out
of source commits, and include focused tests for behavior changes.

Before opening a pull request:

```sh
ruff check src/maintenance test/test_maintenance.py scripts
pytest -q
python -m src.maintenance.evaluate_system
docker compose config --quiet
```

For changes to training or feature definitions, rerun training and inspect the held-out
metrics, saved split IDs, and evaluation charts. Do not tune against the test engines.
For notebook changes, execute all cells and confirm there are no error outputs.
For interface changes, exercise the upload-to-report flow with the API running.

The completion work is on branch codex/complete-maintenance-platform.
Do not commit .env files, credentials, runtime caches, or uploaded sensor data.
