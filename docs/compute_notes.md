# Compute notes

## T0 — Repo sanity (laptop, CPU)
- Date: 2026-11-06 · Host: macOS (Darwin, Apple silicon laptop, no GPU) · Python 3.10.22 (uv venv, outside repo) · torch 2.14.1 (CPU)
- Base commit: 88d9c50 (main) + setup.py commit on `akash/extraction`
- `uv pip install -e .` (deps now declared in setup.py): OK
- `python -m pytest -q`: **4 passed, 0 failed** (11.7 s; synthetic data only, no downloads)
- `python scripts/list_models.py`: concat_fusion, gated_fusion, residual_fusion, safe_lstm, safe_mlp, single_layer_mlp

### Known issue
Python ≥3.11 fails on the TrainConfig() dataclass default (`failure_prob/conf/__init__.py:284`, `train: TrainConfig = TrainConfig()` -> "mutable default ... not allowed"); fix = `field(default_factory=TrainConfig)`, pending team approval. Not applied (SAFE baseline file). On 3.12 pytest fails at collection with this error.

## T1–T7
Not run yet. Require a session with CARC access.
