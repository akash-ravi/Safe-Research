# Compute notes

## T0 — Repo sanity (laptop, CPU)
- Date: 2026-11-06 · Host: macOS (Darwin, Apple silicon laptop, no GPU) · Python 3.10.22 (uv venv, outside repo) · torch 2.14.1 (CPU)
- Base commit: 88d9c50 (main) + setup.py commit on `akash/extraction`
- `uv pip install -e .` (deps now declared in setup.py): OK
- `python -m pytest -q`: **4 passed, 0 failed** (11.7 s; synthetic data only, no downloads)
- `python scripts/list_models.py`: concat_fusion, gated_fusion, residual_fusion, safe_lstm, safe_mlp, single_layer_mlp

### Known issue
Python ≥3.11 fails on the TrainConfig() dataclass default (`failure_prob/conf/__init__.py:284`, `train: TrainConfig = TrainConfig()` -> "mutable default ... not allowed"); fix = `field(default_factory=TrainConfig)`, pending team approval. Not applied (SAFE baseline file). On 3.12 pytest fails at collection with this error.

## Environment separation on CARC
The detector repo (this repo) must not share an environment with the OpenVLA collection env, which pins `torch==2.2.0` / `transformers==4.40.1` (`vla-safe/openvla`). If they do share one env, install this repo with `pip install -e . --no-deps` so `setup.py` does not upgrade or replace the pinned torch. Preferred: two separate conda envs (collection vs. detector training/probing).

## Collection metadata must record the attention implementation
Hidden states differ numerically between `flash_attention_2`, `sdpa` and `eager`. The collector's `meta` must therefore include `attn_implementation` (default `flash_attention_2`, matching SAFE `run_libero_eval.py:72`), and the full collection (pilot through A8) must use one implementation throughout. T6 records the value in `t6_report.json`. `VLA_NORM_ATOL` stays at 0.0: re-applying the same norm module to the same input is expected to be bitwise identical; if it is not, report the numbers rather than loosening the tolerance.

## T1–T7
Not run yet. Require a session with CARC access.
