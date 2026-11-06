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

## Intentional deviations from SAFE's rollout code (collector)
- **Per-episode seeding.** SAFE (`run_libero_eval.py`) seeds once at startup, so skipping already-collected episodes on resume would shift the RNG state. The collector instead re-seeds every episode with `sha256(f"{base_seed}:{task_id}:{init_state_idx}")[:4] % (2**31-1)` and records it as `meta.episode_seed`. Decoding is greedy and the init state is fixed, so actions are not expected to depend on RNG state; the parity check (pilot_report.md #9) verifies this empirically.
- **Resume safety.** Bad files are quarantined to `{root}/_quarantine/` (never deleted); temp files are per-job (`*.pt.{SLURM_JOB_ID|pid}.tmp`); episodes are claimed with O_EXCL lock files.
- **Token probabilities** are stored as `token_probs [T,7]` and `token_entropy [T,7]`, computed by the fork's `compute_token_uncertainty_metrics` (softmax over the action-bin slice of the logits; "prob" = max bin probability, equal to the chosen token's probability under greedy decoding when the argmax is an action-bin token). Needed by SAFE's token-uncertainty baselines and the baseline rows.

## T1–T7
Not run yet. Require a session with CARC access.
