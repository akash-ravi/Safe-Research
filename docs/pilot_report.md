# Pilot report (A6) — gates A8 full collection

Status: **not run yet.** Record every number with date, GPU type and commit (hard rule 7).

| # | Check | How | Result |
|---|---|---|---|
| 1 | Hooks preserve generated tokens | Compare token IDs with and without hooks, same execution setup, ≥200 steps | |
| 2 | Every layer index is correct | Per-block equality vs `hidden_states[k]` (T6 / `tests/gpu/test_layer_mapping.py`) on pilot steps | |
| 3 | `final_norm` capture is correct | Per-token `RMSNorm(raw block_32) ≈ final_norm` before pooling (report max abs/rel error) | |
| 4 | Pooling is correct | Recompute first/last/mean from `token_level_sample`, compare | |
| 5 | Alignment and labels survive save/reload | T, s, env_t, action, token IDs, success line up; `label_failure = 1 - success` | |
| 6 | Class balance | Successes and failures per task; flag any task missing a class | |
| 7 | Runtime and storage | Measured steps/sec and bytes/episode → A8 projection | |
| 8 | Interrupted-job recovery | Kill a job mid-episode, resume, verify no duplicates/gaps; also run two concurrent jobs on one task dir and a requeue | |
| 9 | **Parity with SAFE's `run_libero_eval.py`** | `python experiments/robot/libero/parity_check.py --checkpoint $VLA_CKPT --task_id 0 --n_episodes 2 --workdir $SCRATCH/parity` (fork repo, branch `akash/collector`). Same task, init states, checkpoint, attention implementation, and episode-0 seed. Assert identical per-step action token IDs, executed actions, token probs/entropy, env_t, and per-episode length and success. Greedy ⇒ exact equality expected. | |
| 10 | **Out-of-range action tokens** | Per task: count and rate of env steps where any of the 7 chosen token IDs falls outside the action-bin range [vocab_size − n_action_bins + 1, vocab_size] (`meta.n_out_of_range_token_steps`, `meta.out_of_range_token_rate`, per-step `steps.token_out_of_range`). In those steps the fork clips the action and `token_probs` is not the chosen token's probability. SAFE's prob definition is kept unchanged; this only quantifies the issue. Report count/rate per task and flag any task with a non-trivial rate. | |

Feature-match criteria (checks 3–4): cosine similarity alone is not enough. Report token agreement, relative L2 error, norm ratio, and the downstream detector-score difference where applicable.

Attention implementation must be identical for the whole collection (see compute_notes.md).
