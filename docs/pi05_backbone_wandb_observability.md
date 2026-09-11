# B-gradient W&B observability

## Corrected runtime and recovery, 2026-09-09

Full v1 was stopped at update16 by the checker, which resolved the venv Python symlink and imported a wandb namespace from the base runtime. Its actual update10 cloud metrics are correct. The v2 checker preserves the venv launcher with os.path.abspath and checks sys.prefix/Api before training. The unchanged observed trainer is invoked by a thin provenance wrapper; recovery output is full_seed42_v2, the same fresh official seed42 experiment with zero inherited updates. All old sources remain unchanged. See logs/pi05_official_gradient_20260909_recovery for reproduction, actual cloud comparison and current full_retry_v2 identity. The eight known external RobotWin workers and their live pipeline are registered with the existing guard so it releases compute during their work.

The action expert A and the trainable part of shared backbone B receive the same action flow loss. Subtask decoder S receives subtask CE. There is no independent B objective in these experiments.

The2026-09-09 investigation confirmed limited_seed42_v1 has uploaded train/action_flow_mse_all32 and train/subtask_ce; initial cloud inspection included optimizer updates1..290. At that inspection it was before500, so only validation at update0 existed. Validation remains every500. Separate B gradient norms were missing: the old writer exposed only the joint optim/grad_norm_a_b.

## Display and metrics

Saved comparison: https://wandb.ai/xiahy23-tsinghua-university/agentic-openpi-pi05-subtask?nw=3uzlvjh6wm5

- Learning process: existing train action loss is labelled A/B shared; subtask train CE and validation CE remain separate.
- B gradient section: optim/grad_norm_a, optim/grad_norm_b, optim/grad_norm_b_vision, optim/grad_norm_b_language. These are pre-clipping L2 norms after DDP reduction, read only on rank0 at update1 and each10 updates. Vision is vision_tower plus multi_modal_projector; language includes the remaining B and token embeddings. B norm combines both disjoint groups.
- Diagnostics: optim/b_tensors_with_grad and optim/b_grad_tensor_fraction measure allocated non-None gradients, not the fraction with nonzero values. Existing S and joint A+B norms remain.
- Same trainer/step optimizer-update x-axis; no smoothing or run averaging. display_schema remains1; observability_version is2. Older runs have no fabricated historical gradient data.

The observer does not change gradients, random state, losses, optimizer groups, clipping or parameter updates. A+B still use their original joint norm1 clipping, and S retains its separate optimizer/clipping. New diagnostics are not computed every step or on all ranks.

## Deployment to the next training run

The active limited run remains on the original source and completes all5000 updates. Its126 recorded source files were verified unchanged. The old managed scheduler alone is paused while its training child continues. The finite continuation scripts/continue_official_full_observed.py waits for that child to exit successfully, acquires a shared GPU lease before retiring the old owner, verifies the limited candidate, then starts full_seed42_v1 through scripts/train_official_backbone_gradient_observed.py. No extra experiment or seed is added.

Authoritative queue pointer: logs/pi05_official_gradient_20260908/active_attempt.json. Current continuation: logs/pi05_official_gradient_20260909_wandb_fix/handoff_v1. Do not resume the paused old scheduler. The old scheduler may leave an intentional KeyboardInterrupt record at handoff; consult intentional_handoff.json and the authoritative queue before interpreting it as failure.

Full B retains official audited B/A initialization, identical fresh seed42 S, zero inherited updates,5000 steps, global256, micro32/accumulation1/world8, same cached data, LR and optimizers. Its source archive includes the new observer, entry point and continuation/checker. scripts/verify_official_full_wandb_startup.py performs one finite startup comparison at update10 against real cloud history. Failure aborts that new full run with a recorded error. Actual full-run verification is pending until that run starts.

## Verification evidence

All paths below are relative to logs/pi05_official_gradient_20260909_wandb_fix/:

- unit_tests.log: five CPU tests passed; gradients and RNG unchanged, exact clipped AdamW update equality, mixed BF16/FP32 norms, missing/frozen gradients, metric namespace behavior.
- test_observed_handoff.py.check.log: successful and nonzero child exits exercised; pausing only the parent leaves its child running; Linux zombie exit status verified.
- wandb_smoke_cloud_verified.json: actual CPU toy action MSE, CE and gradient diagnostics uploaded and read back identically. This is an isolated engineering display set, not research training evidence.
- original_sources_unchanged.json and trainer.diff:126 original files unchanged; observed entry differs only in logging, observability provenance and handling W&B initialization errors.
- workspace_update.log: updated existing saved view and verified remote layout, axis, smoothing and filters.
- handoff_v1/arm_verified.json: continuation armed, old scheduler paused, same limited child running, full not started at handoff setup.
- handoff_v1/full_wandb_startup.json will be the actual next-full startup evidence when available; its existence and passed field must be checked before claiming full metrics were verified.
