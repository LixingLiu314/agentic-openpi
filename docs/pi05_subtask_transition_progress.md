# R1 transition refinement progress

Updated: 2026-09-07.

## Scope and user instruction

Implementation and training are authorized. Fixed seed42. Stop assistant polling once the formal run reaches50 steps; leave the finite job running. No GPU occupancy/utilization inspection. All GPU work uses the existing reservation supervisor, which retains its handoff/restoration behavior.

## Implemented

- D0 immutable assets: assets/pi05_piper_transition/eggplant_potato/d0_v1.
- Train158 episodes /106079 frames /1106 boundaries; val20 episodes /13515 frames /140 boundaries. Test semantic data unused.
- boundary_index.jsonl, original-label frame tables, handoff_annotations.jsonl, review_queue.json, three-view review.html, protocol.json, archived-R0 event baseline.
- All physical readiness labels remain null/unreviewed. R1a refines original-label boundary recognition; it does not shift labels or assert physical completion.
- S-only weighted per-example CE with globally correct rank/accumulation normalization; fixed50% stable/50% boundary global draws balanced by task/transition/episode/pre-post.
- New decoder-delta checkpoint variant with immutable parent M3 hash, source/runtime/data fingerprints, B/A tensor equality, per-rank optimizer/RNG, exact resume and strict loader.
- Full dense validation and three causal low-frequency sampling phases, first stable arrival metrics, missed/unconfirmed and short-stage accounting. Proxy safeguards and30% low-frequency P95 improvement checked separately from physical readiness.
- Fixed-noise old/new/generated/drop first15 action diagnostics on boundary samples, plus128 natural validation samples.
- First-update input/output figures and durable training logs; W&B bridge disables GPU telemetry.

## Validation and execution

Eight focused unit tests passed. Real eight-rank engineering gate is recorded in logs/pi05_piper_transition/engineering_gate_v1.process.json. It compares two uninterrupted updates with save/resume, verifies frozen tensors, and loads a real policy observation with fixed noise and no external subtask.

Formal planned output: checkpoints/pi05_piper_transition/r1_seed42.
Configuration:1000 updates, eight ranks, microbatch2, accumulation2, global batch32, LR peak1e-5,50-step warmup. Checkpoint50 then250/500/750/1000; complete validation every250. Engineering weights are never used to initialize formal training.

Finite runner: scripts/run_transition_formal.py. After training it verifies the selected checkpoint through the native policy loader and writes pipeline_result.json, candidate.json and RESULT.md. Physical review and robot trial remain separate. No R2/A1/C1 or mechanical-arm actions are automatically scheduled.

## Commands

```bash
# Execute only after the bounded engineering gate passes; each --log must be new.
.venv/bin/python scripts/gpu_reservation.py run --log logs/pi05_piper_transition/r1_seed42.log -- .venv/bin/python scripts/run_transition_formal.py --output checkpoints/pi05_piper_transition/r1_seed42

# Normal service requires a candidate that passed proxy gates.
.venv/bin/python scripts/serve_transition_policy.py --checkpoint checkpoints/pi05_piper_transition/r1_seed42/step_XXXXXX --port 8000

# R0 rollback service; original checkpoint remains intact.
.venv/bin/python scripts/serve_subtask_policy.py --checkpoint checkpoints/pi05_piper_stage1/m3_pilot_seed42/step_003500 --port 8000
```

## Engineering gate follow-up

V1 saved/resumed all eight optimizer/RNG shards successfully, but strict comparison against an uninterrupted process failed (maximum decoder difference5.70e-6; all rank RNG states and update counts match). Failure logs and diagnosis are preserved. V2 fixes deterministic math SDPA and fixed parameter-order gradient allreduce (Ring/Simple, CUBLAS workspace fixed), avoiding DDP bucket-rebuild reduction-order differences. No acceptance tolerance was relaxed. V2 process: logs/pi05_piper_transition/engineering_gate_v2.process.json. Formal training still starts from the untouched M3 baseline after the gate passes.

## Formal launch

V2 eight-rank gate PASSED: decoder parameters exactly equal after restart versus uninterrupted training; frozen B/A unchanged; real validation observation produces finite native50x14 actions, identical repeated fixed-noise output, unchanged input and external-subtask rejection. Formal run launched from original M3, not engineering weights. Process identity: logs/pi05_piper_transition/r1_seed42.process.json; log: logs/pi05_piper_transition/r1_seed42.log; output: checkpoints/pi05_piper_transition/r1_seed42. Observer stops permanently when milestone_50.json appears. Later status is read only on a new user request.

The initial formal dispatch encountered an already-owned managed-job lock before creating a training directory. Its failure record is r1_seed42_lock_busy.process.json. The canonical r1_seed42.process.json now refers to scripts/queue_transition_formal.py, which waits on that lock and starts the same finite1000-step run automatically. Queue log: logs/pi05_piper_transition/r1_seed42_queue.log; actual managed training log: logs/pi05_piper_transition/r1_seed42_queued.log. No unrelated job was stopped, and no GPU utilization/occupancy was queried.

## Assistant monitoring ended at step50

milestone_50.json confirms step_000050 is saved. Step50 loss: 0.04938622564077377. The finite1000-step pipeline continues autonomously. The observer exited without any further training or GPU polling.
