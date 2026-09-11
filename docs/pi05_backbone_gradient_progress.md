# Action-loss backbone ablations

User authorization: 2026-09-08. Seed42 only; two models, no additional frozen-B control.

## Shared recipe

Both start from `checkpoints/pi05_piper_stage1/m3_pilot_seed42/step_003500`, with fresh optimizers. Each performs5000 additional optimizer updates, global batch256, saves and evaluates every500. Cosine schedule: warmup500, peak2.5e-5, decay_steps5000, final2.5e-6. Initial LR matches official Optax (`peak/(warmup+1)`). All enabled S/A/B groups use the same schedule. AdamW follows the referenced TrainConfig defaults: betas0.9/0.95, eps1e-8, weight decay1e-10 and norm clip1; S and A+B are clipped separately to preserve branch ownership.

The Aloha obstacle example supplies the common recipe, not a new dataset or robot convention. Retain the immutable eggplant-potato assets,158 training episodes,20 validation episodes, native Piper14, joint deltas/absolute grippers, three camera views,200 prompt tokens, action horizon50/internal action dimension32. Test remains sealed.

## Two models

| Name | B update | Output |
|---|---|---|
| limited_seed42 | LoRA rank16/alpha32, last2 language blocks, q/k/v/o + gate/up/down; vision/original B frozen | `checkpoints/pi05_piper_backbone_grad/limited_seed42` |
| full_seed42 | All shared B parameters eligible, including vision/projector/embedding/language | `checkpoints/pi05_piper_backbone_grad/full_seed42` |

Both: CE->S only, action loss->A+B only. Discrete generated subtask conditions actions,10% empty-condition dropout, no GT curriculum. Shared adapted B is used for both S context and A prefix. Training uses joint prefix/suffix attention with activation checkpointing. Full B recomputes image features with gradients, rather than accidentally reusing detached features. Inference stays on the original cached prefix route. Unused prefix final outputs cannot receive action supervision merely through unfreezing.

## Implementation and provenance

- New model: `src/openpi/models_pytorch/backbone_gradient.py`.
- Trainer: `scripts/train_backbone_gradient.py`; fresh disjoint S / A+B optimizers; mixed-dtype ZeRO shards; source/runtime/data/parent fingerprints; atomic checkpoint publication; explicit same-configuration resume.
- Complete policy loader: `src/openpi/policies/backbone_gradient_policy.py`; schema4 `action_backbone_v1`. LoRA is merged into the ordinary hierarchy tensor layout for deployment; an unmerged training file preserves resume. No changes to original M3/R1 artifacts.
- Real-model gradient/merged-export gate: `scripts/verify_backbone_gradient.py`.
- Native policy loading gate: `scripts/check_backbone_gradient_checkpoint.py`.
- Finite runner: `scripts/run_backbone_gradient_pair.py`; waits on managed-job lock, then eight-rank save/resume and policy gates for both modes before formal limited then full training. Failure exits through the reservation supervisor, restoring reservations.
- Formal packing: microbatch2 x accumulation16 x8 ranks =256. Engineering gate uses smaller update budgets and is never a formal initializer.
- Selected-candidate loading checks establish finite native50x14 output, unchanged input, repeatable fixed-noise inference and rejection of external subtask. They do not establish physical task success or boundary safety.

## Initial status

Code preparation in progress; CPU real-model gates scheduled. No formal optimizer updates or robot actions have occurred yet. The reservation lock is currently owned by a separate RobotWin pipeline; no unrelated job has been stopped. GPU occupancy/utilization has not been queried.

Queue identity and subsequent factual results will be appended below before handoff.

## Concurrent launch and completed CPU gates

The user forbade stopping the current RobotWin pipeline and authorized attempting concurrent work. The launch-time snapshot showed about0.6GiB allocated on each80GiB A800. Each of our ranks is capped at55% CUDA allocator capacity. This cap is a bound on our allocation, not a guarantee that any future RobotWin workload fits.

Reservation revision3 adds independent shared leases without changing the existing job's control file. Only the8 identity-verified guard workers were restarted; manager and RobotWin stayed active. Holding compute/buffers remain released while either job is active. Six CPU lease-state checks and eight-worker adoption passed; the old source and update record are preserved.

CPU real-model gates passed:

- CE reaches only S; action loss reaches A and enabled B, never S.
- Full B:596 tensors received nonzero gradients, including437 visual tensors.
- Limited B:2,179,072 trainable LoRA parameters; frozen vision. At zero initialization only one LoRA factor initially has gradients, as expected.
- Cached inference and joint training action losses agree in FP32.
- Nonzero LoRA adapters merge into the ordinary model layout with exactly equal model-space actions and generated text.
- Learning-rate values match the official CosineDecaySchedule at warmup/decay boundaries.
- Actual CPU trainer update1 -> atomic checkpoint -> optimizer/RNG/config resume -> update2 completed. Eight-rank save/resume remains a separate launch gate.

Dispatch: `scripts/launch_backbone_gradient_pair.py` started the detached concurrent supervisor. Canonical identity: `logs/pi05_piper_backbone_grad/pair_seed42.process.json`. Managed log: `logs/pi05_piper_backbone_grad/pair_seed42/managed.log`; stage logs and failures are alongside it. The finite sequence runs both eight-rank engineering gates, then formal limited and full sequentially, each on8 GPUs alongside the preserved RobotWin task.

The first formal step500 model receives a CPU policy loading/inference check while training continues. Successful evidence is `first_deployable.json`. A15-minute thread heartbeat notifies only on new loadable checkpoints, failures or completion. No robot actions are part of this run.

## Formal training verified

Both8-rank engineering gates passed: each mode performed a real distributed update, atomically saved all rank optimizer/RNG shards, resumed to update2, and loaded its merged/full checkpoint on CUDA. Both yielded finite50x14 native actions, exact repeated fixed-noise outputs, unchanged observations and rejected external subtask labels. Evidence: `logs/pi05_piper_backbone_grad/pair_seed42/engineering_passed.json` and `{limited,full}_engineering_policy.json`.

The finite runner has started `limited_seed42` formal training. The latest verified snapshot completed12 optimizer updates with global batch256, no invalid generation in that batch, and first-update input/output figures saved. Its actual run_config.json was checked against5000 steps, batch256, save500, seed42, warmup500, peak2.5e-5, decay_lr2.5e-6 and8 ranks. Full mode is scheduled automatically after limited finishes. No training-quality conclusion follows from these startup checks.

Read-only status command:

```bash
cd /media/raid/workspace/surongpeng/ws_lixing/agentic-openpi
.venv/bin/python scripts/status_backbone_gradient_pair.py
```

The original RobotWin reservation had already switched to `holding` at Unix1788836997.1147485 **before** the worker upgrade and our dispatch. Its original PID4167989 was subsequently confirmed absent. We sent no stop signal to that job; its exit reason is not established by this task. The reservation's last_exit.json refers to an older smoke run and must not be used to explain the later pipeline's exit. Earlier prose about the original job remaining active was an initial assumption, superseded by this identity check. The CPU shared-lease checks and actual independent lease are valid; this launch does not establish simultaneous high-load compatibility with RobotWin.

Our separate lease correctly keeps fallback buffers/compute released even though the exclusive control file says `holding`. The55% allocator cap remains, leaving headroom for other work. Do not overwrite another job's control state to manage this run.

A separate server entry is prepared: `scripts/serve_backbone_gradient_piper.py --checkpoint <complete schema4 checkpoint>`. It uses the strict loader, native Piper metadata and deterministic warmup, and defaults to loopback port8000. CLI import/help checks passed; no server or robot actions were started. Robot-side installation/GUI recognition of the new variant has not been done yet; do not present the existing M3/R1 GUI as already supporting it. Deployment needs model.safetensors, metadata.json and assets, plus the matching loader/server sources; training_model.safetensors and optimizer shards are for resume only.
