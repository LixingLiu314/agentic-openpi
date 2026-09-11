# Official π0.5 initialization: three gradient experiments

## Current full recovery

Frozen and limited completed5000 updates, each selected step_003000 with native loading passed. Full v1 was externally stopped after16 updates by a W&B interpreter-check bug; no checkpoint existed. Full v2 restarts from the same official weights and seed, with unchanged model/data/optimization. Resolve logs/pi05_official_gradient_20260908/active_attempt.json for full_retry_v2; do not restart old queues. See the observability document for the corrected checker and protected external-job guard leases.

## 2026-09-09 full-B logging update

The next full run uses the observed trainer; current limited training remains unchanged. Resolve the actual scheduler through logs/pi05_official_gradient_20260908/active_attempt.json. See [B-gradient observability](pi05_backbone_wandb_observability.md) for metrics, the finite queue handoff and validation evidence. Old attempt_v2 is intentionally paused at its managed scheduler only; do not resume it or launch duplicate full training.

This experiment replaces the cancelled M3 continuation experiments. All three arms independently load official π0.5 B/A weights and the same fresh S initialization. No trained project checkpoint contributes weights, optimizer state, or prior training updates.

| Setting | Shared value |
| --- | --- |
| Official input | `checkpoints/pi05_base_pytorch/model.safetensors` |
| Official file SHA256 | `be5b2233cf302a8fd7097e239e0e2c4680fc6f3696a55084983f2225e2d7044e` |
| Architecture | `Pi0Config(pi05=True)`, Gemma2B / action expert300M, horizon50, model action32 / native Piper14 |
| S | New four-layer decoder, width512, 8 heads, identical seed42 initialization |
| Dataset | eggplant-potato; preserved episode split and train-only normalization |
| Steps | 5,000 optimizer updates per arm; all counters start at zero |
| Batch | 32 per GPU × accumulation1 × 8 GPUs = global256 |
| Learning rate | cosine, warmup500, peak2.5e-5, decay5000, end2.5e-6 |
| Optimizer | AdamW, betas(.9,.95), eps1e-8, weight decay1e-10; disjoint S / A+B groups, clipping1 |
| Data loading | audited PIL RGB224 mmap cache, workers4/rank, prefetch2, persistent workers, pinned/nonblocking transfer |
| Validation / save | every500 updates; fixed128 validation frames, 2 flow draws, validation batch2 |
| Subtask conditions | generated autoregressively from the current observation; independent .1 empty-condition dropout |
| Extra training | no teacher-conditioned action warmup, no M1/M2/M3 curriculum, no uncounted S pretraining |
| Seed | 42 only |

The three modes differ only in the permitted action-loss gradient into B:

| Mode | B trainability | S/A |
| --- | --- | --- |
| frozen | All B frozen | CE trains S; action trains A |
| limited | LoRA r16/alpha32 on q/k/v/o and gate/up/down in the last two language blocks; original B and vision frozen | Same |
| full | Shared language/vision B trainable, including SigLIP | Same |

CE never updates B or A. Action loss never updates S. Teacher forcing used to calculate S's cross entropy is separate from the autoregressively generated text that conditions A. The `set_stage("m3")` call inside the model is retained solely as a legacy S/A trainability switch; neither initialization nor exported experiment identity is M3.

The official input audit proves exact official JAX conversion for811 stored tensors. The812th stored tensor is the documented constructor-only, unused expert LM head. This exception is retained transparently; it is not described as an official JAX-derived weight.

## Execution and evidence

New trainer: `scripts/train_official_backbone_gradient.py`. New schema5 variant: `official_pi05_backbone_v1`. Strict native loader: `openpi.policies.official_gradient_policy.create_official_gradient_policy`. The existing robot schema4 loader does not yet recognize this new variant; install the matching entrypoint before eventual robot deployment.

Finite runner: `scripts/run_official_gradient_trio.py`. It performs real CPU initialization/gradient tests for all modes while reservation workers keep the GPUs occupied. Then it obtains a reservation lease and performs eight-rank update/save/resume and native policy checks for all modes. It starts formal frozen, limited, full runs sequentially only after these checks pass. Each formal run starts fresh from official weights, including after the engineering runs. Failures stop the sequence and release its lease so the existing reservation workers resume.

Current attempt is resolved through `logs/pi05_official_gradient_20260908/active_attempt.json`. The registered identity uses PID plus creation time. `phase.json`, `failure.json`, `engineering_passed.json`, each `<mode>_complete.json`, and final `complete.json` describe progress without assistant step polling. Inspect actual evidence before claiming any gate or run completed.

Planned research outputs:

```text
checkpoints/pi05_piper_official_grad/frozen_seed42_v1
checkpoints/pi05_piper_official_grad/limited_seed42_v1
checkpoints/pi05_piper_official_grad/full_seed42_v1
```

Every run writes `initialization.json`, including official weight SHA256, fresh S parameter digest, zero inherited updates and identical-rank initialization. Source/runtime snapshots, optimizer/RNG shards, actual training parameters and immutable data hashes are retained. Limited mode stores an unmerged training file for exact resume and a merged complete inference file. Frozen/limited native checks also compare all603 stored original B tensors and the tied embedding/head alias to the official input.

W&B uses a separate comparison set `official-pi05-backbone-s42-v1`. Do not combine the cancelled M3 continuation runs with this comparison. Preserve the standard separate train/validation metrics, optimizer-update axis, unsmoothed unaveraged curves and collapsed diagnostics/media.

Report a measured remaining duration after bounded startup verification, then check at the estimated completion time. Use the existing `checkpoint` reminder; no fixed short-interval training polling. Checkpoint availability plus native loading is not evidence of physical task success.

## Authorized cleanup

The user requested deleting project models added from September6 onward on the training server. `deletion_manifest.json` records2440 weight/optimizer files totalling3,014,477,260,017 logical bytes; `deletion_result.json` confirms all were deleted after53 identity-checked old experiment/logging processes were stopped. The cleanup includes stage1/M3, transition and old backbone-gradient research/engineering weights. Official inputs/audit source objects, older models, datasets, decoded caches, configurations, reports, sources and Agilex copies were preserved. Historical references to deleted checkpoints do not authorize recovery or restarting obsolete queues.

Saved W&B comparison: https://wandb.ai/xiahy23-tsinghua-university/agentic-openpi-pi05-subtask?nw=3uzlvjh6wm5 . Cloud round-trip verification passed; the view contains only the new official initialization comparison set, with fixed frozen/limited/full colors.

CPU gates completed for all3 modes: all812 stored official base tensors equal after precision conversion; S digest bab5a0bce442eb7e5759a15deea98c1ff0e75b326941e9abff2e3eacc7eb53ee in every arm. CE-to-S isolation, action-to-A/permitted-B isolation and joint/cache parity passed. Full B:596 nonzero action-gradient tensors including437 visual tensors. Limited merged export reproduced actions/text exactly with nonzero adapters. These are implementation checks, not learning-quality results.

Attempt v1 stopped at a native-check assertion that expected604 stored B tensors. All603 unique stored tensors already matched; the remaining state entry is the tied embedding/head alias. Attempt v2 explicitly checks603 stored tensors plus shared storage identity. Only the checker/orchestrator changed; all model/trainer/data sources are unchanged, so completed CPU checks and frozen8-step save/resume evidence are reused with the original source manifest. Formal training still starts independently from official inputs.

All3 eight-rank gates passed in attempt_v2/engineering_passed.json, including save at4, restore to8, native CUDA50x14 output, repeated fixed-noise equality and external-subtask rejection. Frozen/limited original B checks passed603 stored tensors plus the tied alias. Formal frozen was dispatched2026-09-09 00:26 Asia/Shanghai; limited and full are queued behind it. Measured engineering update means:3.6575/3.7014/4.9700 seconds; formal startup measurement and a buffered completion estimate follow separately.

Formal startup verified2026-09-09 00:29 Asia/Shanghai:18 updates from zero, mean3.69636s/update (69.26 global samples/s), average max-rank input wait0.000661s. W&B step1/10 train losses matched local events, and both first-update images plus the per-class Table are present. The first candidate estimate including margin is06:27; existing checkpoint reminder is scheduled once at06:30. Whole-sequence estimate including margin is20:22 (roughly20:00-21:00). See formal_startup_and_eta.json for assumptions; no intermediate formal-step polling.

Matching serving entrypoint scripts/serve_official_gradient_piper.py passed import/help; it has not been started or installed on Agilex. The robot copies of old checkpoints were preserved.
