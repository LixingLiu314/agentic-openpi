# Larger-batch full-B retry and decoded input pipeline

User requested stopping current project experiments, testing less accumulation, and optimizing data loading on 2026-09-08. Global batch256, seed42, M3 step3500 initialization,5000 updates and the existing optimizer/schedule remain fixed.

## Preserved and stopped work

The original full run stopped after620 updates; its last complete checkpoint is step500. The original limited run already completed5000 updates and its selected step1500 remains on Agilex. Both old result directories and all32 fingerprinted sources are unchanged. The R2 cache/training queues were stopped before the backbone pair to prevent automatic dispatch. No RobotWin or unrelated task was signalled. All38 identified project experiment processes exited. Reservation manager/eight workers remain active. See stop_result.json and original_sources_preserved.json in the evidence directory below.

## Actual capacity evidence

Original trainer/data loader, eight ranks, microbatch32 x accumulation1 = global256:8 real full-B updates including optimizer allocation and complete checkpoint save passed. Discarding the first update, mean update time was4.884573 seconds. Historical microbatch2/accumulation16 full-B timing was about10 seconds/update. This is a short measured comparison, not a guarantee for the entire training duration. No GPU telemetry was queried. Allocator cap for the authorized retry is90%, replacing the earlier conservative55% shared-job cap; unrelated jobs remain protected.

## Data pipeline

- Decode each of the534 original train/val videos sequentially once. Test videos remain untouched. Validate source SHA256 against the immutable split.
- Use the exact openpi_client PIL resize from the original input transform, storing uint8 RGB224 and float32 video PTS. A first JAX-resize prototype failed pixel equality and is preserved as a rejected cache; it is not eligible for training.
- Final cache location: .stage1_staging/piper_rgb224_cache_v2. Keep derived caches outside checkpoint assets so checkpoint saving never copies the image cache.
- CachedVideoLeRobotDataset changes only image retrieval. Original LeRobot state/action-window indexing, episode-tail padding, task/subtask sidecars and normalization remain in use. Timestamp selection matches the float32 closest-frame convention; cache/source identities are validated. Read-only memory maps have a bounded per-worker LRU.
- Training uses persistent4 workers/rank, prefetch2, custom pinned batches and nonblocking CPU-to-GPU transfer. Validation keeps the original microbatch2 independently from training, preserving the fixed128-frame/2-draw comparison.
- New trainer is scripts/train_backbone_gradient_largebatch.py; old trainer/model sources remain untouched. Loss routing and model forward are unchanged.

## Gates and restart

Before research launch: exact old/new transformed samples at every episode start/tail plus random and subtask-boundary samples; same-sample CPU read benchmark;8-rank cached training stop4/resume8; all2048 actual scheduled samples/collated tensors exactly equal to the original reader; repeated-original GPU control; native CUDA policy loading. The first cross-independent-run weight-hash gate failed, and an unchanged original-reader repeat also has non-bit-exact weights. All initial losses match exactly; tiny first-gradient differences are observed in both paths. Preserve this numerical reproducibility limitation instead of claiming bit-exact independent GPU trajectories. Failure of data/restore/native checks stops the finite sequence.

Current finite runner: scripts/run_backbone_largebatch_verified.py, identity verified_retry.process.json. The earlier retry.process.json / retry_failure.json record the completed diagnostic attempt; active_attempt.json identifies the current attempt. Research output: checkpoints/pi05_piper_backbone_grad/full_seed42_b32_cache_v1. It starts fresh from original M3; it is not an exact continuation of the old microbatch2 run. Engineering weights are never its initializer. R2 queues remain stopped during this retry.

Direct rank0 W&B schema v1 replaces raw event uploads for this new run. Display set backbone-grad-largebatch-s42; every10 training updates plus step1 and all validation events, explicit optimizer-step axes and separate train/val losses. Final validation-selected checkpoint receives the strict native load gate before candidate.json/complete.json publication.

Evidence directory: logs/pi05_piper_backbone_grad/batch_retry_20260908. The earlier ETA reminder is paused; schedule a single new completion check after the actual restarted throughput is known.

## Current verification

Data equivalence passed547 selected samples plus all2048 actual scheduled engineering inputs. CPU reading improved16.3x(train)/17.5x(val) on the same64 samples. Cached eight-rank update4 save and same-config optimizer/RNG restore through8 completed. Warmed cached update timing is recorded in cached_throughput.json; the native CUDA gate and formal launch are being finalized. Formal completion remains pending.


## Formal restart dispatched

The repeated original-reader control also had non-bit-exact final weights; all three initial action/CE losses matched exactly, and first S-gradient norm differences were below1e-6. The strict weight-hash trial remains recorded as failed, not retroactively rewritten. Byte-exact input checks and same-configuration restore are passed; cross-run bitwise GPU trajectory equivalence is not claimed.

Native CUDA policy gate passed on the cached engineering export: finite50x14, fixed-noise repeat exact, input arrays unchanged, external subtask rejected. The verified finite runner dispatched fresh full_seed42_b32_cache_v1 at 2026-09-08T22:47:23.877638+08:00. Research completion remains pending.

Saved W&B view: https://wandb.ai/xiahy23-tsinghua-university/agentic-openpi-pi05-subtask?nw=e0rximjrisg . The new view isolates the restarted full run from old interrupted traces.


## Formal startup confirmed and follow-up

Actual formal updates3-7 averaged4.82248s/update,53.0848 global examples/s; the mean maximum-rank data wait was0.000955s. This is about2.1x the historical~10s/update full-B throughput. CPU reading improved16-17x, but no additional end-to-end gain from caching alone is claimed: raw batch32 already measured4.88457s/update, close to the cached result. Data contents and all optimizer-update inputs are exact; GPU independent-run weights are not guaranteed bitwise equal.

The direct W&B run is live: full_seed42_b32_cache_v1. Remote history verification matched37 metric values against localJSONL, with distinct train/val axes, no raw event fields, and both first-update images plus the class table uploaded. Local evidence: wandb_live_verification.json.

At startup, estimated remaining time is6.5-7hours, including validation/save/native-gate margin. The existing checkpoint reminder now checks once at2026-09-09 06:00 Asia/Shanghai (2026-09-08 22:00 UTC), replacing the old11:15 check. Do not poll in between; if unfinished then, re-estimate and move the same reminder. No physical robot actions were performed.
