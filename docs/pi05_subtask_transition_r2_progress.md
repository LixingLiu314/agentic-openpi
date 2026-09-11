# R2 boundary-capability exploration

User goal (2026-09-08): autonomously explore on server180 until a model has a substantial, verified improvement at subtask boundaries. Existing training may run first; queue normally. This goal remains active. A prepared implementation, a running job, or a loadable checkpoint does not complete it.

Keep seed42, no GPU occupancy/utilization inspection, no robot actions, and no intermediate training-step polling after formal step50. The finite experiment should write completed results for subsequent review. Do not stop or duplicate the independently authorized backbone-gradient pair or other workloads.

## Evidence and fixed comparison

R1 step1000 is a failed experiment. Its original artifacts remain intact. The new scorer `src/openpi/training/transition_metrics_v2.py` separates confirmation opportunities from actual model failures, retains single-observation errors, scores forward jumps/backtracking/far-stage errors, and includes failed events in a capped latency cost. Tests include a perfect-prediction case, an incorrect short stage, a stuck phase, future-frame exclusion, and refusing to pass an unchanged model.

`assets/pi05_piper_transition/eggplant_potato/r2_v1/protocol.json` freezes this exploration's comparison before R2 results:

- Same M3 parent hash, original dataset split and train-only normalization. Test data remains sealed.
- Original158 train episodes are divided into138 R2 fit episodes and20 internal calibration episodes, stratified by task with seed42. M3 had previously trained on all158; calibration is held out of R2 updates, not claimed to be unseen by its parent. The original20 validation episodes remain outside R2 fitting/calibration.
- Current boundary window remains±15 frames. Strong semantic evidence requires at least+8 percentage points boundary EM and a positive95% paired episode bootstrap interval; stable EM may drop at most1pp.
- Compare independent empty-history dense and three~0.764s causal rollouts. Failures, short-stage correctness, early switching, jumps, regressions and far-stage errors must not worsen. Failure-inclusive low-rate latency cost must improve at least30%.
- Generated-condition native action errors must stay within+5% on boundary and natural samples; strict real-observation loading must pass. A physical-success claim requires separate evidence and is not made by these gates.

Archived R0/R1 predictions were rescored without GPU inference. Perfect annotations now have zero model failures at all rates. R1 still fails the new criteria. Original and version2 reports both remain available.

## Implemented R2 path

`src/openpi/models_pytorch/temporal_subtask.py` retains the original autoregressive decoder and full vocabulary. It adds a trainable temporal encoder, model-owned prior text, continuous state changes, relative observation ages and a three-way annotation-proxy head (keep / advance / reidentify). No classifier-to-label lookup, hardcoded task progression or fixed timeout advancement is used.

History uses up to4 real observations. Older contextual image features preserve4×4 locations per camera plus a prompt summary. The current observation retains the full original B prefix. Dense and low-rate rollouts and the native policy now use the same selector for received observations near0.764s lags (80ms quantization slack, always strictly before current); missing/expired history is masked. Only raw B features and the external vocabulary embedding are detached. The trainable decoder memory projection and T/D remain on the S side. A zero-initialized residual preserves M3 text outputs at initialization, while D supplies temporal gradients. Physical readiness remains unreviewed; this initial experiment learns annotation-based phase transition evidence and must not be described as an audited physical completion model.

`scripts/train_temporal_subtask.py` implements5000 updates, effective batch256, official500-step warmup/cosine2.5e-5→2.5e-6, saves/calibrates every500. B/A stay outside the optimizer. Text CE and completion supervision use global denominators before fixed-order gradient summation. Boundaries/stable samples are mixed50/50. Past model outputs provide active text, histories refresh from causal fit-only rollouts, and10% synthetic wrong-active examples use other fit observations' model outputs. Calibration chooses the completion keep threshold and checkpoint; original validation is evaluated after selection. No label or future observation is used in deployment input.

`src/openpi/policies/temporal_subtask_policy.py` loads the exact M3 parent plus the new temporal/decoder parameters and returns native50×14 actions. State is per session. The protocol requires run ID, monotonically increasing query sequence, observation timestamp, and explicit offline/execute mode. Execute calls acknowledge the previous chunk and actual published steps; missing/out-of-order acknowledgments fail, partial execution resets history. New runs/tasks/expired histories reset. Session metadata is not fed to the model except relative observation ages. `scripts/serve_temporal_policy.py` and `serving/temporal_policy_server.py` provide separate per-connection serving. Real loopback transport tests verify array serialization and connection isolation using a mock policy; real native-model CPU checks are separate. No model server or robot client was deployed or left running.

## Completed engineering checks

-22 focused tests:5 metrics,5 temporal causality/gradient/initial-map tests,3 session/adoption tests,3 history/completion-proxy tests,3 server tests including real loopback transport,3 orchestration/qualification/resume-comparison tests.
- Actual frozen M3 features for2 train and2 validation frames were cached and read back bit-exactly on CPU. These are engineering artifacts, never a formal cache.
- Actual M3 vocabulary/decoder with those recorded features: original text-generation parity, S/T/D gradients, and exact two-update save/resume passed. The initially unsupported tied embedding alias was corrected using the parent safetensors metadata; failed evidence was preserved.
- Learning-rate values match the repository's actual CosineDecaySchedule at initialization, warmup boundaries, midpoint and final step.
- Actual CPU trainer completed update1, saved, restored optimizer/RNG/config, and completed update2. The separate real native-policy gate passed: finite50x14 output, fixed-noise equality after reset, isolated session state, unchanged input, duplicate-request rejection and rejection of external subtask labels. No robot I/O occurred.

CPU evidence root: `logs/pi05_piper_transition/r2_exploration_seed42/`. Core results: `temporal_cpu_gate/gate.json`, `trainer_cpu_gate/metrics.jsonl`; native check target: `native_cpu_gate.json`. These validate implementation, not improved capability.

Latest prelaunch evidence is sealed in `prelaunch_cpu_checks.json`. Updated `trainer_cpu_gate_v3` completed update1/save/resume/update2 with actual source/runtime/data hashes. Its first-update PNG is byte-identical to the visually inspected v2 figure. `native_cached_history_gate_v2.json` confirms bit-exact inputs and outputs over four real observations, with1/2/3/4 valid history frames. `actions_cpu_gate_v1.json` runs the native paired-action evaluator on one real sample (engineering scope only). `failure_replay_cpu_gate.json` runs reconstructed queries1/94/95: both parent M3 and the two-update engineering R2 incorrectly predict final put at94/95, and the quality checks correctly reject them. This is not a formal R2 result. The original pytest import-path failure log remains; corrected invocation uses `PYTHONPATH=scripts:src` and passes22 tests.

## Queued frozen-feature extraction

The existing backbone pair process was checked with PID creation time and live child processes. It is running. The cache job waits behind that pair and all live managed leases. It does not stop any job or query utilization.

- Canonical queue identity: `logs/pi05_piper_transition/r2_exploration_seed42/cache_queue.process.json`.
- Queue source snapshot: `.stage1_staging/r2_cache_source_v1`; source manifest SHA256 `b04be545e2adf24f0b5644ea5196675355e25d4a0d18ef3732f246dbf9cc61e4`.
- Expected cache: `assets/pi05_piper_transition/eggplant_potato/r2_frozen_m3_cache_v1`.
- Exact raw frozen B memory, compact past summaries, masks, normalized state, and separately stored R0 text predictions. Current-frame labels remain sidecars. No test episodes are read.
- Expected106079 train+13515 validation frames /178 episodes; completion marker `cache_complete.json` is emitted only after all episode outputs match these counts.
- This is feature extraction, not formal optimizer training. Code snapshots prevent subsequent implementation changes from altering the waiting cache job.

## Finite training/evaluation queue dispatched

The training queue is registered and its live PID creation identity was verified. It currently waits for the existing cache job; formal R2 optimizer training has not started at this snapshot.

- Identity: `logs/pi05_piper_transition/r2_exploration_seed42/training_queue.process.json`.
- State: `training_queue_state.json`; the verified dispatch state is `waiting_for_cache_job`.
- Immutable source: `.stage1_staging/r2_training_source_v1`,190 Python files, manifest SHA256 `d20d8aa89c586ad0d1ec538f57e5d363202feb8c00e62c9c66e4abf9c7dde87b`. Do not edit this snapshot or duplicate the queue. Its cache dependency snapshot also remains immutable.
- Planned formal output: `checkpoints/pi05_piper_transition/r2_temporal_seed42_v1`.
- Engineering: eight-rank uninterrupted2 updates versus1+resume1; compare all parameter tensors and every rank's optimizer/RNG plus model-owned histories. Run CUDA native50x14 and four-observation cache/live checks. Engineering checks must pass before formal training starts from M3.
- Formal recipe:5000 optimizer updates, global256=micro8×accum4×world8, seed42, save/calibrate500, official warmup500/cosine2.5e-5→2.5e-6, W&B without system telemetry, and actual first-update visualization. No engineering weights initialize formal training.
- Final evaluation: internal-calibration-selected checkpoint only, original20-val semantic tests, all140 boundaries at offsets−3/0/+3/+15 plus128 natural samples for paired native first15 action comparison (R0/R2 generated, with old/new/drop diagnostic conditions at boundaries), and the frozen99-query approximate failure replay. No robot I/O.
- Final artifacts: selected checkpoint metadata, `selected_validation.json`, `semantic_gates.json`, `policy_load_gate.json`, `action_gate.json`, `stress_gate.json`, and `candidate.json`. The root `experiment_result.json` is published after the finite sequence finishes. `training_complete.json` is an earlier training-stage record and does not by itself certify post-training gates.
- The normal loader requires the exact candidate checkpoint/hash and passed semantic/action/native/stress flags. A failing gate retains a failed candidate. A qualified offline candidate still requires actual evidence review before the active exploration goal can be completed; neither queue code nor training code marks the goal achieved.
- Process or engineering failures emit `training_queue_failure.json` / `experiment_failure.json` and preserve phase logs. No automatic retry of a failed variant, no competing launch, and no lowering of quality gates.

## Required work before goal completion

1. Preserve native/transport/history/first-update CPU evidence. The queue's GPU gates additionally compare real formal batch-cache features with native single-observation features at four causal observations, require identical generated text, and bound feature/probability drift. Do not claim that GPU gate before its output exists.
2. The finite pipeline implementation is complete in `launch_temporal_experiment.py`, `queue_temporal_experiment.py`, and `run_temporal_experiment.py`. Read the canonical `training_queue.process.json` and `training_queue_state.json` for actual dispatch state. It waits for the existing cache and all managed jobs, performs eight-rank exact parameter/optimizer/RNG resume and native/history gates, then starts the fixed formal recipe from M3. It never polls training steps; the trainer writes the step50 milestone and runs final evaluation automatically. Queue dispatch is not formal-training completion.
3. Inspect final calibration/validation results against the fixed gates; run broader paired boundary/native action checks across all140 validation events rather than only one episode per transition group.
4. The real failure diagnostic is sealed in `assets/pi05_piper_transition/eggplant_potato/r2_failure_replay_v1/protocol.json`:99 input reconstructions, with each camera's source timestamp no newer than the queried input and all prediction overlays removed. The constraints were fixed before R2 results: no final put across the run, at least one grasp-handle/move-away recognition at queries94/95, at most one invalid generation. The dataset/video audit supports these phase constraints, not reliable-grasp or physical-ready certification. This record is never used for training or calibration. All comparisons remain explicitly approximate because RGB crops are lossy and resized.
5. If evidence fails, preserve the failed variant, diagnose the actual error and continue exploration. Do not lower the fixed success criteria to declare success. Additional variants should use new immutable configurations and remain seed42.
6. Only claim an improved model once the actual selected checkpoint passes substantial paired semantic improvement, non-regression/action/loading gates and the relevant stress diagnostics. Report residual physical-readiness/closed-loop limits. Leave the active goal unfinished until then.


## FIT/data diagnostic during the verified queue wait

Read `docs/pi05_transition_fit_audit_20260908.md` for the complete follow-up. FIT-only grasp spans have median durations1.70s(handle)/1.53s(eggplant)/1.67s(potato), and none loses all D supervision to the uncertainty window. Acting hands vary, so v2 corrects v1 fixed-hand closure offsets. Typical terminal widths differ by object and are not readiness labels. CPU M3/R1 counterfactuals show q70 width sensitivity, but q94/95 remain final-put for every tested width. In the image/state2x2 crossover, q94 images still produce put with q70 state; q70 images with q94 state produce grasp. These local approximate-input tests implicate stage confusion associated with the visual input, not a single-width explanation. They do not establish physical causality or R2 improvement. Both queued source snapshots were fully rehashed unchanged. Evidence: `fit_and_failure_audit_complete.json`.


## limited 完成后的配对失败回放

独立排队的 limited 骨干梯度实验已完成，原有选择器按固定128帧生成条件 native14 flow MSE 选出 step1500，模型 SHA256 为 `86ead156ecc8b236cf91b49385b72c726e7fd4e13e32f74003389186ae52b42f`。原生加载检查通过，full 实验已由既有流程接续；R2 仍保持原队列。

同一128帧中，M3/limited 的文本准确率分别为82.03125%/82.8125%，5帧纠正、4帧新增错误；抓把手的9帧由判对5帧变为8帧。生成条件归一化 native14 flow MSE 从0.0323465006降至0.0302298395，约下降6.54%。这不是完整边界评估，也不是实际首15步动作误差或实机成功率；不能据此宣布目标完成。检查点在失败回放前已固定，没有按回放结果重新选模。

追加的 CPU FP32 诊断沿原生 prepare_context/generate_subtask 路径，对 M3 与已选 limited 检查点各运行同一99个封存近似失败输入，不运行 A、不与机器人通信。沿用原有失败约束和输入校验和，结果只作为文本诊断。脚本为 `scripts/audit_backbone_failure_text.py`；独立冻结源码 `.stage1_staging/limited_failure_text_source_v1` 的 manifest SHA256 为 `659dd86f95183d75ee2b1f53d15df20f7d6afef925f3b7d4eaa32b9eff4177c4`。

过程身份：`logs/pi05_piper_transition/r2_exploration_seed42/limited_selected_failure_text_v1.process.json`。结果目录同名，`fixed_validation_comparison.json` 已保存上述配对比较；完整回放需等待 `report.json`，本记录不预判结果。此诊断没有改动 R2 的封存源码、训练配置或成功标准。
