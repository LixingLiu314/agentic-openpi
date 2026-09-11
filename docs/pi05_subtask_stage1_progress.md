当前进度：首个种子42的 C0/M1/M2/M3/B1/O 及全套验证已完成。G/B1/O 的 H50 joint RMSE 为 0.14097974/0.14211989/0.13529330；G−B1 的主要误差区间仍包含0，G−O的H50 joint差值95%区间为[0.00239752,0.00884741]。第二种子43已完成M1及500步M2，接续M3；种子44仍在后续队列。修复了旧C0缺少engineering_smoke字段导致收尾流程拒绝的问题，10项回归及真实C0报告/首轮完整配对汇总通过。最终收尾序列已恢复，测试集仍未运行。

更新日期：2026-09-07。服务器项目：`/media/raid/workspace/surongpeng/ws_lixing/agentic-openpi`；分支仍为 `upstream-openpi-main`，起点提交 `215abfb217dbac7d5f1273282331b9b1866c0479`。本轮未切换分支、未提交或推送。

| 工作项 | 已完成内容 | 状态 |
|---|---|---|
| 数据审计 | 198 个 parquet、594 个视频；视频头帧数/FPS、首帧解码、数值有限性、帧/标签/时间戳对齐、文件 SHA256 | 完成 |
| 数据划分 | 158/20/20 episodes，每任务 79/10/10；源路径、精确轨迹和三路视频哈希重复分组 | 完成，三类精确重复组均为 0 |
| 归一化 | 只用 106,079 个训练帧；12 关节相对当前 state 的 delta，两个夹爪保持 native absolute；H=50、episode 末尾 repeat-last | 完成，无退化 quantile 维度 |
| Piper 适配 | 严格三相机；native 数值不做 Trossen 翻转或夹爪变换；32 维补齐；输出恢复 14 维 absolute | 完成并测试 |
| 真实窗口检查 | 训练/验证全部 178 条轨迹的 356 个首尾窗口与源 parquet 逐项比较 | 全部一致 |
| 文本预算 | train+val 全部 119,594 帧；global prompt 最大 72 token，GT subtask 条件 prompt 最大 83；目标含 EOS 为 4–8 | 200/16 的设计预算可保留 |
| 可视抽查 | train/val 每任务各一条：episodes 0、18、99、106；阶段中点及边界前后 3 帧图表 | 已生成；4 条阶段图及 18/106 边界图已查看 |
| 单机训练链路 | 严格加载现有 π₀.₅ checkpoint；完整前向、反向、AdamW、验证和原子保存 | CPU FP32 两步通过 |
| 训练恢复 | 小型随机模型恢复优化器、Python/NumPy/Torch RNG 后，下一更新逐元素一致 | 通过 |
| 多卡基础设施 | DDP；FP32/BF16 分 dtype 的 ZeRO-1 AdamW；每 rank 优化器/RNG 分片；原子 checkpoint | 两 rank 精确恢复和真实 8 卡 step 1 → 2 恢复通过 |
| GPU 接管 | 用户明确允许停止全部当前 GPU 任务，并要求任务结束后立即续占卡 | 已停止已识别旧任务；8 卡守护已验证异常/正常退出自动续占卡 |
| C0 | 8 张 A800；有效 batch=32；1,000 updates；按固定验证 flow MSE 选中 step 1000 | 训练和原单位动作离线评估完成，仍无闭环成功率证据 |
| 模型核心 | Subtask decoder、内部生成条件、冻结策略、两类 loss、model-space 推理 | 已实现，真实模型 CPU 梯度/缓存/等价性检查通过 |
| M1–M3 训练器 | 双优化器/独立裁剪、阶段继承、严格恢复、完整 AR 验证、任务/阶段均衡采样 | CPU/两 rank 精确恢复及真实 8 卡 M1/M2/M3 恢复通过；正式 M1/M2/M3 首轮完成 |
| 原生策略 | 输入不含 subtask；内部生成；默认 10 步 flow；返回 [50,14] native 动作和语义/计时 | 真实 CPU 端到端、原单位转换和不修改输入检查通过 |
| 研究实验 | 首轮 G/B1 与 drop/shuffle/GT 条件干预、时间诊断、置黑图像、边界/延迟已完成 | O 活跃；三种子和最终测试仍待完成 |

模型和数据接口已落实到当前官方分支，新增文件路径如下（均相对服务器项目根目录）：

- `src/openpi/policies/piper_policy.py`：Piper 输入/输出、相机映射和关节 mask。
- `src/openpi/training/stage1_data.py`：可校验 manifest、重复分组、LeRobot 一致的动作窗口。
- `src/openpi/training/local_lerobot_dataset.py`：当前 LeRobot 版本的稀疏 episode 索引兼容层。
- `src/openpi/training/config.py`：`pi05_piper_stage1` 与专用数据配置。
- `src/openpi/training/data_loader.py`：本地 root、split manifest、显式 video backend 和子集兼容层。
- `scripts/prepare_subtask_data.py`：全量审计、划分和训练集归一化。
- `scripts/inspect_subtask_data.py`：文本长度统计与阶段/边界图表。
- `scripts/train_subtask_pytorch.py`：当前仅实现 M0；支持单机或 DDP、优化器状态分片、累积梯度、固定验证、完整保存/恢复。
- `scripts/smoke_stage1_distributed.py`：多进程累积梯度和 checkpoint 恢复验证。
- `src/openpi/training/stage1_data_test.py`、`stage1_training_test.py`：数据、往返、划分、恢复回归测试。
- `src/openpi/models_pytorch/subtask_decoder.py`：4 层、宽度 512、8 heads 的自回归 decoder，新增 19,972,096 个参数；外部冻结词表不注册进 decoder 参数。
- `src/openpi/models/subtask_tokenizer.py`：共享的全局/条件 prompt 和监督目标编码；只序列化真实 14 维 state，包含 EOS 的目标预算为 16，超长明确报错。
- `src/openpi/models_pytorch/pi05_subtask_pytorch.py`：一次图像编码、两次共享 VLM prefix、S/A 参数所有权、独立 loss、内部 greedy 生成和 model-space 动作推理。
- `src/openpi/models_pytorch/subtask_decoder_test.py`、`src/openpi/models/subtask_tokenizer_test.py`：生成器、mask、梯度、EOS、目标及模板测试。
- `scripts/smoke_pi05_subtask.py`：真实 π₀.₅ 的完整梯度隔离、官方空条件等价性和 cache 只读检查。

本轮还修正了 `src/openpi/models_pytorch/pi0_pytorch.py` 中前向预处理始终 `train=True` 的问题，现在遵循 `self.training`。正常训练行为保留，`eval()` 时关闭随机图像增强，便于固定噪声的验证。

实际暴露并修复了两个环境/数据问题：默认 TorchCodec 在该环境无法加载 FFmpeg 共享库，Piper 配置改用已实读验证的 PyAV；当前 LeRobot 对 episode 子集构造紧凑边界数组，却以原始 episode ID 索引它，局部子类只转换动作窗口边界索引，并保留视频路径和元数据中的原始 ID。第一次 GPU 试跑严格权重加载成功，随后在 collate 接口错误处退出；该错误也已修复，失败记录保留。

CPU 真实模型试跑使用现有 `checkpoints/pi05_base_pytorch/model.safetensors`，显式 `pi05=True`，共 3,616,757,520 个模型参数。两次更新分别约 20.75 秒和 19.25 秒。用于链路检查的固定验证只有 2 帧、每帧 1 个噪声/time draw，14 个机器人维度的**归一化 flow MSE** 从 0.179715 到 0.165339；32 维 MSE 从 0.079248 到 0.073617。这不是物理单位动作误差，也不是有效的泛化结论。两步权重仅是 smoke checkpoint，不能作为正式 C0。

CPU 完整 checkpoint：`checkpoints/pi05_piper_stage1/m0_cpu_smoke_seed42/step_000002`，保存模型、优化器、RNG、配置、源码/数据/基础权重指纹和规范化资产。单机源码随后增加了 DDP，因此该旧 smoke 的完整训练恢复会被严格源码指纹检查拒绝；它仍可作为单独权重/数值检查的输入。正式多卡 run 将使用更新后的配置和 checkpoint schema，不复用旧 smoke 的训练身份。

资产与结果位置：

- `assets/pi05_piper_stage1/eggplant_potato/{split.json,norm_stats.json,audit.json}`。
- split SHA256：`ee6bcedf3c32e00b060037010573cacdd14a0319632d015d2b7ca22922dcec64`。
- norm SHA256：`811151009cd2e276b86d1c1abf09a02ad3166aa50ea50d590c47695c8e8b73f4`。
- `logs/pi05_subtask_stage1/data_window_verification.json`。
- `logs/pi05_subtask_stage1/data_inspection/report.json` 与 8 张联系表。
- `logs/pi05_subtask_stage1/m0_cpu_smoke.log`、`m0_cpu_smoke_process.json`。
- `checkpoints/pi05_piper_stage1/m0_cpu_smoke_seed42/metrics.jsonl`。
- `logs/pi05_subtask_stage1/distributed_cpu_smoke/step_000001`：两 rank 小模型恢复验证产物。
- `logs/pi05_subtask_stage1/decoder_full_vocabulary_smoke.json`：使用真实冻结词表 `[257152, 2048]`、合成 prefix 的 decoder 数值和梯度检查。
- `logs/pi05_subtask_stage1/hierarchy_cpu_smoke.json`：使用真实图像/state/global task 和 CPU smoke 权重的完整模型检查。

离线回归命令（从服务器项目根目录运行）：

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 \
  .venv/bin/python -m pytest \
  src/openpi/training/data_loader_test.py \
  src/openpi/training/stage1_data_test.py \
  src/openpi/training/stage1_training_test.py \
  -q -k 'not test_with_real_dataset'
```

数据/训练基础设施已有 12 项通过，1 项需要其他公开数据集的既有测试未运行；decoder 和文本编码器另有 8 项通过。两 rank CPU 分布式验证已完成：全局 batch 梯度一致，AdamW 状态保存/恢复后下一次更新逐元素一致，每个 rank 的随机数状态正确恢复。新增和修改文件的 Ruff 与补丁空白检查已执行，后续变更继续按受影响范围验证。

真实模型集成检查确认：subtask CE 只向 S 产生梯度；flow loss 只向 A 产生梯度；VLM 无梯度且外层 `train()` 后仍处于 eval；S/A 参数集合互斥且均不拥有词表；空 subtask 条件的 flow loss 和两步 Euler 动作与官方路径在设定浮点容差内一致；action loss 反向和去噪没有修改 prefix KV。当前 A 参数组实际为 430,098,464 个参数，包含原 action expert 与动作/时间投影等模块。检查使用 CPU FP32、一个真实训练帧和两步 Euler，GPU BF16/默认十步及多卡仍需后续验证。

早期未训练 S 的内部推理产生了 `[1, 50, 32]` 有限动作，greedy 达到长度上限后按设计使用空条件。后续训练器和策略检查已经接入完整标签 batch、阶段调度、双优化器和原单位输出；这些少量更新的 CPU 权重仍只用于工程检查。C0 现已建立，是否学会生成需以正在运行的 GPU 过拟合和后续自然验证集结果判断。

真实 8 卡短测已完成：microbatch=1、每次更新全局 8 个样本，step 1/2 更新分别约 0.98/1.32 秒；rank 0 的峰值显存分别约 16.88/17.06 GiB。第 1 步退出并严格恢复优化器/RNG 后，第 2 步正常更新、验证、保存和结束。这里只证明真实多卡恢复链路可运行；小型两 rank 对照另外验证了恢复后下一更新逐元素一致。下面 C0 pilot 使用全局 batch=32、1,000 updates、学习率 2e-5、warmup 100、每 100 步固定验证（128 帧、2 个固定 draw），根据验证曲线决定后续预算。

```bash
JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
  .venv/bin/python scripts/gpu_reservation.py run \
  --log logs/pi05_subtask_stage1/m0_pilot_seed42_unique.log -- \
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train_subtask_pytorch.py \
  --output checkpoints/pi05_piper_stage1/m0_pilot_seed42 \
  --steps 1000 --batch-size 1 --accumulation 4 --workers 2 \
  --cpu-threads 4 --warmup 100 --lr 2e-5 \
  --eval-every 100 --eval-samples 128 --eval-draws 2 --checkpoint-every 100
```

用户随后明确授权：“都允许停止，但要记住这个GPU不能空闲，如果我这边的任务跑完了就要立刻续上占卡程序”。已按 PID、创建时间、所属账户、工作目录和启动命令核实原 Ray/RLinf launcher，终止该 launcher 与后代进程；没有停止 tmux server。记录在 `logs/pi05_subtask_stage1/authorized_gpu_takeover_result.json`。此前自动审批关于未获授权共享任务的顾虑已由这次明确授权解决。

`scripts/gpu_reservation.py` 已启动常驻 manager 和 8 个 worker。无训练任务时每卡保留约 67.42 GiB buffer（驱动显示约 69,574 MiB 总占用）；训练交接时释放大块显存，保留 64 MiB buffer 和轻量 heartbeat，持续持有 CUDA context。每 0.25 秒检查受管训练 PID 及创建时间；任务结束或失败后恢复约 85% 显存占用，manager 会重启退出的 worker。异常退出、正常 checkpoint 停止、恢复完成后的续占卡均已实测。状态目录为 `logs/pi05_subtask_stage1/gpu_reservation`。此守护保证任务间的资源占用，并不让 GPU 始终满算力运算；服务器重启后仍需重新启动 manager。

真实多卡暴露并修复两个基础设施问题：官方模型参数含 FP32/BF16，单个 ZeRO 优化器不接受混合 dtype，新增 `src/openpi/training/stage1_optimizer.py`，分别分片且保持原参数精度和 AdamW 更新规则；默认优化器状态全局汇总需要广播大型 Python 序列化对象，改成每 rank 直接保存 `training_rank_000.pt` 至 `training_rank_007.pt`，全部成功后原子发布 checkpoint。完整训练恢复要求相同 world size、配置、数据和源码指纹。单机仍使用 `training.pt`。

新增实现与检查产物：

- `src/openpi/training/subtask_batch.py`：`SubtaskTrainingBatch`，标签留在 observation 外；均衡 subtask 采样按优化器 step 可复现；M3 的 25%/50%/75%/100% 预测条件调度；生成失败保持空条件。
- `src/openpi/models_pytorch/pi05_subtask_pytorch.py`：`set_stage` 和 DDP 可调用的训练 `forward`；M1 只训练 S，M2/M3 的 S/A 参数互斥，CE 与 flow loss 可相加反传后各自优化。
- `scripts/smoke_subtask_training.py` 与 `logs/pi05_subtask_stage1/training_forward_cpu_smoke.json`：真实 CPU FP32 模型三阶段前向/反向通过；同时改变监督文本和 target token 后，纯预测条件下固定随机数的 action loss 逐元素不变。
- `logs/pi05_subtask_stage1/distributed_local_shards_cpu_smoke`：两 rank 混合 dtype 的分片 checkpoint 精确恢复检查。
- `checkpoints/pi05_piper_stage1/m0_ddp8_smoke_v3_seed42/step_000001` 和 `step_000002`：真实 8 卡保存及恢复产物；每个完整 checkpoint 约 19.51 GiB。这两步权重仍是工程检查产物，不是 C0。
- `logs/pi05_subtask_stage1/m0_ddp8_smoke_v3.log` 和 `m0_ddp8_smoke_v3_resume.log`：8 卡完整检查日志。最初 dtype 失败和 v2 汇总过慢的日志保留。
- `checkpoints/pi05_piper_stage1/m0_pilot_seed42`：当前 C0 pilot 目录；完整启动命令记在 `logs/pi05_subtask_stage1/m0_pilot_seed42_start.process.json`。不要重复运行上面的示例命令覆盖该目录。

pilot 在 batch=32 的第 1/2 步用时约 2.89/2.27 秒，rank 0 记录峰值显存约 21.25/23.33 GiB；这里的峰值是 PyTorch 分配量，不等于驱动总占用，也不是所有 rank 的最大值。第 2 步 checkpoint 已成功保存。完整续训日志为 `logs/pi05_subtask_stage1/m0_pilot_seed42_run.log`，命令与 supervisor PID 记录在同名 `.process.json`，计划步数仍为 1,000；退出后仍由常驻守护自动占卡。前两步处于学习率 warmup，固定 128 帧 × 2 draws 的归一化 14 维 flow MSE 约 0.29935 → 0.29930，仅记录链路表现，不据此判断训练效果。

C0 已完成并选择 `checkpoints/pi05_piper_stage1/m0_pilot_seed42/step_001000`，选择记录为 `logs/pi05_subtask_stage1/c0_selection.json`。权重 SHA256 为 `c108e69fd72c0c2030b6c4d7e254f16d380719b661616e10f6cd0e0bee7c387f`。固定验证 14 维归一化 flow MSE 在 step 0/100/300/500/700/900/1000 分别约为 0.299350/0.087618/0.045760/0.040153/0.038783/0.038302/0.038236。末段仍有小幅下降；这里只建立首轮可复用 C0，不能声称所有训练预算已充分收敛。

`scripts/eval_pi05_baseline.py` 已使用 8 卡、128 个自然验证帧、每帧 2 个固定噪声、10 步 flow 采样完成动作评估。结果在 `logs/pi05_subtask_stage1/c0_native_eval_val/report.json` 和各 rank 明细中。H50 的 native 关节 RMSE 约 0.154284，前 10 个有效帧约 0.062782；对应夹爪 RMSE 约 0.019560/0.012972；以 0.045 为开合阈值的 H50 夹爪准确率约 95.14%。这些数值是数据源原生单位，未确认控制器物理单位。连续夹爪输出约 53.87% 落在示教范围 [0,0.09] 之外，当前统计未记录越界幅度，后续须补充幅度/容差统计并核对控制器限制，不能只依据二分类准确率判断可直接真机执行。主评估保留未裁剪预测，后续所有方法沿用相同处理。

该 C0 检查的 `model.sample_actions` 时间 p50/p95 约 266.34/284.04 ms，包含模型内图像/prefix 与去噪，**不含输入转换，因此不是完整 policy 的端到端延迟**。跨/不跨 subtask 边界、逐任务和各维度结果已分别保存。完整策略延迟与语义边界评估仍属后续 M4。

本轮新增核心文件：

- `scripts/train_subtask_hierarchy.py`：M1–M3 独立训练入口，保留已运行 M0 的源码身份；M1 只更新 S，M2/M3 更新互斥 S/A；各自裁剪 1.0。M1 按任务/阶段组合采样，动作阶段保持自然帧分布。M2 更新数不超过 M2+M3 的 20%，M3 四段预测比例为 25%/50%/75%/100%。
- `src/openpi/training/hierarchy_training.py`：双优化器、完整 checkpoint 和分布式自回归验证。阶段切换继承已有分支 optimizer，新增分支单独初始化；各阶段重新开始自己的 LR schedule。同阶段恢复还原每 rank RNG、采样步号和调度位置。
- 研究阶段要求父 run 已完成；工程/过拟合权重不能初始化正式研究 run。M1 按 macro-F1（并以 CE 打破平局）选模，M2/M3 按预测条件动作误差选模。验证只用自然验证集；过拟合检查明确使用训练子集，不伪装成泛化验证。
- 新 checkpoint schema=3，记录阶段、累计 S/A 更新次数、C0/父权重、split/norm、tokenizer 模型指纹和源文件指纹，并归档源文件副本。选中的 C0 及其 run 也补充了经原始指纹逐项核对的 `sources/`，没有修改权重或原配置。
- `src/openpi/policies/subtask_policy.py`：默认正常部署要求研究 M3 checkpoint；工程权重需要显式 debug 选项。正常 `infer` 拒绝 subtask/action 监督字段，返回 native 14 维动作、普通字符串/标量语义信息及完整分段计时。
- `scripts/smoke_subtask_policy.py`：真实无标签、10 步采样、手工原单位重建对照、不修改调用者输入检查；结果为 `logs/pi05_subtask_stage1/native_policy_cpu_smoke.json`。该单帧 CPU 检查不用于 GPU 延迟结论。
- `scripts/verify_hierarchy_freeze.py`：逐 tensor 比较真实 checkpoint。CPU M1 的全部 812 个原模型 tensor 不变；CPU M2/M3 的 604 个冻结 tensor 不变，208 个 A tensor 和 80 个 S tensor 均有更新。GPU M1 首步也验证了 812 个原模型 tensor 全部不变。报告为 `m1/m2/m3_checkpoint_freeze_cpu.json` 和 `m1_checkpoint_freeze_gpu.json`。
- `src/openpi/training/hierarchy_training_test.py` 和 `scripts/smoke_hierarchy_distributed.py`：在已有 momentum/weight decay 的情况下核对隔离，验证阶段继承和下一更新精确恢复。四项新单元检查和三阶段两 rank 检查通过。

真实 CPU trainer 产物：`m1_trainer_cpu_smoke_seed42/step_000002`、`m2_trainer_cpu_smoke_seed42/step_000001`、`m3_trainer_cpu_smoke_seed42/step_000004`（均位于 `checkpoints/pi05_piper_stage1`）。M3 在第 2 步保存并恢复，随后调度继续为 75% 和 100%；累计 S/A 更新为 7/5。它们仅有少量更新，生成仍出现截断，不能用作研究结果。之后加入源码归档和父阶段门槛，早期工程 checkpoint 的严格原配置续训需恢复它对应的源码；权重加载与阶段转换已验证。

当前 GPU 工作是 `checkpoints/pi05_piper_stage1/m1_overfit32_ddp8_seed42`：从选中 C0 全新初始化 S，32 帧覆盖 16 个任务/阶段组合，8 卡有效 batch=32，500 步，S 学习率 1e-4、warmup 25，每 50 步对全部 32 帧完整自回归评估。首步保存和严格恢复已通过；日志为 `logs/pi05_subtask_stage1/m1_overfit32_ddp8_start.log` 与 `m1_overfit32_ddp8_run.log`。恢复后约 0.27 秒/更新，rank 0 峰值 PyTorch 显存约 8.33 GiB。第 50 步 CE 约 1.195，已有 EOS 输出但 exact match 仍为 0；需要等完整过拟合结果，不可只看 teacher-forced CE 宣称生成成功。

接下来先验收 32 帧自回归过拟合，再从同一 C0 全新初始化正式 M1（不要沿用过拟合头），随后进行 M2/M3 和 B1/O/G 等匹配对照。正式研究开跑前，补充运行库版本/本地 Transformers 补丁指纹以完善复现信息；新增字段不得破坏正在运行实验的严格恢复身份。M4 仍需补充全量语义、边界、完整 policy 延迟、干预、时间捷径/无图像诊断及三种子复跑；最终测试集只用于约定的最终报告。真机闭环需要实际设备和控制器约定，不把离线误差替代成功率。

数据仍有三项研究限制：无采集 session ID，因此不能声称跨 session 泛化；精确重复检查不能保证近重复场景隔离；控制器关节和夹爪单位在真机部署前仍需核对。可视抽查显示阶段中点与任务语义大致一致，但相邻阶段边界前后图像经常非常相似，因此应保留方案中的边界延迟/提前/抖动指标，不把单帧阶段识别等同于完整任务规划能力。

32 帧 GPU 过拟合最终验收：`m1_overfit32_ddp8_seed42` 已完成 500 步。从第 250 步到第 500 步完整自回归 exact match/macro-F1 均为 1.0，未知和截断率均为 0；第 500 步 CE 约 1.2012e-5。`logs/pi05_subtask_stage1/m1_overfit500_checkpoint_freeze_gpu.json` 逐项证实 C0 全部 812 个 tensor 未变化。前文第 50 步为历史中间状态。

正式 M1：`checkpoints/pi05_piper_stage1/m1_pilot_seed42`，日志 `logs/pi05_subtask_stage1/m1_pilot_seed42.log`。8 卡 batch 2、accumulation 2，有效 batch 32；计划 2,000 步，S lr=1e-4、warmup 200。从选中 C0 全新初始化 S，没有使用过拟合头；每 100 步对自然验证集固定 128 帧完整自回归评估。最新核实第 800 步 CE 0.0887676、exact match 0.7734375、macro-F1 0.7126245、未知/截断率 0，8 rank 与占卡守护正常。研究 run 已归档 25 个项目 Python 源文件、9 个运行库模块及版本；在其仍需精确恢复期间不修改被记录的源文件。

下一步在新文件中实现 M4 连续帧边界评估、native 动作干预和完整 GPU 策略延迟；正式 M2/M3 需先核实 M1 完成与选模。初始动作预算拟为 M2 500 步 + M3 3,500 步，二者共 4,000 次 A 更新，GT 预热占 12.5%；B1/O 对照须匹配同一 C0、批次分布、A 更新数及学习率阶段。此预算是首轮 pilot，按验证结果评估，不使用测试集调参。


主线恢复记录（2026-09-07）：

- M1 选模记录：`logs/pi05_subtask_stage1/m1_selection.json`；完整语义评估：`logs/pi05_subtask_stage1/m1_dense_val/report.json`。所选权重为 `m1_pilot_seed42/step_001900`。逐项检查显示 604 个 B tensor 和 208 个 A tensor 均未更新，见 `m1_research_checkpoint_freeze.json`。
- 全量自然验证 exact match 0.8383278、macro-F1 0.7851513，显式目标物标签上的目标物准确率 0.9570725，未知输出率 0。三路图像置黑、保留 state/task 的干预得到 exact match 0.1113578、macro-F1 0.0341913；这是输入干预，未重新训练无视觉模型。
- 全 20 条轨迹有 140 个真值切换，138 个匹配、2 个漏检。±10 帧内比例 0.4142857，匹配切换平均有符号偏差 +7.087 帧，绝对偏差 p50/p95 为 12/27.15 帧；110 个延迟、24 个提前、4 个精确。预测总切换 348 次，其中未匹配变化 210 次，平均每轨迹 10.5 次。边界附近 2,940 帧准确率 0.5612245，远离边界 10,575 帧为 0.9153664。原始预测不做平滑；具体匹配定义随报告保存。
- `scripts/eval_subtask_time_baseline.py` 只用训练 episodes 的阶段起点中位数拟合“任务＋已过帧数”，不用图像/state/未来 episode 长度，也不调验证阈值。完整验证 exact match 0.7172031、macro-F1 0.5966069，见 `logs/pi05_subtask_stage1/time_baseline_val/report.json`。
- 新增 `subtask_curriculum.py`：M3 的选模与正常部署必须已累积至少 20% 不使用真值条件的动作更新；M3 必须从 M2 完整 warmup 终点继承。M2=500、M3=3,500 时，最终纯预测区间为 875 次更新。课程、离线指标及双优化器相关 10 项 CPU 回归已通过。
- `m1_eval_smoke_ddp8` 的无标签策略短测使用 8 个样本，完整 `infer` 耗时 p50/p95 约 366.6/380.3 ms，样本过少且尚无最终 M3 权重，不作为最终延迟结论。初始执行 10 帧的 333 ms 周期需要根据最终测量调整。
- 新增 B1/O action-only 控制模型与训练入口 `src/openpi/training/action_control.py`、`scripts/train_subtask_action_control.py`，匹配同一 C0、冻结 B、4,000 次 A 更新、M2/M3 的学习率重启及采样重启，当前在做 CPU 工程验证，尚未形成研究结果。
- 正式 M2/M3 不使用工程 checkpoint。8 卡工程 M2 为 `m2_trainer_ddp8_smoke_seed42`，首步 batch 32 用时约 1.05 秒、rank0 峰值 PyTorch 显存约 9.89 GiB；step1 保存完成，正在按原配置恢复 step2。

GPU 守护最新约定：用户要求占卡时每卡计算利用率和显存占用都达到 80%，已把旧的轻量 heartbeat 改成持续 8192×8192 FP16 矩阵计算，显存目标仍为 85%。稳定状态的 60 秒逐秒采样全通过，通常约 98–99% 计算利用率和 84.9% 显存占用。正常退出约 0.82 秒恢复双阈值；异常退出后也恢复，但严格后续测试捕获过一次 74–83% 短暂计算下降，失败日志保留在 `gpu_reservation_compute_verification`，不宣称该严格测试全通过。用户随后明确表示“占卡不需要这么完美，差不多得了，现在恢复主线”，因此保留 revision 2，不再进一步调优占卡负载。当前 manager/worker PID 以 `gpu_reservation/*.json` 和真实进程身份为准；每个 GPU 训练仍必须经 `gpu_reservation.py run` 交接。


主线动作阶段更新（2026-09-07，正式 M3 中途）：

- 正式 M2 已从 M1 所选 step1900 完成 500 步，terminal checkpoint 为 `m2_pilot_seed42/step_000500`。固定 128 帧 × 2 draws、生成条件下的 native14 归一化 flow MSE 从 0.03913321 降至 0.03580473；选中 step500。累计 S/A 更新为 2400/500。`m2_research_checkpoint_freeze.json` 确认 B 的 604 个 tensor 不变，A 的 208 个和 S 的 80 个 tensor 均有更新。
- M2/M3 真实八卡工程恢复链路均已完成；M3 在 step4 停止、恢复到 step8，比例连续为 75%/100%，仅末步通过选模课程门槛。工程权重没有用于正式模型。
- 正式 M3 已从上述 M2 terminal 接上，输出 `m3_pilot_seed42`，预算 3500 步，8 卡有效 batch32。最新核实 step1600 固定验证生成条件 flow MSE 0.03332100、语义 exact match 0.8046875、macro-F1 0.7227985、未知和截断率 0。这是中途结果，尚不具备选模资格。有限序列 `continue_stage1_action_phase.py` 完成 M3 后会逐 tensor 检查冻结组并退出，占卡守护接回。
- B1 action-only 对照 CPU 五步工程训练、step2→5 严格恢复完成；B604 与 S80 不变，A208 更新。相同两个验证帧上的独立动作 evaluator flow 值 0.03160570 与 trainer 一致，50×14 输出有限。该微型检查仅证明实现链路，不能作为对照质量结论。真值条件 O 的 CPU 五步检查正在进行，正式 B1/O 尚未运行。
- `eval_subtask_offline.py` 与 `eval_action_control.py` 现在保存逐样本 `native_rank_*.npz`，含预测、目标、frame index、draw、condition、valid horizon，以便按最终策略延迟确定统一执行长度后复算；新增归档尚需真实运行核实。最终 native 动作比较与 GPU 完整策略延迟尚待 M3 和匹配对照完成。
- 基础权重来源审计 `audit_pi05_base_weights.py` 正在独立 CPU 进程下载官方 GCS `pi05_base` 固定 generation 参数对象，按 MD5/CRC32C 校验，再用官方转换逻辑与本地 base 逐 tensor 比对。报告目标 `official_base_weight_audit.json`，当前下载尚未完成，不能宣称已经确认逐值一致。
- W&B watcher 和 M3 同步进程存活；本地 metrics.jsonl 仍是持久记录。


后续有限序列已接续（2026-09-07）：

- 新入口 `scripts/continue_stage1_controls.py` 的进程记录为 `logs/pi05_subtask_stage1/controls_phase_sequence.process.json`，日志同名 `.log`。该进程按 PID 和创建时间等待 `continue_stage1_action_phase.py`，随后要求 M3 完整 3500 步、累计 A4000 和冻结检查通过；不会提前抢占正在训练的八卡。它是一次性序列，已有日志/输出时拒绝覆盖；失败即退出并由守护续占卡。
- 执行顺序：`m3_native_dense_val`（全量自然语义、置黑图像、128×2 四种动作条件、128 次完整策略计时）→ `c0_comparable_native_val` → B1/O 各自八卡 step2 保存后恢复 step5 → B1 4000 步与统一评估 → O 4000 步与统一评估。所有 GPU 子任务仍经同一 reservation supervisor。
- O 的 CPU 五步已完成，两个固定验证帧 flow native14 为 0.03109671；仅为工程检查。B1/O 对照冻结 B 与无用 S，只更新 A。八卡检查将逐 tensor 比较恢复前后权重；正式对照还将与同一 C0 比较冻结 B。
- 对照和 M3 使用相同有效 batch32、自然帧采样、500+3500 的动作更新预算及 LR/sampler 阶段重启；正式 O 只作为使用真值条件的离线上界，不能当成可部署策略。
- 新动作归档会逐样本核对 index/draw/condition/valid horizon、50×14 形状和有限值，并从保存的预测/目标重新计算全部现有 native 指标。统一执行长度和最终效果结论等完整 M3/B1/O 结果后确定。
- M3 W&B： https://wandb.ai/xiahy23-tsinghua-university/agentic-openpi-pi05-subtask/runs/04f99c0cf63e0f0c 。本地 `metrics.jsonl` 与 checkpoint 是持久依据。


M3 完成与统计协议（2026-09-07）：

- `m3_pilot_seed42/step_003500` 完成并选中；固定 128 验证帧的 CE 0.06701879、exact match 0.8203125、macro-F1 0.7276079、未知/无效率 0，生成条件 native14 flow MSE 0.03234650。完整验证与动作指标仍等 `m3_native_dense_val/report.json`，不要把这 128 帧指标写成全量结果。
- `m3_research_checkpoint_freeze.json` 确认 B604 全部未变，A208 和 S80 全部有更新。最后 875 次 M3 更新无 GT 动作条件，占累计 A4000 的 21.875%；仍保留 10% 条件置空。
- `assets/pi05_piper_stage1/eggplant_potato/research_protocol_v1.json` 已固定三个种子 42/43/44、阶段预算、选模、数据指纹、干预、指标与统一执行长度选择规则。三个种子共享同一 C0，因此种子方差衡量给定 C0 后的 S/A 训练随机性，不包括 M0 数据适配的方差。
- 新增 `training/subtask_comparison.py` 和 `scripts/summarize_subtask_comparison.py`。从保存的完整动作窗口复算统一执行长度指标；配对检查覆盖帧、draw、条件、去噪步数、目标值、split/norm、研究权重、种子与 A4000 预算。统计按任务分层重采样整条 episode，所有方法/种子使用相同重采样；噪声 draw 先按帧平均，报告训练种子的均值/样本标准差和条件于已拟合模型的轨迹 bootstrap 区间。不能将它解释为覆盖所有训练随机性的总体置信区间。三项数值/配对回归通过。
- `pilot_comparison_manifest.json` 等待四类模型的统一评估产物。统一执行帧数依照完整策略 p95 加 20% 余量，在 10/15/20/25/50 中选最小可满足值；选择仅用验证延迟，并对所有方法/种子/最终测试固定。视频读取与传输延迟不在当前计时范围，不能直接承诺机器人控制实时性。
- `scripts/continue_stage1_replications.py` 已作为有限序列启动，记录为 `replications_sequence.process.json`。它等待首轮 B1/O 训练及评估成功、官方 base 逐值审计通过，然后依次运行种子 43/44 的 M1/M2/M3、M1/M3 全量验证和 B1/O 匹配对照。每项失败即停止，全部 GPU 子进程仍经占卡守护。完成后生成 `three_seed_comparison_manifest.json`；尚未启动测试集评估。
- 官方权重下载改为单独入口 `audit_pi05_base_ranges.py`。一个 34,945,019 字节的官方对象从 1 MiB 前缀断点续下并通过 MD5，实测约 3.75 秒。原顺序下载进程已按 PID/创建时间核实后停止，全部验证文件和已有 partial 前缀保留；新入口使用固定 generation、严格 Content-Range、每对象 6 路范围下载，仍调用原始转换和逐 tensor 比对。当前日志 `official_base_weight_audit_ranges.log`，结果仍目标 `official_base_weight_audit.json`；下载/校验未全完成前，不宣称权重来源已逐值确认。


完整 M3/C0 离线验收与来源审计（2026-09-07）：

- `m3_native_dense_val/report.json`：全量 13,515 帧 exact match 0.85364410、macro-F1 0.79233905、显式目标物标签准确率 0.96622097，未知/无效生成率 0。三路图像置黑但保留任务/state 后 exact match 0.20577137、macro-F1 0.06977953；这是输入干预，不是重训无图像模型。
- M3 边界：140 个 GT 切换中匹配137、漏检3；±10 帧内比例0.49285714，匹配有符号偏差均值+4.8613帧，绝对偏差 p50/p95=10/23帧。预测变化400次、未匹配263次，平均每轨迹13.15次；边界附近 exact match0.57482993，远处0.93115839。相对 M1，切换误差改善但多余切换增加，不能宣称阶段跟踪已稳定。
- 同一 M3 checkpoint、同一128帧×2噪声：G / G-drop / G-shuffle / G+GT 的 H50 native joint RMSE 分别0.14097974 / 0.14223922 / 0.14640432 / 0.13855609；G+GT 是当前G权重的条件干预，不是独立训练的O模型。shuffle在同任务内置换，改变93.75%的这些评估条件。此结果显示文本条件影响动作，但训练预算匹配的收益结论仍需B1/O与种子复跑。
- G 的 H50 native gripper RMSE0.01862871，前10有效帧 joint/gripper RMSE0.05473419/0.01223723。G原始夹爪输出54.40%位于示教区间之外，但越界幅度均值0.00010868、最大0.00183932，超过0.001的比例0.10547%，超过0.01为0。保留未裁剪数据，物理控制单位尚未核实。
- C0 统一 evaluator 与原始 native 评估的关节/夹爪 RMSE 数值一致：H50 joint0.15428421、gripper0.01955990；前10 joint0.06278214、gripper0.01297158。补充的夹爪越界均值0.00014832、最大0.00287272，超过0.001比例0.72656%，超过0.01为0。
- 完整策略计时128帧/每rank3次warmup：M3 p50/p95=359.55/386.98ms；其中S生成32.58/37.99ms。C0完整官方Policy计时306.75/351.34ms。包括CPU输入转换/tokenizer、图像/prefix、全部flow步和native输出，不包含视频解码/传感器传输。先前C0的266.34/284.04ms仅为模型采样时间，不能与本次完整策略时间混用。最终统一执行长度尚需B1及其余种子的完整验证延迟。
- M3共1,024条、C0共256条动作窗口归档均已逐值复算现有指标并核对唯一身份和有限50×14形状。`evaluation_sources_v1/` 保存两次已完成评估的全部报告源码指纹对应文件，后续加入测试协议入口不会丢失旧评估实现。
- B1/O八卡都完成step2保存、恢复到step5：冻结B604和S80全部不变，A208均更新。正式 `b1_pilot_seed42` 已由有限序列启动；工程权重没有参与正式初始化。
- 官方权重审计最终报告 `official_base_weight_audit.json`：共有812个存储tensor，811个JAX来源tensor经官方转换、保存dtype对齐后逐值一致，无不匹配。PaliGemma embedding/lm_head 的共享别名已解析验证；`paligemma_with_expert.gemma_expert.lm_head.weight` 是官方转换器strict=False加载留下的初始化参数，无官方JAX来源，明确列为不做来源一致性声明。
- `unused_expert_head_behavior.json` 使用真实M3验证观测、CPU FP32：把该未使用head置为NaN并注册调用即报错的hook后，完整subtask与两步flow动作前后逐元素一致，空条件action loss仍有限。源码显示动作路径只调用expert.model，不调用此head。原始覆盖检查失败日志和源码均保留；最终审计入口为 `audit_pi05_base_weights_v2.py`，没有修改任何模型权重。
- `training/evaluation_protocol.py` 为两个 evaluator 新增 `--split test --test-protocol ...` 校验：必须在sealed清单中固定checkpoint/metadata/weights哈希、split/norm、统一执行长度、评估参数及world size。默认仍是val。三项协议检查与三项配对统计检查全部通过；当前没有sealed最终测试清单，尚未运行测试集模型评估或用测试指标选模；前期全量数据完整性审计不属于模型选模。


最终交付接续与调用说明（2026-09-07）：

- 新增 `docs/pi05_subtask_stage1_usage.md`，说明正常 `create_subtask_policy` 输入/输出、实际 `ok/empty/truncated` 状态、原单位动作、未校准语义分数、动作分块和严格恢复。模型调用仅输出数组，不驱动机器人；实际执行长度由最终协议单独确定。
- 新增 `scripts/benchmark_subtask_policy.py`，将策略独立加载到各GPU，测量batch1完整推理、PyTorch峰值allocated/reserved及调用增量，记录外部GPU进程和源码。正常G加载不使用engineering bypass。还会用实际首个输出检查全部50个ActionChunkBroker切片、语义/计时元数据保持及reset/replan行为。静态检查已通过，真实GPU测量排在三个种子验证完成后，当前不能宣称该测量已完成。
- `scripts/finalize_subtask_stage1.py` 的有限进程已启动，记录为 `finalization_sequence.process.json`，当前按真实进程身份等待 `replications_sequence`。全部三种子验证完成后，先按验证生成条件flow误差选部署候选，再运行独立G/C0/B1部署测量。依照所有验证完整策略p95加20%余量，在既定执行帧数候选中选统一值。
- 最终序列会先生成三种子验证比较，再固定全部测试checkpoint/metadata/weights哈希、参数和world size，写入 `final_test_protocol.json`；随后只运行约定的C0、三个种子M1/G/B1/O测试与统一比较。测试仍未开始；最终协议也尚未生成。该序列的失败会中止后续步骤，既有日志/输出不会被覆盖，占卡守护继续接管。
- 新增 `scripts/eval_subtask_time_final.py`：最终测试使用已经保存在 `time_baseline_val/report.json` 的训练期阶段起点中位数，不重新拟合或按测试标签调阈值；参数报告和评估源码也受最终协议指纹固定。
- `scripts/plot_subtask_validation.py` 绘制G与B1在相同累计A更新数上的完整已观测验证曲线及差值。首个快照为 `matched_validation_snapshot_1788774187`，包含B1到2500步的结果，PNG已查看，JSON保留原数值。该单种子中途曲线显示正负波动，不作为最终显著性结论，不据此修改预算。


等预算 B1 完成与首轮配对分析（2026-09-07）：

- `b1_pilot_seed42/step_004000` 正式完成并选中，权重 SHA256 `b99c6cc09a449d824b3cb048d97df34dfd7fbaceb43d347b77c113d4d684163f`；冻结 B604 全部不变，A208 更新。统一评估 256 条动作归档身份、形状、有限值与指标复算通过。现有 controls 序列已自动启动 `o_pilot_seed42`，不重复启动。
- B1 统一验证 flow native14=0.03260968、H50 joint/gripper RMSE=0.14211989/0.01881906、前10有效帧=0.05455012/0.01216213；完整策略 p50/p95=301.95/340.79ms。不同 batch 的训练期验证 flow=0.03264437 单独保留，不能混用数值口径。
- 新增 `scripts/analyze_subtask_pilot_pair.py`：只读取 G/B1 验证归档，验证研究阶段、同一 seed/C0/A4000、权重哈希、split/norm、帧/噪声/目标配对，按轨迹做2000次任务分层bootstrap。10个点估计与原报告逐项复算一致（误差<1e-12）。结果 `g_b1_seed42_validation_pair.json`。
- G−B1 的 H50 joint RMSE 差值=-0.00114015，95%区间[-0.00486573,+0.00206456]；前10有效帧joint差值=+0.00018407，区间[-0.00095398,+0.00118163]。主要误差区间均包含0，不能声称动作增益已确立。分任务/跨阶段结果仅探索性描述，不用于改变预算、选模或执行长度。
- `docs/pi05_subtask_stage1_seed42_interim.md` 是可审阅阶段记录；完整G/B1训练曲线更新为 `matched_validation_seed42_complete_b1`，旧2500步快照保留。最终仍等待O和43/44后按固定协议封存/测试；独立部署峰值显存与客户端runtime gate尚待排队执行。


首轮 O 完成、第二种子推进与 C0 兼容修复（2026-09-07）：

- `o_pilot_seed42/step_004000` 正式完成，权重SHA256 `ed53eaa69c2c5c8efc1f6399fa3a4233c51aa6c6892f26f217403a43bef97f63`；冻结B604不变，A208更新。`o_native_val`的256条归档复算通过。归一化flow MSE=0.03148055，H50 joint/gripper RMSE=0.13529330/0.01814635，前10有效帧=0.05424863/0.01239633。
- `pilot_seed42_first10_diagnostic_comparison.json` 完成全部G/B0/B1/O与G的三个条件干预配对统计。10帧只保留既定诊断口径，不决定最终统一执行长度。G−O H50 joint差值0.00568644，episode-bootstrap区间[0.00239752,0.00884741]；G−B1主要误差区间仍包含0。G−shuffle H50 joint差值-0.00542458，区间[-0.00917725,-0.00167189]。这些是同一个种子的验证结果，完整结论仍需三种子/最终测试，且多个指标未做多重比较校正。
- `m1_pilot_seed43`完成2000步并选中step2000；全量验证exact match=0.83551609，macro-F1=0.77302482，未知/无效率0。`m2_pilot_seed43`完成500步，生成条件固定验证flow=0.03627760，B604未变；现有replications序列继续M3，无重复启动。
- 报告试跑发现真正的收尾阻断：M0 schema2元数据没有后来新增的`engineering_smoke`字段，原benchmark、统计、seal/test入口会误拒绝或KeyError。仅暂停了尚在等待且没有子任务的finalizer；O和replications始终继续运行。新增`training/research_checkpoint.py`统一处理：现代研究权重仍需显式False；旧M0只接受固定research protocol选定路径、完整训练步数、pi05配置、split/norm及逐文件权重SHA256一致的C0。未知/工程/其他旧M0仍拒绝，未修改checkpoint元数据或模型权重。
- 旧元数据、篡改/错误C0拒绝、最终测试请求入口及配对统计共10项回归通过。真实C0已通过新报告入口和完整首轮统计入口。修改前源码归档在`evaluation_sources_pre_legacy_c0_fix`，修复及进程交接记录为`legacy_c0_compatibility_fix.json`；所有训练源码指纹均未修改。
- `finalization_sequence.process.json`已登记新的等待进程，当前PID754854、创建时间1788778869.85，日志仍追加到`finalization_sequence.log`，旧进程记录保留。后续会完成所有profile、seal、测试统计后生成`three_seed_validation_report`与`final_research_report`。
- 新增`scripts/write_subtask_research_report.py`：按实际报告汇总逐种子动作、语义、每阶段recall、切换、条件干预和profile，并保存全部输入JSON哈希。缺失产物的验证快照明确为complete=false；无sealed测试协议/不完整三种子要求均在生成最终输出前拒绝，两项拒绝检查通过。当前真实快照为`research_report_validation_snapshot_v2`；最终测试报告尚未生成。


正式策略入口验收与性能证据核对（2026-09-07）：

- `scripts/smoke_research_policy_entrypoints.py` 已实际执行结束；报告`research_policy_entrypoints_cpu.json`与逐阶段日志保留。正式C0/G/B1从正常加载器恢复，使用一帧真实val观察、固定噪声、10步flow；三者输出均为有限50×14，重复动作逐元素一致，state/图像输入未修改，客户端50帧切片、元数据保持、reset/replan全部通过。G不使用engineering override，自行生成`reach the handle of the lid`/`ok`，外部subtask字段被拒绝。此为CPU契约证据，不作为GPU延迟/显存或泛化指标。
- 当前第二种子训练继承的26个项目源码指纹全部与实际文件一致；根W&B watcher及`m3_pilot_seed43`同步进程均存活。模型和训练源码未为本次探针修改。
- 已定位真实首步可视化：`m1_overfit32_ddp8_seed42/first_step`包含manifest、两条NPZ、各两张输入/动作图；`observed_cpu_smoke_seed42/first_step`也保留工程图。可视化来源为checkpoint replay，rank0首microbatch，计算float32，不能宣称为当时训练前向的原样记录。
- 性能证据范围需保持准确：`observability_data_benchmark.json`测量8个重复真实帧、64次CPU读取/collate，warm decoded cache与未缓存PyAV为320.90/7.68 examples/s，约41.77倍；不是全数据训练或GPU吞吐加速比。`observability_gpu_sample.json`仅为此前10秒GPU利用率采样。
- 原先准备的`scripts/benchmark_subtask_throughput.py`入口可正常导入，但日志/输出中尚无实际GPU基准产物。因此该项仍待补测，不能将脚本存在视为完成。现有三种子训练/最终评估之后，使用相同C0、同32个已解码真实样本、8卡，对比microbatch2×accum2和microbatch4×accum1，各5步warmup+30步测量；只写测量结果、不保存训练checkpoint，不改当前研究run配置。所有GPU执行仍经reservation supervisor。该项属于剩余交付，需在宣称全部完成前补齐并如实报告结果。


待测八卡吞吐基准准备完成（2026-09-07）：`benchmark_subtask_throughput.py` 现要求恰好8卡、同一协议选定C0、训练split/norm身份一致、32个互异有效训练观察及新输出路径。两种microbatch划分的全局样本覆盖已在CPU逐项核对，静态检查和入口导入通过。记录每个测量更新的最慢rank耗时、mean/median/p95、峰值allocated、GPU采样、固定样本、权重/metadata/数据/源码指纹、配置顺序及测量前GPU进程。两个配置各自从相同decoder状态和新optimizer开始；不宣称不同batch分区的训练轨迹逐位等价。准备记录为`throughput_benchmark_readiness.json`，原始脚本保存在`throughput_sources_original`。此处仍只是准备，真实GPU报告尚未产生，须在当前研究序列结束后执行。
