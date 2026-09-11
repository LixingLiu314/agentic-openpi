# M3 seed42 Piper 真机部署

目标机器：`agilex@10.1.122.249`。仓库：`/home/agilex/agentic-openpi`。

本次使用 `m3_pilot_seed42/step_003500`，任务指令为 `Put the eggplant into the box`。模型根据三路 RGB、当前关节状态和全局任务生成 subtask，然后使用生成的 subtask 预测动作；客户端不输入 subtask 真值。

## 权重与代码

权重目录：

```text
/home/agilex/agentic-openpi/checkpoints/pi05_piper_stage1/m3_pilot_seed42/step_003500
```

`model.safetensors` 共 7,552,992,416 字节，预期 SHA-256：

```text
64600bed0b60e0529ac2721edb55717ac1c80fba2a607292857668a35e7a9275
```

迁移内容包含权重、metadata、归一化/数据划分资产，以及训练时的源码和运行时快照。优化器分片只用于恢复训练，本次推理部署不需要。

真机已切到 `upstream-openpi-main`，官方基线为 `215abfb217dbac7d5f1273282331b9b1866c0479`，再同步训练服务器当前模型代码。模型改动尚未提交 Git。原 `lixing` 分支保留；原 Piper 文件保存在 `/home/agilex/.cache/openpi-deploy/m3_seed42_20260907/original_robot_files`。

模型服务使用仓库 `.venv`（Python 3.11、torch 2.7.1+cu126、transformers 4.53.2）。ROS 客户端使用用户指定的 conda `aloha`（Python 3.8）。不要在 `aloha` 中升级安装整个模型依赖栈。

## 已准备的运行入口

日常启动、复位、停止和输出格式以 [README 真机推理章节](../README.md#本项目m3-piper-真机推理) 为准。
以下命令在真机上执行。先在另一个终端执行 `bash scripts/serve_m3_piper.sh`，等待 `M3 ready`。
默认仅监听本机 `ws://127.0.0.1:8000`；确需网络调用时给服务脚本传 `--host 0.0.0.0`。

只运行观测和模型推理，不下发动作：

```bash
cd /home/agilex/agentic-openpi
bash scripts/run_m3_eggplant.sh --max-steps 30
```

执行一次有步数上限的真机试验：

```bash
cd /home/agilex/agentic-openpi
bash scripts/run_m3_eggplant.sh --execute --max-steps 900
```

实机执行默认先平滑复位，并记录连续视频；仅在明确需要跳过复位时使用 `--no-reset`。
只读运行不执行复位。该前台命令用 Ctrl+C 停止并保存结果。
历史部署使用的后台进程停止脚本只读取旧 PID 记录，不适用于新的前台启动方式：

```bash
cd /home/agilex/agentic-openpi
python3 scripts/stop_m3_robot_client.py
```

停止动作客户端会保留模型服务和驱动。客户端在一次实机试验开始前复位，不会自动重试动作或重启已结束的试验。

模型服务入口为 `scripts/serve_subtask_policy.py`。启动时执行三个固定噪声调用检查输出和重复性后开放端口；
默认使用合成观测预热，也可用 `--warmup-observation` 指定已有真实 NPZ。合成预热不代表任务效果验证。
默认不创建服务日志文件，逐次记录统一由客户端写入 `run.json`。`--log-dir` 仅保留为可选旧版调试接口。

每轮目录为 `logs/inference/<日期_时间_纳秒>/`，默认只包含 `video.mp4` 和 `run.json`。
JSON 合并运行状态、复位报告、预测/动作记录和视频时间戳；每 5 秒后台更新，结束时完整保存。
不再默认生成逐次 NPZ、图片、JSONL 或输入静帧回放。视频独立于模型请求按 30Hz 录制；
JSON 的 `queries[].video_input` 保存最近输入帧的时间和相机时间戳匹配情况。
旧试验的文件保留，以下章节是它们当时的格式和实测记录。

## 真机输入输出

- 相机：`cam_high` ← `/camera_f/color/image_raw`；`cam_left_wrist` ← `/camera_l/color/image_raw`；`cam_right_wrist` ← `/camera_r/color/image_raw`。输入为原始 RGB HWC 图像，服务端处理缩放。
- 状态和动作：`[左臂6关节, 左夹爪, 右臂6关节, 右夹爪]`，共 14 维。关节使用驱动原生弧度值，夹爪使用米制开度。
- 输出是绝对关节目标，已完成反归一化和关节 delta 转绝对值。不要重复转换或套用 Trossen 的符号/夹爪变换。
- 一次推理输出 50 步，客户端默认执行前 15 步后重新观察。每个动作段内按 30 Hz 发送；当前采用阻塞推理，因此动作段之间存在推理停顿，并非连续 30 Hz 闭环。
- 客户端将夹爪命令限于 `[0, 0.09]`，并保留未经限幅的原始模型动作。传感器超过 1 秒未更新、推理超过 5 秒、动作非有限值，或关节目标与当前测量相差超过 0.35 rad 时停止本次客户端。
- subtask 的 EOS 只表示句子生成结束，不表示物理任务完成。

## 日志与验证状态

所有首次部署记录位于：

```text
/home/agilex/agentic-openpi/logs/robot_m3_seed42_20260907
```

- `checkpoint_verification.json`：完整权重和匹配源码/运行时校验；仅在迁移和核验成功后生成。
- `server_warmup.json`：真实传感器快照上的 GPU 推理输出、固定噪声重复性、耗时及显存；仅在检查通过后生成。
- `policy_server.log`、`server_queries.jsonl`：服务日志和每次生成的 subtask。
- `robot_client.log`、`robot_client.process.json`：本次助手启动的动作客户端。
- 每个试验输出目录内保存初始三路相机、每次请求的状态/原始动作、subtask、耗时和已执行步数。`completed.json` 表示达到所设步数上限，不能作为任务成功标签。

当前已完成三路真实 RGB/双臂关节采集、模型依赖导入、6 个 Transformers 补丁哈希检查，以及独立测试端口上的 ROS/WebSocket 通信检查。模型权重和实际动作验证结果以此次部署最后生成的报告为准。

## 首次试验与视频记录补充

完整权重哈希、24 个 checkpoint 记录的项目源码和 6 个 Transformers 运行时源码均已核验一致。正式模型在 RTX4090 上加载、固定噪声重复推理以及 WebSocket 调用均通过，服务使用端口 8000。

首次动作试验记录了 36 次请求，至少完整发送了 525 步动作。随后观测超过 1 秒未更新，客户端退出。20:41:04 系统日志显示右腕相机 CC1T35300A3 所在 USB Hub 断连、重新枚举；之后右臂 CAN 为 ERROR-PASSIVE，驱动有发送失败。客户端已停止，模型服务仍保留。盒体在试验中倾倒，此次不能视为任务成功；没有自动重启动作。

**首次动作试验没有连续视频，也没有逐次推理图像。** `trial_first` 只保存了初始图像、每次状态/动作和 subtask 日志，另有 `during_trial` 的人工采集快照，无法还原全部画面。`first_trial_subtask_timeline.json` 保存原试验的输出时间线。

后续已补充：每个 `query_XXXX.npz` 保存真正送入模型的三路 RGB、状态和原始动作；`queries.jsonl` 保存该次输入的快照时间、各传感器时间戳、响应时间、subtask 和状态。退出时自动渲染 `subtask_preview.mp4`，失败仍保留原始文件。这是**按推理采样时刻对齐的回放**，不是连续 30fps 相机录像；相邻推理间保持上一张输入画面，不插造帧。

另录制了 `readonly_subtask_preview.mp4`：14.9 秒、20 次推理，没有任何动作下发。该视频是首次试验之后的新观察预览，20 次均输出 `Grasp the handle of the lid`，不能当作原动作试验的录像。

## 第二轮：复位与连续录像

用户明确要求恢复复位后再次运行。停止旧驱动并清空 CAN 发送积压、重新启用两条总线后，双臂恢复 ERROR-ACTIVE，且各收到约 200Hz 的新鲜原始反馈。随后以不自动使能的方式重启驱动。

复位目标依据固定训练集中 158 个 episode 的原生起始状态：双臂 12 个关节回到零位，两个夹爪为 0.07 米。使用最小跃度插值，默认至少 6 秒，轨迹峰值关节速度不超过 0.25rad/s；到位后才开始推理。旧 Trossen 的 `[0,-1.5,1.5,...]`/夹爪 `4.0` 不适用于本模型。

第二轮目录为 `logs/robot_m3_seed42_20260907/trial_reset_video_02`。实测复位 6 秒、180 个插值步，最终最大关节误差 0.01490rad、夹爪误差 0.000420m；之后完成 60 次推理及 900 步动作，正常达到步数上限并退出。本轮结束时双臂 CAN 仍正常，没有触发观测超时。

- `continuous_subtask.mp4`：52.27 秒，1440×576、30fps、1568 帧，包含复位和动作全过程。叠加的是最近一次已返回的 subtask，同时标明对应输入时刻和输出年龄。
- `subtask_preview.mp4`：45.8 秒，包含 60 组实际模型输入及对应 subtask；用于严格检查输入与输出的对应。
- `continuous_frames.jsonl`：连续视频每帧时刻、相机时间戳、最新预测及其输入/响应时刻。
- `reset_report.json`、`review_summary.json`：复位实测及 subtask 转换时间。

本轮输出为 `reach the handle of the lid` 24 次和 `Put the lid on the box` 36 次。连续视频约 21.14 秒首次切到放回盒盖，24–34 秒之间反复切换。完成 900 步仅表示运行结束，不等于任务成功或预测正确。

硬件配置核实：USB `1-12:1.0` → `can_right`，`1-13:1.0` → `can_left`，均 1 Mbit/s。相机序列号为右 `CC1T35300A3`、左 `CC1T35300AV`、前 `CC1T35300YS`。
