# M3 Piper 推理流程重整与反馈故障排查

日期：2026-09-08。部署位置：`agilex@10.1.122.249:/home/agilex/agentic-openpi`。

## 结论与证据

用户提供的 `logs/inference/20260908_083229_777510107/run.json` 显示：该轮在 reset 的第 128 帧停止，尚无模型请求。右臂 joint5 的 ROS 值从开始到报错均为 `0.4116435 rad`，拟发送的 reset 指令为 `0.0612091 rad`，因此计算出 `0.3504344 rad` 的差值。

随后同时监听 ROS 和原始 SocketCAN：右臂 joint5 实际 CAN 反馈为 `0.073927672 rad`，而 ROS 仍为 `0.411643512 rad`；右臂 ROS 控制模式为 0，CAN 控制模式为 1。ROS 时间戳仍持续更新。左臂原始关节值与 ROS 一致。这证明了右臂驱动反馈链路正在发布旧值，单看 ROS 时间戳不足以证明反馈有效。并不能据此认定模型产生了异常动作；当轮模型尚未调用。

新检查器在旧驱动上直接复现 `ROS/CAN feedback mismatch: right joint5 ROS=0.4116 CAN=0.0739`。确认无动作发布者后，正常结束旧 Piper roslaunch，使用 `mode:=1 auto_enable:=false` 重启。两臂随后通过连续 5 秒一致性检查。导致旧驱动缓存异常的 SDK 内部故障机制没有独立复现，不能声称已永久修复 SDK；新流程能在启动和控制期间识别该状态。

证据位于机器人项目下：

- `logs/pipeline_refactor_20260908/before_driver_restart/run.json`：旧驱动的独立 CAN / ROS 对比。
- `logs/pipeline_refactor_20260908/driver_recovery.json`：旧、新驱动进程身份及恢复记录。
- `logs/pipeline_refactor_20260908/after_driver_restart/run.json`：重启后只读检查。
- `logs/pipeline_refactor_20260908/regression_tests.log`：19 项回归检查结果。
- `logs/pipeline_refactor_20260908/readonly_verification.json`：真实模型/相机联调及无指令发布验证。

## 对稳定版本的参考与实现

参考仓库为 `/home/agilex/workspace/xiahongyu/agentic-openpi`，分支 `aloha-real-robot-eval`，HEAD `aae748bdd6b3468e827383083a3d23fc959d3401`。阅读了 `examples/aloha_real/piper_main.py`、`piper_env.py`、`piper_real_env.py`，参考它的 ROS 传感器快照、环境 `reset/step` 与策略循环分工、原生 14 维关节通道。参考仓库未修改。

当前项目的部署代码分工：

| 文件 | 职责 |
| --- | --- |
| `scripts/run_m3_eggplant.sh` | 加载 conda aloha/ROS，完整传递客户端参数 |
| `scripts/run_subtask_piper.py` | 参数、准备、模型连接、推理和有限动作块循环 |
| `scripts/piper_robot.py` | ROS 传感器、发布者管理、使能握手、reset/step |
| `scripts/piper_can_feedback.py` | 只接收 SocketCAN，按 CAN ID 检查内核时间戳、位置一致性、使能及故障 |
| `scripts/piper_reset.py` | 无 ROS 副作用的反馈驱动复位规划 |
| `scripts/piper_run_log.py` | 原子 JSON 记录、预览/执行时间区分 |
| `scripts/subtask_live_video.py` | 既有独立连续录像线程，保持原实现 |
| `scripts/piper_pipeline_test.py` | 无真实控制发布的回归检查 |

主要变化：

1. 逐组校验两臂 CAN 状态、关节、夹爪、6 个电机的收包时间。一个新的 CAN ID 不会掩盖另一组旧值。ROS 与 CAN 关节容差为 `0.05 rad`、夹爪为 `0.01 m`；这用于识别反馈链路错误，不替代动作差值限制。
2. 执行前要求两臂全部 12 个电机的原始 CAN 使能位确认；ROS 订阅者已连接不再被视为电机已使能。
3. reset 保留当前 M3 的零关节、`0.07 m` 夹爪目标，至少 6 秒、关节峰值 `0.25 rad/s`。拟发送指令超前反馈超过 `0.08 rad` 或夹爪超前超过 `0.012 m` 时暂停轨迹时间。持续有跟踪误差但 2 秒无有效位置变化时报告具体关节/夹爪停滞；另有总超时。原 `0.35 rad` 硬限制保留。
4. 发布周期使用单调时钟，调度迟延不会导致补发一串追赶指令。控制读取位置时不再复制三路图像，但仍检查相机和关节新鲜度。
5. 执行客户端持有独占锁并检查现有指令发布者；退出时释放发布者，不自动回零、重启动作或失能释放负载。
6. 脚本先保存客户端参数、清空位置参数后加载环境，解决 `--help` 被 catkin setup 错误处理的问题。服务刚启动时最多等待 60 秒连接，连接成功和模型身份验证后才使能/复位；执行中断连不自动重连或重放动作。
7. `run.json` 升为 schema 3：真实发布记录在 `action_step_times`，只读步记录在 `preview_step_times`。正常推理输出仍为一个 MP4 和一个 JSON；`--check-only` 只生成 JSON。

当前 checkpoint、每次预测 50 步/执行 15 步、块内 30 Hz、M3 原生关节和米制夹爪约定保留。没有套用旧 banana 任务的 `[-1.5,1.5]` 复位关节和 `4.0` 夹爪数值。

## 验证结果与范围

- 19 项回归检查全部通过：复现右 joint5 旧反馈、单个 CAN ID 过期、缺失反馈、单电机未使能、碰撞故障、正常/缓慢/卡住的 reset、调度延迟、动作超限不发布、只读权限、旧 ROS 时间戳、夹爪裁剪、JSON 收尾、服务启动重试与超时。
- 部署脚本语法、Python 3.8 编译、原命令入口及 `--help` 通过。
- 真实相机 + 现有 M3 服务完成 10 次请求、150 个预览步；单次推理往返约 183–215 ms，10 次 subtask 均为 `reach the handle of the lid`，状态为 `ok`。
- 独立 ROS 观察器确认整个只读运行没有动作/使能消息，也没有相关指令发布者。
- 视频 `1440×576 / 30fps`，212 帧、7.07 秒、0 个遗漏时间槽，最大采集间隔约 40.96 ms。ffprobe 帧数与 JSON 一致，输出目录只有 `video.mp4`、`run.json`。
- 服务预热期间的一次连接拒绝也保留在 `logs/pipeline_refactor_20260908/readonly_model/`，该失败正常保存视频和 JSON；通过的联调位于 `readonly_model_v2/`。

本次没有执行真实 reset 或模型动作。reset 跟踪逻辑通过模拟反馈回归，实际双臂与相机通过只读检查；这不等于新版本已经通过实机动作试验，也不等于任务成功率验证。

## 使用

```bash
cd /home/agilex/agentic-openpi

# 只检查，不使能、不推理、不录像
bash scripts/run_m3_eggplant.sh --check-only

# 真实模型与相机预览，不发送动作
bash scripts/run_m3_eggplant.sh --max-steps 150

# 场景准备好后，复位并执行
bash scripts/run_m3_eggplant.sh --execute
```

如果以后出现 `ROS/CAN feedback mismatch`，先正常退出 Piper 驱动，使用 `roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=false` 重新启动，再跑 `--check-only`。不要重复启动驱动，也不要用增大关节限制替代反馈恢复。

修改前文件保存在机器人 `/home/agilex/.cache/openpi-deploy/pipeline_refactor_20260908/`。当前代码已直接安装到机器人项目；保留原有未提交改动，没有提交或切换分支。
