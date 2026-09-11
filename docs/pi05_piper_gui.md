# Piper 简洁推理 GUI

部署：`agilex@10.1.122.249:/home/agilex/agentic-openpi`，2026-09-08。

## 启动与操作

在 Agilex 桌面终端执行：

```bash
cd /home/agilex/agentic-openpi
bash scripts/start_aloha_eval_gui.sh
```

启动脚本加载 conda `aloha`、ROS Noetic 和机器人工作空间，使用已有 PyQt5。参考入口为旧仓库
`/home/agilex/workspace/xiahongyu/agentic-openpi/scripts/start_aloha_eval_gui.sh`，旧仓库未改动。

1. **选择任务**：下拉菜单预置“茄子放入盒子”“红薯放入盒子”。编辑名称和任务指令，点击“保存任务 / 保存修改”新增或更新任务。
2. **选择 checkpoint**：支持本机目录浏览、输入路径、下拉历史、刷新扫描。当前后端使用完整 M3 checkpoint；R1 的 decoder-only 增量目录会明确拒绝，不会默默换成另一模型。
3. **连接 / 加载模型**：已有相同模型时接入；本机端口无服务时在后台启动所选 checkpoint。模型路径和 M3/native14 合同均校验。若已有不同模型，可先“停止模型”再加载。远程地址只连接已有服务。
4. **设置运行参数**：选择“只读推理”或“执行动作”，设置本轮步数上限、每次采用动作数（1–50）、执行前复位和视频录制。
5. **开始推理**：模型确认就绪后启动当前命令行客户端。运行期间锁定参数，避免本轮任务与界面配置不一致。
6. **停止本轮 / Esc**：中断本轮并等待 MP4 和 JSON 保存。加载模型期间同样可以取消待启动的推理。关闭运行中的窗口会先停止并收尾，不自动复位，也不开始下一轮。

“检查反馈”不请求模型、不使能；“仅复位（动作）”明确执行原生复位，可独立于模型服务使用。
窗口启动默认只读，立即显示相机，但不会自动启动模型、推理、复位或动作。

配置存放在 `~/.config/agentic-openpi/piper_gui.json`。可使用 `--settings` 指定独立配置文件；
`--host`、`--port`、`--checkpoint`、`--task` 可设初始值。运行模式不会保存为下一次的默认动作模式。

## 显示内容

- 三路实时 RGB：前视、左腕、右腕。显示画面年龄，过期画面明确标记。
- 当前 subtask、解码状态、模型分数（非概率）、结果年龄。
- 处理步数 / 上限、模型请求数、往返延迟、运行阶段、进度条。
- 轨迹预测、子目标、完成状态三个预留展示块，当前明确显示“尚未接入”。
- 简要控制台日志和“打开本轮输出目录”。

推理信息和图像均为展示用途。没有手工标注、点击画轨迹、数字键覆盖 subtask 或人工推理信息输入通道。
任务指令编辑是正常的 task prompt 输入。

正常每轮输出仍为 `logs/inference/<timestamp>/video.mp4` 和 `run.json`。
“检查反馈”只生成 JSON。GUI 启动的模型服务日志另存 `logs/gui/`。

## 实现边界

- `scripts/piper_eval_gui.py`：Qt 布局、相机订阅、按钮和任务生命周期。
- `scripts/piper_gui_support.py`：任务配置、checkpoint 扫描/校验、参数列表构造、模型进程身份管理。
- `scripts/piper_gui_telemetry.py`：客户端到 GUI 的本机 UDP 状态消息；非阻塞、独立运行 token，不接收控制指令。
- `scripts/start_aloha_eval_gui.sh`：隔离参数后加载 conda/ROS，启动 GUI。
- `scripts/piper_gui_test.py`：无真实控制的 GUI 回归检查。

GUI 通过参数列表调用 `scripts/run_m3_eggplant.sh`，不拼接用户输入为 shell 命令。沿用当前客户端的
原始 CAN / ROS 一致性检查、使能确认、reset 和动作硬限制。新增 `--expected-checkpoint` 在客户端真正开始
执行前核对服务路径，避免界面选择与实际服务不一致。

只停止当前项目、当前用户且 PID 启动时间匹配的模型进程；有其他当前项目推理客户端时拒绝停止模型。
停止本轮只向该客户端独立进程组发 SIGINT，覆盖脚本准备阶段；视频编码器独立于该信号组，由客户端正常关闭输入并完成 MP4 收尾。
停止或关闭 GUI 后模型继续可用；用户点击“停止模型”才结束服务。

## 验证

- 17 项 GUI 检查通过：任务新增/编辑/重载、参数原样传递、只读与复位按钮边界、checkpoint 不匹配、
  decoder-only 目录拒绝、PID 身份与活动客户端保护、非阻塞遥测、运行中参数锁定、取消待启动推理等。
- 19 项既有 pipeline 回归通过。
- 使用真实 Qt 按钮、真实三路相机和现有 M3 服务完成只读 10 次请求 / 150 步；视频约 7.1 秒、213 帧、零遗漏时间槽。
- 实际“停止本轮”按钮和运行中关闭窗口均验证了 `interrupted` 状态以及一个 MP4 / 一个 JSON 的完整收尾。
- 独立 ROS 观察器在验证中未收到动作/使能消息，未观察到相应指令发布者。
- 在 1480×930 和 1120×760 检查布局；开始、停止按钮固定可见，设置区域可滚动。
- 本次没有执行真实复位或模型动作，没有切换现有运行模型进行 GPU 加载试验。

证据位于机器人 `logs/gui_validation_20260908/`，其中 `unit_tests.log`、`pipeline_regression.log`、
`live_verification.json` 和 `live_verification_v1.json` 分别记录检查结果、关闭和停止按钮验证。
中间发现的编码器停止失败也保留在对应 `logs/inference/` 运行中；最终编码器进程隔离后重新验证通过。

修改前客户端、日志模块、视频模块和 README 已保存在 `/home/agilex/.cache/openpi-deploy/gui_20260908/`。
