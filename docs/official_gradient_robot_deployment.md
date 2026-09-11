# 三组官方起点模型：部署与客户端检查

2026-09-09晚，北京时间。三组均完成5000个优化器更新，验证选中的模型现已全部安装在Agilex。

| 模型 | Agilex checkpoint |
| --- | --- |
| Frozen | /home/agilex/agentic-openpi/checkpoints/pi05_piper_official_grad/frozen_seed42_v1/step_003000 |
| Limited | /home/agilex/agentic-openpi/checkpoints/pi05_piper_official_grad/limited_seed42_v1/step_003000 |
| Full | /home/agilex/agentic-openpi/checkpoints/pi05_piper_official_grad/full_seed42_v2/step_001000 |

本次补传Full完整B/S/A推理权重、metadata、归一化/划分资产、源码溯源及候选记录，共65个文件，逐文件长度和SHA256与服务器一致。无需optimizer、额外LoRA或M3 parent。三组完整权重SHA256也已重新核对一致。

Full SHA256：b1637560db46dd1a10e568cd6b776ef04bb2d80a92e308c09e29cff952373457。

## 软件检查通过

- Full在机器人项目.venv实际CPU FP32加载通过。两次固定噪声推理输出有限50×14动作、结果一致、输入未改动，拒绝外部subtask注入。
- 20个项目推理源码与6个Transformers核心文件匹配Full归档版本。
- 正式ROS + conda aloha环境的75项无运动回归检查通过。首次直接运行时遗漏ROS环境，并混入模型环境测试，导致导入失败；按正式启动环境运行上述75项后全部通过，无需修改客户端源码。
- 实际客户端connect_policy、序列化协议和GUI validate_server经临时端口18765完成合成输入推理。GUI catalog识别三组模型。
- CPU协议检查延长了接收超时，单次约21.57秒；不验证正式CUDA延迟或客户端默认5秒时限，也不是任务成功率测试。
- 临时服务已退出。日志中一次InvalidMessage来自就绪检查的TCP探测，随后实际WebSocket请求成功。

## 真机目前未就绪

三路相机均为640×480 RGB，新鲜图像延迟约9–13毫秒。但只读预检因缺少双臂关节状态而失败。ROS驱动仍在，日志持续报告Stale CAN feedback；两路CAN接口虽为ERROR-ACTIVE，独立3秒被动监听双臂均无帧。

需先检查双臂供电/CAN物理反馈，再在当前场景确认后处理驱动。本次没有重启驱动、使能、复位或发布动作。8000端口的其他Dataarm插管服务保持不变。本项目加载时请使用空闲端口，并保持GUI与模型端口一致。

## 启动入口

在Agilex桌面终端执行：

    cd /home/agilex/agentic-openpi
    bash scripts/start_aloha_eval_gui.sh

旧GUI重新打开后选择上述模型。GUI本身只显示相机；Start会执行真实动作，当前反馈未恢复前不要开始。

仅加载Full服务的示例（先确认8002空闲）：

    bash scripts/serve_m3_piper.sh --checkpoint checkpoints/pi05_piper_official_grad/full_seed42_v2/step_001000 --port 8002

模型使用项目.venv，GUI/客户端使用conda aloha。沿用schema5严格身份校验、原生Piper14维、复位保护与正常推理仅MP4+JSON记录合同。

证据：Agilex logs/full_official_deploy_20260909/；服务器 logs/agilex_full_transfer_20260909/；控制工作区 context/full_deployment_20260909/。历史9月10日00:30 ETA已被实际完成状态覆盖，本次未修改原Windows提醒调度。
