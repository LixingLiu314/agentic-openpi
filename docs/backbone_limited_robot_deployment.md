# Limited action-to-backbone checkpoint on Agilex

本次迁移模型：有限开放 `action loss → B`，seed42。原训练完成5000次新增更新，按固定验证 native14 flow MSE 选择 `step_001500`，选择误差为0.030229839527237345。它是完整部署权重，LoRA已合并；不需要额外adapter、训练优化器或另一个M3模型文件。

Agilex目标：`agilex@10.1.122.249`，分支 `upstream-openpi-main`，仓库 `/home/agilex/agentic-openpi`。

模型目录：

```text
/home/agilex/agentic-openpi/checkpoints/pi05_piper_backbone_grad/limited_seed42/step_001500
```

权重大小：7,552,992,264 bytes。

SHA256：

```text
86ead156ecc8b236cf91b49385b72c726e7fd4e13e32f74003389186ae52b42f
```

## 启动 GUI

在Agilex桌面终端执行：

```bash
cd /home/agilex/agentic-openpi
bash scripts/start_aloha_eval_gui.sh \
  --checkpoint /home/agilex/agentic-openpi/checkpoints/pi05_piper_backbone_grad/limited_seed42/step_001500
```

已有GUI需要关闭后重新打开，才能载入新增的格式识别。界面显示“有限开放 B · step1500 · 实验模型”。选择任务后使用“连接/加载模型”。若端口仍是旧模型，先停止当前推理，再使用“停止模型”，随后连接新模型。打开GUI只显示相机；只有用户按下开始才执行现有动作流程。

现有复位与记录约定保留：推理前按配置复位，复位不写视频；推理输出仍为原速 `video.mp4` 与 `run.json`。CAN/ROS反馈检查、动作幅度检查和停止机制不变。

## 单独启动模型服务

```bash
cd /home/agilex/agentic-openpi
bash scripts/serve_m3_piper.sh \
  --checkpoint /home/agilex/agentic-openpi/checkpoints/pi05_piper_backbone_grad/limited_seed42/step_001500 \
  --port 8000
```

原有服务入口现已按metadata分发M3、R1和`action_backbone_v1`，无需R1专用override或parent参数。模型服务使用仓库 `.venv`，GUI/机器人客户端使用conda `aloha`。端口须由用户在GUI中释放旧服务后再启动。本次迁移不停止旧模型服务，不执行机器人动作。

## 实现与检查

- 原训练匹配的 `backbone_gradient_policy.py` 严格验证权重、归一化、数据划分和tokenizer，再加载完整B/S/A参数。
- GUI目录扫描、服务分发、握手和客户端兼容新的真实 `stage=backbone_grad`，不把新模型伪装为M3或R1。
- GUI握手匹配路径、mode、variant、完整权重SHA、parent SHA和experimental字段。新模型保留实验身份；加载成功不代表真机任务成功率合格。
- 70项无动作检查通过，覆盖新格式以及已有M3/R1、GUI、复位/停止/记录行为。证据在 `logs/backbone_deploy_limited_1500/regression_checks_v2.log`。
- CPU实模型检查和传输校验以同目录的 `deployment_manifest.json`、`robot_cpu_load_gate.json` 为准。CPU检查使用合成观测，验证实际加载、有限50×14动作、固定噪声复现、输入不变、拒绝外部subtask；不作为场景任务质量结果。
- Robot代码改动前的备份：`/home/agilex/.cache/openpi-deploy/backbone_limited_1500_20260908/original`。

全量B组是另一个独立实验，其最终候选尚未产生时不能声称已完成或已迁移。本次目录仅含limited验证集选中的模型。

## 本次实际完成结果

迁移后完整SHA256一致；70项无动作回归检查通过；Agilex真实CPU FP32模型加载及两次合成观测推理通过，50×14动作有限、固定噪声逐元素相同、输入不变、拒绝外部subtask。详细结果见 `logs/backbone_deploy_limited_1500/deployment_manifest.json`。未执行机器人动作或停止原模型服务。
