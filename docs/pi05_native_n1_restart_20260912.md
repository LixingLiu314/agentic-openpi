# N1 释放占卡进程后的重新启动记录

日期：2026-09-12。用户授权：确认先前进程是占卡程序后停止，并继续N1训练。

结果：占卡程序已正常停止；新八卡工程与完整模型检查通过，N1正式训练已启动并完成首10步核验。11:17:57唯一有界启动读取为14/5000，8rank存活；预计19:05左右完成，19:30单次跟进。

## 占卡程序已确认并停止

完整审阅外部toolkits/gpu_hold.py：只分配显存、随机矩阵乘法和睡眠，不加载数据/模型、不训练或保存checkpoint。其父run_robotwin_hybrid_optimization_a800.sh在10:32:05记录FINISHED后启动8个holder，末尾仅wait，无自动重启逻辑；操作时父进程下只有这8个holder。

逐项核验PID、创建时间、完整argv、cwd、父子关系和源码SHA后，11:06:35向375186–375193发送SIGTERM，8个均正常退出，父脚本自然退出。未向父/tmux/其他任务/本项目guard发信号，未删除或修改外部项目文件。实际证据：服务器logs/pi05_native_n1_20260912/attempt_02/{holder_audit,holder_stop}.json。

操作脚本和授权范围已提交：12620c95a73e446cd6b701451eec4def1e7331cb。N1模型修复仍是e005ecf的版本，没有再次改动模型/训练数学。

## 新运行身份及固定配方

- 服务器项目：/media/raid/workspace/surongpeng/ws_lixing/agentic-openpi。
- 新控制root：logs/pi05_native_n1_20260912/attempt_02。
- 11:07:36真实派发；launcher PID417027，created1789182455.36。进程身份仅为启动记录，操作时必须重验，不可照抄PID。
- 正式输出：checkpoints/pi05_piper_native/n1_action_stop_seed42_v1。
- 数据：Datasets/eggplant_potato_reach_arm_v1，178train/20val、无test、train-only norm、15类left/right arm标签。
- 官方fresh pi05 B/A，seed42，5000 updates，global256，首测8卡micro32/accum1，每500验证和保存；正式不继承工程更新。
- VLM原生语言层与绑定词表输出subtask；没有独立Subtask decoder、query或跨帧记忆。CE→B、flow→A；A不读GT或生成的subtask。
- 唯一W&B视图：https://wandb.ai/xiahy23-tsinghua-university/agentic-openpi-pi05-subtask?nw=2r09j4dx7vn。

## 验证与交付状态

CPU真实小尺寸HF、投影形状、因果/梯度/优化器恢复及数据契约检查已通过。新八卡工程从官方fresh重新执行4→保存→恢复至8，B/A各8次更新、8份rank文件完整，全部rank恢复各自B/A optimizer和随机状态。随后用这次工程checkpoint完成完整原生/梯度检查：

| 检查 | 结果 |
| --- | --- |
| teacher logits及原生Action一致性 | 最大差均0 |
| 50×14原生输出、固定噪声重复/session/reset、输入不变 | 通过 |
| 外部subtask拒绝、未来GT隔离、prefix cache不变 | 通过 |
| CE梯度 | B28.49566、A0；视觉/投影/语言/词表分别非零 |
| Action loss梯度 | B0、A0.786307 |

正式流程11:13:51启动。812个存储tensor与指定官方文件装载逐项核对（不混同811个JAX来源tensor的历史转换审计）；初始B/A计数0，无独立文本网络。11:17:36首10步W&B/local两loss、B/A梯度与媒体全部通过；以下是第10步启动指标，不代表最终效果：

- Subtask CE：9.29140675。
- Action flow MSE（all32，global-only）：0.143628754。
- B/A裁剪前梯度范数：130.505798 / 0.578730762，两组独立clip1。

11:17:57一次有界读取正式14/5000，8个rank真实存活；145份计算源码及正式归档源码哈希全部匹配。训练SHA12620c95a73e446cd6b701451eec4def1e7331cb；之后只更新文档，不改变训练源码。

已视觉检查首个实际batch的输入和动作图，三路相机、任务、GT标签、原生14维状态/动作与“N1文字只展示，不条件化A”的注释清晰。第一步生成空串/truncated及未拟合动作曲线保留，不把启动检查等同语义/动作效果成功。

完整收据：root/engineering_passed.json、engineering_native.json、formal.process.json、startup_handoff.json；正式输出startup_verified.json及first_update/。工程步骤没有算入正式5000预算。

## ETA与跟进

steps3–14实测平均4.1851916秒/update；剩余4986步纯训练20867秒，约5小时48分钟。另留2小时用于验证、存盘和最终原生检查，点估计19:05，预期约19:00–20:00；共享服务器负载和原生生成验证开销会影响实际时间。

按OpenAI Docs核对计划任务设置，更新原唯一vla-action-stop为9月12日19:30北京时间单次检查当前N1；若尚未完成，届时一次有界重估并延后同一个提醒。此前不轮询中间step，不增设短周期检查。提醒需要控制机/桌面应用可运行，服务器训练本身为独立后台流程。

前一天数值检查失败和今天attempt_01的三档OOM证据全部保留。Q1未实现/未派发，无机器人部署、服务切换或动作。
