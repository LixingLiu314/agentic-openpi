# Reach-arm 交付与两组语义实验（2026-09-10）

用户授权：已训练完成的候选转移到 Agilex，再设计并启动两组实验。没有机器人运动授权。

## 已完成候选

limited_recurrent_seed42_v1 完成5000步及最终诊断；最终权重SHA256 2fcba5cab1f10985987d644bad84506bdddfc31c48c944c3a7f53f32440486af。优先交付step005000，验证最佳3000仅作可选对照。本次68个部署文件已在Agilex校验，27项实际模型源码/Transformers依赖一致；GUI/客户端94项无运动测试通过，实际CPU推理、独立会话重复及严格GUI/client身份检查通过，耗时115秒；另1项真实WebSocket隔离/重连检查通过。

新178/20验证：正常S EM0.87530、reach actor0.91642、首帧19/20；generated flow native14 0.035431。保持同观测切换目标后，S角色正确28/40（14/20场景、每场景两次相同S），动作角色19/40。28次S正确中9次动作仍错，支持S判断和A执行文字两处均需研究。这些是关节运动主导代理指标，不能当作真机成功率。新旧split/norm不同，不将不同轮flow直接解释为升降。

## 两组实验

两组均官方fresh pi05 B/A、seed42同一全新递归S、limited末两语言层LoRA16/32、四个512维记忆、unroll4。相同数据/178train20val/无test/train-only norm/精确RGB缓存；均5000updates/global256/8卡micro32accum1/workers4，warmup500/peak2.5e-5/end2.5e-6，每500保存验证；最终5000优先。A结构与参数布局、S递归结构及推理路径完全相同，均仅生成离散文本进入原B/A接口。当前数据文件及标签不改，arm后缀仍仅reach。

1. `semantic_s_seed42_v1`：S关键语义加权。只从train标签词表自动计算token编辑距离；合并最近距离及+1范围的差异token位置，权重4，其他token/EOS权重1；每样本按权重总和归一，再batch平均。保留原非加权CE作可比指标。没有人工阶段图/任务专属词表/额外标签。
2. `semantic_s_actionrank_seed42_v1`：同样S损失，再增加系数0.1的动作文字ranking hinge，margin0.01（归一化native14 flow MSE）。正条件仍为模型实际生成文本，仅状态ok、与当前真标签完全相同、未被0.1dropout置空且附近标签稳定时启用；从train现有词表中选最近的另一标签作负条件。不替换错误S为GT、不把GT放入推理输入，也不编造负条件的正确动作。正负共用观测/全局任务/真实动作/flow noise/time；约束Lneg至少比Lpos高margin，满足即停止该项惩罚。每rank最多4对，按update轮转选择。动作窗口未来49帧和过去7帧标签须一致（episode端裁剪），排除阶段边界附近的动作兼容歧义。

训练稳定性保护：CE及新增加权只到S；action主损失和ranking只到A及允许B，绝不进S。主action loss继续all32平均；ranking仅比较native14，左右臂维度对称，无arm mask或隐状态注入。联合A+B clipping1、S clipping1保持；单独记录main flow、普通CE、weighted CE、ranking hinge和实际pair数，禁止把主损失和辅助项混用同名指标。

预检：两组实际八卡4→8保存恢复；第二组工程阶段使用明确标记的正确文字fixture，强制micro32+4负条件/卡的真实最大额外分支，确保S尚不会生成时也检验额外容量。fixture仅engineering_smoke，正式配置断言false。正式均重新官方初始化，绝不继承工程权重。额外核验加权CE/辅助项梯度归属、原joint action逐元素还原后均值精确相等、LoRA合并推理一致、603个原B张量未变、CPU/native会话重复。

输出 `checkpoints/pi05_piper_semantic/{semantic_s,semantic_s_actionrank}_seed42_v1`；控制 `logs/pi05_semantic_pair_20260910`，新schema8/official_pi05_recurrent_semantic_v1，实验字段experiment区分两组。普通推理结构相同，但加载器严格验证新训练身份，后续部署需要schema8 GUI支持。

两组先通过工程检查，再按顺序每组8卡训练与最终诊断；全部经gpu_reservation run_concurrent。保护其他任务，禁止GPU占用查询和短周期训练轮询。实际启动后依据工程耗时和已完成候选约7小时端到端时间估计ETA，按ETA一次跟进。

最终评估保留全因果阶段/角色/边界、native14 flow、memory-reset及原goal/换goal/正确actor/删除actor/反转actor；新增换goal后提供正确actor的离线诊断，拆分S错误与A未执行文字。验证只作比较，不代表六类任务泛化或真机成功。


## 执行与初步工程证据

2026-09-10 19:25:57（北京时间）已通过run_concurrent派发有限队列，launcher初始PID2489239/created1789039556.22。身份操作必须重新核对，不用历史PID盲目操作。第一次SSH派发前连接关闭，经核验未产生队列/identity/source manifest后才重试，未重复启动。

第一组工程真实八卡4→8恢复通过，6个非首次更新平均3.7104秒；weighted CE只到S和memory，普通action及新增native14 ranking各自只到A和limited B，208个A参数张量/18个B张量有非零梯度。原action joint损失与逐元素实现平均值完全一致；merged/unmerged实际推理一致；603个原B存储张量不变。第二组专门强制每卡4对额外分支测容量；其fixture跳过普通生成，因此不把该工程耗时直接当正式吞吐。

刚完成候选真实5000步平均4.9124秒、median3.7132秒，总训练update24561.9秒，含所有验证/存盘/最终门槛端到端25321.1秒（约7.03小时）。估计新两组须考虑这些实际长尾及第二组额外动作分支，不能只用median给出过短ETA。存储余量约4.5TiB，20个完整checkpoint按现有规模约326GiB，容量充足。

W&B规范视图已包含当前候选基线和两组新实验，7张默认主图，附加损失独立折叠，不改历史run配置或指标：[比较视图](https://wandb.ai/xiahy23-tsinghua-university/agentic-openpi-pi05-subtask?nw=hbjosycla4b)。Agilex已预装schema8两组新模型catalog/loader支持（尚无其训练权重），94项客户端检查及1项真实WebSocket会话隔离检查通过。不能把预装支持说成两个新模型已经完成或部署。


## 正式派发（2026-09-10 19:39:24）

两组工程检查全部通过，第二组每个update确有global32对（每rank4对）额外动作分支；八卡4→8保存恢复、全部原B冻结存储检查、实际梯度/merged native/会话检查通过。两组共有S初始化hash一致。第一组semantic_s已启动正式训练，第二组完成后自动接续；正式8个rank进程身份已核对，配置engineering_smoke=false、engineering_condition_fixture=false，官方fresh/5000/global256与新数据一致。

第一组预计9月11日约03:00完成；两组整体预计9月11日11:00–13:00（北京时间），受实际更新长尾及额外动作分支影响。已建立当前任务一次性heartbeat `vla`，9月11日13:00检查两组完成/失败/最终诊断；若未完成，仅一次有界进度重估并延后同一个跟进，不做固定短周期轮询。此时已交付候选无需重复部署或重启训练。提醒存于当前控制机Codex app，并非旧Windows调度。


正式启动核验已通过：8个实际训练rank，官方fresh且工程fixture关闭；首1/10步各10项本地/W&B指标一致（主CE、main flow、S/A/B梯度、LR与3项语义指标），首更新图像/动作图与类别表、首帧角色验证均已上传。步骤3–10平均3.7265秒。初次核验碰到W&B第10步异步上传尚未可读，仅检查器退出；稍后一次重试通过，未停止或修改训练。最后一次启动读取到step49，随后停止中间进度读取，按既定13:00 ETA跟进。
