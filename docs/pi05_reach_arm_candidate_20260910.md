# Limited + 递归 S + reach-arm 候选训练

用户在2026-09-10明确要求启动“候选”。本次只训练一个候选，不运行原标签对照。

服务器仓库：`/media/raid/workspace/surongpeng/ws_lixing/agentic-openpi`。

- 正式输出：`checkpoints/pi05_piper_reach_arm/limited_recurrent_seed42_v1`。
- 控制与检查记录：`logs/pi05_reach_arm_20260910`。
- 模型身份：schema7、`official_pi05_recurrent_reach_arm_v1`；新加载器为`src/openpi/policies/reach_arm_subtask_policy.py`。不以旧schema6/frozen加载器解释新权重。

## 固定配方

从已审计官方pi05 B/A权重开始，S及其递归模块seed42全新初始化，不继承任何已训练模型、优化器或计数器。正式5000个optimizer updates，8卡×micro32×accum1=global256，unroll4；每rank4个数据worker。warmup500，peak2.5e-5，cosine到5000，end2.5e-6；每500步保存和验证。

S保留四个512维、仅由观测更新的记忆token；不引入真实历史标签或未来动作输入。CE只更新S；action loss只更新A和limited B。B仅末两层语言块attention q/k/v/o与MLP gate/up/down的LoRA，rank16/alpha32。维持S与A+B两个norm1裁剪组；分别测量A/B裁剪前梯度，不改变更新规则。

action仍使用100%生成的离散subtask文字、0.1独立置空；没有GT条件课程、角色词重加权、额外actor头或S隐状态注入。A结构、32维/50步模型输出和Piper原生14维转换保持。

## 数据

`Datasets/eggplant_potato_reach_arm_v1`，资产`assets/pi05_piper_reach_arm_v1/eggplant_potato`。仅接近盖子、茄子、紫薯标签加臂；其余标签保持。178训练/20验证，无test，seed42按完整episode及任务/布局/角色分层，统计只来自新训练集。

- split文件SHA：`e86b8d7f91a60c6b120cfbe821b882980b56f8dc99dfe701e80124f7581c2bab`
- norm SHA：`e42de696d2a699111c2bb1022c9de74acd47706a147a5cfa9c8fe9d7ff1f366d`
- actor修改清单SHA：`b95c4a31ac5bb39f56a7702ef9d270bfb8c1f5ce74447c8725cbfeadaf83146b`

新缓存`.stage1_staging/piper_rgb224_reach_arm_v1`：对原缓存534个视频的源、RGB和时间戳内容逐项哈希校验后链接复用，另60个视频实际解码，共594。新manifest绑定新split，未仅替换旧缓存哈希。与原始读取路径比对587个变换后样本精确一致，涵盖全部198段首尾、抽样边界与随机帧。缓存引用的原数据和原RGB缓存应保留。

## 验证与交付

正式启动前经过真实8卡4步保存、恢复至8步，再验证原生加载、两个独立会话/重置重复、合并与未合并推理一致、实际CE/action梯度分离、603个原始B存储张量未变。工程8步仅用于流程检查，正式训练重新从官方起点开始。

每500步验证保留因果episode重放，并增加全部episode首帧、起始一秒和reach起点的动作flow覆盖；单独报告reach actor、初始phase+actor、忽略臂后缀的phase正确率。与旧划分/旧稀疏指标不直接作严格对比。

最终**step005000为主要候选**，较早验证最佳仅保留参考。有限runner在5000完成后自动检查原生policy，再做验证集正常记忆/每次重置、同图换全局任务和正确/删除/反转actor文字的离线动作干预。关节主导臂指标与flow误差是代理指标，不是真机成功率。完整推理权重导出已合并LoRA；精确训练恢复另外使用未合并权重和8个rank各自的优化器/RNG/记忆状态。

W&B使用既定用户项目及规范视图，已核对早先用户“之后都按照规范配置wandb”的持续授权。新视图：<https://wandb.ai/xiahy23-tsinghua-university/agentic-openpi-pi05-subtask?nw=1ig80uudfmj>。正式run ID `limited_recurrent_seed42_v1`，display_set `official-pi05-reach-arm-s42-v1`。首批日志将核对本地与云端一致；工程检查不进入research曲线。展示工具故障不应误停健康的训练。

全部GPU作业由`run_concurrent`登记真实进程租约，保留其他任务与guard，不查询GPU占用。训练在服务器独立进程运行；操作前核对PID与creation time。正式启动后根据实际耗时给ETA，在该时间跟进，不做固定频率中途轮询。没有机器人动作；本轮尚未将新schema7安装到Agilex GUI。

## 当前入口

工程：`scripts/run_reach_arm_candidate.py --stage gates`，通过后正式：同一入口`--stage formal`。不要重复启动；先读上述控制目录的phase、process identity、failure/exit、engineering_passed和complete记录。

新文件源码已在工程派发时冻结于`source_manifest.json`。正式训练的`run_config.json`与源码/runtime快照进一步固定数据、环境与实现；运行期间不能修改这些文件。工程初始launcher身份在`gates_launcher.process.json`，正式launcher身份另保存在`formal_launcher.process.json`。
