# N1：VLM原生Subtask生成，并行Action-stop

日期：2026-09-11。用户选择：只进行N1，明确使用left/right arm版本数据。N0及其他对照不安排。

最新修复状态（2026-09-12）：用户授权修复并重新训练。旧N1 attempt_02实际完成8卡工程4→8，但在原生teacher logits严格检查退出，正式未开始；下方“等待前驱”仅为历史。修复报告见docs/pi05_native_n1_repair_20260912.md。新root logs/pi05_native_n1_20260912/attempt_01；不复用旧source_manifest或工程更新，重过全部门槛后official fresh formal。真实启动与首10步收据另行更新。

状态：2026-09-11 15:13北京时间已真实排队。有效root logs/pi05_native_n1_20260911/attempt_02，phase=waiting_predecessor，PID1256324/created1789110811.06；dispatch_verified.json读回queued=true、GPU可见性覆盖为null。派发commit3d911c91a1ffad120d766a5476efe9cb8ce871d7，核心实现5f3408a。当前action_stop_seed42_v2及其最终检查成功结束后，才执行N1八卡门槛，再fresh正式训练。完整官方模型及八卡GPU门槛尚未执行。旧attempt_01因CPU-only环境继承问题只停止了尚无子进程/GPU工作的waiter，证据保留；不再使用其历史waiting状态。

## 1. 唯一新实验

建议实验名：native_parallel_action_stop。新实现必须使用独立variant/schema和输出目录；不得借用当前parallel或旧recurrent身份。

- Subtask由VLM自身的预训练语言层和原生词表输出路径进行自回归生成。不新增、沿用或变相恢复独立Subtask Transformer decoder。
- 文字训练采用移位GT输入及causal mask，仅目标subtask token（含EOS、不含PAD）计算普通CE；原始观测前缀、Action与任何能影响Action的特征都不能读取GT输出区。
- CE更新B的参与计算路径，包括视觉、投影、语言和原生词表输出权重。绑定的输入/输出词表参数只登记一次，不沿用旧独立S词表视图detach规则。
- Action读取普通图像、当前状态和全局任务对应的各层B K/V；每一条Action到B梯度通路截断。Flow loss只更新原有Action expert及其动作/时间投影。
- Action训练和推理均不读取GT或生成subtask；文字仅用于输出、验证和训练B。固定权重/观测/噪声，改变GT输出区或关闭文字生成不能改变Action。
- 首轮单帧观测，无跨帧递归记忆、上一subtask输入、learnable query、grounding、ranking或动作前缀加权。与当前在训模型比较时公开记忆机制差异，不能将差异全部归因于原生语言生成。
- 不进行N0冻结B对照，不恢复Action-limited/full、旧decision或semantic队列；不自动加入显式subtask条件化Action的第二阶段。

## 2. 固定left/right arm数据

服务器项目根目录：/media/raid/workspace/surongpeng/ws_lixing/agentic-openpi。

- Dataset root：Datasets/eggplant_potato_reach_arm_v1。
- Repo ID：local/eggplant_potato_reach_arm_v1。
- Assets：assets/pi05_piper_reach_arm_v1/eggplant_potato。
- Split：上述assets中的split.json，固定seed42、178个train episode、20个val episode，无独立test；不重新划分。
- Normalization：上述assets中的norm_stats.json，仅基于现有train划分；不重新计算，不混用旧split的norm。
- RGB cache：.stage1_staging/piper_rgb224_reach_arm_v1；实现时仍须验证缓存身份与数据/划分匹配。
- 原生动作定义：12维关节相对当前状态的delta和2维绝对夹爪；50步窗口，训练保持32维填充表示，原生14维另评估。

仅三个reach阶段增加with the left/right arm：

1. reach the handle of the lid
2. reach the eggplant
3. reach the sweet potato

例如：reach the handle of the lid with the right arm；reach the eggplant with the left arm。其他阶段保持原始标签，不给所有阶段重复加臂，不改标注、原始轨迹或视频。左右臂标签是离线示范监督，不能把推断标签时用到的未来动作引入模型观测。

本次已实际核对的SHA256：

- split.json：e86b8d7f91a60c6b120cfbe821b882980b56f8dc99dfe701e80124f7581c2bab
- norm_stats.json：e42de696d2a699111c2bb1022c9de74acd47706a147a5cfa9c8fe9d7ff1f366d
- meta/reach_arm_annotations.jsonl：b95c4a31ac5bb39f56a7702ef9d270bfb8c1f5ce74447c8725cbfeadaf83146b

READY.json状态为ready_for_training_configuration；数据入口沿用reach_arm_data.data_config的完整数据契约校验，但新模型身份不得复用该模块的历史recurrent variant。

## 3. 固定训练配方

- 官方fresh pi05 B/A；不继承当前实验或历史候选的训练权重、优化器、计数。
- seed42，5000 optimizer updates，global batch256，8卡；优先micro32/accum1，实际容量检查后允许保持global256的等价累积配置。
- B、A唯一参数所有权，各自AdamW和独立clip norm1；无独立S优化器。betas0.9/0.95、eps1e-8、weight_decay1e-10。
- warmup500，peak LR2.5e-5，cosine end LR2.5e-6；每500验证和保存，最终step005000为主要候选。
- 普通CE和普通50×32 flow系数均为1；不按标量loss大小配平。语言输出预算16 tokens，须对实际tokenizer、输出边界和EOS做长度校验，不能静默截断arm后缀。
- 保持现有完整episode划分及因果采样；无跨帧模型记忆，不再把unroll4作为递归反传合同。
- 原生文字生成的吞吐/显存必须实测，不承诺与当前独立S方案同耗时。

## 4. 实施与派发门槛

1. 核验官方checkpoint的原生词表/输出权重与绑定关系、tokenizer、输出边界；禁止随机初始化输出头后冒充原生能力。
2. 教师强制与逐token缓存路径一致；因果mask无未来答案泄漏，原始前缀及Action看不到subtask目标区；原有Action位置编号不被文字suffix移动。
3. 分loss实测：CE到B视觉/投影/语言非零且A为零，flow到A非零且B为零；共享参数无重复优化器所有权。
4. 固定观测/权重/噪声，替换目标文本、关闭生成等不改变原生Action输出；保留严格原生加载与可解释的数值等价检查。
5. 真实8卡容量、4→8保存恢复、初始化与计数、DDP/accum缩放、新schema加载检查通过，才启动正式5000步；工程权重不作为正式初始化。
6. 新模型/训练入口使用独立文件，不修改action_stop_seed42_v2的指纹源码；实现完成先测试、按明确路径git commit，记录启动SHA与源码/数据/runtime身份。
7. GPU工作沿用run_concurrent，不查询GPU占用、不停止其他任务或guard、不打断当前实验。若要接续，必须明确记录真实等待/派发进程、身份及依赖，不能仅写计划就报告已排队。
8. 正式派发后按项目规范核验首10步W&B/local两loss、B/A梯度与实际输入/输出示例，基于实际耗时给ETA；不短周期轮询。当前提醒不因本次方案落地自动修改。

## 5. 评估与结论边界

- Subtask：自由自回归生成的整句准确率、macro指标、三个reach阶段的actor/object、首帧和阶段边界。无GT历史输入，不以teacher-forcing CE代替生成质量。
- Action：普通all32与native14、前25/完整50、关节与夹爪分别报告；同观测/噪声进行可比分析。额外诊断未实际完成前不得宣称已有结果。
- B/A裁剪前梯度、裁剪比例、相对更新；采样相对更新必须标注sampled，视觉/语言梯度观测须区分门槛验证与持续曲线。
- 当前旧模型仅作实用参考，记忆/生成路径/词表优化等差异应公开。只做N1，没有匹配N0，不能声称已经因果证明CE→B改善动作。
- 20个val episode、单seed仅作初步实验；按episode而非相关帧统计不确定性。未做受控真机实验前不宣称选臂/选物成功率提升。
- 本协议不包含push、模型部署、机器人服务切换或实际机器人动作。

## 6. N1独立实现与门槛状态

- 模型src/openpi/models_pytorch/native_subtask.py，stage native_subtask，schema11，variant official_pi05_native_subtask_n1_v1。全部模型参数来自base；不创建S网络，不创建S优化器。原生lm_head与输入词表保持同一Parameter，由B拥有一次。
- 普通pi05全局任务/原生14维state prefix保持原样（原模板以Action:结尾）。文字使用单独的causal流，固定cue为`\nSubtask: `，随后GT右移输入；仅目标16位置计算CE，含EOS不含PAD。此模板是本项目N1的明确选择，不声称逐字复现论文未公开的训练模板。文字流共享各层VLM对象并读取原prefix K/V；原prefix和Action永不读取文字流，Action位置不因文字流变长而改变。
- 部署先计算普通prefix cache；文字逐token使用私有text-only KV cache，不修改供Action使用的prefix cache。不论是否生成文字，给定相同普通观测/权重/噪声，Action不变。输出无跨观察/session记忆。
- CPU预检logs/pi05_native_n1_20260911/attempt_01/cpu_preflight.json已通过。真实小尺寸PaliGemma/Gemma验证原生联合/增量文字等价、因果与动作隔离、视觉/投影/语言/词表真实CE梯度、双optimizer保存恢复和共享词表；pidfd用真实短生命周期CPU子进程验证退出事件。没有使用GPU进行这些检查。
- 完整官方权重812个存储tensor的装载核对、真实8卡32×1容量/4→8恢复、实际梯度/原生50×14与新loader，由scripts/run_native_n1.py在当前launcher及最终检查成功结束后自动执行。检查失败则N1退出留证，不启动正式训练，不停止其他实验或guard。正式始终official fresh，不承接工程checkpoint。
- root logs/pi05_native_n1_20260911/attempt_01；正式输出checkpoints/pi05_piper_native/n1_action_stop_seed42_v1；等待依赖当前attempt_04/formal_launcher.process.json，以PID+创建时间+真实argv验证，而非假设相对/绝对路径字符串一致。pidfd事件等待，无中间step/GPU轮询。
- 专用W&B视图：https://wandb.ai/xiahy23-tsinghua-university/agentic-openpi-pi05-subtask?nw=2r09j4dx7vn 。已保存并API读回7张主图及折叠诊断/速度/示例；当前未formal启动，不存在N1 loss数据。首10步脚本自动对账云端与本地两loss、B/A梯度、三类media，并用实测速率保存startup_eta.json；验证/存盘/最终门槛暂留2小时估计余量，真实耗时不保证。
- 当前比较须公开：N1不再使用4层独立S或4×512递归记忆，改为原生B语言层/共享词表、单帧、unroll1；数据流packing也随之变化。这不是只改变一项变量的匹配消融，不据单N1宣称CE→B的因果收益。
