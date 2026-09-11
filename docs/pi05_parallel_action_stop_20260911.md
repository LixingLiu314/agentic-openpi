# 首轮修订：Subtask 更新 VLM，Action 仅更新动作专家

日期：2026-09-11。状态：用户已明确授权修改训练代码并启动新实验；新实现已提交，按本文工程门槛验证后派发正式训练。当前实际阶段以 `logs/pi05_parallel_action_stop_20260911/attempt_03/phase.json` 和对应完成/失败记录为准。

服务器权威路径：`docs/pi05_parallel_action_stop_20260911.md`。本修订覆盖 `pi05_parallel_multitask_redesign_20260911.md` 中首轮三组、共用LoRA槽位、两种B梯度配平与三组调度计划；不改写历史实验。

## 1. 唯一首轮实验

首轮只实施和训练 `parallel_action_stop`。Subtask CE 更新共享VLM B与S；action loss只更新A，不回传B。Action-limited/full延后，不自动列入接续队列。

此前测得的S/A各自参数梯度差异不等于两种loss在共享B上的梯度差异。本次决定是先消除B上的多loss竞争，建立简洁基线，不宣称已实测共享B梯度相差悬殊。

```text
三路图像 + 当前state + 全局任务
                  |
               VLM B
              /     \
   末层观测特征       各层prefix K/V
          |               |
     递归Subtask S      Stop-gradient
          |               |
       CE loss         Action expert A
       更新S+B             |
                       Flow loss
                        只更新A
```

S生成文字仅用于输出/评估，不拼回B，不作为A的输入。A继续读取当前观测的各层prefix K/V，不能简化成只读末层特征。只在A读取B特征的分支detach，S读到的同一观测特征保留完整梯度图。不能全局no_grad或冻结B。

所有A能够读取的B派生特征都需要检查，包括各层K/V、可能复用的投影/embedding路径；S/A参数不共享，B参数不进入A优化器。不能只detach一处末层输出就声称action已完全隔离。

首轮采用官方B基础参数的全量CE更新，不额外安装原三组比较预留的LoRA。若以后恢复limited/full，需重新定义匹配参数化；不能把有/无adapter的历史结果冒充仅梯度掩码不同的严格对照。

## 2. Loss与优化器

`L_S = mean_examples(mean_valid_tokens(token_CE))`

`L_A = mean_batch,time,dim((velocity - (noise - actions))**2)`

普通CE包含EOS、不包含PAD，普通flow是50步全32维均值；原生14维和前15/25/50误差另记。无semantic token加权、grounding、ranking、前缀加权或subtask条件dropout。

`g_B = dL_S/dB`

`g_S = dL_S/dS`

`g_A = dL_A/dA`

首轮系数 `lambda_S=lambda_A=1`。可以在正确detach后将loss求和backward，或对两个隔离目标分别反传；必须在任何参数更新前完成本次所需梯度。两条loss不在同一参数上竞争，因此取消上一稿按B梯度比例0.3校准lambda_S的步骤；不按CE/flow的标量大小凑相等，也不引入动态多任务平衡。

S/B/A参数唯一所有权、各自AdamW状态，每参数每update更新一次。每组独立clip norm1，避免S与B或A间的联合裁剪干扰。相同既定LR schedule作为起点；仍要实测B的CE梯度、相对参数更新、S/B裁剪比例及验证退化。独立梯度不保证最优学习率或特征稳定性，不能把结构隔离说成性能保证。

S复用B词表时，词表的S查表/输出打分视图继续detach；CE经B观测路径训练视觉/投影/语言和任务/state embedding。验证CE→视觉等实际梯度，不能让直接词表更新替代预期的观测语义训练。

## 3. 保留的共同设置

- 官方fresh B/A，seed42全新S，所有optimizer/counters从零；不继承旧候选。
- S：4层512维、8 heads、FFN2048、16token，4×512观测驱动记忆，unroll4；跨update detach carry、新会话/任务reset，真实历史标签不进入记忆。
- 现有reach-arm178train/20val/no-test、train-only norm、精确RGB缓存、原生Piper12关节delta+2绝对夹爪。
- 单组5000 optimizer updates/global256/8卡，优先最终step005000，每500验证/存盘。
- AdamW betas0.9/0.95、eps1e-8、weight_decay1e-10；warmup500、peak2.5e-5、decay5000、end2.5e-6；S/B/A先共用schedule。
- micro32accum1、workers4是首测配置；完整CE→B及unroll4的新图需要实际容量/吞吐确认。不保证旧速度或内存可直接沿用。
- 暂不加learnable query、共享历史、S→A隐状态、定位/ranking。以后按单独变量讨论。
- 正式输出 `checkpoints/pi05_piper_parallel/action_stop_seed42_v1`，variant `official_pi05_parallel_multitask_v1`，schema10；工程权重不作为正式训练起点或研究候选。

## 4. 仍然存在的两项耦合

训练中的B会被CE更新，因此A接收到的特征随训练变化。Action-stop并非Frozen VLM；语义更新可以间接影响动作，既可能有益也可能使A适应困难。

推理中的S记忆只属于S；参数共享不会把本次会话记忆即时传给A。固定B/A、当前观测及噪声，重置S记忆或改变S文字不应改变A输出。若任务需要历史动作条件，应另行设计共享历史，而不是悄悄恢复文字桥。

## 5. 工程门槛

1. CE-only：S和参与计算的B视觉/投影/语言具有真实梯度，A梯度为None/零且A参数无更新。
2. Action-only：A有梯度，B与S梯度为None/零且参数无更新。检查完整每层K/V和所有替代通路，不能仅看一个参数。
3. 合并训练与分别计算两个目标的梯度一致；唯一optimizer所有权、独立clip与每参数单次update通过。跨rank和accum的缩放正确。
4. 当前官方初始化、相同普通task/state prompt、相同观测/噪声，A数值与原生direct-action一致；detach只改变反向，不应改变前向。
5. teacher-forcing只在S；真实/生成subtask不进入B/A。不同S输出/记忆reset不改变同一次A预测；噪声随机数与S调用隔离。
6. 全局因果性/会话隔离，真实8卡4→8步保存恢复、native50×14加载/固定噪声/输入不变、新schema检查通过后，才可启动正式训练。

GPU门槛和训练仍通过reservation run_concurrent；不查询GPU占用，不停止其他任务/guard。只读/模拟验证不代表机器人动作授权。

## 6. 指标与结论边界

保留前稿的S文本/actor/object/阶段/边界与记忆消融，动作native14/all32/前15/25/50/关节夹爪分开、同图换全局目标、配对选臂/选物/全任务结果。文字oracle/empty对动作应成为“不应影响”的工程检查，而非新模型的性能改善消融。

记录普通train/val CE与flow；B-from-CE、S与A裁剪前梯度/裁剪比例/相对参数更新，B视觉/语言分开。Action→B在门槛中必须为零；不持续浪费算力计算被禁止的action→B梯度来做平衡。没有两种共享B梯度的cosine，标N/A，不填伪造0。

单组不能证明CE→B改善动作，也不能比较action回传范围。日后若需因果证据，先考虑匹配的S→B detach对照；此处不自动增加新正式实验。旧串联模型作为实用历史参照，公开其文字/历史通路差异。

W&B使用新global-only动作语义/display_set，首10步一次local/cloud核验，之后按真实耗时给ETA并沿用单次跟进。规范视图已创建并回读验证：<https://wandb.ai/xiahy23-tsinghua-university/agentic-openpi-pi05-subtask?nw=i0indjgwejl>。旧失败decision接续器不自动恢复。

## 7. 本轮状态

- 用户先后授权实现/启动、并要求提交先前修改和固化提交习惯。历史快照与Git规范为 `08cb9014025cc6a52cf2776df3d5c25425c0829f`；并行Action-stop实现为 `d85537170c1dc04c4b43666f6813b4564edd058c`。原三组设计作为历史方案保留，首轮以本文件为准。
- 新模型/训练/原生加载/评估/W&B及有限派发入口使用独立 `*parallel*` 文件；未改历史模型/训练源、数据、标签或归档运行指纹。训练共享一次普通观测prefix，A在每层仅以detached B K/V读取观测；S使用保留梯度的末层特征及自身递归状态。
- CPU真实attention算子的小模型梯度路由与跨观测记忆反传已通过；真实八卡micro32×accum1×8已从4步检查点恢复至8步，S/A/B各8次更新。工程原生加载与实际大模型梯度检查结果见运行目录，不能把工程8步当成正式训练或性能证明。
- attempt_01因前向比较检查失败退出，未启动正式训练。同权重原生cached动作与新路径逐元素一致，旧joint与cached本身有相同最大0.019366差异。修订checker不扩大native容差：要求native有效prefix/动作精确相等，并用禁用TF32的FP32检查joint等价；不把全masked PAD输出当作S有效特征。模型/训练数学未变，attempt_02使用新源码指纹重验八卡门槛。CPU另已核验loss合并反传等于分别计算后求和。所有失败证据保留。
- 正式训练从官方B/A和匹配seed42新S重新初始化，5000步；优先最终checkpoint。有限runner在训练完成后执行严格原生加载并写入 `candidate_ready.json` / `complete.json`；只表示可加载实验候选，不表示真机成功。失败写入对应 `gates_failure.json` / `formal_failure.json`，不自动恢复旧队列。
- 首10步的实际本地/云端loss、S/A/B梯度和示例核验保存在正式输出的 `startup_verified.json`。梯度曲线是S/A/B各组裁剪前范数；相对更新是每张量前16个元素的采样统计，不是全参数精确相对范数。视觉/语言独立梯度由工程门槛验证，当前未作为持续训练曲线记录。额外前缀、关节/夹爪、目标干预及记忆消融诊断仍需独立结果，不能把当前普通验证当成这些分析已完成。
- attempt_03另纳入独立 `visualize_parallel_step.py`，首步图注明确S文字只显示、不条件化A；模型与训练数学不变。`run_config.json`保留原启动Git HEAD与各源文件哈希；恢复保留此启动SHA，当前HEAD单独记入恢复收据，仅文档提交不影响精确恢复，但任何配置/源/runtime/data变化仍拒绝。相应CPU回归与新八卡门槛作为当前版本证据；不修改在训指纹源码。
- 未部署到Agilex、启动新模型服务或操作机器人。
