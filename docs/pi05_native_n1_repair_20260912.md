# N1 teacher-forcing 数值一致性修复与重新训练

2026-09-12。授权：修复上次失败并重新开始N1训练；Q1仍只是待确认方案。

当前结论（10:41北京时间）：原数值问题修复成功，完整模型检查通过；实际重启八卡工程时受新外部显存保留任务影响，三档容量全部失败，formal未启动。实现commit e005ecfa857670e9a45e11070cc1c9991acebb99。详见末节；不能把“修复通过”和“正式训练已启动”混为一谈。

## 原因：相同特征，不同词表投影形状

原attempt_02已实际完成8卡micro32/accum1的4→8保存恢复，但在完整模型检查中退出，未启动formal。故障不是OOM或NaN。旧日志、工程checkpoint及source_manifest保留。

同一份工程step_000008权重（SHA256 aa57379b3aa3f38de2fa7e3487c349fa7df8aa0bc7efca7beb2980b51d0501c3）诊断：

| 对照 | 实测结果 |
| --- | --- |
| joint / cached 词表投影输入 | [1,16,2048] / [1,20,2048] |
| 有效目标隐藏特征及最终norm后特征 | 逐元素完全一致 |
| 不同形状BF16词表投影 | 197/2,828,672项不同，最大绝对差0.25 |
| 相同形状BF16词表投影 | 逐元素完全一致 |
| 关闭TF32的FP32词表投影 | 逐元素完全一致 |
| 目标位置argmax | 无变化 |
| joint CE / 旧cached CE | 1.2566772699 / 1.2566773891 |
| joint Action / 原生cached Action | 逐元素完全一致 |

证据：logs/pi05_native_n1_repair_20260912/diagnostic.json及diagnostic.log；脚本scripts/diagnose_native_n1_parity.py。完整原始隐藏特征在同一次进程用hook提取，先验证特征相同再做词表投影控制，不是通过提高容差掩盖来源差异。上述FP32只指输出投影控制，不冒称整套backbone在FP32完成此诊断。

## 修复范围

- cached teacher-forcing现在先选目标行，再做原生norm/lm_head，与joint训练保持相同矩阵形状；teacher_logits_cached统一checker及验证CE。
- 训练joint_outputs与loss数学不变，生成token路径不变，无新参数/decoder/query，不改B/A梯度合同。BF16验证CE可能有最后几位变化；历史指标不回填改写。
- 严格完整模型门槛仍要求有效teacher logits和Action逐元素相同（rtol=0、atol=0）。CPU真实两层HF模型新增batch2、变长目标、投影形状回归，保留独立HF参考、逐token cache、因果、梯度、optimizer恢复检查。
- 派发/预检允许真实已退出且5000及最终native gate成功的前驱；拒绝失败/缺失终态、复用PID、改变源码、重复root和已存在formal输出。空CUDA_VISIBLE_DEVICES修复保留。

## 验证及新运行

- 修改前审计旧两组7条process记录均不再存活，formal输出不存在；原计算源码6份独立备份。证据pre_repair.json及original_sources/。
- 新CPU预检已通过：logs/pi05_native_n1_20260912/attempt_01/cpu_preflight.json。实际left/right arm15类、最长11tokens含EOS，train119816帧，固定178/20及train-only norm。
- 完整修复后检查已通过logs/pi05_native_n1_repair_20260912/repaired_native.json：有效teacher logits与原生Action最大差均0；CE梯度B28.49566/A0（vision9.78218/projector2.65295/language25.03927/head9.03215），flow梯度B0/A0.786307。50×14、独立会话重复/reset、输入不变、外部subtask拒绝、GT未来隔离、prefix cache不变均通过。这些是旧工程权重的修复验证，不是新八卡恢复；随后新run仍必须重新执行8卡4→8及新checkpoint完整原生/梯度门槛。
- 新权威root：logs/pi05_native_n1_20260912/attempt_01。formal输出checkpoints/pi05_piper_native/n1_action_stop_seed42_v1，official fresh、seed42、5000/global256、首测micro32×1×8；保存/验证每500，最终5000优先。
- dispatch_verified.json证明已派发；engineering_passed.json证明新门槛；formal输出startup_verified.json证明真实首10步云端/本地loss、B/A梯度与三类media核验。没有对应真实收据不能宣称完成。
- 保持run_concurrent、保护其他任务及guard、不查GPU占用、不部署机器人。实现测试commit在派发之前；仅一次有界正式启动读取估ETA，然后更新原唯一完成提醒。

## 实际重启结果：外部显存保留阻塞

10:33:39派发新attempt_01，10:40:05有限工程流程退出，10:41:04做一次终态审计：

| 每卡microbatch | 梯度累积 | GPU数 | Global batch | 结果 |
| --- | --- | --- | --- | --- |
| 32 | 1 | 8 | 256 | 首次update完成前CUDA OOM |
| 16 | 2 | 8 | 256 | 首次update完成前CUDA OOM |
| 8 | 4 | 8 | 256 | 首次update完成前CUDA OOM |

三档均完成官方fresh装载并进入真实训练计算，但完成optimizer updates均0。新4→8恢复未完成，不能复用前一天工程状态声称本次成功。正式输出目录尚不存在，无正式loss/W&B startup或可用ETA。launcher及3个torchrun记录均已核对不再存活，145份指纹源码不变。

OOM异常中的外部进程占约56.07GiB/卡。按异常PID做只读CPU进程身份核查，发现8个gpu_hold.py进程，PID375186–375193，创建时间1789180324.93；完整argv含`/media/raid/workspace/surongpeng/ws_liyan/robotwin2-runtime/openvla-oft/bin/python toolkits/gpu_hold.py --memory-fraction 0.70 --sleep-ms 20`。父268393/created1789178761.73是ws_liyan/RLinf的`run_robotwin_hybrid_optimization_a800.sh`。这些不是本项目的guard workers。未查询GPU占用API，未向外部任务/guard发信号。

按既有共存规范登记外部pipeline的真实PID+创建时间lease，避免N1退出后本项目guard再次与其争用；不修改外部pipeline源码或行为。登记证据logs/pi05_native_n1_repair_20260912/external_pipeline_lease.json。此登记随真实进程退出失效，不能使用虚构常驻身份。

权威终态：logs/pi05_native_n1_20260912/attempt_01/capacity_terminal.json，配套b32/b16/b8_capacity_failure.json与原始日志。原phase.json只记录最后到达阶段，failure.json/exit.json和真实身份才决定当前状态。

需要资源协调或明确授权后才能处理外部保留任务并重试；没有自动再派发、没有新等待队列、Q1未启动。唯一完成提醒保持暂停，不能在未开训时制造完成ETA。原失败权重、这次工程目录与所有日志保留。
