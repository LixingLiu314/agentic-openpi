当前实现的正常入口是 `openpi.policies.subtask_policy.create_subtask_policy`。输入为当前三路相机、当前 state 和全局任务，输出为内部生成的 subtask 以及原单位动作块。输入中不放 subtask 真值。模型调用只产生数组，不连接或驱动机器人。

服务器项目根目录：`/media/raid/workspace/surongpeng/ws_lixing/agentic-openpi`。运行环境使用项目 `.venv`。当前 GPU 作业由现有序列统一交接；独立 GPU 调用也应通过 `scripts/gpu_reservation.py run --log <唯一日志> -- <命令>` 执行。

下面展示已训练 M3 的调用接口。三个种子实验结束后的最终候选，以 `assets/pi05_piper_stage1/eggplant_potato/final_test_protocol.json` 中的部署候选记录为准；当前首轮权重已经满足正常推理的阶段门槛。

```python
import numpy as np
from openpi.policies.subtask_policy import create_subtask_policy

policy = create_subtask_policy(
    "checkpoints/pi05_piper_stage1/m3_pilot_seed42/step_003500",
    device="cuda:0",
    num_steps=10,
)

# 三幅图像为 RGB uint8 HWC；state 沿用数据集原来的 14 维顺序和数值单位。
observation = {
    "images": {
        "cam_high": rgb_high,
        "cam_left_wrist": rgb_left_wrist,
        "cam_right_wrist": rgb_right_wrist,
    },
    "state": np.asarray(native_state, dtype=np.float32),
    "prompt": "Put the eggplant into the box",
}
result = policy.infer(observation)
assert result["actions"].shape == (50, 14)
print(result["subtask"], result["subtask_status"])
print(result["policy_timing"])
```

`actions` 已经完成反归一化，12 个关节维度从相对当前 state 的预测恢复为 absolute；第 6、13 维为原始夹爪命令。调用方不要再次归一化或反归一化。当前评估保留未裁剪输出；控制器单位、轴序和可执行范围需在真机接入时与数据来源核对。

语义返回值为 `subtask`（普通字符串）、`subtask_status` 和 `subtask_score`。实际状态只有 `ok`、`empty`、`truncated`：后两种情况返回空文本并走训练过的条件置空动作路径。达到 EOS 表示本条句子结束，不表示机器人任务完成。`subtask_score` 是未校准的平均 token log probability，不能直接作为成功率或置信概率阈值。当前接口没有实现独立的硬件超时中断。

`policy_timing` 记录输入转换、图像编码、第一次 prefix、subtask 自回归生成、第二次 prefix、flow 采样、输出转换及总耗时。CUDA 上分段计时会同步 GPU；总耗时不包含相机传输和视频解码。部署显存通过 `scripts/benchmark_subtask_policy.py` 独立测量，避免训练或离线 loss 的缓存混入结果。

预测长度固定为 50 帧，实际执行长度由最终协议单独确定。可用现有客户端分块器按该长度重规划：

```python
import json
from openpi_client.action_chunk_broker import ActionChunkBroker

with open("assets/pi05_piper_stage1/eggplant_potato/final_test_protocol.json") as stream:
    protocol = json.load(stream)

broker = ActionChunkBroker(policy, action_horizon=protocol["execution_frames"])
one_step = broker.infer(observation)
assert one_step["actions"].shape == (14,)
```

分块器在用完约定帧数后才调用下一次策略推理；期间返回的是此前动作块的切片，语义字符串和计时标量保持原值。新的观察应由调用方随控制步提供。这个接口本身不实现机械臂通信或实时调度。

训练分为 M1、M2、M3，入口 `scripts/train_subtask_hierarchy.py`；B1/O 对照入口为 `scripts/train_subtask_action_control.py`。模型参数所有权为：冻结 C0 参数组，新 decoder S 只接受 CE，action expert/动作时间投影 A 只接受 flow loss。S/A 使用不同 optimizer、各自裁剪梯度，两个 optimizer 不共享参数。动作条件通过独立 BOS 自回归生成获得，不使用 teacher-forcing 的逐 token argmax。缺失标签以空 target mask 表示，不进入观察；当前数据中的标签均完整。

训练恢复使用对应 run 的原始参数加 `--resume`，保持 world size、batch/accumulation、数据和源码/运行库身份相同。每个 rank 的 optimizer/RNG 分片随完整 checkpoint 一起保存。当前三个一次性序列的进程及日志分别为 `controls_phase_sequence`、`replications_sequence`、`finalization_sequence`，都位于 `logs/pi05_subtask_stage1`。现有输出和失败日志保留，接续时核实真实 PID/创建时间和完成事件，不重复启动已有序列。

离线评估默认使用验证集。G/M1 使用 `scripts/eval_subtask_offline.py`；C0/B1/O 使用 `scripts/eval_action_control.py`。`--split test` 必须同时给出已固定的 `--test-protocol`，校验所选权重和 metadata 的哈希、数据划分、归一化、执行长度、参数和 world size。最终序列会先完成全部验证，再固定清单并运行测试；不使用测试结果重新选权重或调整训练预算。

每次动作评估保存完整原单位预测/目标、frame index、noise draw、条件及有效 horizon 的 NPZ，并从这些数组复核指标。统一比较入口 `scripts/summarize_subtask_comparison.py` 按 episode 做配对 bootstrap，分别报告三个种子的结果、均值和标准差。离线 flow/native 误差、阶段识别率和条件干预均不等同于闭环任务成功率。

基础权重核验见 `logs/pi05_subtask_stage1/official_base_weight_audit.json`：811 个 JAX 来源 tensor 与官方转换结果逐值匹配；唯一没有 JAX 来源的 action-expert LM head 是转换器初始化后保留的未使用参数，已通过 NaN 扰动和调用 hook 检查。此例外不改变当前 VLA 的预测，也没有修改训练权重。

正常研究权重入口已补充实际 CPU 验收：`logs/pi05_subtask_stage1/research_policy_entrypoints_cpu.json` 覆盖首轮 C0、G、B1，均使用正式加载器；G不设置`allow_engineering`。同一真实验证观察与固定噪声的两次10步flow输出逐元素一致，动作50×14且有限，输入不变，50帧客户端切片、元数据、reset及replan通过。G自行生成`reach the handle of the lid`，状态`ok`，并拒绝显式外部subtask字段。该检查证明调用契约，GPU延迟/显存仍以已排队的独立测量为准。
