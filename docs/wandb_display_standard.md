# W&B 显示规范 v1

适用范围：agentic-openpi 后续训练、消融和评估。2026-09-08 起按此规范接入；正在运行且需要精确恢复的实验通过独立日志桥接迁移显示，不修改其训练源码或原始日志。

## 1. 日常怎么看

使用命名的 **saved view**，不要从项目中全部历史指标自动生成的页面判断结果。当前视图为 `Pi05 Backbone gradients s42 v1`，[打开当前视图](https://wandb.ai/xiahy23-tsinghua-university/agentic-openpi-pi05-subtask?nw=2lxpka702pl)，链接同时记录在 `logs/wandb_display_v1/workspace.json`。

- 第一排：验证集动作 Flow MSE、subtask 完全匹配率、subtask Macro-F1。动作误差越小越好，两个 subtask 指标越大越好。
- 第二部分：train action loss、train subtask CE、val subtask CE、学习率。train 和 val 分图。
- 梯度、生成异常率、速度、首步示例和逐类别表默认折叠。
- 当前蓝色是 `B-limited | s42`，橙色是 `B-full | s42`。全量实验真正开始后才显示数据，不制造空的训练记录或虚构曲线。
- 横轴从本次初始化后的第 0 次 optimizer update 起计数。当前两个实验均从 M3 step3500 初始化，横轴 500 表示新增500次更新，不能解释为历史累计训练到500步。
- 两组尚未训练到同一步时，只在相同训练预算上比较。128帧、2次噪声抽样的固定验证结果是离线估计，不是真机任务成功率。

## 2. Run 身份与可比性

| 字段 | 规范 |
| --- | --- |
| entity / project | `xiahy23-tsinghua-university / agentic-openpi-pi05-subtask` |
| name | 简短、有含义的 `实验变体 | s<seed>`；例如 `B-limited | s42`。目录、完整时间戳、超参放 config |
| id | 稳定身份，与输出目录和不可变配置绑定；精确恢复使用相同 ID，新实验必须新 ID |
| group | 同一对比实验的 display_set；不能把 M1、M3、R1、smoke 混作一个实验 |
| job_type | `train`、`evaluation`、`engineering`；旧日志显示迁移专用 `metrics-view` |
| tags | `display-v1`、research/engineering、实验族、变体、seed；工程检查不得进入研究主视图 |
| config 必填 | `display_schema=1`、`display_set`、`experiment_family`、`variant`、dataset、seed、global_batch_size、num_train_steps、parent/checkpoint hash、split/normalization hash、优化器和 LR schedule、梯度路径、精度、world/microbatch/accumulation |

只对相同数据划分、归一化、评估样本、噪声抽样、action维度和训练预算的结果直接比较。参数、样本数、模型结构、checkpoint路径、来源、事件名只进 config/summary/table，不逐项生成 scalar 图。精确恢复必须检查配置与源码指纹。

同一训练实验只允许一个直接 W&B writer，限 rank0。`metrics-view` 仅用于旧格式迁移，必须记录 `view_only=true`、`source_run_url`、`source_metrics` 和原配置校验值；不能与原 run 一起计入实验样本量。

## 3. 指标字典与横轴

所有曲线的显式横轴为 `trainer/step`，单位是已完成的 **optimizer updates**。日志传输 `_step`、JSONL 行号、microbatch、GPU rank、验证调用次数不能作为训练步数。每条曲线记录显式携带横轴，不借用上一条事件的步数。

| 指标 | 含义 / 方向 | 默认显示 |
| --- | --- | --- |
| `val/action_flow_mse_native14` | generated subtask 条件下，归一化 native14 flow MSE；越低越好 | 第一排 |
| `val/subtask_exact_match` | 完整生成文本与标签完全匹配的帧比例；越高越好 | 第一排 |
| `val/subtask_macro_f1` | 各类别 F1 的宏平均；越高越好 | 第一排 |
| `train/action_flow_mse_all32` | 本次更新、全局 batch 聚合后的32维 action训练损失 | 学习过程 |
| `train/subtask_ce` | 本次更新的 subtask token CE | 学习过程 |
| `val/subtask_ce` | 固定验证集 subtask token CE | 学习过程 |
| `optim/lr` | 当前 S/A/B 共用 LR | 学习过程 |
| `optim/grad_norm_s`、`optim/grad_norm_a_b` | clipping 前的梯度 L2 范数 | 折叠诊断 |
| `val/invalid_generation_rate`、`val/unknown_rate` | 解码异常、生成文本不在任务词表的比例 | 折叠诊断 |
| `train/invalid_generation_rate`、`train/empty_condition_rate` | 全局生成异常比例、训练条件置空比例 | 折叠诊断 |
| `perf/update_seconds` | 单次 optimizer update 耗时，不含随后的验证/存盘 | 折叠速度 |
| `perf/examples_per_second` | effective global batch / update_seconds | 折叠速度 |
| `progress/fraction` | optimizer step / 本次总更新数 | run summary；不占主图 |
| `val/action_flow_mse_all32` | 原评估同时记录的32维 flow MSE | 保留，不占默认图 |
| `media/subtask_per_class` | step、文本类别、F1、recall、support 的单张 Table | 折叠示例 |
| `media/first_update_inputs`、`media/first_update_actions` | 首次真实更新示例，只记录一次 | 折叠示例 |

准确率、F1、异常率统一使用0–1，图上注明 fraction；例如0.82表示82%，不得有的 run 写0.82、有的写82。训练32维损失与验证native14误差不能同名、不能当成同一量直接叠加。Flow MSE 也不能标成真实动作位置误差或任务成功率。

禁止无区别 `flatten()` 任意嵌套字典或 `wandb.log(raw_event)`；必须显式注册并映射指标。损失语义变化（例如加权CE、不同维度/单位/验证协议）要新增名字或版本，不能沿用旧名字混画。S/A/B各自学习率不同的实验应显式新增 `optim/lr_s` 等。

边界切换实验需建立专用附加区，区分边界 EM、失败计入后的延迟、切换失败率、早切/夹爪保护和稳定阶段回退；报告单位、事件定义、样本量与协议版本。不能仅靠普通 EM/F1 宣称切换延迟已解决，不能用不存在的数据填空图。

## 4. 采样、布局与 summary

- 本地 JSONL 是完整事实源，先落盘再上传。新训练默认每10个 optimizer updates上传训练 scalar，step1、step0验证和每500步验证/存盘必须保留；按配置声明 cadence。当前迁移保留已有每步数据，不重采样。
- 每个 scalar 明确是全局 batch 的均值还是计数；不得把单 rank 或 microbatch 误当全局。分布式计数先求和，再计算比例。
- 默认关闭自动生成 panels；5个有序区域：验证效果、学习过程、诊断、速度、示例/类别表。前两区共7张曲线展开，其余折叠。主要图3列，同一指标对比各变体。
- 默认 raw 曲线、smoothing=none、保留异常点、不跨 run 求平均。需平滑时保留原始轨迹并说明方法；验证仅11个点时不要用平滑掩盖波动。
- 参数统计、run状态、latest checkpoint、best checkpoint及选择准则放 summary。各指标的独立 min/max 不能假装来自同一个 best checkpoint；选择记录必须带 step 和准则。当前选择准则是 `val/action_flow_mse_native14` 最小。
- 逐类别结果只用 Table；首步示例只上传一次；不上传权重/优化器大文件。关闭自动代码扫描和 GPU/system telemetry，避免重复采集与无关图表。
- 训练失败或显示同步失败要区分。同步桥的 W&B run status 表示同步进程状态，训练状态以 `source_status` 和原始训练产物为准。

## 5. 新训练如何接入

公共实现：`src/openpi/training/wandb_standard.py`。在新训练代码启动前接入并纳入源码指纹；不修改正在训练或等待精确恢复的旧源码。

```python
import wandb
from openpi.training.wandb_standard import configure_run, log_event

# rank0 only; config 中按上文补齐身份和不可变参数。
run = wandb.init(
    entity="xiahy23-tsinghua-university",
    project="agentic-openpi-pi05-subtask",
    id=stable_run_id, name=display_name,
    group=config["display_set"], job_type="train",
    tags=["display-v1", "research", config["experiment_family"]],
    config=config, resume="allow",
    settings=wandb.Settings(x_disable_stats=True, disable_git=True, save_code=False),
)
configure_run(run)
# 先持久化原 event；按 cadence 上传。不要再次 log 原始 event。
log_event(run, {"event": "train", "step": optimizer_step,
                "loss_subtask": subtask_ce, "loss_action": action_flow_mse_all32,
                "lr": learning_rate}, config)
```

此 mapper 当前支持已核验的 backbone 事件语义。新增训练器应显式适配并验证其实际损失含义，禁止为了接入而把不兼容量改名冒充。未来训练的 `steps` 和 `global_batch_size` 也要提供给 mapper；摘要中另存标准 `num_train_steps`。

首次接入必须通过：train/val同step不串线、横轴是optimizer步、0–1单位、Table不展开、非有限值拒绝、断点恢复不重复；并抽查实际 W&B history 与原始JSONL一致。

视图创建/更新使用独立工具环境，不修改训练环境的依赖：

```bash
cd /media/raid/workspace/surongpeng/ws_lixing/agentic-openpi
PYTHONPATH=src:scripts .stage1_staging/wandb_display_env/bin/python \
  scripts/configure_wandb_workspace.py \
  --display-set backbone-grad-m3-s42 \
  --name 'Pi05 Backbone gradients s42 v1'
```

新比较组使用新的 `--display-set`、`--name` 和 `--receipt logs/<experiment>/wandb_workspace.json`。同一 receipt 重运行会更新同一个保存视图；不覆盖个人工作区、不批量删除历史 run。依赖锁定和验证见 `logs/wandb_display_v1/verification.json`。

## 6. 本次迁移

旧 `train_backbone_gradient.py` 把原始 event 直接上传，train/val 共用 `loss_subtask`，未声明optimizer步横轴，历史自动图过多。修复方式：新增 `scripts/sync_wandb_standard.py` 从 limited/full 各自的 durable JSONL 持续读取，使用稳定的 `viewv1-*` 展示 ID，配置独立 saved view。原模型、超参、优化器、源码、原 run/history不变。

桥接只覆盖正式 `limited_seed42` 与 `full_seed42`；不重启训练、不新增种子/模型实验、不读GPU遥测。只在对应源目录有记录时创建展示 run。桥接每30秒检查新行，跳过尚未完成的JSONL末行；重启依据W&B transport step去重；单实例锁防止第二个 writer。有限训练的进程身份消失或两组结束时退出，不无限监视项目目录。

运行身份、日志与视图地址都在 `logs/wandb_display_v1/`。已有实例运行时不要再启动；故障修复后可用相同命令恢复：

```bash
PYTHONPATH=src:scripts .stage1_staging/wandb_display_env/bin/python \
  scripts/sync_wandb_standard.py --root "$PWD"
```

官方 API 依据：[自定义指标横轴](https://docs.wandb.ai/models/ref/python/experiments/run#define_metric)、[Workspaces API](https://docs.wandb.ai/models/ref/wandb_workspaces/workspaces)。这里只使用可重复执行的 Python 模板和 saved view，不依赖 Enterprise Workspace Templates。

## 7. 本次验收记录

6项CPU检查通过；实际云端2063个训练点、5个验证点与源JSONL数值完全一致，重启显示writer后train/val各自无重复步。类别Table与首步两张图片已上传。已通过认证SDK读回5区布局、optimizer横轴、关闭自动图/平滑/聚合；浏览器未登录，未做页面截图验收。32个活动训练源码哈希保持一致。详细证据：`logs/wandb_display_v1/verification.json`。全量实验尚未产生源日志时不显示曲线，其后续数据由同一桥接自动接入。
