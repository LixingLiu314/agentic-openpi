# 约十小时窗口内的两组 S 对照实验

用户于2026-09-09约22:01（北京时间）授权安排两组实验，约十小时后在 Agilex 实测。目标交付时间为2026-09-10约08:00。本文覆盖此前“仅规划、未授权训练”的状态；不授权机器人动作。

两组均使用当前 eggplant-potato 数据，固定 train158/val20，封存 test20不读取。保持原图像缓存、归一化、任务和 subtask 标注、原生 Piper 动作约定。

| 项目 | 无状态对照 | 递归状态候选 |
| --- | --- | --- |
| 目录 | pi05_piper_recurrent_s/stateless_seed42_v1 | pi05_piper_recurrent_s/recurrent_seed42_v1 |
| 初始化 | 官方 pi05 B/A；全新 seed42 S | 同一官方 B/A、相同共有 S 参数；新增状态模块独立确定性初始化 |
| B | 冻结 | 冻结 |
| S | 原文字解码器，无跨步状态 | 原文字解码器 + 4×512 状态、观测交叉注意力及 GRU 更新 |
| A | 原 action expert，沿用文字经 B 编码的条件路径 | 同左；不接收 S 隐状态或 memory token |
| 采样 | 8条流×4时刻/rank，沿 episode 顺序取样 | 完全相同；跨训练块保留状态、截断梯度 |
| 总批量 | 8卡×32=256，accumulation1 | 同左 |
| 训练预算 | 4000 optimizer updates | 4000 optimizer updates |

原方案5000步/组按实测约3.65秒/步，仅训练就超过10小时。为了给验证、模型传输和实机加载检查留时间，本轮统一缩短为4000步；这是用户当前时间窗口下的配对预算调整。warmup500、峰值2.5e-5、末值2.5e-6，cosine在4000步结束；每500步保存验证，仍只使用seed42，不继承现有模型/优化器/更新计数。

当前观测间隔在0.5–1秒内抖动；状态只由观测更新，不输入真实历史标签，不存历史图片或人工事件表。共有解码器对每次请求只投影一次当前特征，两组均采用相同计算优化；无状态输出与原解码器已做精确等价测试。

训练损失与梯度：CE仅训练S和新增状态模块；action仅训练A，B冻结；100%生成文字条件，独立0.1空条件dropout。优化器分支、损失归一化及原动作监督不变。隐状态按数据流保存到各rank的断点文件，采样器按已完成步数确定性重放，不受worker预取影响。

每500步对全部20个val episode进行约0.7秒间隔的因果回放，并加入原固定128动作验证帧；每帧A验证仍使用固定的两组噪声/时间。S不接收参考标签。选模分数预先固定为：0.5×(1−normalized EM)+0.5×(1−macro F1)+5×native14 flow MSE，越低越好。保存所有分项、参考转移的漏切换、上一采样点提前切换和切换延迟；不把任意回退直接定义为错误。最终候选可另做清空记忆与正确/空文字条件诊断。

训练通过服务器现有 run_concurrent 资源协调器运行。其他任务与守护进程保持原状，不查询GPU利用率。CPU单元检查、真实八卡保存/恢复、官方共有S初始化一致性、冻结B603个存储张量一致性、CE/action梯度隔离和原生50×14推理是正式队列前置检查。

Agilex已增加schema6两组识别与按WebSocket连接隔离的记忆状态。新运行、连接断开和任务变化重置；GUI元数据探测不影响活动会话；端口清理覆盖新增服务入口。旧GUI需重新打开。完成事件通过inotify与单条SSH流交给控制机，按候选逐个传输，校验所有文件与权重，做实际CPU推理检查后记录就绪；不启动动作或重启驱动。

实时权威记录：训练服务器 logs/pi05_recurrent_20260909/{phase.json,engineering_passed.json,formal_launcher.process.json,stateless_complete.json,recurrent_complete.json,formal_failure.json}；控制机 context/recurrent_experiments_20260909/deployment/{watcher_state.json,stateless_ready.json,recurrent_ready.json,failure.json,complete.json}；Agilex logs/recurrent_deployment_20260909。

代码为新的 recurrent_subtask、recurrent_sequence、recurrent_evaluation 模块及 train_recurrent_subtask/run_recurrent_pair 入口。旧实验源码不改。实际是否启动与完成，以上述记录和进程身份为准，不能把本文当作训练完成证明。
