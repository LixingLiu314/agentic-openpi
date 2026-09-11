# Reach 阶段左右臂标签：数据处理完成

2026-09-10。按用户最新范围，只给接近盖子、接近茄子和接近紫薯的标签增加左右臂。此前评估中给抓取、移动、释放等整条操作链加臂的建议不再采用。

服务器仓库：`/media/raid/workspace/surongpeng/ws_lixing/agentic-openpi`。

## 生成结果

- 新数据：`Datasets/eggplant_potato_reach_arm_v1`，repo ID 为 `local/eggplant_potato_reach_arm_v1`。
- 新资产：`assets/pi05_piper_reach_arm_v1/eggplant_potato`，含新版 `split.json`、`norm_stats.json`、`audit.json`、`native_loader_check.json` 和 `READY.json`。
- 198 个 episode、132939 帧；修改396个 reach片段、48148帧，另外84791帧的 subtask 字符串逐字保留。原来的12种字符串变为15种：3种 reach 各拆成左右两种，另9种保持原文。

| 原标签 | 加 left arm 的片段 | 加 right arm 的片段 |
|---|---:|---:|
| `reach the handle of the lid` | 98 | 100 |
| `reach the eggplant` | 50 | 49 |
| `reach the sweet potato` | 50 | 49 |

后缀统一为 ` with the left arm` 或 ` with the right arm`。例如 `reach the handle of the lid with the right arm`。盖子的 grasp、移盖、抓物、移动物体、release 和放盖都不加臂信息。保留原标签中 `sweet potato` 的命名。

## 自动判断与完整性

每个指定 reach 片段单独判断操作臂：左右各六个关节相对片段起点（含前一帧）的最大位移范数大于0.1 rad且大于对侧两倍；state和action必须得出相同结果，再与此前独立保存的角色审计交叉核对。396个片段全部通过，无需人工强制填补或根据物体左右位置硬编码。

全部198个新 parquet 在写入后回读，除 `subtask` 外的所有列、schema metadata均与原表精确相等；只有指定原标签所在的帧发生文字变化。原198个 parquet及原meta文件哈希未变。594段视频逐文件哈希、帧数、帧率和首帧解码通过，与原始审计记录一致。

原始数据目录 `Datasets/eggplant_potato_gripper_binary` 保留。新版本的 `videos` 是指向原视频目录的绝对符号链接，未重复复制或转码；迁移新数据时必须保留该源视频目录或显式复制视频，不能只搬走新的 parquet。新版本仅复制/更新运行所需的 meta 与 data，不复制原始辅助目录。

`meta/reach_arm_annotations.jsonl` 保存396条修改的episode、起止帧、原文字、新文字、actor、判断数值和校验依据；`meta/reach_arm_provenance.json` 保存来源和脚本哈希。新标签最大11 tokens（含EOS），通过当前16-token编码和解码回环。

## 9:1 与统计

按照此前已确定的规则，新划分为178段训练、20段验证，无test。固定seed42，按任务、已审核布局和臂角色分层；每种任务89段训练、10段验证。训练集开盖右臂90/左臂88，验证集右臂10/左臂10。

完整episode之间无重叠且覆盖全部198段；核对source_path、state/action精确哈希和三相机视频组合哈希，未发现需要跨集合绑定的重复组。数据没有采集session ID，不能据此证明不同采集场次隔离或画面完全独立。

归一化仅使用新训练集119816帧；验证集13123帧不参与统计。沿用已有OpenPI RunningStats、50步同episode尾端重复动作窗口、12关节相对当前state的delta与绝对夹爪定义。额外从原始训练数据独立计算state均值/标准差，与生成统计一致。旧划分和旧模型统计未覆盖。

注意：LeRobot的 `meta/info.json` 中 `train: 0:198` 表示全数据可用范围，实际实验划分由上面的 `assets/.../split.json` 显式指定。

## 真实读取检查

现有 `create_torch_dataset` → `SplitLeRobotDataset` → `SubtaskTrainingDataset` → `collate_subtask` 在CPU通过，无模型加载和训练。

- train/val各覆盖8种任务与布局组合，实际读取各56个、共112个训练样本，包含首帧、两种reach阶段的开始/结束、紧邻的原标签阶段，以及episode末帧。
- 全部6种新reach字符串和未修改阶段均覆盖；标签解码回环正确，global task保留，真实标签/actor未进入模型Observation。
- 实际绝对state及50步action窗口逐项与原始数据一致，包括最后一帧的尾端重复；处理后的state32、action50×32有限，批处理target为2×16。

数据已可供下一轮训练配置选择。没有更改现有训练配置默认值、模型或推理行为，没有启动训练或机器人动作。

下一轮必须同时使用新数据root、repo ID、新split和新norm，不能仅替换一个路径。旧 `piper_rgb224_cache_v2` 绑定旧split哈希且未覆盖全部新训练episode，不能直接伪装成适配新划分；如启用缓存，应另建或经内容校验扩展新版本。现有原生视频读取路径已通过本次检查。

## 复现与校验

服务器新增 `scripts/prepare_reach_arm_data.py`、`scripts/verify_reach_arm_data.py`；准备脚本拒绝覆盖已存在的数据和资产目录。准备流程使用临时目录，全量完整性检查后再将版本置为可见；`READY.json` 表示真实加载检查也已通过。

- split文件SHA256：`e86b8d7f91a60c6b120cfbe821b882980b56f8dc99dfe701e80124f7581c2bab`
- norm文件SHA256：`e42de696d2a699111c2bb1022c9de74acd47706a147a5cfa9c8fe9d7ff1f366d`
- 修改清单SHA256：`b95c4a31ac5bb39f56a7702ef9d270bfb8c1f5ce74447c8725cbfeadaf83146b`

控制机证据：`/home/robot/workspace/VLA/context/reach_arm_data_20260910/`。本报告同步服务器 `docs/pi05_reach_arm_data_20260910.md`。
