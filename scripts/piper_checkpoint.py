"""Lightweight deployment manifests for the Python 3.8 GUI and model server.

Catalog inspection does not hash multi-GB weights on the Qt thread. The actual
policy loader verifies both weight files before constructing a usable policy.
"""
import json
from pathlib import Path

R1_VARIANT = "r1_boundary_ce_v1"
BACKBONE_VARIANT = "action_backbone_v1"
OFFICIAL_VARIANT = "official_pi05_backbone_v1"
DECISION_VARIANT = "official_pi05_recurrent_decision_v1"
DECISION_EXPERIMENTS = ("decision_prefix", "decision_grounded")
SEMANTIC_VARIANT = "official_pi05_recurrent_semantic_v1"
SEMANTIC_EXPERIMENTS = ("semantic_s", "semantic_s_actionrank")
REACH_ARM_VARIANT = "official_pi05_recurrent_reach_arm_v1"
RECURRENT_VARIANT = "official_pi05_recurrent_s_v1"
OFFICIAL_SHA256 = "be5b2233cf302a8fd7097e239e0e2c4680fc6f3696a55084983f2225e2d7044e"
ROOT = Path(__file__).resolve().parents[1]


def _metadata(path):
    return json.loads((Path(path)/"metadata.json").read_text(encoding="utf-8"))


def _m3_info(path, metadata):
    if metadata.get("schema_version") != 3 or metadata.get("stage") != "m3":
        raise ValueError("请选择完整 M3 checkpoint 或 R1 boundary decoder checkpoint")
    if metadata.get("config", {}).get("engineering_smoke", True):
        raise ValueError("工程测试 checkpoint 不能用于正常部署")
    for name in ("model.safetensors", "assets/eggplant_potato/norm_stats.json", "assets/eggplant_potato/split.json"):
        if not (path/name).is_file():
            raise ValueError("checkpoint 缺少 " + name)
    return {"path":str(path), "stage":"m3", "kind":"m3", "step":metadata.get("completed_steps"),
            "experimental":False}


def checkpoint_info(path, root=None, parent_checkpoint=None):
    path = Path(path).expanduser().resolve()
    root = Path(root or ROOT).resolve()
    metadata = _metadata(path)
    if metadata.get("variant") in (BACKBONE_VARIANT, OFFICIAL_VARIANT, RECURRENT_VARIANT, REACH_ARM_VARIANT, SEMANTIC_VARIANT, DECISION_VARIANT):
        decision = metadata["variant"] == DECISION_VARIANT
        semantic = metadata["variant"] == SEMANTIC_VARIANT
        reach_arm = metadata["variant"] in (REACH_ARM_VARIANT, SEMANTIC_VARIANT, DECISION_VARIANT)
        recurrent = metadata["variant"] in (RECURRENT_VARIANT, REACH_ARM_VARIANT, SEMANTIC_VARIANT, DECISION_VARIANT)
        official = metadata["variant"] in (OFFICIAL_VARIANT, RECURRENT_VARIANT, REACH_ARM_VARIANT, SEMANTIC_VARIANT, DECISION_VARIANT)
        if parent_checkpoint is not None:
            raise ValueError("完整 backbone checkpoint 不需要 --parent-checkpoint")
        config = metadata.get("config", {})
        expected = (9, "recurrent_subtask") if decision else (8, "recurrent_subtask") if semantic else (7, "recurrent_subtask") if reach_arm else (6, "recurrent_subtask") if recurrent else (5, "official_backbone_grad") if official else (4, "backbone_grad")
        if (metadata.get("schema_version"), metadata.get("stage")) != expected:
            raise ValueError("Backbone checkpoint 格式不匹配")
        modes = ("limited",) if reach_arm else ("frozen",) if recurrent else ("frozen", "limited", "full") if official else ("limited", "full")
        if reach_arm and (config.get("arm"), config.get("label_version")) != ("recurrent", "reach_arm_v1"):
            raise ValueError("reach-arm 候选的记忆或标签版本不匹配")
        if semantic and (config.get("experiment") not in SEMANTIC_EXPERIMENTS or config.get("engineering_condition_fixture") is not False):
            raise ValueError("语义实验身份或正式生成条件不匹配")
        if decision:
            if config.get("experiment") not in DECISION_EXPERIMENTS or config.get("engineering_condition_fixture") is not False:
                raise ValueError("选臂/选物实验身份不匹配")
            for name in ("READY.json","decisions.json","reviewed_points.json"):
                if not (path/"assets/decision"/name).is_file():raise ValueError("缺少实验数据身份："+name)
        if recurrent and config.get("arm") not in ("stateless", "recurrent"):
            raise ValueError("未知的 S 实验类型")
        if config.get("engineering_smoke", True) or config.get("mode") not in modes:
            raise ValueError("Backbone checkpoint 不是支持的正式训练模式")
        if official and (config.get("initialization") != "official_pi05_base"
                         or config.get("inherited_training_updates") != 0
                         or config.get("official_weights_sha256") != OFFICIAL_SHA256
                         or config.get("parent_weights_sha256") != OFFICIAL_SHA256):
            raise ValueError("Official checkpoint 的官方初始化身份不匹配")
        if not isinstance(metadata.get("completed_steps"), int) or metadata["completed_steps"] <= 0:
            raise ValueError("Backbone checkpoint 没有完成训练更新")
        for key, value in (("weights_sha256", metadata.get("weights_sha256")),
                           ("parent_weights_sha256", config.get("parent_weights_sha256"))):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Backbone 缺少有效权重身份：" + key)
        for name in ("model.safetensors", "assets/eggplant_potato/norm_stats.json", "assets/eggplant_potato/split.json"):
            if not (path/name).is_file():
                raise ValueError("checkpoint 缺少 " + name)
        result = {"path":str(path), "stage":expected[1], "kind":"recurrent_subtask" if recurrent else "official_backbone_grad" if official else "backbone_grad", "variant":metadata["variant"],
                "mode":config["mode"], "step":metadata["completed_steps"], "experimental":True,
                "weights_sha256":metadata["weights_sha256"], "parent_weights_sha256":config["parent_weights_sha256"]}
        if official:
            result.update(initialization="official_pi05_base", inherited_training_updates=0)
        if recurrent:
            result["arm"] = config["arm"]
        if reach_arm:
            result["label_version"] = config["label_version"]
        if semantic or decision:
            result["experiment"] = config["experiment"]
        return result
    if metadata.get("schema_version") != 1 or metadata.get("variant") != R1_VARIANT:
        if parent_checkpoint is not None:
            raise ValueError("--parent-checkpoint 仅用于 R1 增量模型")
        return _m3_info(path, metadata)
    config = metadata.get("config", {})
    if config.get("engineering_smoke", True) or config.get("seed") != 42:
        raise ValueError("R1 需要 seed42 正式训练的 checkpoint")
    if metadata.get("frozen_base_equal") is not True or not metadata.get("base_tensor_hash") or (
            metadata["base_tensor_hash"] != metadata.get("initial_base_tensor_hash")):
        raise ValueError("R1 冻结基座身份检查失败")
    if not metadata.get("decoder_sha256") or not (path/"decoder.safetensors").is_file():
        raise ValueError("R1 缺少 decoder.safetensors 或权重校验值")

    if parent_checkpoint is not None:
        parent = Path(parent_checkpoint).expanduser()
        parent = parent if parent.is_absolute() else root/parent
    else:
        recorded = Path(config.get("initialize_from", ""))
        if not config.get("initialize_from"):
            raise ValueError("R1 缺少 initialize_from 基座路径")
        candidates = []
        # Training-server absolute paths are relocated under this robot checkout.
        if "checkpoints" in recorded.parts:
            candidates.append(root.joinpath(*recorded.parts[recorded.parts.index("checkpoints"):]))
        candidates.append(recorded if recorded.is_absolute() else root/recorded)
        parent = next((item for item in candidates if (item/"metadata.json").is_file()), None)
        if parent is None:
            raise ValueError("找不到对应 M3 基座；请部署 initialize_from 对应的完整 checkpoint，或使用 --parent-checkpoint")
    parent = parent.resolve()
    parent_meta = _metadata(parent)
    _m3_info(parent, parent_meta)
    for key in ("split_sha256", "norm_sha256"):
        if not config.get(key) or config[key] != parent_meta.get("config", {}).get(key):
            raise ValueError("R1 与 M3 基座的数据身份不一致："+key)
    if not config.get("parent_weights_sha256"):
        raise ValueError("R1 缺少基座权重校验值")
    candidate_path = path.parent/"candidate.json"
    candidate = json.loads(candidate_path.read_text()) if candidate_path.is_file() else {}
    matches = candidate.get("checkpoint") == path.name and candidate.get("decoder_sha256") == metadata["decoder_sha256"]
    qualified = matches and all(candidate.get(key) is True for key in (
        "proxy_gates_passed", "action_gate_passed", "label_proxy_p95_reduction_30percent", "policy_load_gate_passed"))
    return {"path":str(path), "stage":"m3", "kind":"r1", "variant":R1_VARIANT,
            "step":metadata.get("completed_steps"), "parent_checkpoint":str(parent),
            "parent_weights_sha256":config["parent_weights_sha256"], "decoder_sha256":metadata["decoder_sha256"],
            "experimental":not qualified, "candidate_qualified":bool(qualified),
            "candidate_status":candidate.get("status", "not_evaluated") if matches else "not_evaluated"}


def require_native_policy_contract(metadata):
    """Shared GUI/client check; keep stage identity instead of relabeling as M3."""
    stage, variant = metadata.get("stage"), metadata.get("variant")
    recognized = stage == "m3" and variant in (None, R1_VARIANT)
    recognized |= stage == "backbone_grad" and variant == BACKBONE_VARIANT and metadata.get("mode") in ("limited", "full")
    recognized |= (stage == "official_backbone_grad" and variant == OFFICIAL_VARIANT
                   and metadata.get("mode") in ("frozen", "limited", "full")
                   and metadata.get("initialization") == "official_pi05_base"
                   and metadata.get("inherited_training_updates") == 0
                   and metadata.get("parent_weights_sha256") == OFFICIAL_SHA256)
    recognized |= (stage == "recurrent_subtask" and variant == RECURRENT_VARIANT
                   and metadata.get("mode") == "frozen" and metadata.get("arm") in ("stateless", "recurrent")
                   and metadata.get("initialization") == "official_pi05_base"
                   and metadata.get("inherited_training_updates") == 0
                   and metadata.get("parent_weights_sha256") == OFFICIAL_SHA256)
    recognized |= (stage == "recurrent_subtask" and variant == REACH_ARM_VARIANT
                   and metadata.get("mode") == "limited" and metadata.get("arm") == "recurrent"
                   and metadata.get("label_version") == "reach_arm_v1"
                   and metadata.get("initialization") == "official_pi05_base"
                   and metadata.get("inherited_training_updates") == 0
                   and metadata.get("parent_weights_sha256") == OFFICIAL_SHA256)
    recognized |= (stage == "recurrent_subtask" and variant == SEMANTIC_VARIANT
                   and metadata.get("mode") == "limited" and metadata.get("arm") == "recurrent"
                   and metadata.get("label_version") == "reach_arm_v1"
                   and metadata.get("experiment") in SEMANTIC_EXPERIMENTS
                   and metadata.get("initialization") == "official_pi05_base"
                   and metadata.get("inherited_training_updates") == 0
                   and metadata.get("parent_weights_sha256") == OFFICIAL_SHA256)
    recognized |= (stage == "recurrent_subtask" and variant == DECISION_VARIANT
                   and metadata.get("mode") == "limited" and metadata.get("arm") == "recurrent"
                   and metadata.get("label_version") == "reach_arm_v1"
                   and metadata.get("experiment") in DECISION_EXPERIMENTS
                   and metadata.get("initialization") == "official_pi05_base"
                   and metadata.get("inherited_training_updates") == 0
                   and metadata.get("parent_weights_sha256") == OFFICIAL_SHA256)
    if not recognized or metadata.get("state_dim") != 14 or metadata.get("action_horizon") != 50:
        raise ValueError("服务不是支持的 M3 / R1 / Backbone / Official native14 / horizon50 Piper 模型")
