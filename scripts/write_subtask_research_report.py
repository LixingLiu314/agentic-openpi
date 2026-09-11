# ruff: noqa: RUF001
# Chinese report prose intentionally uses Chinese punctuation.
"""Render factual stage-one results; incomplete validation remains a snapshot."""

import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from openpi.training.research_checkpoint import require_research_checkpoint
from openpi.training.stage1_data import manifest_digest
from openpi.training.stage1_data import sha256_file

LOGS = Path("logs/pi05_subtask_stage1")
ASSETS = Path("assets/pi05_piper_stage1/eggplant_potato")
SEEDS = [42, 43, 44]


class Inputs:
    def __init__(self):
        self.hashes = {}
        self.missing = []

    def read(self, path):
        path = Path(path)
        if not path.exists():
            self.missing.append(str(path))
            return None
        result = json.loads(path.read_text())
        self.hashes[str(path)] = sha256_file(path)
        return result


def locations(split, seed):
    if split == "test":
        return {
            method: LOGS / f"{name}_test_seed{seed}" / "report.json"
            for method, name in [("M1", "m1"), ("G", "g"), ("B1", "b1"), ("O", "o")]
        }
    suffix = "" if seed == 42 else f"_seed{seed}"
    return {
        method: LOGS / f"{name}{suffix}" / "report.json"
        for method, name in [
            ("M1", "m1_dense_val"),
            ("G", "m3_native_dense_val"),
            ("B1", "b1_native_val"),
            ("O", "o_native_val"),
        ]
    }


def table(lines, columns, rows):
    lines.extend(["", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)])
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    lines.append("")


def number(value, digits=6):
    return "待完成" if value is None else f"{value:.{digits}f}"


def percent(value):
    return "待完成" if value is None else f"{100 * value:.2f}%"


def mean_std(values, *, percentage=False):
    values = np.asarray(values, dtype=float) * (100 if percentage else 1)
    if not len(values):
        return "待完成"
    unit = "%" if percentage else ""
    if len(values) == 1:
        return f"{values.mean():.4f}{unit}（仅1种子）"
    return f"{values.mean():.4f} ± {values.std(ddof=1):.4f}{unit}（n={len(values)}）"


def validate_model_report(report, method, seed, split, protocol, frame_count, inputs, verify_weights):
    if report["split"] != split or report["split_sha256"] != protocol["split_sha256"]:
        raise ValueError("Report split identity differs from protocol")
    if report["norm_sha256"] != protocol["norm_sha256"]:
        raise ValueError("Report normalization differs from protocol")
    checkpoint = Path(report["checkpoint"])
    metadata = inputs.read(checkpoint / "metadata.json")
    if metadata is None:
        raise ValueError("Need a research checkpoint")
    require_research_checkpoint(checkpoint, metadata)
    if metadata["stage"] != {"G": "m3", "M1": "m1", "B1": "control_b1", "O": "control_o", "C0": "m0"}[method]:
        raise ValueError("Checkpoint stage differs from declared method")
    if method != "C0":
        if metadata["config"]["seed"] != seed:
            raise ValueError("Checkpoint seed differs from declared seed")
        if metadata["config"]["c0_weights_sha256"] != protocol["c0_weights_sha256"]:
            raise ValueError("Checkpoint C0 differs from protocol")
    if method in {"G", "B1", "O"} and metadata["counters"]["action"] != 4000:
        raise ValueError("Action budget mismatch")
    if method in {"M1", "G"}:
        if report["semantics"]["frames"] != frame_count or report["no_image"]["semantics"]["frames"] != frame_count:
            raise ValueError("Semantic report does not cover the complete split")
        if sum(row["support"] for row in report["semantics"]["per_class"].values()) != frame_count:
            raise ValueError("Semantic class supports do not sum to the split")
    if verify_weights and sha256_file(checkpoint / "model.safetensors") != report["weights_sha256"]:
        raise ValueError("Checkpoint weights changed")


def main(args):
    inputs = Inputs()
    inputs.hashes["src/openpi/training/research_checkpoint.py"] = sha256_file(
        Path("src/openpi/training/research_checkpoint.py")
    )
    protocol = inputs.read(ASSETS / "research_protocol_v1.json")
    split_manifest = inputs.read(ASSETS / "split.json")
    if manifest_digest(split_manifest) != protocol["split_sha256"]:
        raise ValueError("Split manifest differs from research protocol")
    episode_ids = set(split_manifest["splits"][args.split])
    frames = sum(row["length"] for row in split_manifest["episodes"] if row["episode_index"] in episode_ids)
    final = args.split == "test"
    seal = inputs.read(ASSETS / "final_test_protocol.json") if final else None
    if final and (seal is None or seal["status"] != "sealed"):
        raise ValueError("Final report requires a sealed final test protocol")
    reports = {
        seed: {method: inputs.read(path) for method, path in locations(args.split, seed).items()} for seed in SEEDS
    }
    c0_path = LOGS / ("c0_test" if final else "c0_comparable_native_val") / "report.json"
    c0 = inputs.read(c0_path)
    time_baseline = inputs.read(LOGS / f"time_baseline_{args.split}/report.json")
    summary_path = LOGS / f"three_seed_{'test' if final else 'validation'}_comparison.json"
    summary = inputs.read(summary_path)
    for seed, methods in reports.items():
        for method, report in methods.items():
            if report is not None:
                validate_model_report(report, method, seed, args.split, protocol, frames, inputs, final)
                if final and report["test_protocol"]["sha256"] != sha256_file(ASSETS / "final_test_protocol.json"):
                    raise ValueError("Test report was generated from another protocol")
    if c0 is not None:
        validate_model_report(c0, "C0", None, args.split, protocol, frames, inputs, final)
    if time_baseline is not None:
        if time_baseline["split"] != args.split or time_baseline["split_sha256"] != protocol["split_sha256"]:
            raise ValueError("Time baseline split differs")
        if time_baseline["semantics"]["frames"] != frames:
            raise ValueError("Time baseline does not cover the complete split")
    if summary is not None:
        if summary["training_seeds"] != SEEDS or summary["frames"] != 128 or summary["episodes"] != len(episode_ids):
            raise ValueError("Matched action summary is incomplete")
        manifest = inputs.read(Path(summary["manifest"]))
        if manifest is None or sha256_file(Path(summary["manifest"])) != summary["manifest_sha256"]:
            raise ValueError("Action comparison manifest changed")
        if final and summary["execution_frames"] != seal["execution_frames"]:
            raise ValueError("Final execution length differs")
    profile_paths = (
        [
            LOGS / f"deployment_profile_{method}_seed{seal['deployment_candidate_seed']}/report.json"
            for method in ["g", "b0", "b1"]
        ]
        if final
        else sorted(LOGS.glob("deployment_profile_*/report.json"))
    )
    profile_records = [inputs.read(path) for path in profile_paths]
    if final:
        for report in [c0, *[report for methods in reports.values() for report in methods.values()]]:
            if report is not None:
                checkpoint = Path(report["checkpoint"])
                entry = seal["checkpoints"][str(checkpoint.resolve())]
                if (
                    report["weights_sha256"] != entry["weights_sha256"]
                    or sha256_file(checkpoint / "metadata.json") != entry["metadata_sha256"]
                    or report["test_protocol"]["sha256"] != sha256_file(ASSETS / "final_test_protocol.json")
                ):
                    raise ValueError("Selected final checkpoint or protocol changed")
        for profile in profile_records:
            if profile is not None and (profile["samples"] != 128 or profile["world_size"] != 8):
                raise ValueError("Deployment profile is incomplete")
            if profile is not None:
                selected = seal["checkpoints"][str(Path(profile["checkpoint"]).resolve())]
                if profile["weights_sha256"] != selected["weights_sha256"]:
                    raise ValueError("Deployment profile weights differ from the selected test checkpoint")
    complete = not inputs.missing
    if (final or args.require_complete) and not complete:
        raise ValueError("Required results are incomplete: " + ", ".join(inputs.missing))
    phase = "最终测试结果" if final else ("完整三种子验证结果" if complete else "验证阶段快照，三种子尚未完成")
    lines = [
        f"π₀.₅ subtask 一阶段：{phase}。",
        "",
        f"生成时间：{datetime.datetime.now(datetime.UTC).isoformat()}。以下数值均从已完成的报告读取，缺失项明确标记为待完成。",
        "",
        "实现路径为当前三路图像、14维 state 和全局任务 → 独立自回归 subtask Transformer → 内部文本条件 → flow action expert。S 由 CE 更新、A 由 flow loss 更新，B 冻结；两组 optimizer 参数不重叠。工程验收与研究效果分别判断。",
        "",
        f"当前划分为 {args.split}，共 {len(episode_ids)} 条 episode、{frames:,} 帧。三个种子 42/43/44 共享同一 C0，测量的是给定 C0 后的训练随机性，不包括 M0 的种子方差。G/B1/O 均额外执行 4,000 次 A 更新，但 G 还有独立 S 训练，因此不声称总计算量相同。",
    ]
    action_rows = []
    for seed, methods in reports.items():
        for method in ["G", "B1", "O"]:
            report = methods[method]
            score = (
                (report["action_conditions"]["generated"]["overall"] if method == "G" else report["overall"])
                if report
                else {}
            )
            action_rows.append(
                [
                    seed,
                    method,
                    *[
                        number(score.get(key))
                        for key in [
                            "flow_native14_normalized",
                            "native_joint_rmse_h50",
                            "native_gripper_rmse_h50",
                            "native_joint_rmse_first10",
                            "native_gripper_rmse_first10",
                        ]
                    ],
                ]
            )
    lines += [
        "",
        "各种子的动作误差如下；前10有效帧是诊断窗口，最终统一执行长度由验证延迟确定。原单位数值未裁剪，夹爪与关节分开报告。",
    ]
    table(lines, ["种子", "方法", "flow MSE", "关节 RMSE H50", "夹爪 RMSE H50", "关节前10", "夹爪前10"], action_rows)
    if c0:
        lines.append(
            f"共享 C0 的 H50 关节/夹爪 RMSE 为 {c0['overall']['native_joint_rmse_h50']:.6f}/{c0['overall']['native_gripper_rmse_h50']:.6f}。G 相对 C0 的下降包含额外动作训练的收益，subtask 增量主要由 G/B1 对照评估。"
        )
    if summary:
        lines += [
            "",
            f"统一执行长度 E={summary['execution_frames']} 帧。下面为三种子均值±样本标准差；配对区间使用2,000次任务分层 episode bootstrap，噪声 draw 先按帧平均。区间条件于已拟合的 checkpoint，不把相邻帧或噪声 draw 视为独立样本。",
        ]
        keys = [
            "flow_native14_normalized",
            "joint_rmse_h50",
            "gripper_rmse_h50",
            "joint_rmse_execution",
            "gripper_rmse_execution",
            "gripper_accuracy_execution",
            "gripper_excursion_gt_1e_3_execution",
        ]
        table(
            lines,
            ["指标", "G 均值±SD", "B1 均值±SD", "G−B1", "配对95%区间"],
            [
                [
                    key,
                    *[
                        f"{summary['methods'][m][key]['mean_across_seeds']:.6f} ± {summary['methods'][m][key]['seed_standard_deviation']:.6f}"
                        for m in ["G", "B1"]
                    ],
                    number(summary["paired_comparisons"]["G_minus_B1"][key]["difference"]),
                    " / ".join(
                        number(x)
                        for x in summary["paired_comparisons"]["G_minus_B1"][key]["paired_episode_bootstrap_ci95"]
                    ),
                ]
                for key in keys
            ],
        )
        lines.append(
            "差值以 G−B1 定义，除夹爪准确率外均越低越好。完整 B0/O/置空/打乱/GT 干预对比及各自区间保存在随附 JSON；GT 干预是同一 G 权重，O 是单独训练的真实条件对照，二者不可混用。"
        )
    else:
        lines += ["", "完整三种子配对统计尚未生成，此快照不据单种子点估计宣称稳定动作收益。"]
    semantic_rows, boundary_rows = [], []
    for seed, methods in reports.items():
        for method in ["M1", "G"]:
            report = methods[method]
            if report is None:
                semantic_rows.append([seed, method, *(["待完成"] * 6)])
                continue
            s, t = report["semantics"], report["transition_metrics"]
            semantic_rows.append(
                [
                    seed,
                    method,
                    *[
                        percent(s[k])
                        for k in [
                            "exact_match",
                            "macro_f1",
                            "target_object_accuracy",
                            "unknown_rate",
                            "invalid_generation_rate",
                        ]
                    ],
                    percent(report["no_image"]["semantics"]["exact_match"]),
                ]
            )
            boundary_rows.append(
                [
                    seed,
                    method,
                    f"{t['matched_switches']}/{t['gt_switches']}",
                    t["missed_switches"],
                    percent(t["within_tolerance_fraction_all_gt"]),
                    number(t["signed_error_mean_matched"], 2),
                    "/".join(number(x, 2) for x in t["absolute_error_p50_p95_matched"]),
                    number(t["unmatched_changes_per_episode"], 2),
                ]
            )
    table(
        lines,
        ["种子", "方法", "exact match", "macro-F1", "目标物准确率", "未知率", "无效率", "置黑图像 EM"],
        semantic_rows,
    )
    if time_baseline:
        s = time_baseline["semantics"]
        lines.append(
            f"任务+已过时间诊断的 EM/macro-F1 为 {percent(s['exact_match'])}/{percent(s['macro_f1'])}，仅使用训练期阶段起点中位数，不使用真实未来 episode 长度。置黑图像保留任务/state，是输入干预而非重训无视觉模型；目标物准确率只统计标签明确包含目标物的帧。"
        )
    table(
        lines,
        ["种子", "方法", "匹配/GT切换", "漏检", "±10帧内比例", "有符号均值/帧", "绝对误差p50/p95", "额外变化/episode"],
        boundary_rows,
    )
    lines.append(
        "切换使用未经平滑的逐帧预测，按相邻GT边界中点划分区间后匹配直接 old→new 变化。匹配误差不涵盖漏检，所以漏检与额外变化同时报告；EOS 只代表文本结束，不代表任务完成。"
    )
    generated_reports = [methods["G"] for methods in reports.values() if methods["G"]]
    if generated_reports:
        labels = sorted(generated_reports[0]["semantics"]["per_class"])
        table(
            lines,
            ["阶段标签", "G recall 均值±SD", "G F1 均值±SD"],
            [
                [
                    label,
                    *[
                        mean_std([r["semantics"]["per_class"][label][key] for r in generated_reports], percentage=True)
                        for key in ["recall", "f1"]
                    ],
                ]
                for label in labels
            ],
        )
        intervention_rows = []
        for seed, methods in reports.items():
            if methods["G"]:
                for condition, score in methods["G"]["action_conditions"].items():
                    intervention_rows.append(
                        [
                            seed,
                            condition,
                            number(score["overall"]["native_joint_rmse_h50"]),
                            number(score["overall"]["native_gripper_rmse_h50"]),
                        ]
                    )
        table(lines, ["种子", "同一G权重的条件", "关节 RMSE H50", "夹爪 RMSE H50"], intervention_rows)
    latency_rows = []
    if not final:
        if c0 and c0["complete_policy_ms_p50_p95"]:
            latency_rows.append(["共享C0", *[number(x, 2) for x in c0["complete_policy_ms_p50_p95"]], "无S"])
        for seed, methods in reports.items():
            for method in ["G", "B1"]:
                report = methods[method]
                if report is not None:
                    timing = (
                        report["timing"]["p50_p95_ms"]["infer_ms"]
                        if method == "G"
                        else report["complete_policy_ms_p50_p95"]
                    )
                    decoder = (
                        "/".join(number(x, 2) for x in report["timing"]["p50_p95_ms"]["subtask_ms"])
                        if method == "G"
                        else "无S"
                    )
                    latency_rows.append([f"{method} seed{seed}", *[number(x, 2) for x in timing], decoder])
        table(lines, ["验证完整策略", "p50/ms", "p95/ms", "S p50/p95 ms"], latency_rows)
    profiles = [
        [
            profile["checkpoint"],
            *[number(x, 2) for x in profile["policy_ms_p50_p95"]],
            number(profile["max_peak_allocated_gib"], 3),
            number(profile["max_peak_reserved_gib"], 3),
        ]
        for profile in profile_records
        if profile is not None
    ]
    if profiles:
        table(lines, ["独立部署策略", "p50/ms", "p95/ms", "峰值 allocated/GiB", "峰值 reserved/GiB"], profiles)
    else:
        lines += ["", "单策略部署显存和客户端分块运行检查尚待执行，当前没有该项完成结论。"]
    lines += [
        "",
        "完整策略计时包括CPU转换/tokenizer、视觉编码、prefix、S生成（如有）、全部flow步及原单位输出；不包括相机/网络传输或机械臂控制。PyTorch显存不等同于驱动总占用，profile记录额外GPU进程。",
        "",
        "数据来自固定两任务、固定八阶段流程，没有采集 session ID，精确去重不能保证近重复场景隔离。当前没有物理闭环任务成功率、控制器单位确认或随机中途启动/暂停恢复/位置扰动的真机实验结果。离线误差、语义准确率与条件敏感性都不能替代这些结果。",
    ]
    if inputs.missing:
        lines += ["", "未完成产物：", "", *[f"- `{path}`" for path in inputs.missing]]
    lines += [
        "",
        "所有读取的报告及固定协议 SHA256 记录在 `report_inputs.json`；原始逐帧预测、动作数组及完整统计保留在服务器 logs。报告生成不修改模型、训练预算、选模或测试协议。",
        "",
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.md").write_text("\n".join(lines))
    (args.output / "report_inputs.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "complete": complete,
                "missing": inputs.missing,
                "inputs_sha256": inputs.hashes,
                "source_sha256": sha256_file(Path(__file__)),
                "weight_files_rehashed": final,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {"output": str(args.output), "complete": complete, "split": args.split, "missing": len(inputs.missing)}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    main(parser.parse_args())
