"""Create/update a curated saved W&B view. Never overwrite the personal workspace."""
import argparse
import json
from pathlib import Path

from openpi.training.wandb_standard import AXIS


def save_workspace(root, *, display_set="official-pi05-reach-arm-s42-v1", name="Pi05 Limited recurrent reach-arm s42 v1", receipt=None):
    import wandb
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    entity, project = "xiahy23-tsinghua-university", "agentic-openpi-pi05-subtask"
    receipt = receipt or root / "logs/wandb_display_v1/workspace.json"
    runs = list(wandb.Api().runs(f"{entity}/{project}", filters={"config.display_set": display_set, "config.display_schema": 1}))
    colors = {"frozen": "#64748B", "limited": "#2563EB", "full": "#EA580C"}

    def plot(title, key, unit, bounds=(None, None)):
        return wr.LinePlot(title=title, x=AXIS, y=[key], title_x="Optimizer updates since initialization",
                           title_y=unit, range_y=bounds, smoothing_type="none", smoothing_factor=0,
                           ignore_outliers=False, aggregate=False, groupby=None, groupby_rangefunc="none",
                           legend_position="south", max_runs_to_show=8)

    def section(title, plots, opened=True, columns=3):
        for index, panel in enumerate(plots):
            panel.layout = wr.Layout(x=(index % columns) * (24 // columns), y=(index // columns) * 6, w=24 // columns, h=6)
        return ws.Section(name=title, panels=plots, is_open=opened,
                          layout_settings=ws.SectionLayoutSettings(columns=columns, rows=2))

    sections = [
        section("01 验证效果 · 优先看这三张", [
            plot("动作 Flow MSE ↓ | validation, normalized native14", "val/action_flow_mse_native14", "Normalized flow MSE (14 dims)"),
            plot("Subtask 完全匹配率 ↑ | validation", "val/subtask_exact_match", "Fraction (0–1)", (0, 1)),
            plot("Subtask Macro-F1 ↑ | validation", "val/subtask_macro_f1", "Macro-F1 (0–1)", (0, 1)),
        ]),
        section("02 学习过程 · train 与 val 分开", [
            plot("Action loss ↓ | train, all32", "train/action_flow_mse_all32", "Flow MSE (32 dims)"),
            plot("Subtask CE ↓ | train", "train/subtask_ce", "Token cross entropy"),
            plot("Subtask CE ↓ | validation", "val/subtask_ce", "Token cross entropy"),
            plot("学习率 | S / A / B 共用 schedule", "optim/lr", "Learning rate"),
        ]),
        section("03 诊断 · 需要时展开", [
            plot("S 梯度范数 | clipping 前", "optim/grad_norm_s", "L2 norm"),
            plot("A+B 梯度范数 | clipping 前", "optim/grad_norm_a_b", "L2 norm"),
            plot("生成异常率 ↓ | validation", "val/invalid_generation_rate", "Fraction (0–1)", (0, 1)),
            plot("词表外 subtask 比例 ↓ | validation", "val/unknown_rate", "Fraction (0–1)", (0, 1)),
            plot("条件置空比例 | training dropout", "train/empty_condition_rate", "Fraction (0–1)", (0, 1)),
            plot("生成异常率 ↓ | train", "train/invalid_generation_rate", "Fraction (0–1)", (0, 1)),
        ], False),
        section("04 速度 · 不含验证及存盘", [
            plot("单次 optimizer update 耗时", "perf/update_seconds", "Seconds / update"),
            plot("训练吞吐量 | effective global batch / update time", "perf/examples_per_second", "Examples / second"),
        ], False, 2),
        section("05 示例与类别表 · 需要时展开", [
            wr.MediaBrowser(title="首次更新 · 输入图像与文本", media_keys=["media/first_update_inputs"], num_columns=2),
            wr.MediaBrowser(title="首次更新 · 动作预测示例", media_keys=["media/first_update_actions"], num_columns=2),
            wr.MediaBrowser(title="逐 subtask F1 / recall / support", media_keys=["media/subtask_per_class"], num_columns=1),
        ], False, 2),
    ]
    sections.insert(3, section("03b 阶段与选臂 · validation", [
        plot("起始阶段与操作臂同时正确 ↑", "sequence/initial_phase_actor_accuracy", "Fraction (0–1)", (0, 1)),
        plot("Reach 操作臂正确率 ↑", "sequence/reach_actor_accuracy", "Fraction (0–1)", (0, 1)),
        plot("忽略臂后缀的阶段正确率 ↑", "sequence/phase_only_exact_match", "Fraction (0–1)", (0, 1)),
        plot("A 梯度范数 | clipping 前", "optim/grad_norm_a", "L2 norm"),
        plot("Limited B 梯度范数 | clipping 前", "optim/grad_norm_b", "L2 norm"),
    ], False))
    settings = ws.WorkspaceSettings(x_axis=AXIS, smoothing_type="none", smoothing_weight=0,
                                   ignore_outliers=False, sort_panels_alphabetically=False,
                                   group_by_prefix="first", max_runs=8)
    runset = ws.RunsetSettings(
        filters=f"Config('display_schema') = 1 and Config('display_set') = '{display_set}'",
        groupby=[], run_settings={"limited_recurrent_seed42_v1": ws.RunSettings(color="#2563EB")},
        pinned_columns=["run:displayName", "config:mode", "config:initialization", "config:seed", "summary:last_optimizer_step",
                        "summary:source_status", "config:global_batch_size"])
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old["display_set"] != display_set:
            raise ValueError("Use a different --receipt for a new comparison set")
        workspace = ws.Workspace.from_url(old["url"])
        workspace.sections, workspace.settings, workspace.runset_settings = sections, settings, runset
        workspace.name = name
    else:
        workspace = ws.Workspace(entity=entity, project=project, name=name, sections=sections,
                                 settings=settings, runset_settings=runset, auto_generate_panels=False)
    workspace.save()
    # Keep the returned URL even if read-back validation fails; retries update it.
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({"url": workspace.url, "display_set": display_set,
                                   "verified_remote_roundtrip": False}, indent=2) + "\n")
    loaded = ws.Workspace.from_url(workspace.url)
    assert loaded.auto_generate_panels is False
    assert loaded.settings.x_axis == AXIS
    assert len(loaded.sections) == 6
    for section_item in loaded.sections:
        for panel in section_item.panels:
            if isinstance(panel, wr.LinePlot):
                assert getattr(panel.x, "name", panel.x) == AXIS
                assert panel.smoothing_type == "none" and panel.aggregate is False
    receipt.parent.mkdir(parents=True, exist_ok=True)
    data = {"url": workspace.url, "display_set": display_set, "name": name,
            "sections": [{"name": s.name, "open": s.is_open, "panels": len(s.panels)} for s in loaded.sections],
            "run_ids": [r.id for r in runs], "auto_generate_panels": loaded.auto_generate_panels,
            "wandb": wandb.__version__, "verified_remote_roundtrip": True}
    receipt.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(data, ensure_ascii=False), flush=True)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--display-set", default="official-pi05-reach-arm-s42-v1")
    parser.add_argument("--name", default="Pi05 Limited recurrent reach-arm s42 v1")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    save_workspace(args.root.resolve(), display_set=args.display_set, name=args.name, receipt=args.receipt)
