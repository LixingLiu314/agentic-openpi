# Experiment Log

Use this ledger to record the exact data, code, and runtime context for each conditional Bridge experiment.

| Experiment | Branch / Commit | Config Name | Dataset Path | Chunk Size (N) | Episodes / Frames | Prompt / Condition Format | Machine / Node | W&B Run URL | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Subtask | `feat/cond-subtask` / `TODO` | `pi05_bridge_subtask` | `Datasets/bridge_cond_aligned_lerobot` | `5` | `38454 / 1298818` | `<task>, subtask: <subtask>` | `TODO` | `TODO` | Running | `TODO` |
| Traj | `feat/cond-traj` / `e3f5a2e` | `pi05_bridge_traj` | `Datasets/bridge_cond_aligned_lerobot` | `5` | `38454 / 1298818` | `<task>, traj: <traj_cot>` | `172.31.0.189` | `TODO` | Running | `TODO` |
| Subgoal | `TODO` | `pi05_bridge_subgoal` | `Datasets/bridge_cond_aligned_lerobot` | `5` | `38454 / 1298818` | `<task>` + future GT subgoal image (`image_0_subgoal_gt`) | `TODO` | `TODO` | Planned | `TODO` |
| All COT | `TODO` | `pi05_bridge_all_cot` | `Datasets/bridge_cond_aligned_lerobot` | `5` | `38454 / 1298818` | `<task>, subtask: <subtask>, traj: <traj_cot>` + future GT subgoal image | `TODO` | `TODO` | Planned | `TODO` |

## Verification Records

Record one probe output per experiment before trusting long-running results.

| Date | Config Name | Probe Command | Dataloader Samples | Batches / Epoch | Prompt Check | Image Keys Check | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `TODO` | `pi05_bridge_subtask` | `uv run scripts/probe_dataloader.py --config-name pi05_bridge_subtask` | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |
| `TODO` | `pi05_bridge_traj` | `uv run scripts/probe_dataloader.py --config-name pi05_bridge_traj` | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |
| `TODO` | `pi05_bridge_subgoal` | `uv run scripts/probe_dataloader.py --config-name pi05_bridge_subgoal` | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |
| `TODO` | `pi05_bridge_all_cot` | `uv run scripts/probe_dataloader.py --config-name pi05_bridge_all_cot` | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |

## Run Details

- Dataset generation script commit: `TODO`
- Dataset generation command: `TODO`
- Dataset report path: `Datasets/bridge_cond_aligned_lerobot/meta/prepare_bridge_conditions_report.json`
- Normalization stats source: `TODO`
- Known caveats: `TODO`
