# Aloha Trajectory Projection And CoT Tools

This folder contains the scripts for projecting robot FK trajectories into the
front-camera image and converting those trajectories into text CoT prompts.

These scripts expect a LeRobot-format absolute Aloha dataset. For raw HDF5
input, run `tools/conversion/convert_hdf5_to_lerobot_absolute.py` first.

## Contents

- `mobile_aloha_link6_fk.py`
  - Lightweight FK for the AgileX mobile Aloha arm chain.
  - Provides `fk_link6`, `fk_gripper_center`, and `fk_full`.
- `aloha_fk_trajectory_utils.py`
  - Shared JSON, calibration, FK, distortion, and projection helpers.
- `generate_aloha_fk_trajectory.py`
  - Projects arm FK points into front-camera pixels and writes trajectory JSON.
- `generate_aloha_bimanual_cot_prompts.py`
  - Builds `cot_text_prompts.json` in `Left: ... Right: ...` format.
- `visualize_aloha_fk_trajectory.py`
  - Renders one annotated trajectory video.
- `visualize_all_aloha_fk_trajectories.py`
  - Batch-renders all annotated videos from a trajectory JSON.
- `project_fk_to_pixels.py`
  - Small single-episode debugging projector.
- `run_aloha_bimanual_pipeline.sh`
  - Convenience script for FK trajectory, CoT prompt, and visualization stages
    after HDF5 screening/conversion are already complete.
- `data/`
  - Small debug JSON outputs from prior FK projection checks.

## Dependencies

The projection and CoT scripts need:

```bash
pip install numpy pyarrow tqdm
```

Video rendering needs:

```bash
pip install av pillow tqdm
```

The scripts intentionally do not require OpenCV for visualization.

## Calibration Inputs

By default, `generate_aloha_fk_trajectory.py` reads calibration JSON files from:

```text
tools/calibration/data/intrinsics_front_charuco.json
tools/calibration/data/front_in_left_base_from_left_charuco_final.json
tools/calibration/data/front_in_right_base_from_left_charuco_final.json
```

The FK module default is:

```text
tools/trajectory/mobile_aloha_link6_fk.py
```

Coordinates in `all_frame_eef_positions` are stored in a 256x256 resized image
space by default. Raw 640x480 front-camera pixels are also stored for
visualization under `all_frame_eef_positions_raw*`.

## One-Command Trajectory Pipeline

Run:

```bash
bash tools/trajectory/run_aloha_bimanual_pipeline.sh playground/Datasets/<dataset_name>
```

Environment overrides:

```bash
DATASET_PATH=playground/Datasets/<dataset_name> \
COT_WORKERS=16 \
VIDEO_WORKERS=4 \
bash tools/trajectory/run_aloha_bimanual_pipeline.sh
```

This script processes every episode in the converted LeRobot dataset. Bad raw
HDF5 episodes should already have been excluded by
`tools/conversion/convert_hdf5_to_lerobot_absolute.py --screen-report`.

## Step 1. Generate Bimanual Trajectory JSON

```bash
python tools/trajectory/generate_aloha_fk_trajectory.py \
  --dataset-path playground/Datasets/<dataset_name> \
  --include-arms both \
  --output playground/Datasets/<dataset_name>/trajectory_data/fk_bimanual.json
```

Important options:

- `--primary-arm right`
  - The `all_frame_eef_positions` compatibility field follows the right arm.
- `--include-arms both`
  - Writes `all_frame_eef_positions_by_arm.left/right`.
- `--drop-invisible-for-ref`
  - Optional; omit out-of-image primary-arm points from the compatibility field.
- `--fk-function fk_gripper_center`
  - Default; projects an approximate gripper-center point.

## Step 2. Generate Bimanual CoT Prompts

```bash
python tools/trajectory/generate_aloha_bimanual_cot_prompts.py \
  --dataset_path playground/Datasets/<dataset_name> \
  --traj_json playground/Datasets/<dataset_name>/trajectory_data/fk_bimanual.json \
  --output_path playground/Datasets/<dataset_name>/trajectory_data/cot_text_prompts.json \
  --image_size 256 \
  --left_gripper_dim 6 \
  --right_gripper_dim 13 \
  --num_workers 16
```

Output prompt format:

```text
Left: Go along (...). Right: Go along (...), close gripper, (...).
```

The gripper threshold defaults to an adaptive midpoint per episode. Use explicit
thresholds if the gripper scale is known:

```bash
--left_gripper_threshold 0.04 --right_gripper_threshold 0.04
```

## Step 3. Visualize One Episode

```bash
python tools/trajectory/visualize_aloha_fk_trajectory.py \
  --dataset-path playground/Datasets/<dataset_name> \
  --traj-json playground/Datasets/<dataset_name>/trajectory_data/fk_bimanual.json \
  --episode 0 \
  --arm both
```

Default output:

```text
playground/Datasets/<dataset_name>/trajectory_data/visualizations/episode_000000_both_trajectory.mp4
```

## Step 4. Visualize All Episodes

```bash
python tools/trajectory/visualize_all_aloha_fk_trajectories.py \
  --dataset-path playground/Datasets/<dataset_name> \
  --traj-json playground/Datasets/<dataset_name>/trajectory_data/fk_bimanual.json \
  --arm both \
  --num-workers 4
```

Default output directory:

```text
playground/Datasets/<dataset_name>/trajectory_data/visualizations_all
```

The batch renderer writes `render_report.json` with success/failure details.

## Visualize Prompt Trajectory Points

Use this when debugging traj/triple-cot annotation quality. Unlike the FK
trajectory videos above, this renderer parses the actual sparse points inside
`cot_text_prompts.json` or a parquet `traj_cot` column, then overlays only the
prompt points and gripper events that are fed to the model.

For `aloha_letter`, render the frames where trajectory annotation is expected
in eval (`N=60` by default):

```bash
python tools/trajectory/visualize_traj_prompt_points.py \
  --dataset-path playground/Datasets/aloha_letter \
  --episode 0 \
  --update-every 60 \
  --frame-policy direct-window
```

Default output:

```text
playground/Datasets/aloha_letter/trajectory_data/prompt_point_visualizations/
  episode_000000_direct-window/
    episode_000000_frame_000000_prompt_000000.png
    episode_000000_frame_000060_prompt_000060.png
    ...
    episode_000000_prompt_points_sheet.jpg
    prompt_point_report.json
```

To visualize the prompt that the current GUI semi-blocking trajectory pipeline
actually applies to each current frame, including the one-window lag, use:

```bash
python tools/trajectory/visualize_traj_prompt_points.py \
  --dataset-path playground/Datasets/aloha_letter \
  --episode 0 \
  --frames 0 60 120 180 \
  --frame-policy eval-blocking
```

In `eval-blocking`, frame `60` still uses prompt frame `0`, frame `120` uses
prompt frame `60`, and so on. This is useful for checking whether stale
trajectory prompts are hurting triple-cot behavior.

If a dataset already has the training-time `traj_cot` column, force exact
parquet prompt visualization with:

```bash
python tools/trajectory/visualize_traj_prompt_points.py \
  --dataset-path playground/Datasets/<dataset_name> \
  --episode 0 \
  --prompt-source parquet \
  --traj-column traj_cot
```

Add `--render-video` to write an MP4 overlay for the full episode, or use
`--show-prompt-text` to include the raw prompt text at the bottom of each
rendered image.

To render all prompt-point overlay videos, mirroring `visualizations_all`:

```bash
python tools/trajectory/visualize_all_traj_prompt_points.py \
  --dataset-path playground/Datasets/aloha_letter \
  --update-every 60 \
  --frame-policy direct-window \
  --num-workers 4
```

Default output:

```text
playground/Datasets/aloha_letter/trajectory_data/prompt_point_visualizations_all/direct-window/
  episode_000000_direct-window_prompt_points.mp4
  episode_000001_direct-window_prompt_points.mp4
  ...
  render_report.json
```

Use `--frame-policy eval-blocking` if you want every rendered frame to show the
same one-window-lag prompt used by the current GUI trajectory pipeline.

## Debug Single-Episode Projection

Use this when checking a calibration quickly:

```bash
python tools/trajectory/project_fk_to_pixels.py \
  --dataset-root playground/Datasets/<dataset_name> \
  --episode 0 \
  --arm right \
  --intrinsics tools/calibration/data/intrinsics_front_charuco.json \
  --extrinsics tools/calibration/data/front_in_right_base_from_left_charuco_final.json \
  --fk-module tools/trajectory/mobile_aloha_link6_fk.py \
  --fk-function fk_gripper_center \
  --frame-start 0 \
  --frame-stop 10 \
  --output /tmp/fk_pixels_episode0_right.json
```
