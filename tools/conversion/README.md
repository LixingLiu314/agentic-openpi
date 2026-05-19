# HDF5 To LeRobot Absolute Conversion

This folder contains the HDF5-to-LeRobot conversion step for Aloha data
collected under `~/data/aloha_pipeline/<task_name>/`.

The converter is intentionally for the absolute dataset variant:

- HDF5 `/observations/qpos` is copied directly to LeRobot
  `observation.state`.
- HDF5 `/action` is copied directly to LeRobot `action`.
- No delta action, offset-to-current-state, normalization, or gripper
  binarization is applied during conversion.
- `meta/modality.json` marks all left/right arm and gripper state/action fields
  with `absolute: true`.

## Main Script

```text
tools/conversion/convert_hdf5_to_lerobot_absolute.py
```

It writes a local LeRobot v2.1-style dataset layout:

```text
<out_dir>/
  data_quality/
    hdf5_screen_report.json
    bad_hdf5_episodes.txt
  meta/
    info.json
    modality.json
    episodes.jsonl
    tasks.jsonl
    episodes_stats.jsonl
    stats_gr00t.json
    conversion_report.json
  data/
    chunk-000/
      episode_000000.parquet
  videos/
    chunk-000/
      observation.images.cam_high/
        episode_000000.mp4
      observation.images.cam_left_wrist/
        episode_000000.mp4
      observation.images.cam_right_wrist/
        episode_000000.mp4
```

The `data_quality/` directory is screening metadata, not a generated LeRobot
shard. When `--screen-report` points inside `<out_dir>/data_quality/`,
`--overwrite` preserves that directory while rebuilding the generated LeRobot
outputs. The underscore naming, `chunks_size`, `data_path`, `video_path`, and
`modality.json` metadata are written to match the StarVLA LeRobot loader and the
trajectory scripts under `tools/trajectory/`.

## Dependencies

Run in the data-processing environment that has HDF5 and video dependencies:

```bash
pip install h5py numpy pandas pyarrow opencv-python
```

The script also calls `ffmpeg` to encode mp4 videos:

```bash
ffmpeg -version
```

Use `--ffmpeg-bin /path/to/ffmpeg` if `ffmpeg` is not on `PATH`.

## Recommended Use

For normal use, call the end-to-end pipeline instead of this script directly:

```bash
python tools/pipeline/run_aloha_hdf5_pipeline.py \
  --tasks custom \
  --dataset-name aloha_custom \
  --overwrite
```
```bash
python tools/pipeline/run_aloha_hdf5_pipeline.py \
  --tasks <task_name1> <task_name2> \
  --dataset-name <dataset_name> \
  --overwrite
```

The pipeline writes a screen report, then passes it to this converter so bad
HDF5 episodes are skipped automatically.

## Manual Conversion

Screen raw HDF5 task folders first:

```bash
python tools/data_quality/screen_hdf5_episodes.py \
  --root ~/data/aloha_pipeline \
  --tasks <task_name1> <task_name2> \
  --output playground/Datasets/<dataset_name>/data_quality/hdf5_screen_report.json \
  --bad-list playground/Datasets/<dataset_name>/data_quality/bad_hdf5_episodes.txt \
  --num-workers 8
```

Then convert the same task folders. The converter reads the screen report and
skips every entry in `bad_episodes`. If `--overwrite` is used, the converter
keeps `<out_dir>/data_quality/` and rebuilds only the generated LeRobot
outputs:

```bash
python tools/conversion/convert_hdf5_to_lerobot_absolute.py \
  --src-dir ~/data/aloha_pipeline \
  --tasks <task_name1> <task_name2> \
  --out-dir playground/Datasets/<dataset_name> \
  --screen-report playground/Datasets/<dataset_name>/data_quality/hdf5_screen_report.json \
  --robot-type aloha_piper_absolute \
  --episode-index-mode sequential \
  --overwrite
```

For a single task, `--episode-index-mode preserve` can keep source HDF5 episode
ids. For multiple task folders, use `sequential` to avoid collisions because
each task folder normally starts at `episode_0.hdf5`.

## Episode Index Policy

Default mode is:

```text
--episode-index-mode auto
```

`auto` preserves source episode ids when they are globally unique. For example,
`episode_000246.hdf5` becomes `episode_000246.parquet`.

For a multi-task HDF5 tree where each task folder starts again at
`episode_000000.hdf5`, use:

```bash
--episode-index-mode sequential
```

That writes contiguous output ids and avoids collisions.

## Task Names

If `~/data/aloha_pipeline/pipeline_meta.json` exists, task text is read from it.
Otherwise:

- A flat source directory uses the directory name with underscores replaced by
  spaces.
- One-level task subdirectories use each task folder name.

To force one task name for a flat source directory:

```bash
--task-name "pick object and place it on target"
```

## Image Inputs

The default image group is:

```text
/observations/images
```

Camera order defaults to:

```text
cam_high, cam_left_wrist, cam_right_wrist
```

Any additional cameras are appended in sorted order. To force a subset/order:

```bash
--camera-names cam_high cam_left_wrist cam_right_wrist
```

JPEG-compressed frames are decoded with OpenCV. For uncompressed HDF5 image
arrays, set the color layout if needed:

```bash
--raw-image-color rgb
```

The default is `bgr`, matching OpenCV camera/video conventions.

## Output Metadata

Important generated files:

- `meta/info.json`
  - LeRobot path patterns, feature metadata, camera names, fps, and explicit
    `state_action_encoding` set to absolute.
- `meta/modality.json`
  - Split mapping for left/right arms and grippers:
    - left arm: `0:6`
    - left gripper: `6:7`
    - right arm: `7:13`
    - right gripper: `13:14`
- `meta/stats_gr00t.json`
  - Aggregate statistics for `observation.state` and `action`, so the StarVLA
    loader does not need to recompute them on first training load.
- `meta/conversion_report.json`
  - Source root, filters, resolved episode-index mode, converted count, and
    failure count.

## Continue The Pipeline

After conversion, generate FK trajectory JSON, CoT prompts, and annotated videos
from every converted episode:

```bash
bash tools/trajectory/run_aloha_bimanual_pipeline.sh playground/Datasets/<dataset_name>
```
