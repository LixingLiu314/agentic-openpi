# Tools Layout

This directory is split by workflow instead of keeping all scripts and generated
assets in one flat folder.

## Directory Map

- `tools/calibration/`
  - Camera calibration, hand-eye calibration, front/wrist transfer calibration,
    calibration images, manifests, and solved calibration JSON files.
  - Start here when camera intrinsics or extrinsics need to be recomputed.

- `tools/trajectory/`
  - FK projection into image pixels, trajectory JSON generation, trajectory CoT
    prompt generation, and trajectory visualization.
  - This is the normal path after calibration is already available.

- `tools/data_quality/`
  - Raw HDF5 episode screening utilities for `~/data/aloha_pipeline`.
  - Use this before converting raw HDF5 data into LeRobot format or before
    running trajectory generation.

- `tools/conversion/`
  - HDF5 to local LeRobot v2.1-style dataset conversion for the Aloha absolute
    state/action dataset.
  - Use this after raw HDF5 screening and before trajectory projection.

- `tools/pipeline/`
  - End-to-end orchestration from raw HDF5 task folders to LeRobot, trajectory
    JSON, CoT prompts, and annotated videos.
  - This is the recommended entry point after a new data collection.

## Generated Cache

`tools/__pycache__/` may appear after Python runs. It is generated cache and can
be ignored.

## Typical Order

Most runs should use the end-to-end pipeline:

```bash
python tools/pipeline/run_aloha_hdf5_pipeline.py \
  --tasks <task_name1> <task_name2> \
  --dataset-name <dataset_name> \
  --overwrite
```

The script expects raw HDF5 folders at:

```text
~/data/aloha_pipeline/<task_name>/
```

It writes all generated outputs to:

```text
playground/Datasets/<dataset_name>/
```

Important outputs:

```text
playground/Datasets/<dataset_name>/data_quality/hdf5_screen_report.json
playground/Datasets/<dataset_name>/data_quality/bad_hdf5_episodes.txt
playground/Datasets/<dataset_name>/meta/info.json
playground/Datasets/<dataset_name>/trajectory_data/fk_bimanual.json
playground/Datasets/<dataset_name>/trajectory_data/cot_text_prompts.json
playground/Datasets/<dataset_name>/trajectory_data/visualizations_all/
```

The conversion step reads the screen report and skips bad HDF5 episodes
automatically. `data_quality/` is preserved when conversion is rebuilt with
`--overwrite`; only generated LeRobot outputs are replaced. Trajectory
generation does not need any episode-exclusion arguments.

## Manual Steps

Use these only when debugging one stage.

1. Screen selected raw HDF5 task folders:

   ```bash
   python tools/data_quality/screen_hdf5_episodes.py \
     --root ~/data/aloha_pipeline \
     --tasks <task_name1> <task_name2> \
     --output playground/Datasets/<dataset_name>/data_quality/hdf5_screen_report.json \
     --bad-list playground/Datasets/<dataset_name>/data_quality/bad_hdf5_episodes.txt \
     --num-workers 8
   ```

2. Convert only valid HDF5 episodes into the absolute LeRobot dataset:

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

   The converter copies `/observations/qpos` to `observation.state` and
   `/action` to `action` as absolute values. If the screen report is under
   `<dataset_name>/data_quality/`, `--overwrite` keeps that directory while
   rebuilding `meta/`, `data/`, and `videos/`.

3. Recompute calibration only if hardware/camera mounting changed:

   ```bash
   less tools/calibration/README.md
   ```

4. Generate trajectory JSON, CoT prompts, and videos from the converted dataset:

   ```bash
   bash tools/trajectory/run_aloha_bimanual_pipeline.sh playground/Datasets/<dataset_name>
   ```
