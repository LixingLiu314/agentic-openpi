# End-To-End Aloha HDF5 Pipeline

This folder contains the recommended entry point for processing newly collected
Aloha HDF5 task folders.

Raw data is expected at:

```text
~/data/aloha_pipeline/<task_name>/episode_*.hdf5
```

The pipeline writes the converted base dataset to:

```text
playground/Datasets/<dataset_name>/
```

By default it also writes the training-ready gripper-binary dataset to:

```text
playground/Datasets/<dataset_name>_gripper_binary/
```

## Run

```bash
python tools/pipeline/run_aloha_hdf5_pipeline.py \
  --tasks food_test food_test_2 \
  --dataset-name food_foreact \
  --overwrite
```

Use comma-separated task names if that is easier in shell scripts:

```bash
python tools/pipeline/run_aloha_hdf5_pipeline.py \
  --tasks grasp_the_green_cube_and_place_it_on_the_blue_plate,grasp_the_red_cube_and_place_it_on_the_blue_plate \
  --dataset-name cube_to_blue_plate_absolute \
  --overwrite
```

## What It Does

1. Screens HDF5 files under the requested task folders.
   Operator-provided bad episodes can be added on top of automatic screening.
2. Saves the screen report and bad file list under:

   ```text
   playground/Datasets/<dataset_name>/data_quality/
   ```

3. Converts only valid HDF5 episodes into a LeRobot absolute dataset.
4. Patches per-frame `subtask` labels from each source HDF5 when available.
5. Generates calibrated FK trajectory JSON for both arms.
6. Generates `cot_text_prompts.json` in `Left: ... Right: ...` format.
7. Renders all annotated front-camera trajectory videos.
8. Creates `<dataset_name>_gripper_binary` with binary gripper actions and fixed parquet video columns.
9. Re-encodes final dataset videos, generates `cam_high_subgoal`, and writes `traj_cot` into parquets.

Bad HDF5 episodes are not deleted by this pipeline. They are recorded in the
screen report and skipped during conversion. Manual bad episodes are stored in
the same report with the reason `manual_bad_episode`.

`data_quality/` is persistent screening metadata. When `--overwrite` is used,
the converter preserves this directory and rebuilds only generated LeRobot
outputs such as `meta/`, `data/`, and `videos/`.

## Useful Options

- `--raw-root ~/data/aloha_pipeline`
  - Root containing task folders.
- `--dataset-root playground/Datasets`
  - Parent directory for generated datasets.
- `--screen-only`
  - Only write `data_quality/hdf5_screen_report.json` and
    `data_quality/bad_hdf5_episodes.txt`.
- `--skip-videos`
  - Run through CoT generation but skip annotated video rendering.
- `--skip-screen`
  - Reuse an existing `data_quality/hdf5_screen_report.json`.
- `--active-arms any|left|right|both`
  - Motion criterion used by HDF5 screening.
- `--check-images`
  - Include image length/variance checks in screening.
- `--manual-bad-episodes food_test:3 food_test_2:10:15`
  - Add operator-specified bad source episodes on top of automatic screening.
    Accepted forms include absolute paths, root-relative paths, `task:id`,
    `task:start:stop`, bare `id`, and bare `start:stop`.
- `--manual-bad-list path/to/bad_episodes.txt`
  - Read the same manual bad episode tokens from a text file. `#` comments are ignored.
- `--limit N`
  - Debug conversion on the first N valid episodes after screening.
- `--skip-gripper-binary`
  - Keep only the base absolute dataset and skip final gripper-binary creation.
- `--skip-subgoal`, `--skip-traj-cot-patch`, `--skip-reencode-videos`
  - Disable individual postprocess stages corresponding to scripts step4/5/6.

## Output Files

```text
playground/Datasets/<dataset_name>/
  data_quality/
    hdf5_screen_report.json
    bad_hdf5_episodes.txt
    pipeline_summary.json
  meta/
    info.json
    modality.json
    conversion_report.json
    stats_gr00t.json
  data/
    chunk-000/episode_000000.parquet
  videos/
    chunk-000/observation.images.<camera>/episode_000000.mp4
  trajectory_data/
    fk_bimanual.json
    cot_text_prompts.json
    visualizations_all/
      episode_000000_both_trajectory.mp4
      render_report.json

playground/Datasets/<dataset_name>_gripper_binary/
  meta/
    info.json
    modality.json
  data/
    chunk-000/episode_000000.parquet   # action binarized; includes subtask/traj_cot
  videos/
    chunk-000/observation.images.cam_high/episode_000000.mp4
    chunk-000/observation.images.cam_high_subgoal/episode_000000.mp4
  trajectory_data/
    fk_bimanual.json
    cot_text_prompts.json
```

## Smoke Test

To verify the pipeline without rendering annotated videos:

```bash
python tools/pipeline/run_aloha_hdf5_pipeline.py \
  --tasks <task_name> \
  --dataset-name <dataset_name>_smoke \
  --limit 2 \
  --skip-videos \
  --overwrite
```
