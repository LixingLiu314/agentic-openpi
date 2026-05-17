# End-To-End Aloha HDF5 Pipeline

This folder contains the recommended entry point for processing newly collected
Aloha HDF5 task folders.

Raw data is expected at:

```text
~/data/aloha_pipeline/<task_name>/episode_*.hdf5
```

The pipeline writes one dataset to:

```text
playground/Datasets/<dataset_name>/
```

## Run

```bash
python tools/pipeline/run_aloha_hdf5_pipeline.py \
  --raw-root ~/data/shape_hole_pipeline \
  --tasks put_the_shapes_into_the_matching_holes \
  --dataset-name aloha_shape \
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
2. Saves the screen report and bad file list under:

   ```text
   playground/Datasets/<dataset_name>/data_quality/
   ```

3. Converts only valid HDF5 episodes into a LeRobot absolute dataset.
4. Generates calibrated FK trajectory JSON for both arms.
5. Generates `cot_text_prompts.json` in `Left: ... Right: ...` format.
6. Renders all annotated front-camera trajectory videos.

Bad HDF5 episodes are not deleted by this pipeline. They are recorded in the
screen report and skipped during conversion. Once conversion is finished, all
later stages operate only on valid LeRobot episodes and do not need any manual
episode exclusion.

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
- `--limit N`
  - Debug conversion on the first N valid episodes after screening.

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
