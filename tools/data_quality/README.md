# Raw HDF5 Episode Screening

This folder contains utilities for checking raw Aloha HDF5 episodes before
conversion or training.

## Main Script

```text
tools/data_quality/screen_hdf5_episodes.py
```

The script scans `episode_*.hdf5` files, detects bad episodes, writes a report,
and can move bad files out of the dataset tree.

## Dependencies

The script needs:

```bash
pip install h5py numpy
```

In the current default shell environment these packages may not be installed.
Run the script inside the data-processing conda/venv that has `h5py`.

## What It Checks

For each HDF5 file it checks:

- File can be opened as HDF5.
- A state or action dataset exists.
  - State candidates: `observations/qpos`, `observations/state`,
    `observation.state`, `qpos`, `state`.
  - Action candidates: `action`, `actions`.
- Episode length is at least `--min-frames`.
- State/action arrays do not contain NaN/Inf.
- State/action lengths match.
- Arm motion is above threshold.
  - Left arm dimensions: `0:6`
  - Right arm dimensions: `7:13`
- Optional image checks with `--check-images`.

The default motion criterion marks an episode bad only if neither arm moves
enough:

```text
--active-arms any
```

Use `--active-arms right` if a task should be right-arm only, or `--active-arms
both` if both arms must move.

## Dry Run

Dry run is the default. It does not remove files.

```bash
python tools/data_quality/screen_hdf5_episodes.py \
  --root ~/data/aloha_pipeline \
  --num-workers 8
```

Outputs:

```text
tools/data_quality/hdf5_screen_report.json
tools/data_quality/bad_hdf5_episodes.txt
```

For normal dataset processing, prefer writing the report under the output
dataset directory. The conversion script treats this `data_quality/` directory
as persistent screening metadata:

```bash
python tools/data_quality/screen_hdf5_episodes.py \
  --root ~/data/aloha_pipeline \
  --tasks <task_name1> <task_name2> \
  --output playground/Datasets/<dataset_name>/data_quality/hdf5_screen_report.json \
  --bad-list playground/Datasets/<dataset_name>/data_quality/bad_hdf5_episodes.txt \
  --num-workers 8
```

## Scan One Task

```bash
python tools/data_quality/screen_hdf5_episodes.py \
  --root ~/data/aloha_pipeline \
  --tasks <task_name> \
  --active-arms right \
  --num-workers 8
```

## Move Bad Episodes To Quarantine

This removes bad episodes from their original task directory by moving them to a
timestamped quarantine directory:

```bash
python tools/data_quality/screen_hdf5_episodes.py \
  --root ~/data/aloha_pipeline \
  --num-workers 8 \
  --apply
```

Default quarantine path:

```text
~/data/aloha_pipeline/_removed_bad_episodes/YYYYmmdd_HHMMSS/
```

Because this modifies `~/data/aloha_pipeline`, running it from a sandboxed agent
may require explicit filesystem approval.

## Permanent Delete

Only use this after inspecting the dry-run report:

```bash
python tools/data_quality/screen_hdf5_episodes.py \
  --root ~/data/aloha_pipeline \
  --num-workers 8 \
  --apply \
  --delete
```

The recommended workflow is to use quarantine moves, not permanent delete.

## Next Step

After screening, convert the HDF5 tree with the screen report. Bad episodes are
skipped automatically. If `--overwrite` is used, conversion preserves
`data_quality/` and rebuilds only generated LeRobot outputs:

```bash
python tools/conversion/convert_hdf5_to_lerobot_absolute.py \
  --src-dir ~/data/aloha_pipeline \
  --tasks <task_name1> <task_name2> \
  --out-dir playground/Datasets/<dataset_name> \
  --screen-report playground/Datasets/<dataset_name>/data_quality/hdf5_screen_report.json \
  --robot-type aloha_piper_absolute \
  --overwrite
```

## Tune Motion Thresholds

Defaults:

```text
--min-joint-range 0.03
--min-cumulative-joint-movement 0.15
```

For stricter filtering:

```bash
python tools/data_quality/screen_hdf5_episodes.py \
  --root ~/data/aloha_pipeline \
  --min-joint-range 0.05 \
  --min-cumulative-joint-movement 0.25 \
  --num-workers 8
```

## Optional Image Checks

Image checks sample frames from datasets under `observations/images`,
`observation/images`, or `images`.

```bash
python tools/data_quality/screen_hdf5_episodes.py \
  --root ~/data/aloha_pipeline \
  --check-images \
  --image-sample-count 5 \
  --min-image-std 1.0 \
  --num-workers 8
```
