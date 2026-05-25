# Aloha Eval GUI

A unified PyQt5 evaluation pipeline for the `aloha_food`
`pi05_aloha_food_*` policies.
One window, one task field, one mode dropdown — the backend swaps the right
pre-processing (subtask label / Doubao trajectory / ForeAct subgoal image) so
the on-wire observation matches what the **training** repack transforms produced.

## Modes

| GUI Mode  | Training config                            | What the client adds to the obs                                              |
|-----------|--------------------------------------------|------------------------------------------------------------------------------|
| `basic`   | `pi05_aloha_food_baseline`                 | nothing — `prompt = task`                                                    |
| `traj`    | `pi05_aloha_food_traj`                     | `prompt = "{task}, traj: Left: Go along ... Right: Go along ..."`            |
| `subtask` | `pi05_aloha_food_subtask`                  | `prompt = "{task}, subtask: {label}"`; label chosen from the subtask list    |
| `triple_cot` | `pi05_aloha_food_all_cot`               | prompt includes task + subtask + trajectory; may include subgoal image       |
| `subgoal` | `pi05_aloha_food_subgoal`                  | `subgoal_images = {"cam_high": HxWx3 uint8}` from ForeAct                    |

### Policy checkpoint selection

The **Mode** dropdown owns the policy config. Switching modes automatically sets
the matching `policy.config`:

| GUI Mode  | Auto-selected `policy.config`              |
|-----------|--------------------------------------------|
| `basic`   | `pi05_aloha_food_baseline`                 |
| `traj`    | `pi05_aloha_food_traj`                     |
| `subtask` | `pi05_aloha_food_subtask`                  |
| `triple_cot` | `pi05_aloha_food_all_cot`               |
| `subgoal` | `pi05_aloha_food_subgoal`                  |

The checkpoint field is an editable dropdown. You can type a `policy.dir`, pick
one from the dropdown, use the step spinbox to generate the default checkpoint
path for that mode, or click **Browse...** for a local checkpoint directory.
Manually typed and browsed paths are remembered per mode in:

```
~/.cache/agentic-openpi/eval_gui_checkpoints.json
```

Set `AGENTIC_OPENPI_EVAL_GUI_HISTORY=/path/to/history.json` to use a different
history file. With **start local server** checked, clicking **Connect** starts
`scripts/serve_policy_pytorch.py` with the mode-selected config and the selected
checkpoint path, then connects the runner to it. Uncheck it only when connecting
to an already-running external policy server.

### Strict trajectory format (Mode 2)

Doubao is asked to produce a structured per-arm plan; the client renders it as

```
Left: Go along (x,y), close gripper, (x,y). Right: Go along <br/>  (x,y), (x,y), ...
```

and then runs the **same** `coords_to_loc_tokens` regex used by
`scripts/preprocess_traj.py` to substitute every `(x,y)` for
`<loc{x:04d}><loc{y:04d}>`. Before sending the prompt to the VLA, any
`<br>` / `<br/>` separators are replaced with plain spaces while preserving all
`<locXXXX>` tokens. End-to-end the prompt becomes e.g.

```
place_all_the_food_into_the_plate, traj: Left: Go along <loc0000><loc0554>.
Right: Go along <loc0980><loc0627>, <loc1000><loc0533>, close gripper, <loc0714><loc0271>
```

The internal parser can still consume older trajectory strings that contain
`<br/>`; the final VLA prompt is cleaned so HTML break tags do not leak into the
model input.

When Mode 2 receives a Doubao trajectory, the GUI parses the resulting
`<locXXXX><locYYYY>` points, projects them onto `cam_high`, and shows the
annotated image in the main camera panel. Left-arm points are drawn in magenta;
right-arm points are drawn in yellow/green/red/blue.

In Mode 2, the **Trajectory source (Mode 2)** selector chooses what happens
during blocking refresh steps: **Doubao API** calls the API, while **Manual
Annotation** opens an operator annotation dialog. The dialog is implemented with
native PyQt5 widgets (`QDialog`, `QLabel`/`QPixmap`, and `QPushButton`) rather
than OpenCV GUI calls. It shows the current `cam_high` frame. Select **L (Left
Arm)** or **R (Right Arm)**, click image points to add
`<locXXXX><locYYYY>` waypoints, insert **Open Gripper** or **Close Gripper**
actions as needed, and press **Finish**. The generated text is cached exactly
like a Doubao result and uses the same per-arm loc-token ordering. Press
**Emergency Stop** to discard the annotation, pause inference, and queue the
existing return-to-zero path.

### Auto-suggest reference trajectories

Trajectory Mode can pre-fill the annotation dialog with a reference trajectory
retrieved from the offline training dataset. The runtime never scans raw frames.
Instead, you build a cache once:

```bash
python tools/trajectory/build_trajectory_retrieval_cache.py \
    --dataset-root playground/Datasets/food \
    --overwrite
```

By default this writes:

```
playground/Datasets/food/trajectory_data/cam_high_traj_reference_cache.pt
```

The cache contains:

* one normalized image embedding per `observation.images.cam_high` frame;
* the matching `episode_index` and `frame_index`;
* the GT trajectory text from the parquet `traj_cot` column, falling back to
  `trajectory_data/cot_text_prompts.json` when that column is absent.

At evaluation time, set either the GUI cache field or an environment variable:

```bash
export AGENTIC_OPENPI_TRAJ_RETRIEVAL_CACHE=/path/to/cam_high_traj_reference_cache.pt
# or:
export AGENTIC_OPENPI_TRAJ_RETRIEVAL_DATASET=playground/Datasets/food
```

When `playground/Datasets/food` exists in this repo, the GUI uses its
default cache path automatically if neither environment variable is set.

When the trajectory dialog opens, the GUI encodes the current live `cam_high`
frame once, computes cosine similarity against the cached tensor with PyTorch,
and uses the Top-1 matched frame's GT trajectory as the default text. Press
**Finish** to use it directly, or **Clear** to discard it and annotate a new
trajectory. The dialog log includes the matched episode/frame, score, and
search latency; the intended runtime path is comfortably below 1 second because
all dataset embeddings are already resident in memory.

### Auto-suggest subgoal images

Subgoal Mode and Triple-CoT can also use the same retrieval cache to suggest a
subgoal image. Choose **Reference Retrieval** in the subgoal source panel. At
runtime the client:

1. encodes the current live `cam_high` frame;
2. finds the Top-1 matching dataset frame `n` in the trajectory cache;
3. decodes frame `n + 60` from the matched episode's `cam_high` video;
4. sends that RGB frame as `subgoal_images["cam_high"]`.

The `Retrieval lookahead` spinbox controls the `+60` offset. The default is
60 frames. The embedding search is shared with trajectory auto-suggest; only
one target video frame is decoded per refresh.

### Sub-modes (modes 2 / 3 / 4)

A radio toggle in the GUI controls *how* external inputs (Doubao trajectory,
subtask suggestion, ForeAct subgoal image) are fetched:

* **non-blocking (async)** — every `N` steps a request is fired on a
  background thread; the control loop keeps stepping with the most recent
  cached result.
* **blocking** — every `N` steps the control loop pauses and waits for a
  fresh external input before continuing.

`N` is the **step interval** spinbox (default: 60).

For Mode 3 without an auto-suggest predictor, the "fresh external input" is an
operator subtask selection. In blocking mode the loop waits at step 0 and then
every `N` steps until you press/confirm a subtask key. Pressing the currently
active key again counts as confirmation.

### Subtask labels (Mode 3)

* Default key on init = **1**, with the canonical food placement labels
  pre-populated.
* Labels are **editable** in-place (just type & Enter).
* Click **Add** to append a new subtask field. Click **Remove** to remove the
  highest-numbered field. At least one label is always kept.
* The label list is inside a fixed-height scroll area, so adding many labels
  does not grow the window.
* Click a row's **Use** button to select that subtask. Number keys `1`-`9`
  are shortcuts for matching keys when they exist.
* You can also click *Load JSON…* to swap them at runtime; format:
  `{"1": "label_a", "2": "label_b", ...}`.
* In blocking mode, press/confirm a subtask key or click a row's **Use** button
  to release the next blocked subtask interval with that label. The runtime
  status shows `(waiting)` while the inference loop is paused for operator
  input.
* The `SubtaskPredictor` hook in `doubao_predictor.py` is reserved for a
  future Doubao-driven auto-suggest. The default impl is a no-op so it
  never overrides the human key.

## Run Videos

Each evaluation run automatically records the main camera (`cam_high`) to:

```
test_video/
```

Recording starts when the run starts and stops when the episode ends or you
click **Stop**. A background sampler records `cam_high` at the configured video
FPS independently of the VLA control loop, so blocking waits for manual
trajectory annotation, Doubao, ForeAct, pause/resume, or reset do not create
missing video spans. File names include the selected checkpoint name and a
timestamp, for example:

```
test_video/pi05_aloha_food_subtask_5000_20260507_102100.mp4
```

Videos are encoded as H.264 MP4 with `yuv420p` pixels and `+faststart`, so they
can be opened directly in VS Code and browser-based players.

Each run also writes an append-only VLA input log next to the video:

```
test_video/pi05_aloha_food_subtask_5000_20260507_102100_log.jsonl
test_video/pi05_aloha_food_subtask_5000_20260507_102100_step_00000_subgoal.jpg
```

The JSONL file has one entry per actual VLA server inference request. Each
entry includes the control `step`, `inference_index`, exact `text_prompt`,
state vector, model image keys/shapes, runtime mode metadata, and any saved
ForeAct subgoal image path.

## Continuous Grippers

The runner displays the continuous model output for both grippers in the GUI at
every control step. It sends those continuous values directly to the robot,
without applying a threshold:

* left gripper action = action index `6`
* right gripper action = action index `13`

Set the numeric field beside **Debug Gripper** (default `0.06`) and click the
button to pause inference, return the robot to zero, and send that direct
continuous value to both grippers. This bypasses model inference and is intended
only for hardware calibration/debugging.

## Prerequisites

1. **Robot stack** (Piper ROS) up and homed.
2. **ForeAct server** (only for `subgoal` mode), on `10.1.119.68`:
   ```bash
   python server_foreact.py
   ```
3. **Doubao API key** (for API-driven `traj` mode; not needed when using
   **Manual Annotation** for Mode 2 blocking runs):
   ```bash
   export VOLCENKEY="<your-volc-ark-api-key>"
   ```
4. **PyQt5** available in the `uv run python` environment.
5. **ffmpeg** on `PATH` for automatic H.264 MP4 run videos.

## Run

```bash
bash scripts/start_aloha_eval_gui.sh \
    --mode subtask \
    --host 127.0.0.1 --port 8000 \
    --task place_all_the_food_into_the_plate
```

## Keyboard shortcuts

| key     | action                                                |
|---------|-------------------------------------------------------|
| 1-9     | set subtask key when that key exists (mode 3)         |
| Space   | pause / resume                                        |
| H       | 回零 / Return-to-Zero (works whether running or idle) |
| D       | dump model inputs (next inference)                    |
| Q       | quit                                                  |

## Return-to-Zero (回零)

Implemented identically to `scripts/eval_banana.sh --reset-only` (which calls
`environment.reset()` per `examples/aloha_real/piper_main.py:111-113`).

* If the run loop is **active**: a reset event is queued and drained on the
  next loop iteration (no race with `env.step`).
* If the runner is **idle / paused**: a small worker thread directly calls
  `env.reset()` under an env-lock.

## Model-input inspection

The automatic `test_video/*_log.jsonl` file is the lightweight per-run trace of
every VLA inference input. It is intended for comparing prompts, states, modes,
and ForeAct subgoal images across a full episode.

Click **⬇ Dump inputs** (or press **D**) at any time. The next inference
call snapshots the *exact post-repack* payload to disk:

```
debug_inputs/step_0000_<timestamp>/
    instruction.txt           # full prompt (incl. ", traj: ..." / ", subtask: ...")
    cam_high.jpg              # 224x224 model input (CHW->HWC)
    cam_left_wrist.jpg
    cam_right_wrist.jpg
    raw_cam_high.jpg          # original ROS frame (pre-resize)
    subgoal_cam_high.jpg      # if Mode 4
    meta.json                 # state vector, mode, step #, image keys
```

Use this to verify there is no train/eval mismatch in either the visual
preprocessing or the prompt formatting.

## Architecture

```
eval_gui.py          PyQt5 main window
eval_runner.py       EvalRunner thread: env <-> ModeHandler <-> WebsocketPolicy
modes.py             ModeHandler subclasses; build observation matching training
foreact_client.py    msgpack-numpy WS client to server_foreact.py
doubao_predictor.py  Volcengine Ark Vision client + SubtaskPredictor hook
trajectory_retrieval.py  Offline cache builder + runtime Top-1 reference search
```

The handlers reproduce — at inference time — the *post-repack* shape of
`LeRobotAlohaDataConfig` / `LeRobotAlohaWithSubgoalDataConfig` from
`src/openpi/training/config.py`:

* `AppendSubtaskToPrompt`  ↔  `f"{prompt}, subtask: {label}"`
* `AppendTrajCotToPrompt`  ↔  `f"{prompt}, traj: {loc_token_text}"`
* `coords_to_loc_tokens`   ↔  `<loc{x:04d}><loc{y:04d}>` (clamped 0..1023)
* `subgoal_images[<src-cam>]` is HxWx3 uint8 RGB; the policy-side transforms
  map it to `subgoal_base_0_rgb` per `subgoal_camera_map`.

## Troubleshooting

* **回零 button does nothing** — click *Connect* first; the env has to exist.
* **Cameras black** — robot stack not running (no images on ROS topics).
* **Doubao timeout** — set `VOLCENKEY`; the prompt falls back to `task` only.
* **ForeAct unreachable** — check `10.1.119.68:5100`; the run continues without
  a fresh subgoal image until ForeAct becomes reachable again.
* **Train/eval mismatch** — check the GUI Mode and checkpoint selector first,
  then open the latest `debug_inputs/step_*/` folder and inspect
  `instruction.txt` against a training sample. Mode 2 prompts should contain
  clean text and `<locXXXX>` tokens only; `<br>` / `<br/>` tags are stripped
  before the prompt is sent.
