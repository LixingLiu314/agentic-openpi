# Aloha Eval GUI

A unified PyQt5 evaluation pipeline for the four `pi05_aloha_banana_*` policies.
One window, one task field, one mode dropdown — the backend swaps the right
pre-processing (subtask label / Doubao trajectory / ForeAct subgoal image) so
the on-wire observation matches what the **training** repack transforms produced.

## Modes

| GUI Mode  | Training config                            | What the client adds to the obs                                              |
|-----------|--------------------------------------------|------------------------------------------------------------------------------|
| `basic`   | `pi05_aloha_banana`                        | nothing — `prompt = task`                                                    |
| `traj`    | `pi05_aloha_banana_traj`                   | `prompt = "{task}, traj: Left: Go along ... Right: Go along <br/>  ..."`     |
| `subtask` | `pi05_aloha_banana_subtask_segment`        | `prompt = "{task}, subtask: {label}"`; label chosen from the subtask list    |
| `subgoal` | `pi05_aloha_banana_subgoal_base`           | `subgoal_images = {"cam_high": HxWx3 uint8}` from ForeAct                    |

### Policy checkpoint selection

The **Mode** dropdown owns the policy config. Switching modes automatically sets
the matching `policy.config`:

| GUI Mode  | Auto-selected `policy.config`              |
|-----------|--------------------------------------------|
| `basic`   | `pi05_aloha_banana`                        |
| `traj`    | `pi05_aloha_banana_traj`                   |
| `subtask` | `pi05_aloha_banana_subtask_segment`        |
| `subgoal` | `pi05_aloha_banana_subgoal_base`           |

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
`<loc{x:04d}><loc{y:04d}>`. End-to-end the prompt becomes e.g.

```
put banana in the green plate, traj: Left: Go along <loc0000><loc0554>.
Right: Go along <br/>  <loc0980><loc0627>, <loc1000><loc0533>, close gripper, <loc0714><loc0271>
```

The `<br/>` and the **two spaces** that follow it are intentional and match
the training corpus byte-for-byte.

### Sub-modes (modes 2 / 3 / 4)

A radio toggle in the GUI controls *how* external inputs (Doubao trajectory,
subtask suggestion, ForeAct subgoal image) are fetched:

* **non-blocking (async)** — every `N` steps a request is fired on a
  background thread; the control loop keeps stepping with the most recent
  cached result.
* **blocking** — every `N` steps the control loop pauses and waits for a
  fresh external input before continuing.

`N` is the **step interval** spinbox (defaults: 6 for traj, 30 for subgoal,
30 for subtask auto-suggest).

For Mode 3 without an auto-suggest predictor, the "fresh external input" is an
operator subtask selection. In blocking mode the loop waits at step 0 and then
every `N` steps until you press/confirm a subtask key. Pressing the currently
active key again counts as confirmation.

### Subtask labels (Mode 3)

* Default key on init = **1**, with the canonical banana labels
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

Recording starts when the run loop starts and stops when the episode ends or
you click **Stop**. File names include the selected checkpoint name and a
timestamp, for example:

```
test_video/banana_subtask_seg_lr5e5_5000_20260507_102100.mp4
```

Videos are encoded as H.264 MP4 with `yuv420p` pixels and `+faststart`, so they
can be opened directly in VS Code and browser-based players.

## Binary Grippers

The runner displays the raw continuous model output, thresholded binary state,
and final hardware command for both grippers in the GUI at every control step.
The raw model output is interpreted as gripper opening width: larger values
mean the gripper should be more open. Before sending commands to the robot it
thresholds the two gripper action dimensions:

* left gripper raw output = action index `6`
* right gripper raw output = action index `13`
* raw value `> gripper_threshold` -> binary `1` (open)
* raw value `<= gripper_threshold` -> binary `0` (close)

The binary state is then mapped to the Piper hardware command values:

* binary `1` -> `gripper_open` (default `4.0`)
* binary `0` -> `gripper_close` (default `0.0`)

The default threshold is `2.0`, and the GUI **Gripper threshold** spinbox can be
adjusted while the run is active. You can also set the startup value with:

```bash
--gripper_threshold 2.0
```

## Prerequisites

1. **Robot stack** (Piper ROS) up and homed.
2. **ForeAct server** (only for `subgoal` mode), on `10.1.119.68`:
   ```bash
   python server_foreact.py
   ```
3. **Doubao API key** (only for `traj` mode):
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
    --task "put banana in the green plate"
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
  `instruction.txt` against a training sample. Whitespace matters, especially
  the two spaces after `<br/>`.
