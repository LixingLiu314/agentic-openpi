"""GUI-based evaluation pipeline for the Aloha (cobot_magic Piper) real robot.

Four evaluation modes are supported, matching the four training-time data
variants in ``src/openpi/training/config.py``:

  - basic    : pi05_aloha_banana_*           (no extra inputs)
  - traj     : pi05_aloha_banana_traj         (prompt suffixed with ", traj: ...")
  - subtask  : pi05_aloha_banana_subtask_segment (prompt suffixed with ", subtask: ...")
  - subgoal  : pi05_aloha_banana_subgoal_base    (extra ``subgoal_images`` field)

The transforms that build the final prompt / inject the subgoal image at
training time live in ``repack_transforms`` (training-only). At inference
the client must emit data in the **post-repack** format directly, which is
exactly what ``modes.py`` does.
"""
