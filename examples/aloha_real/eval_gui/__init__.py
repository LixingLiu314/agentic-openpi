"""GUI-based evaluation pipeline for the Aloha (cobot_magic Piper) real robot.

Five evaluation modes are supported, matching the training-time data
variants in ``src/openpi/training/config.py``:

  - basic    : pi05_aloha_three_object_baseline (no extra inputs)
  - traj     : pi05_aloha_three_object_traj     (prompt suffixed with ", traj: ...")
  - subtask  : pi05_aloha_three_object_subtask  (prompt suffixed with ", subtask: ...")
  - triple_cot : semi-block prompt with task + subtask + trajectory CoT
  - subgoal  : pi05_aloha_three_object_subgoal  (extra ``subgoal_images`` field)

The transforms that build the final prompt / inject the subgoal image at
training time live in ``repack_transforms`` (training-only). At inference
the client must emit data in the **post-repack** format directly, which is
exactly what ``modes.py`` does.
"""
