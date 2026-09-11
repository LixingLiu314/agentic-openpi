"""A deployable M3 checkpoint must include the final ground-truth-free budget."""


def generated_only_updates(completed_m3_steps, planned_m3_steps):
    if planned_m3_steps < 4 or not 0 <= completed_m3_steps <= planned_m3_steps:
        raise ValueError("Invalid M3 curriculum position")
    first_generated_only_step = (3 * planned_m3_steps + 3) // 4
    return max(0, completed_m3_steps - first_generated_only_step)


def selection_eligible(stage, completed_steps, planned_steps, action_updates):
    if stage != "m3":
        return True
    if action_updates < completed_steps:
        raise ValueError("Action update counter is inconsistent with M3 progress")
    # Integer comparison avoids rounding at the required 20% boundary.
    pure_updates = generated_only_updates(completed_steps, planned_steps)
    return pure_updates > 0 and 5 * pure_updates >= action_updates
