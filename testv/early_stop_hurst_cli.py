from __future__ import annotations

from early_stop_hurst import analyze_sequence


def apply_two_stage_rule(result, *, threshold, consecutive, observation_points):
    """Require two low-Hurst confirmations separated by a fixed wait."""
    if consecutive < 1:
        raise ValueError("consecutive must be positive")
    if observation_points < 1:
        raise ValueError("observation_points must be positive")

    phase = "first"
    hits = 0
    remaining = 0
    first_confirmation_step = None
    observation_end_step = None
    second_confirmation_check = None

    for check in result["checks"]:
        hurst = check["hurst"]
        if hurst is None:
            continue

        if phase == "observation":
            remaining -= 1
            if remaining == 0:
                observation_end_step = check["global_step"]
                phase = "second"
                hits = 0
            continue

        hits = hits + 1 if hurst <= threshold else 0
        if hits < consecutive:
            continue

        if phase == "first":
            first_confirmation_step = check["global_step"]
            phase = "observation"
            remaining = observation_points
            hits = 0
        else:
            second_confirmation_check = check
            break

    updated = dict(result)
    if second_confirmation_check is None:
        stop_index = None
        stop_step = None
    else:
        stop_index = second_confirmation_check["index"]
        stop_step = second_confirmation_check["global_step"]

    updated.update(
        {
            "two_stage": True,
            "observation_points": observation_points,
            "first_confirmation_step": first_confirmation_step,
            "observation_end_step": observation_end_step,
            "second_confirmation_step": stop_step,
            "should_stop": stop_step is not None,
            "first_stop_index": stop_index,
            "first_stop_position": (
                stop_index + 1 if stop_index is not None else None
            ),
            "first_stop_global_step": stop_step,
        }
    )
    return updated
