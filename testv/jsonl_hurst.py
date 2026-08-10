from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from metrics import dfa_hurst


def read_losses(file_path, *, limit=None):
    """Read only the ``loss`` field from each JSONL record."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    path = Path(file_path)
    losses = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Line {line_number}: invalid JSON: {error.msg}"
                ) from None

            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number}: expected a JSON object")
            if "loss" not in record:
                raise ValueError(f"Line {line_number}: missing 'loss' field")

            value = record["loss"]
            if isinstance(value, bool):
                raise ValueError(
                    f"Line {line_number}: loss must be a finite number"
                )
            try:
                loss = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Line {line_number}: loss must be a finite number"
                ) from None
            if not np.isfinite(loss):
                raise ValueError(
                    f"Line {line_number}: loss must be a finite number"
                )

            losses.append(loss)
            if limit is not None and len(losses) >= limit:
                break

    if not losses:
        raise ValueError(f"No loss values found in {path}")
    return losses


def _validate_options(*, window, check_interval, threshold, patience):
    if window < 16:
        raise ValueError("window must be at least 16")
    if check_interval < 1:
        raise ValueError("check_interval must be positive")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if patience < 1:
        raise ValueError("patience must be positive")


def _analyze(losses, *, window, check_interval, threshold, patience):
    _validate_options(
        window=window,
        check_interval=check_interval,
        threshold=threshold,
        patience=patience,
    )

    checks = []
    consecutive_hits = 0
    first_stop_index = None

    for index in range(window - 1, len(losses)):
        step = index + 1
        if (step - window) % check_interval != 0:
            continue

        hurst = dfa_hurst(losses[step - window: step])
        hit = hurst is not None and hurst <= threshold
        consecutive_hits = consecutive_hits + 1 if hit else 0
        triggered = consecutive_hits >= patience
        checks.append(
            {
                "index": index,
                "sequence_position": step,
                "loss": losses[index],
                "hurst": hurst,
                "consecutive_hits": consecutive_hits,
                "triggered_now": triggered,
                "reason": (
                    None if hurst is not None else "insufficient_signal"
                ),
            }
        )
        if triggered and first_stop_index is None:
            first_stop_index = index

    latest_check = checks[-1] if checks else None
    latest_hurst = latest_check["hurst"] if latest_check else None
    if len(losses) < window:
        reason = "insufficient_history"
    elif latest_hurst is None:
        reason = "insufficient_signal"
    else:
        reason = "ready"

    return {
        "method": "hurst",
        "available": True,
        "ready": reason == "ready",
        "reason": reason,
        "count": len(losses),
        "window": window,
        "check_interval": check_interval,
        "threshold": threshold,
        "patience": patience,
        "should_stop": first_stop_index is not None,
        "first_stop_index": first_stop_index,
        "first_stop_position": (
            first_stop_index + 1 if first_stop_index is not None else None
        ),
        "latest_hurst": latest_hurst,
        "latest_check": latest_check,
        "checks": checks,
    }
