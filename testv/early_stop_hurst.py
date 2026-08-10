from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jsonl_hurst import _analyze
from smoothing import causal_ewma


@dataclass(frozen=True)
class _LossRecord:
    original_index: int
    loss: float
    global_step: int
    max_steps: int


def _parse_step(value, *, line_number):
    if not isinstance(value, str) or "/" not in value:
        raise ValueError(
            f"Line {line_number}: invalid 'global_step/max_steps' field"
        )
    current_text, maximum_text = value.split("/", 1)
    try:
        current = int(current_text)
        maximum = int(maximum_text)
    except ValueError:
        raise ValueError(
            f"Line {line_number}: invalid 'global_step/max_steps' field"
        ) from None
    if current < 0 or maximum < 1 or current > maximum:
        raise ValueError(
            f"Line {line_number}: invalid 'global_step/max_steps' field"
        )
    return current, maximum


def _read_records(file_path, *, limit=None):
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    path = Path(file_path)
    records = []
    skipped_lines = []
    expected_max_steps = None

    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Line {line_number}: invalid JSON: {error.msg}"
                ) from None

            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number}: expected a JSON object")
            if "loss" not in value:
                skipped_lines.append(line_number)
                continue

            loss_value = value["loss"]
            if isinstance(loss_value, bool):
                raise ValueError(
                    f"Line {line_number}: loss must be a finite number"
                )
            try:
                loss = float(loss_value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Line {line_number}: loss must be a finite number"
                ) from None
            if not np.isfinite(loss):
                raise ValueError(
                    f"Line {line_number}: loss must be a finite number"
                )

            global_step, max_steps = _parse_step(
                value.get("global_step/max_steps"),
                line_number=line_number,
            )
            if expected_max_steps is None:
                expected_max_steps = max_steps
            elif max_steps != expected_max_steps:
                raise ValueError(
                    f"Line {line_number}: inconsistent maximum step count"
                )

            records.append(
                _LossRecord(
                    original_index=len(records),
                    loss=loss,
                    global_step=global_step,
                    max_steps=max_steps,
                )
            )
            if limit is not None and len(records) >= limit:
                break

    if not records:
        raise ValueError(f"No loss records found in {path}")
    return records, skipped_lines


def _analyze_records(records, *, warmup_fraction, window, check_interval,
                     threshold, patience, smoothing_span):
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")

    max_steps = records[0].max_steps
    warmup_end_step = max_steps * warmup_fraction
    smoothed_losses = causal_ewma(
        [record.loss for record in records],
        span=smoothing_span,
    )
    included = [
        (record, smoothed_losses[index])
        for index, record in enumerate(records)
        if record.global_step > warmup_end_step
    ]
    base = _analyze(
        [loss for _, loss in included],
        window=window,
        check_interval=check_interval,
        threshold=threshold,
        patience=patience,
    )

    for check in base["checks"]:
        record = included[check["index"]][0]
        check["analysis_index"] = check["index"]
        check["index"] = record.original_index
        check["global_step"] = record.global_step

    relative_stop_index = base["first_stop_index"]
    if relative_stop_index is None:
        original_stop_index = None
        stop_global_step = None
    else:
        stop_record = included[relative_stop_index][0]
        original_stop_index = stop_record.original_index
        stop_global_step = stop_record.global_step

    base.update(
        {
            "total_loss_records": len(records),
            "warmup_fraction": warmup_fraction,
            "warmup_end_step": warmup_end_step,
            "warmup_skipped_records": len(records) - len(included),
            "analysis_count": len(included),
            "smoothing_span": smoothing_span,
            "first_stop_analysis_index": relative_stop_index,
            "first_stop_index": original_stop_index,
            "first_stop_position": (
                original_stop_index + 1
                if original_stop_index is not None
                else None
            ),
            "first_stop_global_step": stop_global_step,
            "latest_check": base["checks"][-1] if base["checks"] else None,
        }
    )
    if records[-1].global_step <= warmup_end_step:
        base["ready"] = False
        base["reason"] = "warmup"
    return base


def analyze_sequence(file_path, *, warmup_fraction=0.10, window=50,
                     check_interval=1, threshold=0.51, patience=5,
                     smoothing_span=1):
    """Analyze a full log after excluding the first 10% of training steps."""
    records, skipped_lines = _read_records(file_path)
    result = _analyze_records(
        records,
        warmup_fraction=warmup_fraction,
        window=window,
        check_interval=check_interval,
        threshold=threshold,
        patience=patience,
        smoothing_span=smoothing_span,
    )
    result.update(
        {
            "source": str(Path(file_path)),
            "skipped_lines": skipped_lines,
        }
    )
    return result
