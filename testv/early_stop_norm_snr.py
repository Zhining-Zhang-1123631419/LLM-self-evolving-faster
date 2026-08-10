from __future__ import annotations

import json
from pathlib import Path
import statistics

from smoothing import causal_ewma


def read_records(file_path):
    """Read step, loss, token accuracy, and gradient norm from JSONL."""
    records = []
    with Path(file_path).open("r", encoding="utf-8-sig") as stream:
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
            if not isinstance(record, dict) or "loss" not in record:
                continue

            required = ("grad_norm", "token_acc", "global_step/max_steps")
            missing = [field for field in required if field not in record]
            if missing:
                raise ValueError(
                    f"Line {line_number}: missing {', '.join(missing)}"
                )
            step_field = record["global_step/max_steps"]
            if not isinstance(step_field, str) or "/" not in step_field:
                raise ValueError(
                    f"Line {line_number}: invalid 'global_step/max_steps'"
                )
            try:
                step_text, max_steps_text = step_field.split("/", 1)
                records.append(
                    {
                        "step": int(step_text),
                        "max_steps": int(max_steps_text),
                        "loss": float(record["loss"]),
                        "token_acc": float(record["token_acc"]),
                        "grad_norm": float(record["grad_norm"]),
                    }
                )
            except (TypeError, ValueError):
                raise ValueError(
                    f"Line {line_number}: invalid metric value"
                ) from None

    if not records:
        raise ValueError(f"No training records found in {file_path}")
    return records


def analyze_norm_snr(records, *, warmup_fraction=0.10, window=100,
                     reference_points=20, relative_threshold=0.20,
                     patience=10, smoothing_span=1):
    """Compute rolling gradient-norm SNR and predict the first cutoff."""
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if window < 2:
        raise ValueError("window must be at least 2")
    if reference_points < 2:
        raise ValueError("reference_points must be at least 2")
    if relative_threshold <= 0:
        raise ValueError("relative_threshold must be positive")
    if patience < 1:
        raise ValueError("patience must be positive")

    max_steps = int(records[0]["max_steps"])
    if any(int(record["max_steps"]) != max_steps for record in records):
        raise ValueError("Inconsistent maximum step count")

    warmup_end_step = max_steps * warmup_fraction
    smoothed_gradient_norms = causal_ewma(
        [float(record["grad_norm"]) for record in records],
        span=smoothing_span,
    )
    included = [
        {
            **record,
            "smoothed_grad_norm": smoothed_gradient_norms[index],
        }
        for index, record in enumerate(records)
        if int(record["step"]) > warmup_end_step
    ]
    if len(included) < window + reference_points:
        raise ValueError(
            "Not enough post-warmup records for the window and reference"
        )

    values = []
    for index in range(window - 1, len(included)):
        gradient_norms = [
            float(record["smoothed_grad_norm"])
            for record in included[index - window + 1: index + 1]
        ]
        mean = statistics.fmean(gradient_norms)
        variance = statistics.fmean(
            (value - mean) ** 2 for value in gradient_norms
        )
        norm_snr = mean ** 2 / (variance + 1e-12)
        values.append(
            {
                "step": int(included[index]["step"]),
                "norm_snr": norm_snr,
            }
        )

    reference_values = [
        float(value["norm_snr"])
        for value in values[:reference_points]
    ]
    reference = statistics.quantiles(
        reference_values,
        n=10,
        method="inclusive",
    )[8]
    raw_threshold = reference * relative_threshold

    consecutive_hits = 0
    first_stop_step = None
    for value in values[reference_points:]:
        ratio = float(value["norm_snr"]) / reference
        value["relative_norm_snr"] = ratio
        consecutive_hits = (
            consecutive_hits + 1
            if ratio < relative_threshold
            else 0
        )
        value["consecutive_hits"] = consecutive_hits
        if consecutive_hits >= patience:
            first_stop_step = int(value["step"])
            break

    return {
        "max_steps": max_steps,
        "warmup_fraction": warmup_fraction,
        "warmup_end_step": warmup_end_step,
        "window": window,
        "reference_points": reference_points,
        "reference": reference,
        "relative_threshold": relative_threshold,
        "raw_threshold": raw_threshold,
        "patience": patience,
        "smoothing_span": smoothing_span,
        "first_stop_step": first_stop_step,
        "values": values,
    }
