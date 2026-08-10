from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
import statistics
from typing import Any

from early_stop_hurst import analyze_sequence as analyze_hurst
from early_stop_hurst_cli import apply_two_stage_rule
from early_stop_norm_snr import analyze_norm_snr, read_records


@dataclass
class DualSignalStopState:
    """Remember the first Hurst and SNR flags until both have appeared."""

    hurst_flag_step: int | None = None
    snr_flag_step: int | None = None
    cutoff_step: int | None = None

    def update(self, *, step, hurst_triggered=False, snr_triggered=False):
        if hurst_triggered and self.hurst_flag_step is None:
            self.hurst_flag_step = step
        if snr_triggered and self.snr_flag_step is None:
            self.snr_flag_step = step
        if (
            self.cutoff_step is None
            and self.hurst_flag_step is not None
            and self.snr_flag_step is not None
        ):
            self.cutoff_step = step
        return self.cutoff_step is not None


def _validate_fraction(name, value):
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"{name} must be in (0, 1]")
    return fraction


def _read_training_scale(jsonl_path):
    steps = []
    max_steps = None
    with Path(jsonl_path).open("r", encoding="utf-8-sig") as stream:
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

            field = record.get("global_step/max_steps")
            if not isinstance(field, str) or "/" not in field:
                raise ValueError(
                    f"Line {line_number}: invalid 'global_step/max_steps' field"
                )
            try:
                step_text, maximum_text = field.split("/", 1)
                step = int(step_text)
                maximum = int(maximum_text)
            except ValueError:
                raise ValueError(
                    f"Line {line_number}: invalid 'global_step/max_steps' field"
                ) from None
            if max_steps is None:
                max_steps = maximum
            elif maximum != max_steps:
                raise ValueError(
                    f"Line {line_number}: inconsistent maximum step count"
                )
            steps.append(step)

    if max_steps is None or not steps:
        raise ValueError(f"No training records found in {jsonl_path}")
    positive_differences = [
        current - previous
        for previous, current in zip(steps, steps[1:])
        if current > previous
    ]
    record_interval = (
        float(statistics.median(positive_differences))
        if positive_differences
        else 1.0
    )
    return max_steps, record_interval


def _fraction_to_points(*, max_steps, fraction, step_interval, minimum):
    return max(
        minimum,
        math.ceil(max_steps * fraction / step_interval),
    )


def resolve_fraction_config(jsonl_path, *, method, fraction_config):
    """Convert training-step percentages into logged-record counts."""
    if method not in {"hurst", "snr"}:
        raise ValueError("method must be 'hurst' or 'snr'")

    max_steps, record_interval = _read_training_scale(jsonl_path)
    warmup_fraction = float(fraction_config["warmup_fraction"])
    window_fraction = _validate_fraction(
        "window_fraction",
        fraction_config["window_fraction"],
    )
    window = _fraction_to_points(
        max_steps=max_steps,
        fraction=window_fraction,
        step_interval=record_interval,
        minimum=16 if method == "hurst" else 2,
    )
    smoothing_span = 1
    if "smoothing_fraction" in fraction_config:
        smoothing_span = _fraction_to_points(
            max_steps=max_steps,
            fraction=_validate_fraction(
                "smoothing_fraction",
                fraction_config["smoothing_fraction"],
            ),
            step_interval=record_interval,
            minimum=3,
        )

    if method == "hurst":
        check_interval = int(fraction_config["check_interval"])
        if check_interval < 1:
            raise ValueError("check_interval must be positive")
        check_step_interval = record_interval * check_interval
        return {
            "warmup_fraction": warmup_fraction,
            "smoothing_span": smoothing_span,
            "window": window,
            "check_interval": check_interval,
            "threshold": float(fraction_config["threshold"]),
            "patience": _fraction_to_points(
                max_steps=max_steps,
                fraction=_validate_fraction(
                    "patience_fraction",
                    fraction_config["patience_fraction"],
                ),
                step_interval=check_step_interval,
                minimum=1,
            ),
            "observation_points": _fraction_to_points(
                max_steps=max_steps,
                fraction=_validate_fraction(
                    "observation_fraction",
                    fraction_config["observation_fraction"],
                ),
                step_interval=check_step_interval,
                minimum=1,
            ),
        }

    return {
        "warmup_fraction": warmup_fraction,
        "smoothing_span": smoothing_span,
        "window": window,
        "reference_points": _fraction_to_points(
            max_steps=max_steps,
            fraction=_validate_fraction(
                "reference_fraction",
                fraction_config["reference_fraction"],
            ),
            step_interval=record_interval,
            minimum=2,
        ),
        "relative_threshold": float(
            fraction_config["relative_threshold"]
        ),
        "patience": _fraction_to_points(
            max_steps=max_steps,
            fraction=_validate_fraction(
                "patience_fraction",
                fraction_config["patience_fraction"],
            ),
            step_interval=record_interval,
            minimum=1,
        ),
    }


def _parse_step(value, *, line_number):
    if not isinstance(value, str) or "/" not in value:
        raise ValueError(
            f"Line {line_number}: invalid 'global_step/max_steps' field"
        )
    try:
        return int(value.split("/", 1)[0])
    except ValueError:
        raise ValueError(
            f"Line {line_number}: invalid 'global_step/max_steps' field"
        ) from None


def find_eval_at_or_after(jsonl_path, *, cutoff_step):
    """Return the first validation record at or after the cutoff step."""
    if cutoff_step is None:
        return None

    nearest = None
    with Path(jsonl_path).open("r", encoding="utf-8-sig") as stream:
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
            if not isinstance(record, dict) or "eval_loss" not in record:
                continue

            step = _parse_step(
                record.get("global_step/max_steps"),
                line_number=line_number,
            )
            if step < cutoff_step:
                continue
            try:
                candidate = {
                    "eval_step": step,
                    "eval_loss": float(record["eval_loss"]),
                    "eval_token_acc": float(record["eval_token_acc"]),
                }
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    f"Line {line_number}: invalid evaluation metric"
                ) from None
            if nearest is None or step < int(nearest["eval_step"]):
                nearest = candidate
    return nearest


def find_final_eval(jsonl_path):
    """Return the validation record with the greatest global step."""
    latest = None
    with Path(jsonl_path).open("r", encoding="utf-8-sig") as stream:
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
            if not isinstance(record, dict) or "eval_loss" not in record:
                continue

            step = _parse_step(
                record.get("global_step/max_steps"),
                line_number=line_number,
            )
            try:
                candidate = {
                    "eval_step": step,
                    "eval_loss": float(record["eval_loss"]),
                    "eval_token_acc": float(record["eval_token_acc"]),
                }
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    f"Line {line_number}: invalid evaluation metric"
                ) from None
            if latest is None or step > int(latest["eval_step"]):
                latest = candidate
    return latest


def fuse_validation_losses(first_loss, second_loss):
    """Give 30% weight to the first flag and 70% to the second."""
    return 0.3 * float(first_loss) + 0.7 * float(second_loss)


def _hurst_cutoff(path, config):
    options = dict(config)
    observation_points = int(options.pop("observation_points"))
    result = analyze_hurst(path, **options)
    result = apply_two_stage_rule(
        result,
        threshold=float(options["threshold"]),
        consecutive=int(options["patience"]),
        observation_points=observation_points,
    )
    return result["first_stop_global_step"]


def _snr_cutoff(path, config):
    result = analyze_norm_snr(read_records(path), **config)
    return result["first_stop_step"]


def compare_files(files, *, method, config):
    """Apply one early-stop configuration to several JSONL logs."""
    if method not in {"hurst", "snr"}:
        raise ValueError("method must be 'hurst' or 'snr'")

    rows = []
    for model, file_path in files:
        path = Path(file_path)
        cutoff_step = (
            _hurst_cutoff(path, config)
            if method == "hurst"
            else _snr_cutoff(path, config)
        )
        evaluation = find_eval_at_or_after(path, cutoff_step=cutoff_step)
        row = {
            "model": model,
            "method": method,
            "cutoff_step": cutoff_step,
            "eval_step": None,
            "eval_loss": None,
            "eval_token_acc": None,
        }
        if evaluation is not None:
            row.update(evaluation)
        rows.append(row)
    return rows


def compare_dual_signal_files(files, *, hurst_fraction_config,
                              snr_fraction_config):
    """Stop each model after both its Hurst and SNR flags have appeared."""
    rows = []
    for model, file_path in files:
        path = Path(file_path)
        hurst_config = resolve_fraction_config(
            path,
            method="hurst",
            fraction_config=hurst_fraction_config,
        )
        snr_config = resolve_fraction_config(
            path,
            method="snr",
            fraction_config=snr_fraction_config,
        )
        hurst_step = _hurst_cutoff(path, hurst_config)
        snr_step = _snr_cutoff(path, snr_config)

        state = DualSignalStopState()
        events = [
            (step, method)
            for step, method in (
                (hurst_step, "hurst"),
                (snr_step, "snr"),
            )
            if step is not None
        ]
        ordered_events = sorted(events)
        event_evaluations = []
        for step, event_method in ordered_events:
            state.update(
                step=int(step),
                hurst_triggered=event_method == "hurst",
                snr_triggered=event_method == "snr",
            )
            event_evaluations.append(
                find_eval_at_or_after(path, cutoff_step=int(step))
            )

        first_evaluation = (
            event_evaluations[0] if event_evaluations else None
        )
        second_evaluation = (
            event_evaluations[1] if len(event_evaluations) > 1 else None
        )
        final_evaluation = find_final_eval(path)
        fused_eval_loss = None
        loss_gap = None
        loss_gap_percent = None
        if first_evaluation is not None and second_evaluation is not None:
            fused_eval_loss = fuse_validation_losses(
                float(first_evaluation["eval_loss"]),
                float(second_evaluation["eval_loss"]),
            )
        if second_evaluation is not None and final_evaluation is not None:
            second_loss = float(second_evaluation["eval_loss"])
            loss_gap = second_loss - float(final_evaluation["eval_loss"])
            if second_loss != 0.0:
                loss_gap_percent = 100.0 * loss_gap / second_loss

        row = {
            "model": model,
            "hurst_flag_step": state.hurst_flag_step,
            "snr_flag_step": state.snr_flag_step,
            "cutoff_step": state.cutoff_step,
            "eval_step": None,
            "eval_loss": None,
            "eval_token_acc": None,
            "first_eval_step": None,
            "first_eval_loss": None,
            "second_eval_step": None,
            "second_eval_loss": None,
            "fused_eval_loss": fused_eval_loss,
            "final_eval_step": (
                final_evaluation["eval_step"]
                if final_evaluation is not None
                else None
            ),
            "final_eval_loss": (
                final_evaluation["eval_loss"]
                if final_evaluation is not None
                else None
            ),
            "loss_gap": loss_gap,
            "loss_gap_percent": loss_gap_percent,
        }
        if first_evaluation is not None:
            row["first_eval_step"] = first_evaluation["eval_step"]
            row["first_eval_loss"] = first_evaluation["eval_loss"]
        if second_evaluation is not None:
            row["second_eval_step"] = second_evaluation["eval_step"]
            row["second_eval_loss"] = second_evaluation["eval_loss"]
            row.update(second_evaluation)
        rows.append(row)
    return rows


def _format_value(value, *, digits=None):
    if value is None:
        return "-"
    if digits is not None:
        return f"{float(value):.{digits}f}"
    return str(value)


def format_comparison(rows):
    """Format early-stop results from several models as a TSV table."""
    lines = [
        "model\tmethod\tcutoff_step\teval_step\teval_loss\teval_token_acc"
    ]
    for row in rows:
        cutoff = row.get("cutoff_step")
        lines.append(
            "\t".join(
                [
                    str(row["model"]),
                    str(row["method"]),
                    (
                        "not_triggered"
                        if cutoff is None
                        else str(cutoff)
                    ),
                    _format_value(row.get("eval_step")),
                    _format_value(row.get("eval_loss"), digits=8),
                    _format_value(row.get("eval_token_acc"), digits=8),
                ]
            )
        )
    return "\n".join(lines)


def format_dual_comparison(rows):
    """Format the two remembered flags and their combined cutoff."""
    lines = [
        (
            "模型\tHurst标志步\tSNR标志步\t截停步\t"
            "首次评估步\t首次评估loss\t二次评估步\t"
            "二次评估loss\t融合loss\t最终评估步\t"
            "最终loss\tloss差值\t后续改善\t"
            "token准确率"
        )
    ]
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            row.get("fused_eval_loss") is None,
            (
                float(row["fused_eval_loss"])
                if row.get("fused_eval_loss") is not None
                else math.inf
            ),
        ),
    )
    for row in ordered_rows:
        cutoff = row.get("cutoff_step")
        lines.append(
            "\t".join(
                [
                    str(row["model"]),
                    _format_value(row.get("hurst_flag_step")),
                    _format_value(row.get("snr_flag_step")),
                    (
                        "not_triggered"
                        if cutoff is None
                        else str(cutoff)
                    ),
                    _format_value(row.get("first_eval_step")),
                    _format_value(row.get("first_eval_loss"), digits=8),
                    _format_value(row.get("second_eval_step")),
                    _format_value(row.get("second_eval_loss"), digits=8),
                    _format_value(row.get("fused_eval_loss"), digits=8),
                    _format_value(row.get("final_eval_step")),
                    _format_value(row.get("final_eval_loss"), digits=8),
                    _format_value(row.get("loss_gap"), digits=8),
                    _format_value(row.get("loss_gap_percent"), digits=8),
                    _format_value(
                        row.get("eval_token_acc"),
                        digits=8,
                    ),
                ]
            )
        )
    return "\n".join(lines)
