"""Auto-discover all JSONL files in a directory and output dual-signal early-stop CSV.

Usage:
  cd C:/Users/11236/Desktop/testv
  python compare.py C:/Users/11236/Desktop/exported_5_proposals_20260729_181218
"""

from __future__ import annotations

import sys
from pathlib import Path

from compare_early_stop import (
    compare_dual_signal_files,
    format_dual_comparison,
)

HURST_FRACTION_CONFIG = {
    "warmup_fraction": 0.10,
    "smoothing_fraction": 0.03,
    "window_fraction": 0.18,
    "check_interval": 1,
    "threshold": 0.80,
    "patience_fraction": 0.02,
    "observation_fraction": 0.04,
}

SNR_FRACTION_CONFIG = {
    "warmup_fraction": 0.10,
    "smoothing_fraction": 0.03,
    "window_fraction": 0.18,
    "reference_fraction": 0.11,
    "relative_threshold": 0.80,
    "patience_fraction": 0.05,
}


def discover_files(directory: str) -> list[tuple[str, Path]]:
    base = Path(directory)
    if not base.is_dir():
        raise FileNotFoundError(f"Not a directory: {base}")

    pairs: list[tuple[str, Path]] = []
    for p in sorted(base.rglob("*.jsonl")):
        name = p.stem
        if name.lower().startswith("logging_"):
            name = name[len("logging_"):]
        pairs.append((name, p))

    if not pairs:
        raise FileNotFoundError(f"No .jsonl files found under {base}")
    print(f"Found {len(pairs)} file(s): {[label for label, _ in pairs]}")
    return pairs


def csv_from_tsv(text: str) -> str:
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    for line in text.splitlines():
        if not line.strip():
            continue
        writer.writerow(line.split("\t"))
    return buf.getvalue()


def main() -> None:
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = str(Path(__file__).resolve().parent)

    files = discover_files(directory)
    rows = compare_dual_signal_files(
        files,
        hurst_fraction_config=HURST_FRACTION_CONFIG,
        snr_fraction_config=SNR_FRACTION_CONFIG,
    )
    tsv = format_dual_comparison(rows)
    csv_content = csv_from_tsv(tsv)

    out = Path(__file__).resolve().parent / "早停实验结果.csv"
    out.write_text(csv_content, encoding="utf-8-sig")
    print(f"Saved -> {out}")
    print()
    print(csv_content)


if __name__ == "__main__":
    main()
