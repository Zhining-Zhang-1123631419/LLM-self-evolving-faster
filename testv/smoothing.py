from __future__ import annotations


def causal_ewma(values, *, span):
    """Return an EWMA in which each value uses only the observed prefix."""
    if span < 1:
        raise ValueError("span must be positive")

    sequence = [float(value) for value in values]
    if not sequence:
        return []

    alpha = 2.0 / (span + 1.0)
    smoothed = [sequence[0]]
    for value in sequence[1:]:
        smoothed.append(alpha * value + (1.0 - alpha) * smoothed[-1])
    return smoothed
