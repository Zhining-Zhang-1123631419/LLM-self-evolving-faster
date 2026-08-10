from __future__ import annotations

import numpy as np


def gini_coefficient(values):
    """Return the Gini coefficient of non-negative values."""
    x = np.asarray(list(values), dtype=np.float64)
    if x.size == 0:
        raise ValueError("Gini requires at least one value")
    if not np.all(np.isfinite(x)):
        raise ValueError("Gini values must be finite")
    if np.any(x < 0):
        raise ValueError("Gini values must be non-negative")

    total = float(x.sum())
    if total == 0.0:
        return 0.0

    x.sort()
    n = x.size
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float(np.sum((2.0 * ranks - n - 1.0) * x) / (n * total))


def linear_slope(values):
    """Return the least-squares slope against equally spaced steps."""
    y = np.asarray(list(values), dtype=np.float64)
    if y.size < 2:
        raise ValueError("Slope requires at least two values")
    if not np.all(np.isfinite(y)):
        raise ValueError("Slope values must be finite")

    x = np.arange(y.size, dtype=np.float64)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    return float(np.dot(x_centered, y - y.mean()) / denominator)


def dfa_hurst(values):
    """Estimate the DFA scaling exponent for a one-dimensional series."""
    x = np.asarray(list(values), dtype=np.float64)
    if x.size < 16 or not np.all(np.isfinite(x)):
        return None
    if np.allclose(x, x[0]):
        return None

    profile = np.cumsum(x - x.mean())
    candidates = np.array([4, 5, 6, 8, 10, 12, 16, 20, 25, 32], dtype=int)
    scales = candidates[candidates <= x.size // 2]
    fluctuations = []
    valid_scales = []

    for scale in scales:
        segments = x.size // scale
        if segments < 2:
            continue

        residual_energy = []
        positions = np.arange(scale, dtype=np.float64)
        for offset in (0, x.size - segments * scale):
            for segment in range(segments):
                start = offset + segment * scale
                block = profile[start: start + scale]
                coefficients = np.polyfit(positions, block, 1)
                residual = block - np.polyval(coefficients, positions)
                residual_energy.append(float(np.mean(residual ** 2)))

        fluctuation = float(np.sqrt(np.mean(residual_energy)))
        if fluctuation > 0.0 and np.isfinite(fluctuation):
            valid_scales.append(int(scale))
            fluctuations.append(fluctuation)

    if len(valid_scales) < 3:
        return None

    exponent = np.polyfit(
        np.log(np.asarray(valid_scales, dtype=np.float64)),
        np.log(np.asarray(fluctuations, dtype=np.float64)),
        1,
    )[0]
    return float(exponent) if np.isfinite(exponent) else None
