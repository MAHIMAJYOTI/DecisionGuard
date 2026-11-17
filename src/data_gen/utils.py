"""Small helpers shared by the generators.

Kept here so each extract file can read from top to bottom without a
math detour in the middle.
"""

from __future__ import annotations

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Squash any number into a probability-shaped score between 0 and 1.

    Clipped first so a huge input cannot overflow `exp`.
    """

    clipped = np.clip(values, -20, 20)
    return 1.0 / (1.0 + np.exp(-clipped))


def round_money(values: np.ndarray) -> np.ndarray:
    """Rupees and paise. Rounding here keeps CSV and Parquet identical."""

    return np.round(values.astype(float), 2)


def customer_ids(start_at: int, count: int) -> np.ndarray:
    """CUST0000001, CUST0000002, ... never CUST0000000 (that id is the broken key)."""

    numbers = np.arange(start_at, start_at + count)
    return np.array([f"CUST{number:07d}" for number in numbers])


def scaled_defect_count(full_count: int, n_rows: int, full_rows: int) -> int:
    """Scale a defect count from the full portfolio down to the file in hand.

    The full 50,000-row customer file gets exactly `full_count` bad rows.
    A 1,500-row practice file gets about 1,500/50,000 of that. Files of at
    least 30 rows still get one bad row when the scaled count would round to
    zero, so a demo always shows the defect.
    """

    count = int(round(full_count * n_rows / full_rows))
    if n_rows >= 30:
        return max(1, count)
    return max(0, count)


def split_budget(total: int, shares: dict[str, float]) -> dict[str, int]:
    """Turn percentages into integer row counts that add up to `total` exactly."""

    names = list(shares)
    counts = [int(round(total * shares[name])) for name in names]
    counts[-1] += total - sum(counts)
    if counts[-1] < 0:
        donor = max(range(len(counts) - 1), key=lambda index: counts[index])
        counts[donor] += counts[-1]
        counts[-1] = 0
    return dict(zip(names, counts, strict=True))
