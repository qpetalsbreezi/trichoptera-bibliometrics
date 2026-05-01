"""Display rounding for bibliometric tables: one decimal for ratios and percentages."""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from numbers import Integral


def round_one_decimal(value: float) -> float:
    """Round to one decimal using half-away-from-zero (stable for ties at *.05)."""
    if value != value:  # NaN
        return value
    d = Decimal(str(float(value))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return float(d)


def fmt_integerish(value) -> str:
    """Whole-number counts and years (no trailing .0)."""
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value or not math.isfinite(value):
            return ""
        x = float(value)
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{round_one_decimal(x):.1f}"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return str(int(value))
    return str(value)


def fmt_ratio_or_pct(value) -> str:
    """Ratios and percentages: always one decimal (mentor convention)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        if value != value or not math.isfinite(value):
            return ""
        return f"{round_one_decimal(float(value)):.1f}"
    if isinstance(value, Integral):
        return f"{round_one_decimal(float(int(value))):.1f}"
    return str(value)
