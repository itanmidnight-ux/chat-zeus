"""Normalized scoring system."""
from __future__ import annotations

from src.utils import clamp


def score_solution(result: dict[str, float] | None = None, feasibility: float | None = None, efficiency: float | None = None, safety: float | None = None) -> float:
    try:
        if isinstance(result, dict):
            feasibility = float(result.get('feasibility', feasibility or 0.0))
            efficiency = float(result.get('efficiency', efficiency or 0.0))
            safety = float(result.get('safety', safety or 0.0))
        feasibility = clamp(float(feasibility or 0.0), 0.0, 1.0)
        efficiency = clamp(float(efficiency or 0.0), 0.0, 1.0)
        safety = clamp(float(safety or 0.0), 0.0, 1.0)
        return round(feasibility * 0.4 + efficiency * 0.3 + safety * 0.3, 4)
    except Exception:
        return 0.0
