"""Sistema de puntuación multiobjetivo."""
from __future__ import annotations

from src.utils import clamp


def score_solution(feasibility: float, efficiency: float, safety: float) -> float:
    feasibility = clamp(feasibility, 0.0, 1.0)
    efficiency = clamp(efficiency, 0.0, 1.0)
    safety = clamp(safety, 0.0, 1.0)
    return round(feasibility * 0.4 + efficiency * 0.25 + safety * 0.35, 4)
