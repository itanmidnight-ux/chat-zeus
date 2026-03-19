"""Decision policy for selecting the best execution route under resource constraints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.understanding import UnderstandingResult


@dataclass
class DecisionPlan:
    route: str
    engine: str
    steps: list[dict[str, Any]]
    requires_verification: bool
    confidence_hint: float
    memory_budget_mb: int
    latency_budget_ms: int


class DecisionEngine:
    def __init__(self, *, max_memory_mb: int, max_external_queries: int):
        self.max_memory_mb = max_memory_mb
        self.max_external_queries = max_external_queries

    def decide(self, understanding: UnderstandingResult, hot_memory: dict[str, Any] | None = None) -> DecisionPlan:
        hot_memory = hot_memory or {}
        intent = understanding.selected_intent
        hot_hit_bonus = 0.08 if hot_memory else 0.0
        complexity = understanding.estimated_complexity
        route_scores = {
            'converse': 0.22 + (0.62 if intent == 'conversation' else 0.0),
            'clarify': 0.25 + (0.72 if intent == 'clarification_needed' else 0.0),
            'direct': 0.2 + (0.35 if intent in {'simple', 'time', 'date', 'identity'} else 0.0) + hot_hit_bonus,
            'math': 0.25 + (0.7 if intent == 'math' else 0.0),
            'retrieve': 0.2 + (0.45 if intent == 'fact' else 0.0) + (0.15 if understanding.requires_freshness else 0.0),
            'create': 0.18 + (0.42 if intent == 'creation' else 0.0),
            'analyze': 0.18 + (0.42 if intent == 'analysis' else 0.0) + complexity * 0.12,
            'execute': 0.15 + (0.75 if intent == 'execution' else 0.0),
        }
        if hot_memory and intent in {'fact', 'analysis', 'creation', 'conversation'}:
            route_scores['retrieve'] += 0.05
        if understanding.needs_multi_step:
            route_scores['analyze'] += 0.1
            route_scores['create'] += 0.05
        route = max(route_scores, key=route_scores.get)
        engine_map = {
            'converse': 'conversation',
            'clarify': 'clarification_needed',
            'direct': intent,
            'math': 'math',
            'retrieve': 'fact',
            'create': 'creation',
            'analyze': 'analysis',
            'execute': 'execution',
        }
        plan_steps = [
            {'step': 'understand', 'status': 'completed', 'detail': understanding.selected_intent},
            {'step': 'decide', 'status': 'completed', 'detail': route},
            {'step': 'execute', 'status': 'pending', 'detail': engine_map[route]},
            {'step': 'verify', 'status': 'pending', 'detail': 'enabled' if route in {'retrieve', 'analyze', 'create', 'execute', 'converse'} else 'light'},
            {'step': 'learn', 'status': 'pending', 'detail': understanding.pattern_key},
        ]
        memory_budget_mb = max(64, min(self.max_memory_mb, int(96 + complexity * 128)))
        latency_budget_ms = 450 if route in {'direct', 'math', 'converse', 'clarify'} else 1600 if route == 'retrieve' else 2200
        confidence_hint = round(min(0.98, route_scores[route]), 3)
        return DecisionPlan(
            route=route,
            engine=engine_map[route],
            steps=plan_steps,
            requires_verification=route in {'retrieve', 'analyze', 'create', 'execute'},
            confidence_hint=confidence_hint,
            memory_budget_mb=memory_budget_mb,
            latency_budget_ms=latency_budget_ms,
        )
