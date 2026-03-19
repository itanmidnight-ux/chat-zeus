"""Episode, failure, and strategy learning helpers backed by storage."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from src.storage import StorageManager


@dataclass
class StrategySnapshot:
    success_rate: float
    average_score: float
    sample_count: int


class EpisodeLearner:
    def __init__(self, storage: StorageManager):
        self.storage = storage

    def record_episode(
        self,
        *,
        question: str,
        normalized_question: str,
        intent: str,
        route: str,
        tasks: list[str],
        response_text: str,
        confidence: float,
        verification_score: float,
        quality_score: float,
        sources: list[str],
        issues: list[str],
        pattern_key: str,
        memory_hit: bool,
    ) -> None:
        payload = {
            'normalized_question': normalized_question,
            'intent': intent,
            'route': route,
            'tasks': tasks,
            'response_preview': response_text[:280],
            'confidence': confidence,
            'verification_score': verification_score,
            'quality_score': quality_score,
            'sources': sources[:6],
            'issues': issues[:6],
            'pattern_key': pattern_key,
            'memory_hit': bool(memory_hit),
        }
        self.storage.save_episode(question=question, pattern_key=pattern_key, route=route, outcome='success' if quality_score >= 0.55 else 'partial', score=quality_score, payload_json=json.dumps(payload, ensure_ascii=False))
        self.storage.update_strategy_stat(route=route, pattern_key=pattern_key, success=quality_score >= 0.55, score=quality_score, latency_ms=0.0, memory_mb=0.0)
        if issues:
            self.storage.save_failure(question=question, pattern_key=pattern_key, route=route, error_type='verification', message=';'.join(issues[:3]), payload_json=json.dumps(payload, ensure_ascii=False))

    def strategy_snapshot(self, route: str, pattern_key: str) -> StrategySnapshot:
        payload = self.storage.load_strategy_stat(route, pattern_key)
        if not payload:
            return StrategySnapshot(success_rate=0.55, average_score=0.55, sample_count=0)
        sample_count = int(payload.get('sample_count', 0))
        return StrategySnapshot(
            success_rate=float(payload.get('success_rate', 0.55)),
            average_score=float(payload.get('average_score', 0.55)),
            sample_count=sample_count,
        )

    def recent_failures(self, pattern_key: str, limit: int = 3) -> list[dict[str, Any]]:
        return self.storage.load_recent_failures(pattern_key=pattern_key, limit=limit)
