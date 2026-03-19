"""Autonomous reasoning system implementing the requested modular pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import CONFIG
from src.core.executor import TaskExecutor
from src.core.intent import detect_intent_advanced
from src.core.learning import LearningEngine
from src.core.memory import LightweightMemory
from src.engines.fact_engine import FactEngine
from src.utils.filters import clean_input, clean_output
from src.utils.handlers import handle_simple_queries


@dataclass
class AutonomousResult:
    question: str
    intent: str
    tasks: list[str]
    plan: list[dict[str, Any]]
    research: dict[str, Any]
    best_solution: dict[str, Any]
    memory: dict[str, Any]
    response_text: str


class AutonomousReasoningSystem:
    def __init__(self, *_: Any, memory_path=None, **__: Any):
        path = memory_path or (CONFIG.models_dir / 'agent_memory.json')
        self.memory_store = LightweightMemory(path)
        self.learning_engine = LearningEngine(self.memory_store, timeout=CONFIG.internet_timeout_sec)
        self.fact_engine = FactEngine(self.learning_engine)
        self.executor = TaskExecutor(self.fact_engine)

    def main_pipeline(self, question: str) -> str:
        cleaned = clean_input(question)
        direct = handle_simple_queries(cleaned)
        if direct:
            return clean_output(direct)
        intent = detect_intent_advanced(cleaned)
        result = self.executor.execute_task(intent, cleaned)
        if not result:
            result = self.learning_engine.search_and_learn(cleaned)
        if not result:
            result = 'No encontré una respuesta suficiente.'
        bucket = 'facts' if intent == 'fact' else 'solutions'
        self.memory_store.put(bucket, cleaned, clean_output(result), source='pipeline')
        return clean_output(result)

    def process(self, question: str) -> AutonomousResult:
        cleaned = clean_input(question)
        direct = handle_simple_queries(cleaned)
        intent = 'simple' if direct else detect_intent_advanced(cleaned)
        response = clean_output(direct or self.main_pipeline(cleaned))
        best_solution = {'task': intent, 'proposal': response, 'final_score': 1.0}
        return AutonomousResult(
            question=cleaned,
            intent=intent,
            tasks=[intent],
            plan=[{'step': 'main_pipeline', 'status': 'completed'}],
            research={'used_learning': intent == 'fact'},
            best_solution=best_solution,
            memory=self.memory_store.export(),
            response_text=response,
        )
