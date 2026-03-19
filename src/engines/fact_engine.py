"""Fact engine backed by lightweight memory and internet learning."""
from __future__ import annotations

from src.core.learning import LearningEngine


class FactEngine:
    def __init__(self, learning_engine: LearningEngine):
        self.learning_engine = learning_engine

    def search_fact(self, question: str) -> str:
        return self.learning_engine.search_fact(question)

    def search_and_learn(self, question: str) -> str:
        return self.learning_engine.search_and_learn(question)
