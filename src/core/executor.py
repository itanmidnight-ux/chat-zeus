"""Task execution engine for the autonomous reasoning agent."""
from __future__ import annotations

from datetime import datetime

from src.engines.creation_engine import generate_solution
from src.engines.fact_engine import FactEngine
from src.engines.math_engine import solve_math
from src.sandbox.executor import execute_code_safely


class TaskExecutor:
    def __init__(self, fact_engine: FactEngine):
        self.fact_engine = fact_engine

    def get_time(self) -> str:
        return datetime.now().strftime('%H:%M:%S')

    def get_date(self) -> str:
        return datetime.now().strftime('%Y-%m-%d')

    def execute_task(self, intent: str, question: str) -> str:
        if intent == 'math':
            return solve_math(question)
        if intent == 'time':
            return self.get_time()
        if intent == 'date':
            return self.get_date()
        if intent == 'fact':
            return self.fact_engine.search_fact(question)
        if intent == 'creation':
            return generate_solution(question)
        if intent == 'identity':
            return 'Soy un asistente autónomo de razonamiento y ejecución seguro.'
        if intent == 'execution' or (intent == 'analysis' and question.lower().startswith('python:')):
            code = question.split(':', 1)[1] if ':' in question else question
            return execute_code_safely(code)
        if intent == 'analysis':
            return generate_solution(question)
        return self.fact_engine.search_and_learn(question)
