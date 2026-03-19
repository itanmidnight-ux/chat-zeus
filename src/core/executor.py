"""Task execution engine for the autonomous reasoning agent."""
from __future__ import annotations

from datetime import datetime

from src.engines.creation_engine import CreationEngine
from src.engines.fact_engine import FactEngine
from src.engines.math_engine import solve_math
from src.sandbox.executor import execute_code_safely


class TaskExecutor:
    def __init__(self, fact_engine: FactEngine):
        self.fact_engine = fact_engine
        self.creation_engine = CreationEngine()

    def get_time(self) -> str:
        return datetime.now().strftime('%H:%M:%S')

    def get_date(self) -> str:
        return datetime.now().strftime('%Y-%m-%d')

    def execute_task(self, intent: str, question: str, *, context: dict[str, str] | None = None) -> str:
        if intent == 'math':
            return solve_math(question)
        if intent == 'time':
            return self.get_time()
        if intent == 'date':
            return self.get_date()
        if intent == 'fact':
            return self.fact_engine.search_fact(question)
        if intent in {'creation', 'analysis'}:
            return self.creation_engine.build_solution(question, context=context)
        if intent == 'identity':
            name = context.get('assistant_name', 'Chat Zeus') if context else 'Chat Zeus'
            return f'Soy {name}, un agente conversacional modular con memoria de sesión, razonamiento y ejecución controlada.'
        if intent == 'conversation':
            if context and context.get('name'):
                return f"Hola, {context['name']}. Sigo el contexto de esta sesión y puedo ayudarte a razonar, crear o resolver tareas."
            return 'Hola. Puedo conversar, resolver problemas, pedir aclaraciones y ejecutar tareas de forma controlada.'
        if intent == 'execution' or (intent == 'analysis' and question.lower().startswith('python:')):
            code = question.split(':', 1)[1] if ':' in question else question
            return execute_code_safely(code)
        return self.fact_engine.search_and_learn(question)
