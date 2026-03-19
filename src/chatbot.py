"""Main conversational interface and final answer builder."""
from __future__ import annotations

import gc
import json
from typing import Any

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.autonomous_system import AutonomousReasoningSystem
from src.storage import StorageManager
from src.utils import safe_error_message, sanitize_text
from src.utils.filters import clean_output


def build_final_answer(intent: str, best_solution: dict[str, Any]) -> str:
    try:
        proposal = best_solution.get('proposal', 'No se encontró una solución suficiente.')
        task = best_solution.get('task', 'prioridad principal')
        if intent == 'simple':
            return clean_output(proposal).strip()[:220]
        if intent in {'explanatory', 'explicativa'}:
            text = (
                f'La mejor explicación es centrarse en {task}. '
                f'{proposal} '
                'Primero conviene aclarar el objetivo, luego validar restricciones y finalmente elegir una opción verificable.'
            )
            return clean_output(text)
        text = (
            f'Resumen: {proposal} '
            f'Prioridad principal: {task}. '
            'Enfoque recomendado: dividir el trabajo, validar el riesgo crítico y avanzar por etapas seguras. '
            'Resultado esperado: una solución factible, eficiente y segura.'
        )
        return clean_output(text)
    except Exception:
        return 'La mejor opción es avanzar con una solución simple, segura y verificable.'


class ChatbotInterface:
    def __init__(
        self,
        storage: StorageManager,
        autonomous_system: 'AutonomousReasoningSystem',
        background_executor,
        logger,
        *_,
        **__,
    ):
        self.storage = storage
        self.autonomous_system = autonomous_system
        self.background_executor = background_executor
        self.logger = logger

    def answer(self, question: str) -> dict[str, Any]:
        result = self.autonomous_system.process(question)
        response_text = sanitize_text(result.response_text)
        self.storage.append_response_feedback(result.intent, len(response_text.split()), max(0.4, float(result.best_solution.get('final_score', 0.85))))
        self.storage.save_conversation(
            question,
            response_text,
            json.dumps({
                'intent': result.intent,
                'tasks': result.tasks,
                'best_task': result.best_solution.get('task', ''),
                'score': result.best_solution.get('final_score', result.best_solution.get('score', 0.0)),
            }, ensure_ascii=False),
        )
        gc.collect()
        return {
            'response_text': response_text,
            'question_type': result.intent,
            'tasks': result.tasks,
            'plan': result.plan,
            'best_solution': result.best_solution,
        }

    def safe_answer(self, question: str) -> dict[str, Any]:
        try:
            return self.answer(question)
        except Exception as exc:
            self.logger.exception('Error inesperado al responder')
            self.storage.log_error('chatbot', exc.__class__.__name__, str(exc), {'question': question})
            gc.collect()
            fallback = 'No pude completar el análisis, pero la mejor opción es dividir el problema, validar el riesgo principal y avanzar con una solución simple.'
            return {'response_text': sanitize_text(fallback), 'error': safe_error_message(exc)}
