"""Main conversational interface for clean final answers only."""
from __future__ import annotations

import gc
import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.autonomous_system import AutonomousReasoningSystem
from src.storage import StorageManager


class ChatbotInterface:
    def __init__(self, storage: StorageManager, autonomous_system: 'AutonomousReasoningSystem', background_executor, logger, *_: Any, **__: Any):
        self.storage = storage
        self.autonomous_system = autonomous_system
        self.background_executor = background_executor
        self.logger = logger

    def answer(self, question: str) -> dict[str, Any]:
        result = self.autonomous_system.process(question)
        response_text = result.response_text
        self.storage.append_response_feedback(result.intent, len(response_text.split()), 1.0)
        self.storage.save_conversation(
            question,
            response_text,
            json.dumps({'intent': result.intent, 'tasks': result.tasks}, ensure_ascii=False),
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
            self.logger.exception('Unexpected error while answering')
            self.storage.log_error('chatbot', exc.__class__.__name__, str(exc), {'question': question})
            gc.collect()
            return {'response_text': 'No pude completar la solicitud.'}
