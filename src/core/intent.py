"""Advanced intent detection using lightweight semantic features plus rules."""
from __future__ import annotations

from src.core.understanding import SemanticUnderstandingEngine

_UNDERSTANDING = SemanticUnderstandingEngine()


def detect_intent_advanced(question: str) -> str:
    result = _UNDERSTANDING.analyze(question)
    intent = result.selected_intent
    if intent == 'execution':
        return 'analysis'
    if intent == 'simple':
        if result.intent_scores.get('time', 0.0) >= 0.75:
            return 'time'
        if result.intent_scores.get('date', 0.0) >= 0.75:
            return 'date'
        return 'fact'
    return intent
