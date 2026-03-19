"""Advanced intent detection using rules and simple patterns."""
from __future__ import annotations

import re


_MATH_PATTERN = re.compile(r'[-+/*()0-9.^% ]{3,}')


def detect_intent_advanced(question: str) -> str:
    text = question.lower().strip()
    if not text:
        return 'analysis'

    if text.startswith('python:') or text.startswith('code:'):
        return 'analysis'
    if any(token in text for token in ('who are you', 'quien eres', 'qué eres', 'what are you', 'your name')):
        return 'identity'
    if any(token in text for token in ('time', 'hora', 'qué hora', 'que hora')):
        return 'time'
    if any(token in text for token in ('date', 'fecha', 'día', 'dia', 'today')):
        return 'date'
    if any(token in text for token in ('create', 'build', 'design', 'generate', 'write', 'explica', 'explain', 'formula', 'idea')):
        return 'creation'
    if any(token in text for token in ('analyze', 'analysis', 'compare', 'why', 'por qué', 'porque', 'reason')):
        return 'analysis'
    if _MATH_PATTERN.fullmatch(text) or any(marker in text for marker in ('calculate', 'solve', 'cuánto es', 'cuanto es')):
        return 'math'
    if any(token in text for token in ('who', 'what', 'when', 'where', 'richest', 'capital', 'hombre más rico', 'hombre mas rico')):
        return 'fact'
    return 'fact' if '?' in text or len(text.split()) <= 6 else 'analysis'
