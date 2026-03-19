"""Immediate handler for simple real-world queries."""
from __future__ import annotations

from datetime import datetime

_GREETINGS = {'hola', 'buenas', 'hello', 'hi', 'hey'}


def handle_simple_queries(question: str) -> str | None:
    text = question.strip().lower()
    now = datetime.now()
    if text in _GREETINGS:
        return 'Hola. ¿En qué puedo ayudarte?'
    if any(token in text for token in ('qué hora', 'que hora', 'hora', 'time')):
        return f"Son las {now.strftime('%H:%M:%S')}."
    if any(token in text for token in ('fecha', 'date', 'día', 'dia')):
        return f"La fecha es {now.strftime('%Y-%m-%d')}."
    return None
