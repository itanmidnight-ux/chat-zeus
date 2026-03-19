"""Reality handler with immediate exit for trivial real-world queries."""
from __future__ import annotations

from datetime import datetime

from src.utils.filters import clean_input

_GREETINGS = {'hola', 'buenas', 'hello', 'hi', 'hey'}


def handle_simple_queries(question: str) -> str | None:
    try:
        text = clean_input(question).lower()
        now = datetime.now()
        if text in _GREETINGS:
            return 'Hola. ¿En qué puedo ayudarte?'
        if 'time' in text or 'hora' in text:
            return f'Son las {now.strftime("%H:%M")} aproximadamente.'
        if 'date' in text or 'fecha' in text:
            return f'Hoy es {now.strftime("%d/%m/%Y")}. '
        return None
    except Exception:
        return None
