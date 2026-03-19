"""Reality handler para preguntas triviales y directas."""
from __future__ import annotations

from datetime import datetime

from src.utils import sanitize_text

_SIMPLE_GREETINGS = {'hola', 'buenas', 'hello', 'hi', 'hey'}


def handle_simple_queries(question: str) -> str | None:
    text = sanitize_text(question).lower()
    now = datetime.now()
    if text in _SIMPLE_GREETINGS:
        return 'Hola. ¿En qué puedo ayudarte?'
    if 'hora' in text:
        return f'Son las {now.strftime("%H:%M")} aproximadamente.'
    if 'fecha' in text:
        return f'Hoy es {now.strftime("%d/%m/%Y")}. '
    return None
