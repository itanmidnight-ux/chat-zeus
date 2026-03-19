"""Input/output filters for the cognitive system."""
from __future__ import annotations

import re

from src.utils import sanitize_text

_FORBIDDEN_OUTPUT = ["simulation", "ml", "rag", "delta-v", "debug", "log", "checkpoint"]


def clean_input(text: str) -> str:
    try:
        cleaned = sanitize_text(text)
        cleaned = re.sub(r'[`~^*_#|<>]+', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
    except Exception:
        return str(text).strip()


def clean_output(text: str) -> str:
    try:
        cleaned = sanitize_text(text)
        for token in _FORBIDDEN_OUTPUT:
            cleaned = re.sub(rf'\b{re.escape(token)}\b', ' ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'https?://\S+|www\.\S+', ' ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip(' .\n\t')
    except Exception:
        return 'No pude generar una respuesta clara, pero la mejor opción es avanzar con una solución segura y simple.'
