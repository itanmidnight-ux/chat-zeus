"""Filtros agresivos para la salida final."""
from __future__ import annotations

from src.response_control import clean_output
from src.utils import sanitize_text

BLACKLIST = ['delta-v', 'simulación', 'rag', 'ml', 'checkpoint', 'arxiv', 'debug']


def aggressive_filter(text: str) -> str:
    cleaned = clean_output(sanitize_text(text))
    lowered = cleaned
    for term in BLACKLIST:
        lowered = lowered.replace(term, '').replace(term.capitalize(), '').replace(term.upper(), '')
    return sanitize_text(lowered)
