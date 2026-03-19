"""Input/output cleaning filters."""
from __future__ import annotations

import re


def clean_input(text: str) -> str:
    cleaned = str(text or '').lower().strip()
    cleaned = re.sub(r'[`~^*_#|<>\[\]{}]+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()


def clean_output(text: str) -> str:
    cleaned = re.sub(r'\s+', ' ', str(text or '')).strip()
    return cleaned[:1000]
