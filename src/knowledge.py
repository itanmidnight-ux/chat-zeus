"""Gestor de conocimiento local con búsqueda sencilla tipo RAG."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import CONFIG
from src.storage import StorageManager
from src.utils import sanitize_text


@dataclass
class KnowledgeResult:
    snippets: list[dict[str, Any]]
    summary: str
    formulas: list[str]


class KnowledgeManager:
    def __init__(self, storage: StorageManager):
        self.storage = storage

    def retrieve(self, question: str, recent_context: list[dict[str, Any]] | None = None) -> KnowledgeResult:
        recent_context = recent_context or []
        context_terms = ' '.join(item.get('question', '') for item in recent_context[:3])
        query = f"{question} {context_terms}".strip()
        snippets = self.storage.search_knowledge(query)
        if not snippets:
            return KnowledgeResult([], 'No se encontraron documentos locales relevantes; se usará análisis directo y/o búsqueda externa, manteniendo continuidad con el historial reciente si existe.', [])
        compact_parts: list[str] = []
        formulas: list[str] = []
        total_chars = 0
        for doc in snippets:
            piece = sanitize_text(f"{doc['title']}: {doc['content']}")
            total_chars += len(piece)
            if total_chars <= CONFIG.max_inline_context_chars:
                compact_parts.append(piece)
            if '=' in doc['content']:
                formulas.append(doc['content'])
        summary = ' | '.join(compact_parts)
        return KnowledgeResult(snippets, summary, formulas[:5])
