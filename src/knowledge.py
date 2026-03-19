"""Gestor de conocimiento local con búsqueda sencilla tipo RAG."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.storage import StorageManager


@dataclass
class KnowledgeResult:
    snippets: list[dict[str, Any]]
    summary: str


class KnowledgeManager:
    def __init__(self, storage: StorageManager):
        self.storage = storage

    def retrieve(self, question: str) -> KnowledgeResult:
        snippets = self.storage.search_knowledge(question)
        if not snippets:
            return KnowledgeResult([], 'No se encontraron documentos locales relevantes; se usará análisis directo y/o búsqueda externa.')
        summary = ' | '.join(f"{doc['title']}: {doc['content']}" for doc in snippets)
        return KnowledgeResult(snippets, summary)
