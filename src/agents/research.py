"""Agente de investigación local y silencioso."""
from __future__ import annotations

from typing import Any

from src.storage import StorageManager


class ResearchAgent:
    def __init__(self, storage: StorageManager):
        self.storage = storage

    def investigate(self, question: str, plan: list[dict[str, Any]]) -> dict[str, Any]:
        findings = []
        for item in plan[:5]:
            docs = self.storage.search_knowledge(f"{question} {item['task']}", limit=2)
            for doc in docs:
                findings.append({
                    'task': item['task'],
                    'priority': item['priority'],
                    'summary': doc['content'],
                    'source': doc['source'],
                })
        summary = ' '.join(entry['summary'] for entry in findings[:4])
        return {'findings': findings, 'summary': summary.strip()}
