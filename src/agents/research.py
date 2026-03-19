"""Research agent: facts only, no conclusions."""
from __future__ import annotations

from typing import Any

from src.storage import StorageManager


class ResearchAgent:
    def __init__(self, storage: StorageManager):
        self.storage = storage

    def research(self, topic: str) -> dict[str, Any]:
        try:
            docs = self.storage.search_knowledge(topic, limit=3)
            facts = []
            for doc in docs:
                facts.append({'topic': topic, 'fact': doc['content'], 'source': doc['source']})
            keywords = [token for token in topic.split()[:6] if token]
            return {'topic': topic, 'facts': facts, 'keywords': keywords, 'evidence_count': len(facts)}
        except Exception:
            return {'topic': topic, 'facts': [], 'keywords': topic.split()[:6], 'evidence_count': 0}

    def investigate(self, question: str, plan: list[dict[str, Any]]) -> dict[str, Any]:
        findings = []
        summary_parts = []
        for item in plan[:5]:
            payload = self.research(f"{question} {item['task']}")
            for fact in payload.get('facts', [])[:2]:
                findings.append({'task': item['task'], 'priority': item['priority'], 'summary': fact['fact'], 'source': fact['source']})
                summary_parts.append(fact['fact'])
        return {'findings': findings, 'summary': ' '.join(summary_parts[:4]).strip()}
