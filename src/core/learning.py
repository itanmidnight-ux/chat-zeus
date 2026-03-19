"""Internet-backed lightweight learning with compact storage."""
from __future__ import annotations

import re
from typing import Any

import requests

from src.core.memory import LightweightMemory


class LearningEngine:
    def __init__(self, memory: LightweightMemory, timeout: int = 8):
        self.memory = memory
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'ChatZeus/agent'})


    @staticmethod
    def _normalize_query(question: str) -> str:
        lowered = question.lower().strip()
        replacements = {
            'hombre más rico': 'richest person in the world',
            'hombre mas rico': 'richest person in the world',
            'qué hora es': 'current time',
            'que hora es': 'current time',
        }
        for source, target in replacements.items():
            if source in lowered:
                return target
        return question

    def compress_knowledge(self, text: str) -> str:
        cleaned = re.sub(r'\s+', ' ', text).strip()
        cleaned = re.sub(r'\[[^\]]+\]', '', cleaned)
        if len(cleaned) <= 200:
            return cleaned
        sentence = re.split(r'(?<=[.!?])\s+', cleaned)[0].strip()
        compact = sentence if 20 <= len(sentence) <= 200 else cleaned[:197].rstrip() + '...'
        return compact[:200]

    def _fetch_wikipedia_summary(self, question: str) -> str | None:
        search_url = 'https://en.wikipedia.org/w/api.php'
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': question,
            'format': 'json',
            'utf8': 1,
            'srlimit': 1,
        }
        response = self.session.get(search_url, params=params, timeout=self.timeout)
        response.raise_for_status()
        hits = response.json().get('query', {}).get('search', [])
        if not hits:
            return None
        title = hits[0]['title']
        summary_url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(" ", "_")}'
        summary_response = self.session.get(summary_url, timeout=self.timeout)
        summary_response.raise_for_status()
        data = summary_response.json()
        return data.get('extract')

    def _fetch_duckduckgo_answer(self, question: str) -> str | None:
        response = self.session.get(
            'https://api.duckduckgo.com/',
            params={'q': question, 'format': 'json', 'no_html': 1, 'skip_disambig': 1},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        for field in ('AbstractText', 'Answer', 'Definition'):
            value = payload.get(field)
            if value:
                return value
        topics = payload.get('RelatedTopics') or []
        for topic in topics:
            if isinstance(topic, dict) and topic.get('Text'):
                return topic['Text']
        return None

    def search_and_learn(self, question: str) -> str:
        errors: list[str] = []
        normalized_question = self._normalize_query(question)
        for source_name, loader in (
            ('wikipedia', self._fetch_wikipedia_summary),
            ('duckduckgo', self._fetch_duckduckgo_answer),
        ):
            try:
                fetched = loader(normalized_question)
                if fetched:
                    compact = self.compress_knowledge(fetched)
                    if compact:
                        self.memory.put('facts', question, compact, source=source_name)
                        return compact
            except requests.RequestException as exc:
                errors.append(f'{source_name}:{exc.__class__.__name__}')
        if errors:
            return 'No encontré un dato confiable en internet en este momento.'
        return 'No encontré una respuesta útil.'

    def search_fact(self, question: str) -> str:
        cached = self.memory.get('facts', question)
        if cached:
            return cached['value']
        return self.search_and_learn(question)
