"""Internet-backed lightweight learning with validation, provenance, and compact storage."""
from __future__ import annotations

import json
import re
import time
from typing import Any

import requests

from src.core.memory import LightweightMemory
from src.storage import StorageManager


class LearningEngine:
    def __init__(self, memory: LightweightMemory, storage: StorageManager | None = None, timeout: int = 8):
        self.memory = memory
        self.storage = storage
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
            'último': 'latest',
            'ultima': 'latest',
            'última': 'latest',
        }
        for source, target in replacements.items():
            if source in lowered:
                return target
        return question

    def compress_knowledge(self, text: str) -> str:
        cleaned = re.sub(r'\s+', ' ', text).strip()
        cleaned = re.sub(r'\[[^\]]+\]', '', cleaned)
        if len(cleaned) <= 260:
            return cleaned
        sentences = [segment.strip() for segment in re.split(r'(?<=[.!?])\s+', cleaned) if segment.strip()]
        compact = ' '.join(sentences[:2]) if sentences else cleaned[:257].rstrip() + '...'
        return compact[:260]

    def _record_connectivity(self, source_name: str, status: str, latency_ms: float, detail: str) -> None:
        if self.storage is not None:
            self.storage.save_connectivity_event(source_name, status, latency_ms, detail)

    def _register_research(self, question: str, findings: list[dict[str, Any]]) -> None:
        if self.storage is None:
            return
        quality = 0.0
        if findings:
            quality = sum(float(item.get('score', 0.0)) for item in findings) / len(findings)
        payload = {'findings': findings[:8], 'sources': [item.get('source_type', 'unknown') for item in findings[:8]]}
        self.storage.save_research_session(question, json.dumps(payload, ensure_ascii=False), len(findings), quality)

    def _fetch_wikipedia_summary(self, question: str) -> dict[str, Any] | None:
        search_url = 'https://en.wikipedia.org/w/api.php'
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': question,
            'format': 'json',
            'utf8': 1,
            'srlimit': 2,
        }
        started = time.perf_counter()
        response = self.session.get(search_url, params=params, timeout=self.timeout)
        latency_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
        hits = response.json().get('query', {}).get('search', [])
        if not hits:
            self._record_connectivity('wikipedia', 'empty', latency_ms, 'no-hits')
            return None
        title = hits[0]['title']
        summary_url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(" ", "_")}'
        started = time.perf_counter()
        summary_response = self.session.get(summary_url, timeout=self.timeout)
        latency_ms = (time.perf_counter() - started) * 1000.0
        summary_response.raise_for_status()
        data = summary_response.json()
        self._record_connectivity('wikipedia', 'ok', latency_ms, title)
        extract = data.get('extract')
        if not extract:
            return None
        return {'text': extract, 'source_type': 'wikipedia', 'title': title, 'score': 0.78}

    def _fetch_duckduckgo_answer(self, question: str) -> dict[str, Any] | None:
        started = time.perf_counter()
        response = self.session.get(
            'https://api.duckduckgo.com/',
            params={'q': question, 'format': 'json', 'no_html': 1, 'skip_disambig': 1},
            timeout=self.timeout,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
        payload = response.json()
        text: str | None = None
        for field in ('AbstractText', 'Answer', 'Definition'):
            value = payload.get(field)
            if value:
                text = value
                break
        if text is None:
            topics = payload.get('RelatedTopics') or []
            for topic in topics:
                if isinstance(topic, dict) and topic.get('Text'):
                    text = str(topic['Text'])
                    break
        if not text:
            self._record_connectivity('duckduckgo', 'empty', latency_ms, 'no-answer')
            return None
        self._record_connectivity('duckduckgo', 'ok', latency_ms, 'instant-answer')
        return {'text': text, 'source_type': 'duckduckgo', 'title': question, 'score': 0.58}

    def _clean_candidate(self, text: str) -> str:
        cleaned = re.sub(r'https?://\S+', ' ', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def _validate_candidates(self, question: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        keywords = {token for token in question.lower().split() if len(token) > 3}
        validated: list[dict[str, Any]] = []
        for item in candidates:
            text = self._clean_candidate(str(item.get('text', '')))
            if len(text) < 24:
                continue
            overlap = sum(1 for token in keywords if token in text.lower())
            reliability = min(0.98, float(item.get('score', 0.5)) + min(0.2, overlap * 0.04))
            validated.append({**item, 'text': text, 'score': round(reliability, 4)})
        validated.sort(key=lambda row: row.get('score', 0.0), reverse=True)
        return validated

    def search_and_learn(self, question: str) -> str:
        errors: list[str] = []
        findings: list[dict[str, Any]] = []
        normalized_question = self._normalize_query(question)
        for source_name, loader in (
            ('wikipedia', self._fetch_wikipedia_summary),
            ('duckduckgo', self._fetch_duckduckgo_answer),
        ):
            try:
                fetched = loader(normalized_question)
                if fetched:
                    findings.append(fetched)
            except requests.RequestException as exc:
                errors.append(f'{source_name}:{exc.__class__.__name__}')
                self._record_connectivity(source_name, 'error', float(self.timeout) * 1000.0, exc.__class__.__name__)
        validated = self._validate_candidates(question, findings)
        self._register_research(question, validated)
        if validated:
            best = validated[0]
            compact = self.compress_knowledge(best['text'])
            self.memory.put('facts', question, compact, source=best.get('source_type', 'internet'))
            return compact
        if errors:
            return 'No encontré un dato confiable en internet en este momento.'
        return 'No encontré una respuesta útil.'

    def search_fact(self, question: str) -> str:
        cached = self.memory.get('facts', question)
        if cached:
            return cached['value']
        return self.search_and_learn(question)
