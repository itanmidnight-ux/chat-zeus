"""Lightweight persistent memory for learned facts, patterns, failures, and generated solutions."""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


class LightweightMemory:
    DEFAULT_BUCKETS = ('facts', 'solutions', 'patterns', 'failures', 'episodes')

    def __init__(self, path: Path, limit: int = 500):
        self.path = path
        self.limit = max(10, limit)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.memory: dict[str, OrderedDict[str, dict[str, Any]]] = {
            bucket: OrderedDict() for bucket in self.DEFAULT_BUCKETS
        }
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return
        for bucket in self.DEFAULT_BUCKETS:
            items = payload.get(bucket, {})
            if isinstance(items, dict):
                self.memory[bucket] = OrderedDict(items)
        self._enforce_limit()

    def _save(self) -> None:
        serializable = {bucket: dict(values) for bucket, values in self.memory.items()}
        self.path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding='utf-8')

    @staticmethod
    def make_key(text: str) -> str:
        return hashlib.sha1(text.strip().lower().encode('utf-8')).hexdigest()[:16]

    def _enforce_limit(self) -> None:
        total = sum(len(values) for values in self.memory.values())
        priority = ['episodes', 'failures', 'patterns', 'facts', 'solutions']
        while total > self.limit:
            bucket = next((name for name in priority if self.memory.get(name)), 'facts')
            if self.memory[bucket]:
                self.memory[bucket].popitem(last=False)
            total = sum(len(values) for values in self.memory.values())

    def get(self, bucket: str, query: str) -> dict[str, Any] | None:
        key = self.make_key(query)
        item = self.memory.get(bucket, OrderedDict()).get(key)
        if item is None:
            return None
        self.memory[bucket].move_to_end(key)
        return item

    def put(self, bucket: str, query: str, value: str | dict[str, Any], source: str = 'local') -> None:
        key = self.make_key(query)
        payload = value if isinstance(value, dict) else {'query': query, 'value': value, 'source': source}
        payload.setdefault('query', query)
        payload.setdefault('source', source)
        self.memory.setdefault(bucket, OrderedDict())[key] = payload
        self.memory[bucket].move_to_end(key)
        self._enforce_limit()
        self._save()

    def export(self) -> dict[str, dict[str, dict[str, Any]]]:
        return {bucket: dict(values) for bucket, values in self.memory.items()}
