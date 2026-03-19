"""Búsqueda externa opcional con degradación elegante si no hay red."""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


class ExternalKnowledgeFetcher:
    def __init__(self, timeout_sec: int = 8):
        self.timeout_sec = timeout_sec

    def fetch_formula_hint(self, query: str) -> dict[str, Any]:
        url = 'https://api.duckduckgo.com/?' + urlencode({
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1,
        })
        try:
            with urlopen(url, timeout=self.timeout_sec) as response:
                raw = response.read().decode('utf-8', errors='replace')
            payload = json.loads(raw)
            abstract = (payload.get('AbstractText') or payload.get('Answer') or payload.get('Heading') or '').strip()
            related = payload.get('RelatedTopics') or []
            related_excerpt = ''
            if not abstract and related:
                first = related[0]
                if isinstance(first, dict):
                    related_excerpt = (first.get('Text') or '')[:240]
            excerpt = abstract[:400] if abstract else related_excerpt or 'Sin extracto útil; solo se confirmó accesibilidad externa.'
            return {
                'status': 'ok',
                'source': url,
                'excerpt': excerpt,
            }
        except Exception as exc:
            return {
                'status': 'unavailable',
                'source': url,
                'excerpt': f'No fue posible consultar internet: {exc}',
            }
