"""Búsqueda externa opcional con degradación elegante si no hay red."""
from __future__ import annotations

import urllib.parse
import urllib.request
from typing import Any


class ExternalKnowledgeFetcher:
    def __init__(self, timeout_sec: int = 8):
        self.timeout_sec = timeout_sec

    def fetch_formula_hint(self, query: str) -> dict[str, Any]:
        url = 'https://api.duckduckgo.com/?' + urllib.parse.urlencode({
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1,
        })
        try:
            with urllib.request.urlopen(url, timeout=self.timeout_sec) as response:
                payload = response.read().decode('utf-8', errors='replace')
            return {
                'status': 'ok',
                'source': url,
                'excerpt': payload[:500],
            }
        except Exception as exc:
            return {
                'status': 'unavailable',
                'source': url,
                'excerpt': f'No fue posible consultar internet: {exc}',
            }
