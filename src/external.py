"""Investigación externa multi-fuente con conectividad reforzada, planificación y síntesis."""
from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.storage import StorageManager
from src.utils import clamp


class ExternalKnowledgeFetcher:
    SOURCE_BASE_WEIGHTS = {
        'arxiv': 0.92,
        'crossref': 0.88,
        'wikipedia': 0.66,
        'duckduckgo': 0.52,
    }

    DOMAIN_KEYWORDS = {
        'spacecraft': ['nave', 'cohete', 'rocket', 'orbita', 'orbital', 'launch', 'spacecraft', 'payload', 'tripul', 'astronaut'],
        'physics': ['gravedad', 'drag', 'aerodinam', 'velocidad', 'delta', 'momentum', 'fisica', 'physics'],
        'chemistry': ['combustible', 'fuel', 'propel', 'oxidizer', 'mezcla', 'quim', 'chemistry', 'reaction'],
        'materials': ['material', 'estructura', 'aleacion', 'thermal', 'heat shield', 'fatiga', 'composite'],
        'systems': ['control', 'guidance', 'navigation', 'mission', 'safety', 'riesgo', 'operations'],
        'ocean': ['oceano', 'mar', 'submar', 'underwater', 'ocean'],
        'math': ['matem', 'ecuación', 'equation', 'integral', 'derivada', 'algebra', 'probabilidad', 'estadistica'],
        'geopolitics': ['geopol', 'guerra', 'conflicto', 'presidente', 'estado', 'frontera', 'sancion', 'diplomac'],
        'economics': ['econom', 'mercado', 'inflacion', 'trade', 'gdp', 'moneda', 'finanzas'],
        'computing': ['software', 'algoritmo', 'programacion', 'machine learning', 'ia', 'python', 'código'],
        'biology': ['biolog', 'genet', 'celula', 'evolucion', 'organismo', 'ecosistema', 'medicina'],
        'law': ['ley', 'legal', 'constitucion', 'regulacion', 'juridic', 'tribunal'],
        'aerospace': ['avion', 'aeronáut', 'wing', 'lift', 'aircraft', 'propulsion'],
    }

    def __init__(self, storage: StorageManager, timeout_sec: int = 8, max_queries: int = 18, max_retries: int = 3):
        self.storage = storage
        self.timeout_sec = timeout_sec
        self.max_queries = max_queries
        self.max_retries = max(1, max_retries)
        self.user_agent = 'ChatZeus/3.0 (+https://example.invalid)'

    def _request_text(self, url: str, source_type: str, timeout: int | None = None) -> str:
        timeout = timeout or self.timeout_sec
        last_error = 'unknown-error'
        for attempt in range(1, self.max_retries + 1):
            started = time.perf_counter()
            try:
                request = Request(url, headers={'User-Agent': self.user_agent, 'Accept': 'application/json, text/plain, application/xml'})
                with urlopen(request, timeout=timeout) as response:
                    payload = response.read().decode('utf-8', errors='replace')
                latency_ms = (time.perf_counter() - started) * 1000
                self.storage.save_connectivity_event(source_type, 'ok', latency_ms, f'attempt={attempt}')
                return payload
            except HTTPError as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                last_error = f'http-{exc.code}'
                self.storage.save_connectivity_event(source_type, 'http_error', latency_ms, last_error)
            except URLError as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                last_error = f'network:{exc.reason}'
                self.storage.save_connectivity_event(source_type, 'network_error', latency_ms, last_error)
            except Exception as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                last_error = f'unexpected:{exc}'
                self.storage.save_connectivity_event(source_type, 'unexpected_error', latency_ms, last_error)
            time.sleep(min(0.35 * attempt, 1.2))
        raise RuntimeError(f'{source_type} request failed after {self.max_retries} attempts: {last_error}')

    def _open_json(self, url: str, source_type: str) -> dict[str, Any]:
        return json.loads(self._request_text(url, source_type))

    def _open_text(self, url: str, source_type: str) -> str:
        return self._request_text(url, source_type)

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r'[a-záéíóúñ0-9]{4,}', text.lower())
        stopwords = {
            'para', 'como', 'esto', 'esta', 'estas', 'estos', 'desde', 'sobre', 'quiero', 'pueda', 'debe', 'todas',
            'todos', 'respuesta', 'buscar', 'búsqueda', 'datos', 'internet', 'teoria', 'teorías', 'programa',
            'chatbot', 'pregunta', 'preguntas', 'donde', 'hacia', 'tener', 'using', 'with', 'that', 'this', 'mucho',
        }
        return [token for token in tokens if token not in stopwords]

    def _extract_keywords(self, question: str, context: str = '') -> list[str]:
        ranked = [token for token, _ in Counter(self._tokenize(f'{question} {context}')).most_common(16)]
        return ranked[:10]

    def infer_domains(self, question: str, context: str = '') -> list[str]:
        haystack = f'{question} {context}'.lower()
        matches: list[str] = []
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            if any(keyword in haystack for keyword in keywords):
                matches.append(domain)
        return matches or ['systems', 'physics']

    def _intents_for_domains(self, domains: list[str]) -> list[str]:
        intents = ['overview', 'constraints', 'feasibility', 'failure_modes', 'academic', 'quantitative']
        if 'materials' in domains:
            intents.append('materials')
        if 'chemistry' in domains:
            intents.append('propulsion')
        if 'systems' in domains:
            intents.append('architecture')
        if 'geopolitics' in domains or 'law' in domains:
            intents.append('policy')
        if 'economics' in domains:
            intents.append('market')
        if 'biology' in domains:
            intents.append('mechanism')
        if 'computing' in domains:
            intents.append('implementation')
        return intents

    def _health_adjusted_weights(self, source_weights: dict[str, float] | None = None) -> dict[str, float]:
        weights = dict(source_weights or self.SOURCE_BASE_WEIGHTS)
        connectivity = self.storage.connectivity_profile()
        for source, base in list(weights.items()):
            profile = connectivity.get(source)
            if not profile:
                continue
            success = profile.get('success_rate', 0.0)
            latency = profile.get('avg_latency_ms', 0.0)
            latency_factor = 1.0 if latency <= 0 else clamp(1.15 - latency / 4000.0, 0.55, 1.15)
            reliability_factor = clamp(0.45 + success, 0.45, 1.35)
            weights[source] = round(clamp(base * latency_factor * reliability_factor, 0.15, 0.99), 4)
        return weights

    def plan_queries(
        self,
        question: str,
        context: str = '',
        preferred_domains: list[str] | None = None,
        source_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        keywords = self._extract_keywords(question, context)
        domains = self.infer_domains(question, context)
        intents = self._intents_for_domains(domains)
        source_weights = self._health_adjusted_weights(source_weights)
        ranked_sources = [name for name, _ in sorted(source_weights.items(), key=lambda item: item[1], reverse=True)]
        ranked_sources = [item for item in ranked_sources if item in self.SOURCE_BASE_WEIGHTS]
        if preferred_domains:
            preferred = [item for item in preferred_domains if item in self.SOURCE_BASE_WEIGHTS]
            ranked_sources = preferred + [item for item in ranked_sources if item not in preferred]
        keyword_head = ' '.join(keywords[:4])
        keyword_short = ' '.join(keywords[:3])
        keyword_long = ' '.join(keywords[:5])
        domain_head = ' '.join(domains[:3])
        query_templates = {
            'overview': question,
            'constraints': f'{question} engineering constraints {keyword_head}',
            'feasibility': f'{question} feasibility analysis {domain_head}',
            'failure_modes': f'{question} risks failure modes safety {keyword_short}',
            'academic': f'{question} research paper study {keyword_long}',
            'quantitative': f'{question} calculations equations models quantitative analysis',
            'materials': f'{question} materials thermal loads structure',
            'propulsion': f'{question} propulsion chemistry fuel oxidizer performance',
            'architecture': f'{question} systems architecture guidance operations verification',
            'policy': f'{question} policy regulation strategic implications',
            'market': f'{question} economics market incentives costs tradeoffs',
            'mechanism': f'{question} biological mechanism causal explanation evidence',
            'implementation': f'{question} algorithm implementation architecture best practices',
        }
        tasks: list[dict[str, Any]] = []
        for intent in intents:
            for source in ranked_sources:
                tasks.append({'intent': intent, 'query': query_templates[intent][:260], 'source': source, 'weight': source_weights.get(source, 0.4)})
                if len(tasks) >= self.max_queries:
                    return {'domains': domains, 'keywords': keywords, 'intents': intents, 'tasks': tasks, 'source_weights': source_weights}
        return {'domains': domains, 'keywords': keywords, 'intents': intents, 'tasks': tasks[: self.max_queries], 'source_weights': source_weights}

    def _search_duckduckgo(self, query: str) -> list[dict[str, Any]]:
        url = 'https://api.duckduckgo.com/?' + urlencode({'q': query, 'format': 'json', 'no_html': 1, 'skip_disambig': 1})
        payload = self._open_json(url, 'duckduckgo')
        findings: list[dict[str, Any]] = []
        abstract = (payload.get('AbstractText') or payload.get('Answer') or payload.get('Heading') or '').strip()
        if abstract:
            findings.append({'title': payload.get('Heading') or query, 'snippet': abstract[:500], 'source': payload.get('AbstractURL') or url, 'source_type': 'duckduckgo'})
        for item in payload.get('RelatedTopics') or []:
            if isinstance(item, dict) and item.get('Text'):
                findings.append({'title': item.get('FirstURL', query), 'snippet': item['Text'][:320], 'source': item.get('FirstURL') or url, 'source_type': 'duckduckgo'})
            if len(findings) >= 3:
                break
        return findings

    def _search_wikipedia(self, query: str) -> list[dict[str, Any]]:
        url = 'https://en.wikipedia.org/w/api.php?' + urlencode({'action': 'opensearch', 'search': query, 'limit': 3, 'namespace': 0, 'format': 'json'})
        payload = self._open_json(url, 'wikipedia')
        findings: list[dict[str, Any]] = []
        if len(payload) >= 4:
            for title, snippet, link in zip(payload[1], payload[2], payload[3]):
                findings.append({'title': title, 'snippet': (snippet or title)[:320], 'source': link, 'source_type': 'wikipedia'})
        return findings

    def _search_crossref(self, query: str) -> list[dict[str, Any]]:
        url = 'https://api.crossref.org/works?' + urlencode({'query': query, 'rows': 3, 'select': 'title,DOI,published,container-title'})
        payload = self._open_json(url, 'crossref')
        findings: list[dict[str, Any]] = []
        for item in payload.get('message', {}).get('items', [])[:3]:
            title = ' '.join(item.get('title') or []) or query
            venue = ' '.join(item.get('container-title') or [])
            doi = item.get('DOI', '')
            findings.append({
                'title': title[:220],
                'snippet': f'Referencia académica recuperada desde {venue or "Crossref"} para validar hipótesis de ingeniería.'[:320],
                'source': f'https://doi.org/{doi}' if doi else url,
                'source_type': 'crossref',
            })
        return findings

    def _search_arxiv(self, query: str) -> list[dict[str, Any]]:
        url = 'https://export.arxiv.org/api/query?' + urlencode({'search_query': f'all:{query}', 'start': 0, 'max_results': 3})
        xml_text = self._open_text(url, 'arxiv')
        root = ET.fromstring(xml_text)
        findings: list[dict[str, Any]] = []
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        for entry in root.findall('atom:entry', ns)[:3]:
            title = (entry.findtext('atom:title', default='', namespaces=ns) or '').strip()
            summary = (entry.findtext('atom:summary', default='', namespaces=ns) or '').strip().replace('\n', ' ')
            link = ''
            for node in entry.findall('atom:link', ns):
                href = node.attrib.get('href')
                if href:
                    link = href
                    break
            findings.append({'title': title[:220] or query, 'snippet': summary[:360] or 'Preprint técnico recuperado desde arXiv.', 'source': link or url, 'source_type': 'arxiv'})
        return findings

    def _run_single_search(self, source: str, query: str) -> list[dict[str, Any]]:
        if source == 'duckduckgo':
            return self._search_duckduckgo(query)
        if source == 'wikipedia':
            return self._search_wikipedia(query)
        if source == 'crossref':
            return self._search_crossref(query)
        if source == 'arxiv':
            return self._search_arxiv(query)
        return []

    def _score_finding(self, finding: dict[str, Any], keywords: list[str], domains: list[str], source_weight: float) -> float:
        haystack = f"{finding.get('title', '')} {finding.get('snippet', '')}".lower()
        overlap = sum(1 for keyword in keywords if keyword in haystack)
        domain_overlap = sum(1 for domain in domains if domain in haystack)
        snippet_len = min(len(finding.get('snippet', '')) / 280.0, 1.0)
        score = source_weight * 0.58 + min(overlap / max(len(keywords), 1), 1.0) * 0.24 + domain_overlap * 0.08 + snippet_len * 0.1
        return round(min(score, 0.99), 4)

    def _detect_contradictions(self, findings: list[dict[str, Any]]) -> list[str]:
        contradictions: list[str] = []
        positive_markers = ('feasible', 'viable', 'promising', 'high performance', 'reusable')
        negative_markers = ('challenging', 'limitation', 'risk', 'failure', 'unstable', 'expensive')
        positives = sum(1 for item in findings if any(marker in item.get('snippet', '').lower() for marker in positive_markers))
        negatives = sum(1 for item in findings if any(marker in item.get('snippet', '').lower() for marker in negative_markers))
        if positives and negatives:
            contradictions.append('Las fuentes mezclan señales de viabilidad con riesgos relevantes; conviene validar experimentalmente antes de tratar la idea como resuelta.')
        if not contradictions and findings:
            contradictions.append('No se detectaron contradicciones textuales fuertes, pero la ausencia de conflicto explícito no sustituye verificación técnica rigurosa.')
        return contradictions

    def _build_synthesis(self, plan: dict[str, Any], findings: list[dict[str, Any]], failures: list[str]) -> dict[str, Any]:
        top_findings = findings[:12]
        coverage = Counter(item.get('source_type', 'unknown') for item in top_findings)
        quality_score = round(sum(item.get('score', 0.0) for item in top_findings) / max(len(top_findings), 1), 4)
        feasibility_signal = round(min(0.95, 0.25 + quality_score * 0.75), 4)
        missing_domains = [domain for domain in plan.get('domains', []) if not any(domain in (item.get('title', '') + ' ' + item.get('snippet', '')).lower() for item in top_findings)]
        research_gaps = [f'Faltan evidencias directas para el dominio {domain}.' for domain in missing_domains[:4]]
        if failures:
            research_gaps.append('Parte de la investigación externa falló por conectividad o por límites del proveedor; conviene reintentar o ampliar redundancia de conectores.')
        connectivity = self.storage.connectivity_profile()
        recommended_actions = [
            'Separar el problema en física, materiales, propulsión, control y operaciones antes de concluir factibilidad total.',
            'Convertir los mejores hallazgos en requisitos cuantitativos para la siguiente iteración de simulación.',
            'Mantener un ciclo de hipótesis -> búsqueda -> simulación -> validación en lugar de una sola respuesta definitiva.',
        ]
        return {
            'quality_score': quality_score,
            'feasibility_signal': feasibility_signal,
            'coverage': dict(coverage),
            'contradictions': self._detect_contradictions(top_findings),
            'research_gaps': research_gaps,
            'recommended_actions': recommended_actions,
            'connectivity_profile': connectivity,
        }

    def fetch_research_dossier(
        self,
        query: str,
        context: str = '',
        preferred_domains: list[str] | None = None,
        source_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        plan = self.plan_queries(query, context=context, preferred_domains=preferred_domains, source_weights=source_weights)
        findings: list[dict[str, Any]] = []
        failures: list[str] = []
        for task in plan['tasks']:
            try:
                for finding in self._run_single_search(task['source'], task['query']):
                    finding['intent'] = task['intent']
                    finding['source_weight'] = task.get('weight', self.SOURCE_BASE_WEIGHTS.get(task['source'], 0.4))
                    findings.append(finding)
            except Exception as exc:
                failures.append(f"{task['source']}:{task['intent']}: {exc}")
        for finding in findings:
            finding['score'] = self._score_finding(finding, plan['keywords'], plan['domains'], float(finding.get('source_weight', 0.4)))
        unique: list[dict[str, Any]] = []
        seen = set()
        for item in sorted(findings, key=lambda current: current.get('score', 0.0), reverse=True):
            key = (item.get('title'), item.get('source'))
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        top_findings = unique[:12]
        synthesis = self._build_synthesis(plan, top_findings, failures)
        excerpt = ' | '.join(f"{item['source_type']}[{item.get('intent', 'general')}]: {item['title']} -> {item['snippet']}" for item in top_findings[:4])
        status = 'ok' if top_findings else 'unavailable'
        return {
            'status': status,
            'queries_executed': len(plan['tasks']),
            'sources_consulted': synthesis['coverage'],
            'plan': plan,
            'findings': top_findings,
            'domains': plan['domains'],
            'keywords': plan['keywords'],
            'intents': plan['intents'],
            'source': top_findings[0]['source'] if top_findings else 'multi-source-search',
            'excerpt': excerpt[:1400] if excerpt else ('No se obtuvieron hallazgos externos útiles.' if failures else 'Sin hallazgos externos.'),
            'failures': failures[:10],
            'synthesis': synthesis,
        }

    def fetch_formula_hint(self, query: str) -> dict[str, Any]:
        return self.fetch_research_dossier(query)
