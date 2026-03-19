"""Interfaz conversacional principal que coordina conocimiento, simulación, ML y reportes."""
from __future__ import annotations

from concurrent.futures import TimeoutError as FutureTimeoutError
import gc
import json
import uuid
from typing import Any

from src.calculator import AnalyticalCalculator
from src.config import CONFIG
from src.external import ExternalKnowledgeFetcher
from src.knowledge import KnowledgeManager
from src.ml import LightweightMLModel
from src.optimizer import IterativeOptimizer
from src.reporting import ReportWriter
from src.response_control import build_user_response, estimate_implicit_satisfaction
from src.simulation import SimulationEngine
from src.storage import StorageManager
from src.utils import detect_linker_memory_issue, extract_numeric_value, safe_error_message, sanitize_text
from src.worker import BackgroundExecutor


class ChatbotInterface:
    ENGINEERING_DOMAINS = {'spacecraft', 'physics', 'chemistry', 'materials', 'systems', 'aerospace'}
    ANALYTICAL_KEYWORDS = {
        'derivada', 'integral', 'matriz', 'ecuación', 'ecuacion', 'resuelve', 'resolver', 'solve',
        'geología', 'geologia', 'material', 'stress', 'esfuerzo', 'resistencia', 'sismo', 'sedimento',
    }
    SIMULATION_KEYWORDS = {
        'cohete', 'rocket', 'lanzador', 'orbita', 'orbital', 'trayectoria', 'delta-v', 'delta_v',
        'payload', 'combustible', 'fuel', 'thrust', 'empuje', 'drag', 'propulsion', 'propulsión',
        'masa', 'mezcla', 'altitud', 'stage', 'etapa',
    }

    def __init__(
        self,
        storage: StorageManager,
        knowledge: KnowledgeManager,
        simulation: SimulationEngine,
        calculator: AnalyticalCalculator,
        ml_model: LightweightMLModel,
        external_fetcher: ExternalKnowledgeFetcher,
        optimizer: IterativeOptimizer,
        report_writer: ReportWriter,
        background_executor: BackgroundExecutor,
        logger,
    ):
        self.storage = storage
        self.knowledge = knowledge
        self.simulation = simulation
        self.calculator = calculator
        self.ml_model = ml_model
        self.external_fetcher = external_fetcher
        self.optimizer = optimizer
        self.report_writer = report_writer
        self.background_executor = background_executor
        self.logger = logger

    def _extract_defaults(self, question: str) -> dict[str, float | int]:
        defaults: dict[str, float | int] = {}
        patterns = {
            'payload_mass_kg': r'(?:payload|carga(?:\s+útil)?)\s*=\s*([0-9]+(?:[\.,][0-9]+)?)',
            'fuel_mass_kg': r'(?:fuel|combustible)\s*=\s*([0-9]+(?:[\.,][0-9]+)?)',
            'dry_mass_kg': r'(?:dry|masa\s+seca)\s*=\s*([0-9]+(?:[\.,][0-9]+)?)',
            'thrust_n': r'(?:thrust|empuje)\s*=\s*([0-9]+(?:[\.,][0-9]+)?)',
            'steps': r'(?:steps|pasos)\s*=\s*([0-9]+)',
            'mixture_ratio': r'(?:mixture|mezcla)\s*=\s*([0-9]+(?:[\.,][0-9]+)?)',
            'exhaust_velocity_m_s': r'(?:ve|escape|exhaust)\s*=\s*([0-9]+(?:[\.,][0-9]+)?)',
            'area_m2': r'(?:area|área)\s*=\s*([0-9]+(?:[\.,][0-9]+)?)',
            'drag_coefficient': r'(?:cd|drag)\s*=\s*([0-9]+(?:[\.,][0-9]+)?)',
        }
        for key, pattern in patterns.items():
            value = extract_numeric_value(question, pattern)
            if value is not None:
                defaults[key] = int(value) if key == 'steps' else float(value)
        if 'steps' not in defaults:
            lower_question = question.lower()
            if any(word in lower_question for word in ['rápido', 'rapido', 'ligero', 'simple']):
                defaults['steps'] = 120
            elif any(word in lower_question for word in ['preciso', 'larga', 'detallado', 'optimiza', 'profundo']):
                defaults['steps'] = 360
        if 'steps' in defaults:
            defaults['steps'] = max(60, min(int(defaults['steps']), 480))
        return defaults

    def _profile_question(self, question: str, inferred_domains: list[str]) -> dict[str, Any]:
        lowered = question.lower()
        explicit_general = any(token in lowered for token in ['explica', 'resume', 'analiza', 'compar', 'riesgo', 'viabilidad'])
        analytical_signal = any(token in lowered for token in self.ANALYTICAL_KEYWORDS)
        simulation_signal = any(token in lowered for token in self.SIMULATION_KEYWORDS) or any(
            key in lowered for key in ['payload=', 'fuel=', 'combustible=', 'thrust=', 'empuje=', 'pasos=', 'steps=']
        )
        requires_simulation = simulation_signal or any(domain in self.ENGINEERING_DOMAINS for domain in inferred_domains)
        if analytical_signal and not simulation_signal:
            requires_simulation = False
        if explicit_general and not simulation_signal:
            requires_simulation = False
        focus = 'simulation' if requires_simulation else 'general_analysis'
        return {
            'focus': focus,
            'requires_simulation': requires_simulation,
            'domains': inferred_domains,
            'question_length': len(question),
            'optimization_requested': any(word in lowered for word in ['optimiza', 'optimize', 'mejora', 'improve']),
        }

    def _build_general_analysis_frame(
        self,
        question: str,
        profile: dict[str, Any],
        knowledge_summary: str,
        defaults: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        defaults = defaults or {}
        domains = profile.get('domains') or ['systems']
        decision_axes = ['factibilidad', 'riesgos', 'restricciones', 'evidencia', 'siguientes experimentos']
        questions_to_validate = [
            '¿Qué hipótesis críticas siguen sin evidencia directa?',
            '¿Qué restricciones físicas, económicas o regulatorias limitan la propuesta?',
            '¿Qué variable domina la incertidumbre del análisis?',
            '¿Qué prueba o fuente permitiría refutar la hipótesis principal?',
        ]
        return {
            'run_id': str(defaults.get('run_id', f'analysis_{uuid.uuid4().hex[:10]}')),
            'mode': 'general_analysis',
            'problem_type': 'consulta analítica general',
            'domains': domains,
            'analytical_depth': 'media-alta' if len(question) > 80 else 'media',
            'uncertainty_index': round(min(0.92, 0.35 + len(domains) * 0.07), 3),
            'breadth_score': round(min(0.96, 0.4 + len(domains) * 0.09), 3),
            'decision_axes': decision_axes,
            'questions_to_validate': questions_to_validate,
            'knowledge_summary': knowledge_summary,
            'delta_v_m_s': 0.0,
            'max_altitude_m': 0.0,
            'range_m': 0.0,
            'burn_time_s': 0.0,
            'final_velocity_m_s': 0.0,
            'remaining_fuel_kg': 0.0,
            'payload_mass_kg': 0.0,
            'chemistry': {'estimated_efficiency': 0.0},
            'math': {},
            'materials': {},
            'resource_profile': {
                'chunk_size': 0,
                'max_memory_mb': CONFIG.max_task_memory_mb,
                'estimated_history_bytes': len(knowledge_summary.encode('utf-8')),
                'resumed_from_checkpoint': False,
            },
        }

    def _progress(self, run_id: str, progress: float) -> None:
        self.logger.info('Progreso %s: %.1f%%', run_id, progress * 100)

    def _build_analysis(
        self,
        knowledge_summary: str,
        simulation: dict[str, Any],
        external: dict[str, Any],
        recent_context: list[dict[str, Any]],
        ml_result,
    ) -> tuple[str, str]:
        context_hint = ''
        if recent_context:
            context_hint = f" El historial reciente contiene {len(recent_context)} intercambios y se usó para mantener continuidad temática sin exponer pasos internos."
        synthesis = external.get('synthesis', {})
        analysis = (
            'Se combinó recuperación local de conocimiento científico, una simulación silenciosa de ascenso y trayectoria simplificada, '
            'estimaciones básicas de gravedad, arrastre, propulsión y termodinámica, además de una capa de aprendizaje incremental con checkpoints persistentes. '
            'La consulta activó una investigación web multi-fuente con planificación por dominios, evaluación de evidencia, detección básica de contradicciones, conectividad reforzada con reintentos y priorización adaptativa de fuentes.'
            f' Resumen RAG: {knowledge_summary}.{context_hint}'
            ' El proceso se ejecutó en bloques pequeños para aprovechar CPU y RAM de Linux sin perder estabilidad.'
            f" La investigación complementaria aportó una señal de factibilidad aproximada de {synthesis.get('feasibility_signal', 0.0)} y una calidad media de evidencia de {synthesis.get('quality_score', 0.0)}."
        )
        if simulation.get('mode') == 'general_analysis':
            conclusions = (
                f"El problema se trató como análisis general y no como una simulación física cerrada. El sistema recomienda profundizar en los ejes {', '.join(simulation.get('decision_axes', [])[:6]) or 'principales'}, "
                'resolver primero los vacíos críticos de evidencia y solo después cerrar una respuesta ejecutiva o un plan técnico. '
                'Esto permite responder preguntas de física, química, matemáticas, geopolítica, ingeniería u otros dominios sin quedar limitado a un solo tipo de problema.'
            )
        else:
            conclusions = (
                f"Para esta consulta, el diseño analizado alcanza aproximadamente {simulation['max_altitude_m']} m de altitud máxima con un delta-v de {simulation['delta_v_m_s']} m/s. "
                f"El motor de investigación recomienda trabajar por dominios {', '.join(external.get('domains', [])[:5]) or 'generales'} y priorizar fuentes {', '.join(ml_result.preferred_domains[:4])}. "
                'La siguiente mejora más prometedora es convertir los hallazgos externos y los riesgos detectados en requisitos cuantitativos para nuevas simulaciones y validaciones. '
                'Los resultados son útiles para exploración conceptual avanzada, pero no sustituyen validación de ingeniería de alta fidelidad ni garantizan éxito real del 100 %. '
            )
        return sanitize_text(analysis), sanitize_text(conclusions)

    def _resume_note(self) -> str:
        pending_runs = self.storage.recover_incomplete_runs()
        if not pending_runs:
            return ''
        latest = pending_runs[0]
        return (
            f" Se detectó una ejecución previa no finalizada ({latest['run_id']}) con progreso "
            f"{round(float(latest['progress']) * 100, 1)} %, lo que confirma que el sistema conserva checkpoints reanudables tras reinicios."
        )

    def _degraded_external_result(self, question: str, knowledge_summary: str, reason: str = '') -> dict[str, Any]:
        excerpt = (
            'La investigación web quedó degradada en esta corrida; se priorizó la base local y el razonamiento interno '
            'para no dejar la consulta sin respuesta.'
        )
        if reason:
            excerpt = f'{excerpt} Motivo resumido: {reason}.'
        inferred_domains = self.external_fetcher.infer_domains(question, knowledge_summary)
        return {
            'status': 'degraded',
            'queries_executed': 0,
            'sources_consulted': {},
            'plan': {'domains': inferred_domains, 'keywords': [], 'intents': ['overview'], 'tasks': [], 'source_weights': {}},
            'findings': [],
            'domains': inferred_domains,
            'keywords': [],
            'intents': ['overview'],
            'source': 'local-knowledge-fallback',
            'excerpt': excerpt,
            'failures': [reason] if reason else [],
            'synthesis': {
                'quality_score': 0.0,
                'feasibility_signal': 0.0,
                'coverage': {},
                'contradictions': ['La investigación externa no pudo completarse, así que la respuesta se apoya sobre todo en conocimiento local y heurísticas.'],
                'research_gaps': ['Conviene reintentar con conectividad estable para ampliar evidencia y referencias.'],
                'recommended_actions': [
                    'Reintentar la búsqueda externa cuando haya red disponible.',
                    'Usar la respuesta actual como análisis preliminar y no como validación definitiva.',
                ],
                'connectivity_profile': self.storage.connectivity_profile(),
            },
        }

    def _fallback_response_text(self, question: str, exc: Exception) -> str:
        recent_context = self.storage.recent_conversations(limit=min(3, CONFIG.max_history_messages))
        knowledge = self.knowledge.retrieve(question, recent_context=recent_context)
        domains = self.external_fetcher.infer_domains(question, knowledge.summary)
        profile = self._profile_question(question, domains)
        simulation = self._build_general_analysis_frame(question, profile, knowledge.summary, {'run_id': f'fallback_{uuid.uuid4().hex[:8]}'})
        external = self._degraded_external_result(question, knowledge.summary, safe_error_message(exc))
        calculations = self.calculator.analyze(question, simulation, knowledge.summary)
        ml_result = self.ml_model.predict(simulation, question=question, knowledge_summary=knowledge.summary)
        analysis, conclusions = self._build_analysis(knowledge.summary, simulation, external, recent_context, ml_result)
        payload = {
            'analysis': sanitize_text(
                'Se produjo una degradación controlada durante el procesamiento completo, pero el sistema evitó fallar y generó una respuesta útil en modo de contingencia. '
                + analysis
            ),
            'conclusions': conclusions,
            'simulation': simulation,
            'ml': {
                'prediction': ml_result.prediction,
                'confidence': ml_result.confidence,
                'hypotheses': ml_result.hypotheses,
                'preferred_domains': ml_result.preferred_domains,
                'research_intensity': ml_result.research_intensity,
                'source_weights': ml_result.source_weights,
                'model_state': ml_result.model_state,
                'variables_considered': ml_result.variables_considered,
                'uncertainty_drivers': ml_result.uncertainty_drivers,
                'recommendations': ml_result.recommendations,
                'reliability_score': ml_result.reliability_score,
            },
            'external': external,
            'calculations': calculations,
        }
        analysis_data = self._prepare_response_control_data(question, simulation, calculations, external, ml_result, analysis, conclusions)
        question_type, response_text = build_user_response(question, analysis_data)
        satisfaction = estimate_implicit_satisfaction(question_type, response_text)
        self.storage.append_response_feedback(question_type, len(response_text.split()), satisfaction)
        self.storage.save_conversation(
            question,
            response_text,
            json.dumps({'fallback_error': safe_error_message(exc), 'question_type': question_type, 'response_length': len(response_text.split()), 'satisfaction': satisfaction}, ensure_ascii=False),
        )
        del recent_context, knowledge, domains, profile, simulation, external, calculations, ml_result, payload, analysis_data
        gc.collect()
        return response_text

    def _prepare_response_control_data(
        self,
        question: str,
        simulation: dict[str, Any],
        calculations: dict[str, Any],
        external: dict[str, Any],
        ml_result,
        analysis: str,
        conclusions: str,
    ) -> dict[str, Any]:
        synthesis = external.get('synthesis', {})
        key_points: list[str] = []
        if simulation.get('mode') == 'general_analysis':
            key_points.extend(simulation.get('decision_axes', [])[:3])
        else:
            key_points.extend([
                f"Delta-v aproximado de {round(float(simulation.get('delta_v_m_s', 0.0)), 2)} m/s",
                f"Altitud estimada de {round(float(simulation.get('max_altitude_m', 0.0)), 2)} m",
                f"Tiempo de combustión cercano a {round(float(simulation.get('burn_time_s', 0.0)), 2)} s",
            ])
        key_points.extend(item.get('summary', '') for item in calculations.get('items', [])[:2] if item.get('summary'))
        notable_risks = list(synthesis.get('contradictions', [])[:2]) + list(synthesis.get('research_gaps', [])[:2])
        recommended_actions = list(ml_result.recommendations[:2]) + list(synthesis.get('recommended_actions', [])[:2])
        design_summary = conclusions if simulation.get('mode') != 'general_analysis' else analysis
        summary = conclusions if len(conclusions) <= len(analysis) else analysis
        direct_answer = conclusions.split('. ')[0].strip() if conclusions else analysis.split('. ')[0].strip()
        return {
            'direct_answer': direct_answer,
            'summary': summary,
            'analysis': analysis,
            'conclusions': conclusions,
            'design_summary': design_summary,
            'key_points': [item for item in key_points if item],
            'notable_risks': [item for item in notable_risks if item],
            'recommended_actions': [item for item in recommended_actions if item],
            'question': question,
        }

    def answer(self, question: str) -> dict[str, Any]:
        recent_context = self.storage.recent_conversations(limit=CONFIG.max_history_messages)
        knowledge = self.knowledge.retrieve(question, recent_context=recent_context)
        inferred_domains = self.external_fetcher.infer_domains(question, knowledge.summary)
        profile = self._profile_question(question, inferred_domains)
        defaults = self._extract_defaults(question)
        run_id = f"sim_{uuid.uuid5(uuid.NAMESPACE_DNS, question).hex[:12]}"
        defaults['run_id'] = run_id

        if profile['requires_simulation']:
            simulation_request = self.simulation.build_request(question, defaults)
            simulation_future = self.background_executor.submit(self.simulation.run, simulation_request, self._progress)
            simulation = simulation_future.result()
        else:
            simulation = self._build_general_analysis_frame(question, profile, knowledge.summary, defaults)

        self.ml_model.train_from_result(simulation, question=question, knowledge_summary=knowledge.summary)
        ml_result = self.ml_model.predict(simulation, question=question, knowledge_summary=knowledge.summary)
        self.external_fetcher.max_queries = min(max(self.external_fetcher.max_queries, ml_result.research_intensity // 2), CONFIG.max_external_queries)
        research_context = f"{knowledge.summary} {' '.join(item.get('question', '') for item in recent_context[:3])}".strip()
        external_future = self.background_executor.submit(
            self.external_fetcher.fetch_research_dossier,
            question,
            research_context,
            ml_result.preferred_domains,
            ml_result.source_weights,
        )

        optimization = None
        if profile['requires_simulation'] and profile['optimization_requested']:
            optimization_future = self.background_executor.submit(self.optimizer.optimize, question, CONFIG.optimizer_iterations, self._progress)
            optimization = optimization_future.result()

        try:
            external = external_future.result(timeout=max(12, CONFIG.internet_timeout_sec * 4))
        except FutureTimeoutError:
            external = self._degraded_external_result(question, knowledge.summary, 'timeout en investigación externa')
        except Exception as exc:
            external = self._degraded_external_result(question, knowledge.summary, safe_error_message(exc))
        self.storage.save_research_session(
            question,
            json.dumps(external, ensure_ascii=False),
            len(external.get('findings', [])),
            float(external.get('synthesis', {}).get('quality_score', 0.0)),
        )
        calculations = self.calculator.analyze(question, simulation, knowledge.summary)

        analysis, conclusions = self._build_analysis(knowledge.summary, simulation, external, recent_context, ml_result)
        analysis = sanitize_text(analysis + self._resume_note())
        payload = {
            'analysis': analysis,
            'conclusions': conclusions,
            'profile': profile,
            'knowledge': {
                'summary': knowledge.summary,
                'formulas': knowledge.formulas,
            },
            'simulation': simulation,
            'ml': {
                'prediction': ml_result.prediction,
                'confidence': ml_result.confidence,
                'hypotheses': ml_result.hypotheses,
                'preferred_domains': ml_result.preferred_domains,
                'research_intensity': ml_result.research_intensity,
                'source_weights': ml_result.source_weights,
                'model_state': ml_result.model_state,
                'variables_considered': ml_result.variables_considered,
                'uncertainty_drivers': ml_result.uncertainty_drivers,
                'recommendations': ml_result.recommendations,
                'reliability_score': ml_result.reliability_score,
            },
            'external': external,
            'calculations': calculations,
            'optimization': optimization,
            'recent_context': [{'question': item['question'], 'created_at': item['created_at']} for item in recent_context[:3]],
        }
        report_path = self.report_writer.save(question, payload)
        analysis_data = self._prepare_response_control_data(question, simulation, calculations, external, ml_result, analysis, conclusions)
        question_type, response_text = build_user_response(question, analysis_data)
        satisfaction = estimate_implicit_satisfaction(question_type, response_text)
        self.storage.append_response_feedback(question_type, len(response_text.split()), satisfaction)
        self.storage.save_conversation(
            question,
            response_text,
            json.dumps({
                'profile': profile,
                'simulation': simulation.get('resource_profile', {}),
                'external_status': external.get('status', 'ok'),
                'question_type': question_type,
                'response_length': len(response_text.split()),
                'satisfaction': satisfaction,
            }, ensure_ascii=False),
        )
        payload['report_path'] = str(report_path)
        payload['response_text'] = response_text
        payload['question_type'] = question_type
        payload['response_length'] = len(response_text.split())
        payload['implicit_satisfaction'] = satisfaction
        gc.collect()
        return payload

    def safe_answer(self, question: str) -> dict[str, Any]:
        try:
            return self.answer(question)
        except MemoryError as exc:
            self.logger.error('Se detectó un fallo de memoria; activando respuesta degradada.')
            gc.collect()
            return {'response_text': self._fallback_response_text(question, exc), 'error': safe_error_message(exc)}
        except OSError as exc:
            if detect_linker_memory_issue(exc):
                self.logger.error('Se detectó saturación del linker; activando respuesta degradada.')
                gc.collect()
                return {'response_text': self._fallback_response_text(question, exc), 'error': safe_error_message(exc)}
            raise
        except ZeroDivisionError as exc:
            self.logger.error('Se evitó una división por cero; activando respuesta degradada.')
            gc.collect()
            return {'response_text': self._fallback_response_text(question, exc), 'error': safe_error_message(exc)}
        except Exception as exc:
            self.logger.exception('Error inesperado al responder')
            gc.collect()
            return {
                'response_text': self._fallback_response_text(question, exc),
                'error': safe_error_message(exc),
            }
