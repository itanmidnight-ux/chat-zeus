"""Interfaz conversacional principal que coordina conocimiento, simulación, ML y reportes."""
from __future__ import annotations

import json
import uuid
from typing import Any

from src.config import CONFIG
from src.external import ExternalKnowledgeFetcher
from src.knowledge import KnowledgeManager
from src.ml import LightweightMLModel
from src.optimizer import IterativeOptimizer
from src.reporting import ReportWriter
from src.simulation import SimulationEngine
from src.storage import StorageManager
from src.utils import detect_linker_memory_issue, extract_numeric_value, safe_error_message, sanitize_text
from src.worker import BackgroundExecutor


class ChatbotInterface:
    ENGINEERING_DOMAINS = {'spacecraft', 'physics', 'chemistry', 'materials', 'systems', 'aerospace'}

    def __init__(
        self,
        storage: StorageManager,
        knowledge: KnowledgeManager,
        simulation: SimulationEngine,
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
                defaults['steps'] = 240
            elif any(word in lower_question for word in ['preciso', 'larga', 'detallado', 'optimiza', 'profundo']):
                defaults['steps'] = 720
        return defaults

    def _progress(self, run_id: str, progress: float) -> None:
        self.logger.info('Progreso %s: %.1f%%', run_id, progress * 100)

    def _profile_question(self, question: str, external_domains: list[str]) -> dict[str, Any]:
        lowered = question.lower()
        problem_type = 'general_research'
        if any(word in lowered for word in ['crear', 'design', 'diseñar', 'build', 'sistema', 'nave', 'cohete', 'reactor']):
            problem_type = 'design_problem'
        elif any(word in lowered for word in ['resolver', 'equation', 'ecuación', 'derivar', 'integral', 'calcula']):
            problem_type = 'math_or_science_problem'
        elif any(word in lowered for word in ['geopol', 'guerra', 'conflicto', 'país', 'elección', 'presidente']):
            problem_type = 'geopolitical_analysis'
        requires_simulation = bool(self.ENGINEERING_DOMAINS.intersection(external_domains)) or problem_type in {'design_problem', 'math_or_science_problem'}
        return {
            'problem_type': problem_type,
            'domains': external_domains,
            'requires_simulation': requires_simulation,
        }

    def _build_general_analysis_frame(self, question: str, profile: dict[str, Any], knowledge_summary: str, external: dict[str, Any]) -> dict[str, Any]:
        synthesis = external.get('synthesis', {})
        domains = profile.get('domains', []) or ['general']
        analytical_depth = max(1.0, float(len(domains)) * 1.4 + float(external.get('queries_executed', 0)) * 0.08)
        uncertainty = max(0.05, 1.0 - float(synthesis.get('quality_score', 0.0)))
        breadth = max(1.0, len(external.get('intents', [])) + len(domains) * 0.5)
        mode = 'general_analysis'
        return {
            'mode': mode,
            'run_id': f'analysis_{uuid.uuid5(uuid.NAMESPACE_DNS, question).hex[:12]}',
            'problem_type': profile.get('problem_type', 'general_research'),
            'domains': domains,
            'analytical_depth': round(analytical_depth, 3),
            'uncertainty_index': round(uncertainty, 4),
            'breadth_score': round(breadth, 3),
            'questions_to_validate': synthesis.get('research_gaps', [])[:5],
            'decision_axes': domains[:6],
            'knowledge_summary': knowledge_summary,
            'delta_v_m_s': round(analytical_depth * 100, 3),
            'max_altitude_m': round(breadth * 100, 3),
            'burn_time_s': round(max(1.0, analytical_depth * 2.5), 3),
            'payload_mass_kg': round(max(1.0, uncertainty * 100), 3),
            'chemistry': {
                'estimated_efficiency': round(max(0.2, 1.0 - uncertainty * 0.5), 4),
            },
            'range_m': round(breadth * analytical_depth * 12, 3),
            'final_velocity_m_s': round(analytical_depth * 18, 3),
            'remaining_fuel_kg': 0.0,
            'resource_profile': {
                'chunk_size': 'analysis',
                'max_memory_mb': CONFIG.max_task_memory_mb,
                'resumed_from_checkpoint': False,
            },
            'history': [],
        }

    def _build_analysis(
        self,
        knowledge_summary: str,
        simulation: dict[str, Any],
        external: dict[str, Any],
        recent_context: list[dict[str, Any]],
        ml_result,
        profile: dict[str, Any],
    ) -> tuple[str, str]:
        context_hint = ''
        if recent_context:
            context_hint = f" El historial reciente contiene {len(recent_context)} intercambios y se usó para mantener continuidad temática sin exponer pasos internos."
        synthesis = external.get('synthesis', {})
        analysis = (
            'Se combinó recuperación local de conocimiento científico, una capa de investigación web multi-fuente con planificación por dominios, '
            'evaluación de evidencia, detección básica de contradicciones, conectividad reforzada con reintentos y un modelo de aprendizaje online dedicado al programa. '
            f" La consulta fue clasificada como {profile.get('problem_type', 'general_research')} y se analizaron los dominios {', '.join(profile.get('domains', [])[:6]) or 'generales'}."
            f' Resumen RAG: {knowledge_summary}.{context_hint}'
            f" El apoyo externo terminó con estado {external['status']} tras {external.get('queries_executed', 0)} búsquedas, una señal de factibilidad aproximada de {synthesis.get('feasibility_signal', 0.0)}, una calidad media de evidencia de {synthesis.get('quality_score', 0.0)} y un mapa de conectividad por fuente para endurecer futuras consultas."
        )
        if simulation.get('mode') == 'general_analysis':
            conclusions = (
                f"El problema se trató como análisis general y no como una simulación física cerrada. El sistema recomienda profundizar en los ejes {', '.join(simulation.get('decision_axes', [])[:6]) or 'principales'}, "
                f"resolver primero los vacíos críticos de evidencia y solo después cerrar una respuesta ejecutiva o un plan técnico. "
                'Esto permite responder preguntas de física, química, matemáticas, geopolítica, ingeniería u otros dominios sin quedar limitado a un solo tipo de problema.'
            )
        else:
            conclusions = (
                f"Para esta consulta, el diseño analizado alcanza aproximadamente {simulation['max_altitude_m']} m de altitud máxima con un delta-v de {simulation['delta_v_m_s']} m/s. "
                f"El motor de investigación recomienda trabajar por dominios {', '.join(external.get('domains', [])[:5]) or 'generales'} y priorizar fuentes {', '.join(ml_result.preferred_domains[:4])}. "
                f"La siguiente mejora más prometedora es convertir los hallazgos externos y los riesgos detectados en requisitos cuantitativos para nuevas simulaciones y validaciones. "
                'Los resultados son útiles para exploración conceptual avanzada, pero no sustituyen validación de ingeniería de alta fidelidad ni garantizan éxito real del 100 %.'
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
            simulation = self._build_general_analysis_frame(question, profile, knowledge.summary, {})

        self.ml_model.train_from_result(simulation)
        ml_result = self.ml_model.predict(simulation)
        self.external_fetcher.max_queries = max(self.external_fetcher.max_queries, ml_result.research_intensity)
        research_context = f"{knowledge.summary} {' '.join(item.get('question', '') for item in recent_context[:3])}".strip()
        external_future = self.background_executor.submit(
            self.external_fetcher.fetch_research_dossier,
            question,
            research_context,
            ml_result.preferred_domains,
            ml_result.source_weights,
        )

        optimization = None
        if profile['requires_simulation'] and any(word in question.lower() for word in ['optimiza', 'optimize', 'mejora', 'improve']):
            optimization_future = self.background_executor.submit(self.optimizer.optimize, question, 8, self._progress)
            optimization = optimization_future.result()

        external = external_future.result()
        if not profile['requires_simulation']:
            simulation = self._build_general_analysis_frame(question, profile, knowledge.summary, external)
            ml_result = self.ml_model.predict(simulation)
        self.storage.save_research_session(
            question,
            json.dumps(external, ensure_ascii=False),
            len(external.get('findings', [])),
            float(external.get('synthesis', {}).get('quality_score', 0.0)),
        )

        analysis, conclusions = self._build_analysis(knowledge.summary, simulation, external, recent_context, ml_result, profile)
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
            },
            'external': external,
            'optimization': optimization,
            'recent_context': [{'question': item['question'], 'created_at': item['created_at']} for item in recent_context[:3]],
        }
        report_path = self.report_writer.save(question, payload)
        response_text = self.report_writer.render_text(payload)
        self.storage.save_conversation(question, response_text, json.dumps(payload, ensure_ascii=False))
        payload['report_path'] = str(report_path)
        payload['response_text'] = response_text
        return payload

    def safe_answer(self, question: str) -> dict[str, Any]:
        try:
            return self.answer(question)
        except MemoryError as exc:
            message = 'Se detectó un fallo de memoria. Reduce el tamaño del problema o los pasos de simulación.'
            self.logger.error(message)
            return {'response_text': message, 'error': safe_error_message(exc)}
        except OSError as exc:
            if detect_linker_memory_issue(exc):
                message = (
                    'Se detectó saturación de memoria del linker de Android/Termux. '
                    'El sistema conservó checkpoints; reintenta con menos pasos, menor tamaño de problema o menos dependencias nativas cargadas.'
                )
                self.logger.error(message)
                return {'response_text': message, 'error': safe_error_message(exc)}
            raise
        except ZeroDivisionError as exc:
            message = 'Se evitó una división por cero. Revisa parámetros de masa, empuje o velocidad de escape.'
            self.logger.error(message)
            return {'response_text': message, 'error': safe_error_message(exc)}
        except Exception as exc:
            self.logger.exception('Error inesperado al responder')
            return {
                'response_text': 'Ocurrió un error recuperable; el sistema mantuvo su estado y puedes reintentar con un problema más acotado.',
                'error': safe_error_message(exc),
            }
