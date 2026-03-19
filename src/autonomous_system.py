"""Sistema autónomo modular que orquesta entrada, agentes, aprendizaje y salida."""
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

from src.agents import CriticAgent, MemoryAgent, ReasoningAgent, ResearchAgent, SimulationAgent
from src.core.decomposer import decompose_problem
from src.core.planner import build_research_plan
from src.filters_ext import aggressive_filter
from src.handlers import handle_simple_queries
from src.intent import ANALYTICAL, EXPLICATIVE, SIMPLE, classify_intent
from src.response_control import build_user_response
from src.storage import StorageManager
from src.utils import sanitize_text


@dataclass
class AutonomousResult:
    question: str
    intent: str
    tasks: list[str]
    plan: list[dict[str, Any]]
    research: dict[str, Any]
    best_solution: dict[str, Any]
    memory: dict[str, Any]
    response_text: str


class AutonomousReasoningSystem:
    def __init__(self, storage: StorageManager, memory_agent: MemoryAgent):
        self.storage = storage
        self.memory_agent = memory_agent
        self.research_agent = ResearchAgent(storage)
        self.reasoning_agent = ReasoningAgent()
        self.simulation_agent = SimulationAgent()
        self.critic_agent = CriticAgent()

    def process(self, question: str) -> AutonomousResult:
        normalized = sanitize_text(question)
        direct = handle_simple_queries(normalized)
        if direct:
            return AutonomousResult(normalized, SIMPLE, [], [], {}, {'proposal': direct, 'final_score': 1.0, 'task': 'directo'}, self.memory_agent.load(), aggressive_filter(direct))

        intent = classify_intent(normalized)
        tasks = decompose_problem(normalized)
        plan = build_research_plan(normalized, tasks)
        research = self.research_agent.investigate(normalized, plan)
        hypotheses = self.reasoning_agent.generate_hypotheses(normalized, tasks, research, intent)

        reviewed: list[dict[str, Any]] = []
        for iteration in range(3):
            scenarios = self.simulation_agent.run_scenarios(hypotheses, iteration)
            reviewed = self.critic_agent.evaluate(scenarios)
            if reviewed:
                best = reviewed[0]
                for item in hypotheses:
                    if item['id'] == best['id']:
                        item['feasibility'] = min(1.0, float(item['feasibility']) + 0.015)
                        item['efficiency'] = min(1.0, float(item['efficiency']) + 0.01)
            gc.collect()

        best_solution = reviewed[0] if reviewed else {'proposal': 'No se encontró una solución suficiente.', 'final_score': 0.0, 'task': 'general'}
        memory = self.memory_agent.remember(normalized, best_solution, intent)
        analysis_data = self._build_response_payload(intent, tasks, research, best_solution)
        _, response = build_user_response(normalized, analysis_data)
        response = aggressive_filter(response)
        return AutonomousResult(normalized, intent, tasks, plan, research, best_solution, memory, response)

    def _build_response_payload(self, intent: str, tasks: list[str], research: dict[str, Any], best_solution: dict[str, Any]) -> dict[str, Any]:
        if intent == SIMPLE:
            direct = best_solution.get('proposal', 'Respuesta directa no disponible.')
            return {'direct_answer': direct, 'summary': direct, 'conclusions': direct}
        if intent == EXPLICATIVE:
            summary = f"La idea central es {best_solution.get('task', 'resolver el problema')} de forma gradual. {best_solution.get('proposal', '')}"
            key_points = [
                f"Primero conviene entender {tasks[0]}" if tasks else 'Primero conviene fijar el objetivo',
                'Luego se validan restricciones y riesgos principales.',
                'Finalmente se elige una solución simple y verificable.',
            ]
            return {'summary': summary, 'conclusions': research.get('summary', summary), 'key_points': key_points}
        return {
            'design_summary': best_solution.get('proposal', 'Diseño no disponible.'),
            'summary': f"Conviene priorizar {', '.join(tasks[:3]) or 'objetivo, restricciones y validación'}.",
            'conclusions': f"La mejor ruta es concentrarse en {best_solution.get('task', 'la solución principal')} con foco en seguridad, factibilidad y escalabilidad.",
            'key_points': [f'Prioridad: {task}' for task in tasks[:3]],
            'notable_risks': ['Exceso de complejidad', 'Suposiciones no validadas'],
            'recommended_actions': ['Prototipar primero la parte crítica y validar antes de escalar.'],
        }
