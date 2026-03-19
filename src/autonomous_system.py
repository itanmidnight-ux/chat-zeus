"""Autonomous cognitive system orchestrating the full bounded pipeline."""
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

from src.agents import CriticAgent, MemoryAgent, ReasoningAgent, ResearchAgent, SimulationAgent
from src.chatbot import build_final_answer
from src.core.decomposer import decompose_problem
from src.core.experiment import run_experiments
from src.core.planner import prioritize_tasks
from src.storage import StorageManager
from src.utils.filters import clean_input, clean_output
from src.utils.handlers import handle_simple_queries
from src.utils.intent import classify_intent as classify_english_intent


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
        normalized = clean_input(question)
        try:
            direct = handle_simple_queries(normalized)
            if direct:
                response = clean_output(direct)
                return AutonomousResult(normalized, 'simple', [], [], {}, {'proposal': direct, 'final_score': 1.0, 'task': 'direct'}, self.memory_agent.load(), response)

            intent = classify_english_intent(normalized)
            tasks = decompose_problem(normalized)
            plan = prioritize_tasks(tasks, question=normalized)

            research_blocks = [self.research_agent.research(item) for item in tasks[:5]]
            research = {
                'findings': research_blocks,
                'summary': ' '.join(' '.join(fact.get('fact', '') for fact in block.get('facts', [])[:1]) for block in research_blocks).strip(),
            }

            hypotheses: list[dict[str, Any]] = []
            for task in tasks[:5]:
                hypotheses.extend(self.reasoning_agent.reason({'topic': task, 'keywords': [task], 'research': research}))
            hypotheses = hypotheses[:10] or self.reasoning_agent.reason({'topic': normalized, 'keywords': tasks[:3], 'research': research})

            best_solution = run_experiments(hypotheses, max_iterations=5)
            memory = self.memory_agent.update_memory({**best_solution, 'question': normalized, 'intent': intent})
            response = clean_output(build_final_answer(intent, best_solution))
            gc.collect()
            return AutonomousResult(normalized, intent, tasks, plan, research, best_solution, memory, response)
        except Exception:
            fallback = 'La mejor opción es avanzar con una solución simple, segura y verificable.'
            gc.collect()
            return AutonomousResult(normalized, 'simple', [], [], {}, {'proposal': fallback, 'final_score': 0.0, 'task': 'fallback'}, self.memory_agent.load(), clean_output(fallback))
