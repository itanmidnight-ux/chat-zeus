"""Comprehensive cognitive architecture for Chat Zeus.

This module implements a complete layered cognitive system that follows the
loop:

UNDERSTAND -> INTERPRET -> PLAN -> EXECUTE -> EVALUATE -> LEARN -> RESPOND
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

from src.agents.critic import CriticAgent
from src.agents.reasoning import ReasoningAgent
from src.agents.research import ResearchAgent
from src.agents.simulation import SimulationAgent
from src.core.decision import DecisionEngine
from src.core.executor import TaskExecutor
from src.core.learning import LearningEngine
from src.core.memory import LightweightMemory
from src.core.planner import build_research_plan
from src.core.understanding import SemanticUnderstandingEngine, UnderstandingResult
from src.core.verifier import VerificationEngine, VerificationResult
from src.engines.creation_engine import CreationEngine
from src.storage import StorageManager


@dataclass
class PerceptionResult:
    raw_text: str
    cleaned_text: str
    normalized_text: str
    noise_detected: bool
    broken_sentence: bool


class PerceptionLayer:
    """Pre-process text before semantic interpretation."""

    _SPACE_RE = re.compile(r"\s+")
    _NOISE_RE = re.compile(r"[^0-9a-záéíóúñü.,;:!?()_\-+/*%^=<> ]", re.IGNORECASE)

    def process_input(self, text: str) -> PerceptionResult:
        raw_text = text or ""
        cleaned = raw_text.replace("\n", " ").replace("\t", " ").strip()
        noise_detected = bool(self._NOISE_RE.search(cleaned))
        cleaned = self._NOISE_RE.sub(" ", cleaned)
        cleaned = self._SPACE_RE.sub(" ", cleaned).strip()
        normalized = cleaned.lower()
        broken_sentence = len(normalized.split()) <= 3 or normalized.endswith((" y", " and", " con", " with"))
        return PerceptionResult(
            raw_text=raw_text,
            cleaned_text=cleaned,
            normalized_text=normalized,
            noise_detected=noise_detected,
            broken_sentence=broken_sentence,
        )


class ContextEngine:
    """Short-term memory for dialogue state."""

    def __init__(self, max_history: int = 12):
        self.max_history = max(2, max_history)
        self.context: dict[str, Any] = {
            "history": [],
            "user_data": {},
            "current_goal": None,
        }

    def update(self, user_message: str, system_response: str | None = None, inferred_user_data: dict[str, str] | None = None, goal: str | None = None) -> None:
        self.context["history"].append({"user": user_message, "assistant": system_response or ""})
        self.context["history"] = self.context["history"][-self.max_history :]
        if inferred_user_data:
            self.context["user_data"].update(inferred_user_data)
        if goal is not None:
            self.context["current_goal"] = goal

    def snapshot(self) -> dict[str, Any]:
        return {
            "history": list(self.context["history"]),
            "user_data": dict(self.context["user_data"]),
            "current_goal": self.context["current_goal"],
        }


class LongTermMemorySystem:
    """Compressed persistent knowledge with relevance retrieval."""

    def __init__(self, memory: LightweightMemory, max_size: int = 500):
        self.memory = memory
        self.max_size = max(32, max_size)

    def store(self, bucket: str, query: str, value: str | dict[str, Any], *, source: str = "local") -> None:
        compact = self._compress(value)
        self.memory.put(bucket, query, compact, source=source)

    def retrieve_by_relevance(self, query: str, buckets: tuple[str, ...] = ("facts", "solutions", "patterns")) -> list[dict[str, Any]]:
        terms = {token for token in query.lower().split() if len(token) > 2}
        ranked: list[dict[str, Any]] = []
        exported = self.memory.export()
        for bucket in buckets:
            for item in exported.get(bucket, {}).values():
                haystack = json.dumps(item, ensure_ascii=False).lower()
                score = sum(1 for term in terms if term in haystack)
                if score:
                    ranked.append({**item, "bucket": bucket, "relevance": score})
        return sorted(ranked, key=lambda row: row["relevance"], reverse=True)[:5]

    def _compress(self, value: str | dict[str, Any]) -> str | dict[str, Any]:
        if isinstance(value, dict):
            compact = dict(value)
            if "value" in compact and isinstance(compact["value"], str):
                compact["value"] = self._compress_text(compact["value"])
            if "proposal" in compact and isinstance(compact["proposal"], str):
                compact["proposal"] = self._compress_text(compact["proposal"])
            return compact
        return self._compress_text(value)

    @staticmethod
    def _compress_text(text: str, limit: int = 240) -> str:
        normalized = " ".join(str(text).split()).strip()
        if len(normalized) <= limit:
            return normalized
        sentences = re.split(r"(?<=[.!?])\s+", normalized)
        compact = " ".join(sentences[:2]).strip() or normalized[:limit]
        return compact[:limit].rstrip() + ("..." if len(compact) > limit else "")


class GoalEngine:
    """Translate interpreted meaning into an actionable goal."""

    GOAL_MAP = {
        "math": "solve_problem",
        "fact": "answer_fact",
        "creation": "generate_new_solution",
        "analysis": "analyze_problem",
        "execution": "execute_safely",
        "conversation": "hold_natural_conversation",
        "identity": "describe_identity",
        "clarification_needed": "request_missing_information",
        "simple": "respond_directly",
    }

    def determine_goal(self, understanding: UnderstandingResult) -> str:
        return self.GOAL_MAP.get(understanding.selected_intent, "respond_directly")


class PlanningEngine:
    """Build a structured execution plan from a goal and interpretation."""

    def build_plan(self, goal: str, understanding: UnderstandingResult) -> list[dict[str, Any]]:
        tasks = understanding.tasks or ["understand request", "prepare response"]
        prioritized = build_research_plan(understanding.normalized_question, tasks)
        plan = [{"step": "understand", "status": "completed", "detail": understanding.selected_intent}]
        for item in prioritized[:5]:
            plan.append(
                {
                    "step": item["task"],
                    "status": "pending",
                    "priority": item["priority"],
                    "critical": item["critical"],
                    "goal": goal,
                }
            )
        plan.append({"step": "validate_response", "status": "pending", "priority": 1.0, "critical": True, "goal": goal})
        return plan


class DecisionAgent:
    """Select the best candidate after critique."""

    def choose(self, evaluated: list[dict[str, Any]]) -> dict[str, Any]:
        if not evaluated:
            return {"proposal": "No se generó una solución suficiente.", "final_score": 0.1}
        return max(evaluated, key=lambda item: (item.get("final_score", 0.0), item.get("safety", 0.0)))


@dataclass
class CognitiveCycleResult:
    perception: PerceptionResult
    understanding: UnderstandingResult
    goal: str
    plan: list[dict[str, Any]]
    research: dict[str, Any]
    hypotheses: list[dict[str, Any]]
    simulations: list[dict[str, Any]]
    critiques: list[dict[str, Any]]
    decision: dict[str, Any]
    execution_output: str
    verification: VerificationResult
    learning_notes: dict[str, Any]
    response_text: str
    clarification_question: str | None = None


class InteractiveClarificationEngine:
    def clarify(self, understanding: UnderstandingResult, perception: PerceptionResult) -> str | None:
        if understanding.selected_intent == "clarification_needed":
            return "Necesito más contexto para continuar. ¿Cuál es el objetivo exacto y qué restricciones debo respetar?"
        if perception.broken_sentence and understanding.selected_intent in {"creation", "analysis", "execution"}:
            return "Tu pedido parece incompleto. ¿Puedes indicar objetivo, restricciones y formato esperado?"
        return None


class InternalValidationEngine:
    def __init__(self) -> None:
        self.verifier = VerificationEngine()

    def validate(self, question: str, answer: str, *, source_count: int = 0, executed: bool = False) -> VerificationResult:
        return self.verifier.verify(question, answer, source_count=source_count, executed=executed)


class ResponseEngine:
    def respond(self, *, intent: str, content: str, verification: VerificationResult) -> str:
        response = content.strip()
        if verification.score < 0.45 and "conviene verificar" not in response.lower():
            response = f"{response} Conviene verificar los puntos clave antes de usar esta respuesta."
        if intent == "conversation":
            return response[:420]
        return response


class LearningEngineAdapter:
    """Incremental and meta-learning wrapper around existing storage/memory."""

    def __init__(self, memory: LongTermMemorySystem, learning_engine: LearningEngine, storage: StorageManager):
        self.memory = memory
        self.learning_engine = learning_engine
        self.storage = storage

    def learn(self, question: str, result: dict[str, Any], verification: VerificationResult, pattern_key: str, route: str) -> dict[str, Any]:
        bucket = "solutions" if verification.score >= 0.45 else "failures"
        self.memory.store(bucket, question, result, source=route)
        self.storage.save_learned_pattern(
            pattern_key=pattern_key,
            intent=result.get("intent", "unknown"),
            route=route,
            confidence=max(0.01, float(result.get("final_score", verification.score))),
            support_count=len(self.storage.load_recent_episodes(pattern_key=pattern_key, limit=20)) + 1,
            sample_question=question,
        )
        strategy = self.storage.load_strategy_stat(route, pattern_key) or {}
        success_rate = float(strategy.get("success_rate", 0.0))
        note = "reinforce" if verification.score >= 0.55 else "deprioritize"
        return {"pattern_key": pattern_key, "route": route, "meta_strategy": note, "previous_success_rate": success_rate}


class CognitiveSystem:
    """Complete multi-layer cognitive agent."""

    def __init__(self, storage: StorageManager, long_term_memory: LightweightMemory, task_executor: TaskExecutor, learning_engine: LearningEngine, *, max_history: int = 12):
        self.storage = storage
        self.perception = PerceptionLayer()
        self.context_engine = ContextEngine(max_history=max_history)
        self.long_term_memory = LongTermMemorySystem(long_term_memory, max_size=long_term_memory.limit)
        self.semantic_engine = SemanticUnderstandingEngine()
        self.goal_engine = GoalEngine()
        self.planning_engine = PlanningEngine()
        self.decision_policy = DecisionEngine(max_memory_mb=256, max_external_queries=4)
        self.research_agent = ResearchAgent(storage)
        self.reasoning_agent = ReasoningAgent()
        self.simulation_agent = SimulationAgent()
        self.critic_agent = CriticAgent()
        self.decision_agent = DecisionAgent()
        self.executor = task_executor
        self.creation_engine = CreationEngine()
        self.clarifier = InteractiveClarificationEngine()
        self.validator = InternalValidationEngine()
        self.response_engine = ResponseEngine()
        self.learning_adapter = LearningEngineAdapter(self.long_term_memory, learning_engine, storage)

    def process(self, text: str) -> CognitiveCycleResult:
        perception = self.perception.process_input(text)
        understanding = self.semantic_engine.analyze(perception.cleaned_text)
        goal = self.goal_engine.determine_goal(understanding)
        self.context_engine.update(perception.cleaned_text, inferred_user_data=understanding.inferred_profile, goal=goal)
        clarification = self.clarifier.clarify(understanding, perception)
        if clarification:
            verification = self.validator.validate(perception.cleaned_text, clarification)
            response = self.response_engine.respond(intent="clarification_needed", content=clarification, verification=verification)
            return CognitiveCycleResult(
                perception=perception,
                understanding=understanding,
                goal=goal,
                plan=[{"step": "clarify", "status": "completed"}],
                research={},
                hypotheses=[],
                simulations=[],
                critiques=[],
                decision={"proposal": clarification, "final_score": verification.score, "route": "clarify"},
                execution_output=clarification,
                verification=verification,
                learning_notes={"pattern_key": understanding.pattern_key, "route": "clarify", "meta_strategy": "ask_user"},
                response_text=response,
                clarification_question=clarification,
            )

        plan = self.planning_engine.build_plan(goal, understanding)
        research = self.research_agent.investigate(perception.cleaned_text, build_research_plan(perception.cleaned_text, understanding.tasks))
        hypotheses = self.reasoning_agent.generate_hypotheses(perception.cleaned_text, understanding.tasks, research, understanding.selected_intent)
        simulations = self.simulation_agent.run_scenarios(hypotheses, iteration=1) if hypotheses else []
        critiques = self.critic_agent.evaluate(simulations) if simulations else []
        decision = self.decision_agent.choose(critiques)
        policy = self.decision_policy.decide(understanding, hot_memory=self.long_term_memory.retrieve_by_relevance(perception.cleaned_text)[:1])
        context = self.context_engine.snapshot()["user_data"] | {"assistant_name": "Chat Zeus"}
        execution_output = self._execute(understanding, perception.cleaned_text, context, decision)
        source_count = len(research.get("findings", []))
        verification = self.validator.validate(
            perception.cleaned_text,
            execution_output,
            source_count=source_count,
            executed=policy.engine == "execution",
        )
        learning_notes = self.learning_adapter.learn(
            perception.cleaned_text,
            {"proposal": execution_output, "final_score": decision.get("final_score", verification.score), "intent": understanding.selected_intent},
            verification,
            understanding.pattern_key,
            policy.route,
        )
        response = self.response_engine.respond(intent=understanding.selected_intent, content=execution_output, verification=verification)
        self.context_engine.update(perception.cleaned_text, system_response=response, goal=goal)
        return CognitiveCycleResult(
            perception=perception,
            understanding=understanding,
            goal=goal,
            plan=plan,
            research=research,
            hypotheses=hypotheses,
            simulations=simulations,
            critiques=critiques,
            decision={**decision, "route": policy.route, "engine": policy.engine},
            execution_output=execution_output,
            verification=verification,
            learning_notes=learning_notes,
            response_text=response,
        )

    def _execute(self, understanding: UnderstandingResult, question: str, context: dict[str, str], decision: dict[str, Any]) -> str:
        if understanding.selected_intent in {"creation", "analysis"}:
            solution = self.creation_engine.build_solution(question, context=context)
            if decision.get("proposal"):
                return f"{solution} Alternativa priorizada: {decision['proposal']}"
            return solution
        return self.executor.execute_task(understanding.selected_intent, question, context=context)


def execute_code_safely(code: str, timeout: float = 2.0, memory_limit_mb: int = 64) -> str:
    """Convenience wrapper that exposes the sandbox from the cognitive architecture module."""

    from src.sandbox.executor import execute_code_safely as sandbox_execute

    return sandbox_execute(code, timeout=timeout, memory_limit_mb=memory_limit_mb)
