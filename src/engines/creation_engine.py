"""Creation engine for formulas, architectures, and constrained solution design."""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.utils.filters import clean_input


@dataclass
class CreationPlan:
    objective: str
    constraints: list[str]
    requirements: list[str]
    approach: list[str]
    validation: list[str]


class CreationEngine:
    def build_solution(self, question: str, context: dict[str, str] | None = None) -> str:
        plan = self.plan(question, context=context)
        sections = [
            f"Objetivo: {plan.objective}.",
            f"Restricciones/criterios: {self._join_items(plan.constraints)}.",
            f"Diseño propuesto: {self._join_items(plan.approach)}.",
            f"Verificación: {self._join_items(plan.validation)}.",
        ]
        if plan.requirements:
            sections.insert(2, f"Requisitos detectados: {self._join_items(plan.requirements)}.")
        return ' '.join(section for section in sections if section)

    def plan(self, question: str, context: dict[str, str] | None = None) -> CreationPlan:
        text = clean_input(question)
        context = context or {}
        objective = self._extract_objective(text)
        constraints = self._extract_constraints(text)
        requirements = self._extract_requirements(text, context)
        approach = self._build_approach(objective, constraints, requirements)
        validation = self._build_validation(objective, constraints)
        return CreationPlan(objective, constraints, requirements, approach, validation)

    def _extract_objective(self, text: str) -> str:
        trigger_patterns = [r'(?:design|build|create|generate|write|diseña|disena|crea|genera|escribe)\s+(.+)']
        for pattern in trigger_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip(' .')[:140]
        return text[:140] or 'resolver la solicitud'

    def _extract_constraints(self, text: str) -> list[str]:
        constraints: list[str] = []
        for connector in ('with', 'using', 'under', 'for', 'con', 'para', 'sin', 'limitado a'):
            if connector in text:
                fragment = text.split(connector, 1)[1].split(',')[0].strip()
                if fragment:
                    constraints.append(fragment[:90])
        if re.search(r'\b(low|bajo|ligero|minimal|minimo|mínimo)\b', text):
            constraints.append('mantener bajo consumo de recursos')
        if re.search(r'\b(modular|modularidad)\b', text):
            constraints.append('separar componentes con interfaces limpias')
        return constraints or ['alinear la solución con la petición del usuario']

    def _extract_requirements(self, text: str, context: dict[str, str]) -> list[str]:
        requirements: list[str] = []
        for keyword in ('memory', 'memoria', 'context', 'contexto', 'validation', 'validación', 'conversation', 'conversación', 'safety', 'seguridad'):
            if keyword in text:
                requirements.append(keyword)
        if context.get('name'):
            requirements.append(f"personalizar con el usuario {context['name']}")
        return requirements

    def _build_approach(self, objective: str, constraints: list[str], requirements: list[str]) -> list[str]:
        steps = [
            f"descomponer '{objective}' en módulos responsables de entender, decidir y responder",
            'representar el estado de la sesión con memoria compacta y actualizable',
            'seleccionar herramientas o cálculos según la intención detectada',
        ]
        if requirements:
            steps.append('incorporar validación previa a la salida para cumplir requisitos detectados')
        if any('recursos' in item for item in constraints):
            steps.append('aplicar límites de longitud y pasos para evitar consumo excesivo')
        return steps

    def _build_validation(self, objective: str, constraints: list[str]) -> list[str]:
        checks = [
            f"confirmar que la respuesta siga el objetivo '{objective}'",
            'revisar consistencia entre intención, contexto y salida final',
        ]
        if constraints:
            checks.append('verificar que las restricciones aparezcan reflejadas en la solución')
        return checks

    @staticmethod
    def _join_items(items: list[str]) -> str:
        if not items:
            return 'sin elementos adicionales'
        if len(items) == 1:
            return items[0]
        return '; '.join(items)


def generate_solution(question: str, context: dict[str, str] | None = None) -> str:
    return CreationEngine().build_solution(question, context=context)
