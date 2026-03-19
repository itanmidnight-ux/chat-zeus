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
    domains: list[str]
    hypotheses: list[str]
    approach: list[str]
    validation: list[str]
    optimization: list[str]


class CreationEngine:
    def build_solution(self, question: str, context: dict[str, str] | None = None) -> str:
        plan = self.plan(question, context=context)
        sections = [
            f"Resultado final: {plan.objective}.",
            f"Explicación breve: Dominios clave: {self._join_items(plan.domains)}.",
            f"Hipótesis viables: {self._join_items(plan.hypotheses)}.",
            f"Diseño propuesto: {self._join_items(plan.approach)}.",
            f"Factibilidad y validación: {self._join_items(plan.validation)}.",
        ]
        if plan.requirements:
            sections.insert(3, f"Requisitos detectados: {self._join_items(plan.requirements)}.")
        if plan.optimization:
            sections.append(f"Mejora futura: {self._join_items(plan.optimization)}.")
        return ' '.join(section for section in sections if section)

    def plan(self, question: str, context: dict[str, str] | None = None) -> CreationPlan:
        text = clean_input(question)
        context = context or {}
        objective = self._extract_objective(text)
        constraints = self._extract_constraints(text)
        requirements = self._extract_requirements(text, context)
        domains = self._infer_domains(text, objective)
        hypotheses = self._build_hypotheses(objective, constraints, requirements, domains)
        approach = self._build_approach(objective, constraints, requirements, domains)
        validation = self._build_validation(objective, constraints, domains)
        optimization = self._build_optimization(constraints, domains)
        return CreationPlan(objective, constraints, requirements, domains, hypotheses, approach, validation, optimization)

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

    def _infer_domains(self, text: str, objective: str) -> list[str]:
        domains: list[str] = []
        domain_map = {
            'physics': ('cohete', 'nave', 'orbita', 'delta-v', 'empuje', 'propuls'),
            'engineering': ('sistema', 'arquitectura', 'control', 'estructura', 'dise'),
            'materials': ('material', 'aleaci', 'compuesto', 'térmic', 'termic'),
            'energy': ('energ', 'bater', 'combustible', 'potencia'),
            'control systems': ('control', 'guiado', 'naveg', 'sensor', 'feedback'),
            'memory': ('memoria', 'memory', 'contexto', 'context'),
            'conversation': ('conversa', 'dialog', 'chat'),
            'validation': ('valid', 'verific', 'test'),
        }
        lowered = f"{text} {objective}".lower()
        for name, markers in domain_map.items():
            if any(marker in lowered for marker in markers):
                domains.append(name)
        return domains or ['engineering', 'validation']

    def _build_hypotheses(self, objective: str, constraints: list[str], requirements: list[str], domains: list[str]) -> list[str]:
        hypotheses = [
            f"usar una arquitectura modular para {objective}",
            'mantener memoria compacta con prioridades para contexto útil',
        ]
        if 'physics' in domains or 'energy' in domains:
            hypotheses.append('evaluar balances aproximados de masa, energía y estabilidad con modelos simplificados')
        if requirements:
            hypotheses.append('añadir una etapa final de validación para asegurar que la salida sea utilizable')
        if any('bajo consumo de recursos' in item for item in constraints):
            hypotheses.append('limitar pasos, tamaño de contexto y almacenamiento para evitar sobrecarga')
        return hypotheses[:4]

    def _build_approach(self, objective: str, constraints: list[str], requirements: list[str], domains: list[str]) -> list[str]:
        steps = [
            f"descomponer '{objective}' en módulos responsables de entender, decidir y responder",
            'representar el estado de la sesión con memoria compacta y actualizable',
            'seleccionar herramientas o cálculos según la intención detectada',
        ]
        if requirements:
            steps.append('incorporar validación previa a la salida para cumplir requisitos detectados')
        if 'physics' in domains or 'engineering' in domains:
            steps.append('comparar alternativas y elegir la de mejor factibilidad con aproximaciones simples')
        if any('recursos' in item for item in constraints):
            steps.append('aplicar límites de longitud y pasos para evitar consumo excesivo')
        return steps

    def _build_validation(self, objective: str, constraints: list[str], domains: list[str]) -> list[str]:
        checks = [
            f"confirmar que la respuesta siga el objetivo '{objective}'",
            'revisar consistencia entre intención, contexto y salida final',
        ]
        if 'physics' in domains or 'energy' in domains:
            checks.append('usar estimaciones simplificadas para comprobar estabilidad, energía o desempeño')
        if constraints:
            checks.append('verificar que las restricciones aparezcan reflejadas en la solución')
        return checks

    def _build_optimization(self, constraints: list[str], domains: list[str]) -> list[str]:
        improvements = ['reducir complejidad operacional manteniendo la funcionalidad principal']
        if 'energy' in domains or 'physics' in domains:
            improvements.append('optimizar masa, consumo energético y margen de seguridad')
        if any('bajo consumo de recursos' in item for item in constraints):
            improvements.append('recortar memoria y pasos de cómputo con procesamiento por bloques')
        return improvements[:3]

    @staticmethod
    def _join_items(items: list[str]) -> str:
        if not items:
            return 'sin elementos adicionales'
        if len(items) == 1:
            return items[0]
        return '; '.join(items)


def generate_solution(question: str, context: dict[str, str] | None = None) -> str:
    return CreationEngine().build_solution(question, context=context)
