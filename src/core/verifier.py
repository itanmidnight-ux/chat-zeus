"""Lightweight verification and anti-generic scoring for answers."""
from __future__ import annotations

from dataclasses import dataclass

from src.utils.filters import clean_input


@dataclass
class VerificationResult:
    score: float
    issues: list[str]
    source_count: int
    generic_penalty: float


class VerificationEngine:
    GENERIC_PHRASES = (
        'puedo ayudarte',
        'haz la petición más específica',
        'respuesta útil',
        'more specific',
        'general guidance',
    )

    def verify(self, question: str, answer: str, *, source_count: int = 0, executed: bool = False) -> VerificationResult:
        normalized_question = clean_input(question)
        normalized_answer = clean_input(answer)
        question_tokens = {token for token in normalized_question.split() if len(token) > 3}
        answer_tokens = set(normalized_answer.split())
        overlap = len(question_tokens & answer_tokens)
        coverage = overlap / max(1, min(len(question_tokens), 6))
        issues: list[str] = []
        generic_penalty = 0.0
        if not normalized_answer:
            issues.append('empty_answer')
        if any(phrase in normalized_answer for phrase in self.GENERIC_PHRASES):
            issues.append('generic_answer')
            generic_penalty += 0.25
        if coverage < 0.18 and len(question_tokens) >= 3:
            issues.append('low_topic_overlap')
            generic_penalty += 0.15
        if executed and 'error' in normalized_answer:
            issues.append('execution_error')
            generic_penalty += 0.2
        score = 0.45 + min(0.35, coverage) + min(0.15, source_count * 0.05) - generic_penalty
        return VerificationResult(
            score=round(max(0.05, min(0.99, score)), 4),
            issues=issues,
            source_count=source_count,
            generic_penalty=round(generic_penalty, 4),
        )
