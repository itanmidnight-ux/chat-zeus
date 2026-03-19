"""Interfaz textual ligera y silenciosa optimizada para Linux y Termux."""
from __future__ import annotations

import os
from dataclasses import dataclass

from src.config import CONFIG


@dataclass
class TermuxPalette:
    accent: str = '\033[96m'
    response: str = '\033[92m'
    subtle: str = '\033[90m'
    reset: str = '\033[0m'
    bold: str = '\033[1m'


class TermuxUI:
    def __init__(self, colors: bool | None = None):
        if colors is None:
            colors = self._supports_color()
        self.palette = TermuxPalette() if colors else TermuxPalette('', '', '', '', '')

    def _supports_color(self) -> bool:
        if os.environ.get('NO_COLOR'):
            return False
        term = os.environ.get('TERM', '').lower()
        return bool(term) and term != 'dumb'

    def prompt(self) -> str:
        p = self.palette
        return f"{p.bold}{p.accent}Pregunta > {p.reset}"

    def render_welcome(self) -> str:
        p = self.palette
        return (
            f"{p.bold}{p.accent}Chat Zeus Linux{p.reset}\n"
            f"{p.subtle}Listo para preguntas en lenguaje natural. Perfil detectado: {CONFIG.runtime_profile.cpu_count} CPU y {CONFIG.runtime_profile.total_memory_mb} MB; usa checkpoints silenciosos y muestra solo la respuesta final. Escribe 'salir' para terminar.{p.reset}"
        )

    def render_response(self, text: str) -> str:
        p = self.palette
        return f"{p.response}{text}{p.reset}" if text else text
