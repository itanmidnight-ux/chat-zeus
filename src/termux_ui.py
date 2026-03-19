"""Interfaz textual ligera y silenciosa optimizada para Termux."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TermuxPalette:
    accent: str = '\033[96m'
    response: str = '\033[92m'
    subtle: str = '\033[90m'
    reset: str = '\033[0m'
    bold: str = '\033[1m'


class TermuxUI:
    def __init__(self, colors: bool = True):
        self.palette = TermuxPalette() if colors else TermuxPalette('', '', '', '', '')

    def prompt(self) -> str:
        p = self.palette
        return f"{p.bold}{p.accent}Pregunta > {p.reset}"

    def render_welcome(self) -> str:
        p = self.palette
        return (
            f"{p.bold}{p.accent}Chat Zeus Termux{p.reset}\n"
            f"{p.subtle}Escribe tu consulta científica y presiona Enter. Usa 'salir' para terminar.{p.reset}"
        )

    def render_response(self, text: str) -> str:
        p = self.palette
        return f"{p.response}{text}{p.reset}" if text else text
