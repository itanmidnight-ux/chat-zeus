"""Safe Python sandbox with restricted builtins and timeout protection."""
from __future__ import annotations

import contextlib
import io
import multiprocessing as mp
from typing import Any

_ALLOWED_BUILTINS = {
    'abs': abs,
    'pow': pow,
    'range': range,
    'len': len,
    'min': min,
    'max': max,
    'sum': sum,
    'print': print,
}
_BLOCKED_TOKENS = ('import', 'open(', '__', 'os.', 'sys.', 'socket', 'subprocess', 'requests', 'pathlib', 'eval(', 'exec(')


def _sandbox_worker(code: str, queue: mp.Queue[str]) -> None:
    stdout = io.StringIO()
    safe_globals: dict[str, Any] = {'__builtins__': _ALLOWED_BUILTINS}
    safe_locals: dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, safe_globals, safe_locals)
        queue.put(stdout.getvalue().strip() or str(safe_locals.get('result', '')))
    except Exception as exc:
        queue.put(f'Execution error: {exc}')


def execute_code_safely(code: str, timeout: float = 2.0) -> str:
    normalized = code.strip()
    if not normalized:
        return ''
    lowered = normalized.lower()
    if any(token in lowered for token in _BLOCKED_TOKENS):
        return 'Execution blocked: unsafe code.'
    queue: mp.Queue[str] = mp.Queue()
    process = mp.Process(target=_sandbox_worker, args=(normalized, queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(0.2)
        return 'Execution blocked: timeout.'
    return queue.get() if not queue.empty() else ''
