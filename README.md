# Chat Zeus Linux

Sistema modular en Python 3 optimizado para **Linux de escritorio, Kali, servidores ligeros y también Termux**, con ejecución silenciosa, checkpoints reanudables y uso más eficiente de CPU/RAM que la variante original para Android.

## Características principales

- **Compatibilidad Linux**: perfil automático de CPU y memoria para ajustar workers, chunks de simulación y presupuesto de pasos.
- **Salida silenciosa**: la terminal muestra solo el prompt y la respuesta final en texto plano, sin logs internos.
- **Persistencia transparente**: checkpoints JSON y base SQLite para historial, conocimiento, aprendizaje y reanudación.
- **Reanudación tras reinicios**: simulaciones, optimización y estado ML se guardan en SQLite y en archivos JSON dentro de `models/` y `data/`.
- **ML incremental ligero**: aprendizaje online dedicado, persistencia de pesos y degradación segura si no hay backends opcionales.
- **Simulación científica por bloques**: física, química y matemáticas ligeras con control de memoria y guardado periódico.
- **Optimización iterativa**: muestreo guiado con checkpoints para no recalcular desde cero.
- **Tolerancia a errores**: degradación silenciosa ante `MemoryError`, fallos de subprocesos o problemas de linker.

## Estructura

- `main.py`: entrada principal para ejecutar con `python3 main.py`.
- `src/app.py`: construcción de la aplicación y bucle CLI silencioso.
- `src/chatbot.py`: coordinación de conversación, simulación, ML, búsqueda externa y redacción final.
- `src/ml.py`: aprendizaje incremental ligero con persistencia dual en SQLite y JSON.
- `src/simulation.py`: simulación física/química simplificada en bloques adaptativos.
- `src/optimizer.py`: optimización iterativa con checkpoints.
- `src/storage.py`: SQLite + checkpoints JSON para reanudación.
- `src/termux_ui.py`: interfaz textual ligera y compatible con Linux/Termux.
- `data/chatbot/`, `data/models/`, `data/data/`, `data/logs/`: carpetas compatibles con el flujo pedido.

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

## Ajustes opcionales por entorno

Puedes ajustar el perfil sin tocar el código:

```bash
export CHAT_ZEUS_MAX_WORKERS=4
export CHAT_ZEUS_MAX_TASK_MEMORY_MB=2048
export CHAT_ZEUS_HARD_STEP_CAP=3000
export CHAT_ZEUS_OPT_ITERATIONS=16
python3 main.py
```

## Salida y persistencia

- La consola solo muestra preguntas y respuestas finales en texto plano.
- Los checkpoints quedan en `data/data/checkpoints/`.
- Los reportes quedan en `data/data/reports/`.
- Los pesos del ML quedan en `data/models/lightweight_ml_state.json` además de SQLite.

## Uso

Ejemplos:

- `derivada de x^3 + 2*x en x=4`
- `Analiza un cohete suborbital con payload=150 fuel=260 thrust=19000 steps=1200`
- `Optimiza el diseño de un lanzador ligero para mayor altitud`
- `analiza geología con densidad=2500 espesor=1200`

Para terminar:

```text
salir
```
