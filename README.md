# Chat Zeus Termux

Sistema modular en Python 3 para Android/Termux que actúa como una **supercomputadora simplificada con chatbot**. El objetivo es responder preguntas complejas mediante recuperación de conocimiento local, simulaciones científicas ligeras, aprendizaje incremental y optimización iterativa con checkpoints.

## Estructura

- `main.py`: entrada principal para ejecutar con `python3 main.py`.
- `src/app.py`: construcción de la aplicación y bucle CLI.
- `src/chatbot.py`: coordinación de la conversación, simulación, ML, búsqueda externa y reportes.
- `src/knowledge.py`: recuperación local de conocimiento estilo RAG.
- `src/storage.py`: SQLite + checkpoints JSON para reanudación.
- `src/simulation.py`: simulación física simplificada con tareas pequeñas y guardado de progreso.
- `src/ml.py`: modelo incremental ligero sin dependencias pesadas.
- `src/optimizer.py`: optimización iterativa por muestreo simple.
- `src/reporting.py`: exportación de reportes JSON legibles.
- `data/chatbot/`, `data/models/`, `data/data/`, `data/logs/`: carpetas compatibles con el flujo pedido.

## Instalación en Termux

1. Instala Python en Termux:
   ```bash
   pkg update && pkg install python -y
   ```
2. Opcionalmente instala librerías ligeras si quieres ampliar el sistema:
   ```bash
   pip install numpy sympy pandas requests
   ```
3. Copia este proyecto en tu almacenamiento o dentro del `$HOME` de Termux.

> El sistema actual funciona solo con la biblioteca estándar de Python, precisamente para minimizar problemas de compilación en Android.

## Ejecución

```bash
python3 main.py
```

Ejemplos de consulta:

- `Analiza un cohete suborbital con payload=150 fuel=260 thrust=19000 steps=600`
- `Optimiza el diseño de un lanzador ligero para mayor altitud`
- `Calcula una trayectoria simple y resume las fórmulas relevantes`

## Qué puede calcular

- Simulaciones simplificadas de ascenso vertical con gravedad y arrastre.
- Estimación de `delta-v`, alcance aproximado, tiempo de combustión y altitud máxima.
- Recuperación de fórmulas locales desde una base SQLite embebida.
- Búsqueda externa opcional de pistas/fórmulas con degradación elegante si no hay internet.
- Aprendizaje incremental ligero a partir de resultados previos.
- Optimización iterativa de parámetros con checkpoints y logs.

## Qué no puede calcular con precisión de ingeniería

- CFD avanzada, combustión detallada, dinámica orbital completa o química de alto detalle.
- Modelos de ML grandes tipo LLM local en 4 GB de RAM.
- Resultados certificados para seguridad aeroespacial real.

## Reanudación de simulaciones largas

- Cada corrida guarda checkpoints JSON en `data/data/checkpoints/`.
- También guarda estado resumido en SQLite (`data/data/knowledge.sqlite3`).
- Si Termux se cierra durante una corrida larga, puedes reutilizar el `run_id` guardado en el checkpoint para reanudar el proceso adaptando la consulta o extendiendo el código.
- Los logs se almacenan en `data/logs/chat_zeus.log`.

## Tolerancia a errores

- Manejo de `MemoryError`, `ZeroDivisionError` y fallos generales.
- Si un archivo JSON está corrupto, se ignora y se reconstruye estado desde valores por defecto.
- La aplicación intenta no cerrarse inesperadamente: informa el error y continúa aceptando nuevas consultas.

## Sugerencias para Android/Termux

- Mantén `steps` entre 300 y 3000 para dispositivos con 4 GB RAM.
- Evita múltiples optimizaciones simultáneas.
- Usa `tmux` si planeas procesos de varias horas o días.
