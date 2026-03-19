# Chat Zeus Termux

Sistema modular en Python 3 para Android/Termux que actúa como una **supercomputadora simplificada con chatbot**. Está diseñado para ejecutarse con `python3 main.py`, responder preguntas científicas complejas en modo silencioso y persistir el trabajo automáticamente para reanudar simulaciones largas.

## Características principales

- **Salida silenciosa**: la terminal muestra un encabezado ligero, un prompt claro para escribir y luego solo la respuesta final en texto plano para cada pregunta.
- **Persistencia transparente**: checkpoints JSON y base SQLite para historial, conocimiento, aprendizaje y reanudación.
- **RAG local ligero**: recuperación de fórmulas y conceptos desde una base científica local.
- **Simulación científica simplificada**: gravedad, arrastre, propulsión, trayectoria vertical simplificada y termoquímica básica.
- **ML ligero adaptable**: heurísticas incrementales con autodetección opcional de backends compatibles como TensorFlow Lite o PyTorch.
- **Optimización iterativa**: muestreo ligero con checkpoints, apto para procesos largos en Termux.
- **Tolerancia a errores**: intenta continuar ante datos faltantes, archivos JSON corruptos o fallos parciales.

## Estructura

- `main.py`: entrada principal para ejecutar con `python3 main.py`.
- `src/app.py`: construcción de la aplicación y bucle CLI silencioso.
- `src/chatbot.py`: coordinación de conversación, simulación, ML, búsqueda externa y redacción final.
- `src/knowledge.py`: recuperación local de conocimiento estilo RAG.
- `src/storage.py`: SQLite + checkpoints JSON para reanudación.
- `src/simulation.py`: simulación física/química simplificada con tareas pequeñas y guardado de progreso.
- `src/ml.py`: aprendizaje incremental ligero con degradación elegante si no hay backend móvil.
- `src/optimizer.py`: optimización iterativa por muestreo ligero.
- `src/reporting.py`: generación de la respuesta final en texto plano y reportes JSON.
- `src/termux_ui.py`: interfaz textual ligera con prompt claro y salida limpia para Termux.
- `data/chatbot/`, `data/models/`, `data/data/`, `data/logs/`: carpetas compatibles con el flujo pedido.

## Instalación en Termux

1. Instala Python en Termux:
   ```bash
   pkg update && pkg install python -y
   ```
2. Instala dependencias compatibles y ligeras:
   ```bash
   pip install numpy scipy pandas sympy requests
   ```
3. Opcionalmente añade un backend ML móvil si está disponible para tu dispositivo:
   ```bash
   pip install tflite-runtime
   ```
4. Ejecuta:
   ```bash
   python3 main.py
   ```

> El proyecto funciona sin dependencias pesadas adicionales. Si instalas librerías extra, el sistema las detecta sin exigir cambios en el código.

## Uso

Cada línea de entrada se interpreta como una consulta nueva. El programa mantiene contexto reciente en SQLite y devuelve **solo la respuesta final**.

Ejemplos:

- `Analiza un cohete suborbital con payload=150 fuel=260 thrust=19000 steps=600`
- `Optimiza el diseño de un lanzador ligero para mayor altitud`
- `Calcula una trayectoria simple y resume fórmulas para combustible hipergólico`

Para terminar:

```text
salir
```

## Persistencia y carpetas

Por defecto, el proyecto usa la carpeta `data/` dentro del repositorio. Si quieres emular exactamente el despliegue pedido con rutas tipo `/data/...`, puedes exportar una raíz personalizada antes de ejecutar:

```bash
export CHAT_ZEUS_DATA_ROOT=/data
python3 main.py
```

La estructura creada será:

- `/data/chatbot/`
- `/data/models/`
- `/data/data/`
- `/data/logs/`

## Alcance científico actual

El sistema puede ayudar con preguntas sobre:

- diseño conceptual de naves y lanzadores,
- trayectorias simplificadas,
- gravedad y arrastre básico,
- combustibles y parámetros termoquímicos aproximados,
- hipótesis de mejora y predicciones ligeras.

No sustituye herramientas de alta fidelidad como CFD, análisis estructural detallado ni validación aeroespacial certificada.


## Interfaz en Termux

Al iniciar, la app muestra un encabezado corto y un prompt `Pregunta >` para indicar claramente cuándo introducir la consulta. No se imprimen logs, trazas ni progreso interno en la terminal; esos datos quedan en archivos internos o checkpoints si son necesarios para reanudar procesos largos.
