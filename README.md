# Chat Zeus Termux

Sistema modular en Python 3 para Android/Termux que actúa como una **supercomputadora simplificada con chatbot**. Está diseñado para ejecutarse con `python3 main.py`, responder preguntas científicas complejas en modo silencioso y persistir el trabajo automáticamente para reanudar simulaciones largas.

## Características principales

- **Salida silenciosa**: la terminal muestra un encabezado ligero, un prompt claro para escribir y luego solo la respuesta final en texto plano para cada pregunta.
- **Persistencia transparente**: checkpoints JSON y base SQLite para historial, conocimiento, aprendizaje y reanudación.
- **Reanudación tras reinicios**: cada corrida usa `run_id` estable derivado de la pregunta, de modo que puede continuar después de cerrar Termux o reiniciar Android.
- **RAG local + investigación web multi-fuente profesional**: recuperación local de fórmulas y conceptos, más búsquedas orquestadas en fuentes abiertas como Wikipedia, Crossref, arXiv y DuckDuckGo, con planificación por dominios, cobertura por intenciones, puntuación de evidencia y síntesis de riesgos.
- **Simulación científica simplificada**: gravedad, arrastre, propulsión, trayectoria vertical simplificada y termoquímica básica.
- **ML dedicado y adaptable**: aprendizaje online específico del programa, con pesos propios, memoria persistente, priorización de fuentes, intensidad de investigación por consulta y posibilidad de ampliar con TensorFlow Lite o PyTorch.
- **Optimización iterativa**: muestreo ligero con checkpoints, apto para procesos largos en Termux.
- **Tolerancia a errores**: intenta continuar ante datos faltantes, archivos JSON corruptos, `MemoryError` y saturación del linker de Android (`create_new_page ... MAP_FAILED`).
- **Segundo plano silencioso**: simulación, optimización y búsqueda externa se despachan mediante un ejecutor de fondo y la terminal sigue mostrando solo la respuesta final.
- **Conectividad reforzada**: las búsquedas externas registran latencia, reintentos, tasa de éxito y salud por fuente para endurecer el acceso a internet a lo largo del tiempo.

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
- `src/worker.py`: ejecutor silencioso de tareas en segundo plano, limitado a pocos workers.
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
- `Analiza una etapa con carga útil=90 combustible=210 empuje=16000 mezcla=2.4 pasos=480`

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

Cada simulación y optimización escribe:

- un checkpoint JSON incremental en `data/data/checkpoints/`,
- un estado resumido en SQLite,
- y un reporte final en `data/data/reports/`, incluyendo trazas resumidas de las búsquedas web relevantes.

Si interrumpes Termux y repites la misma consulta, el `run_id` estable permite reusar el progreso guardado.

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

## Nueva capacidad de investigación profunda y profesional

Cada pregunta ahora puede disparar varias reformulaciones automáticas de búsqueda en internet, repartidas entre distintas fuentes abiertas. El motor:

- descompone la pregunta en palabras clave,
- genera consultas orientadas a factibilidad, física, seguridad y literatura técnica,
- agrega hallazgos académicos y generales,
- y usa el módulo ML para decidir cuánta intensidad de investigación conviene aplicar y qué fuentes priorizar.

Esto no equivale a una garantía de verdad absoluta ni a una supercomputadora cuántica real, pero sí deja una base práctica para evolucionar hacia un sistema de investigación asistida mucho más ambicioso.


### Flujo avanzado actual

El motor ya no solo “busca”: ahora intenta operar como un sistema de investigación técnica iterativa. Para cada pregunta puede:

- inferir dominios implicados (por ejemplo física, propulsión, materiales, sistemas),
- generar consultas por intención (overview, constraints, feasibility, failure modes, academic, etc.),
- ponderar fuentes en función del historial de utilidad observado,
- sintetizar contradicciones, vacíos de investigación y siguientes acciones,
- y persistir sesiones de investigación para seguir mejorando el priorizado de fuentes.

Aun así, ningún software serio puede prometer éxito real del 100 % en ingeniería avanzada sin experimentación, validación y revisión humana.
