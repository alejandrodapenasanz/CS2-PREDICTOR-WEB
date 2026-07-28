# Predictor de partidos de tenis — Guion de prompts para Codex

Este documento tiene dos partes:

1. **Lo que configuras tú** antes de empezar (modelo de Codex, razonamiento, prerequisitos, cómo pasar los prompts y qué esperar de verdad).
2. **Los 9 prompts**, numerados, para pasar a Codex **de uno en uno y en orden**.

---

## PARTE 1 — Lo que debes hacer tú antes de empezar

### 1.1 Modelo y razonamiento en Codex

- **Modelo:** usa el modelo de codificación más potente que tengas disponible. A día de hoy el recomendado por defecto en Codex es el buque insignia (`gpt-5.4`); si tu Codex te ofrece uno más nuevo como último general (`gpt-5.5`), úsalo. Ejecuta `/model` en Codex para ver la lista real de tu cuenta y elige el de arriba. (Los nombres cambian según la superficie y la fecha, no me fío de memorizarlos.)
- **Razonamiento:** ponlo al máximo. En Codex el nivel más alto es **`xhigh`** (la escala es `minimal | low | medium | high | xhigh`). Este proyecto tiene decisiones de arquitectura y es muy sensible a fugas de datos, así que el coste extra de tokens compensa (tú dijiste que los tokens te dan igual). Configúralo de una de estas formas:
  - En sesión: `/model` y luego `/reasoning` → elige `xhigh`.
  - Permanente, en `~/.codex/config.toml`:
    ```
    model = "gpt-5.4"
    model_reasoning_effort = "xhigh"
    plan_mode_reasoning_effort = "xhigh"
    ```
  - Por comando puntual: `codex -c model_reasoning_effort="xhigh"`.
- **Modo plan:** para los prompts más difíciles (3-Elo, 6-features, 7-modelos, 9-auditoría) usa `/plan` antes de ejecutar, para que Codex diseñe y te enseñe el plan antes de tocar archivos.
- **Sandbox / permisos:** Codex necesita **poder escribir en el proyecto** y **acceso a red** (para descargar los datos de Sackmann y scrapear). Concédele acceso de escritura al workspace y red. Si te pide aprobación por acción, ve aprobando; si confías, puedes usar un modo más automático.

### 1.2 Prerequisitos en tu máquina

- Python 3.11 o superior y `git` instalados.
- Conexión a internet para el agente.
- Ejecuta Codex **desde la raíz de tu proyecto** (donde ya vive tu `start.ps1` de CS:GO), para que pueda crear la carpeta `TENNIS/` a ese nivel.

### 1.3 Cómo pasar los prompts (IMPORTANTE)

- **De uno en uno y en orden.** No pegues el siguiente hasta terminar el anterior.
- Después de **cada** prompt: (a) deja que Codex ejecute lo que construyó, (b) comprueba tú que corre y que el número que reporta tiene sentido, (c) revisa el `git diff`, y (d) solo entonces pasa al siguiente.
- Codex lee automáticamente el archivo `AGENTS.md` que crea el prompt 1: ahí quedan las reglas permanentes. Aun así cada prompt repite las críticas por seguridad.
- Si un prompt falla o Codex "inventa" algo, no sigas: díselo, que lo corrija, y no avances hasta que esa fase esté sólida. Un error en una fase temprana (sobre todo fugas de datos) contamina todo lo demás.

### 1.4 Expectativa realista (léelo para juzgar bien los resultados)

En predicción honesta (out-of-sample, temporal, sin fugas) el techo real es **~70% de acierto en todos los partidos** y **hasta ~80% en Grand Slams / top players**. Las casas de apuestas están en ~68-72%; igualarlas o batirlas ligeramente ya es nivel profesional. **El 100% no existe** (sorpresas, lesiones, retiradas). Lo que sí perseguimos y sí es alcanzable es la **calibración**: que cuando el sistema diga 80%, ese jugador gane ~80% de las veces. Si en algún backtest ves 90%+ en todos los partidos, **desconfía**: casi siempre es una fuga de datos, no un modelo bueno.

### 1.5 La integración con tu `start.ps1` (la haces tú al final)

Codex tiene **prohibido tocar tu `start.ps1` y tu código de CS:GO**. El proyecto de tenis expondrá su propio lanzador dentro de `TENNIS/`. En el prompt 8 Codex te dará la **línea exacta** que tú añadirás al final de tu `start.ps1` para encadenar el tenis después del scraping de CS:GO. Ese paso lo das tú a mano.

---

## PARTE 2 — Los 9 prompts

> Pega cada bloque tal cual. Están escritos en español; el código y los identificadores irán en inglés.

---

### PROMPT 1 — Andamiaje del proyecto y reglas permanentes

```
Vas a ayudarme a construir, por fases, un predictor profesional de partidos de tenis. Este es el prompt 1 de 9: solo el andamiaje. NO implementes lógica de datos ni de modelo todavía.

REGLAS PERMANENTES DEL PROYECTO (créalas también en un archivo TENNIS/AGENTS.md, porque las leerás en cada sesión):
- Todo el proyecto vive dentro de una carpeta TENNIS/ en la raíz del repositorio. NUNCA crees, edites ni borres nada fuera de TENNIS/. En particular, NO toques start.ps1 ni ningún código de CS:GO que exista en la raíz.
- Ante CUALQUIER duda o ambigüedad (formato de un dato, una decisión de diseño, algo no especificado), PÁRATE y pregúntame. No inventes nombres de columnas, rutas, ni supuestos. Prefiero una pregunta a un supuesto silencioso.
- Documenta TODO: docstring en cada módulo y función; cabecera en cada script explicando qué hace, qué recibe y cómo se ejecuta; y un README que mantendrás actualizado en cada fase.
- Lenguaje: Python. Código modular y testeable, con tests. Nada de scripts monolíticos.
- Regla de oro anti-fugas (la aplicarás en fases posteriores): para predecir un partido con fecha D, solo puedes usar información anterior a D. Tenlo presente desde ya en el diseño.

TAREA DE ESTA FASE (solo andamiaje):
- Crea la estructura: TENNIS/{data/raw, data/processed, src, scripts, tests, docs, models} con .gitkeep donde haga falta.
- TENNIS/AGENTS.md con las reglas de arriba.
- TENNIS/README.md con: objetivo del proyecto, la estructura explicada, y un "Roadmap" con las 9 fases (1 andamiaje, 2 ingesta histórica Sackmann, 3 sistema Elo por superficie y género, 4 scraper Tennis Explorer, 5 mapeo de jugadores, 6 features sin fugas, 7 modelos + validación temporal + calibración, 8 pipeline diario + integración, 9 auditoría anti-fugas).
- TENNIS/requirements.txt con: pandas, numpy, requests, beautifulsoup4, lxml, unidecode, scikit-learn, lightgbm. Crea también un entorno virtual dentro de TENNIS/ y documenta cómo activarlo.
- TENNIS/.gitignore (ignora el venv, datos crudos pesados, caches, __pycache__).
- TENNIS/src/config.py con TODAS las rutas base centralizadas (raw, processed, models, docs) derivadas de la ubicación del propio archivo, para que funcione en cualquier máquina.

Contexto de arquitectura que usaremos en fases siguientes (no lo implementes ahora, solo tenlo en el README como notas de diseño):
- Alcance máximo: hombres (ATP + Challenger + ITF) y mujeres (WTA + ITF).
- DOS universos de rating Elo separados POR GÉNERO (nunca se mezclan hombres y mujeres). Dentro de cada género se agrupan todos los niveles en el mismo pool.
- UN modelo por género (dos en total). El nivel del torneo, la superficie, el best-of y la ronda serán FEATURES, no modelos separados.
- Evaluación y calibración SIEMPRE por segmento (nivel de torneo x superficie), no solo global.
- Incluiremos las cuotas de la casa como feature Y compararemos siempre la probabilidad del modelo con la del mercado.

Al terminar: muéstrame el árbol de TENNIS/, confírmame que no has tocado NADA fuera de TENNIS/, y dime cómo activar el entorno.
```

---

### PROMPT 2 — Ingesta de datos históricos (Jeff Sackmann)

```
Prompt 2 de 9. La fase 1 (andamiaje) ya está hecha; respeta TENNIS/AGENTS.md. Recordatorio: solo dentro de TENNIS/, no toques start.ps1 ni CS:GO, documenta todo, y ante cualquier duda pregunta en vez de inventar.

OBJETIVO: ingerir el histórico de partidos de los repositorios de GitHub de Jeff Sackmann, con alcance máximo y separando por género.

FUENTES (repos públicos de Sackmann):
- Hombres: repo tennis_atp. Incluye partidos de tour principal (atp_matches_YYYY.csv), qualifying y challengers (atp_matches_qual_chall_YYYY.csv) y futures/ITF (atp_matches_futures_YYYY.csv), más jugadores (atp_players.csv) y rankings.
- Mujeres: repo tennis_wta. Partidos (wta_matches_YYYY.csv), qualifying/ITF, jugadoras (wta_players.csv) y rankings.
IMPORTANTE: NO asumas los nombres exactos de los archivos ni de las columnas. Inspecciona primero el repo/los CSV reales y ajústate a lo que exista. Si algún archivo esperado no existe o tiene otro formato, PÁRATE y dímelo.

TAREAS:
- Módulo en TENNIS/src que descargue/actualice esos CSV a TENNIS/data/raw. Debe ser idempotente: no redescargar lo ya presente salvo que se le pida forzar.
- Loaders que devuelvan DataFrames limpios y tipados:
  - Un DataFrame de PARTIDOS que unifique todos los niveles, con dos columnas nuevas explícitas: gender ('M'/'F') y tour_level (p.ej. 'G','M','A','D','F','C' para challenger, 'S'/'ITF' para futures... usa lo que Sackmann codifique y documenta el mapeo). Fechas parseadas a tipo fecha real.
  - Un DataFrame de JUGADORES por género (id, nombre, mano, país/ioc, fecha de nacimiento si está).
- Un diccionario de datos en TENNIS/docs/data_dictionary.md explicando cada columna que vamos a usar (winner_id, loser_id, surface, tourney_date, tourney_level, round, best_of, rank, rank_points, y las que añadas).
- Un test en TENNIS/tests que compruebe que los loaders cargan, que existen las columnas clave, que gender y tour_level están poblados, y que las fechas son fechas.

Al terminar, ejecuta la ingesta y repórtame en una tabla: por género y por tour_level, cuántos partidos hay y el rango de fechas. Si algo no cuadra (huecos, años faltantes), señálamelo en vez de taparlo.
```

---

### PROMPT 3 — Sistema de Elo por superficie y por género

```
Prompt 3 de 9. Fases 1-2 hechas; respeta TENNIS/AGENTS.md. Recordatorio: solo dentro de TENNIS/, no toques start.ps1 ni CS:GO, documenta todo, pregunta si dudas. Esta fase es CRÍTICA y muy sensible a fugas: haz /plan primero y enséñamelo antes de tocar archivos.

CONTEXTO: el Elo es, según la literatura, la feature individual más predictiva. Lo calcularemos bien y sin fugas.

REQUISITOS DE DISEÑO:
- DOS universos de rating completamente separados por género. Los ratings de hombres y de mujeres nunca se mezclan ni se comparan.
- Dentro de cada género, TODOS los niveles (tour + challenger + ITF) alimentan el MISMO pool de Elo, porque los resultados de niveles bajos informan la fuerza del jugador cuando sube de nivel.
- Para cada jugador mantén: un Elo GENERAL y un Elo POR SUPERFICIE (hard, clay, grass, carpet si existe). Documenta cómo combinas general + superficie (p.ej. una mezcla ponderada) y por qué.
- Usa una formulación de Elo de tenis bien documentada (por ejemplo el enfoque estilo FiveThirtyEight/Sackmann: rating inicial 1500, K-factor dinámico que decrece con el número de partidos jugados). NO inventes parámetros en silencio: elige valores razonables, DOCUMÉNTALOS en TENNIS/docs/elo.md con la fórmula exacta, y déjalos como constantes configurables.

ANTI-FUGAS (obligatorio):
- Recorre los partidos en ORDEN CRONOLÓGICO estricto. El Elo de un jugador ANTES de un partido con fecha D solo puede reflejar partidos con fecha < D.
- Guarda un histórico consultable: función get_elo(player_id, gender, surface, as_of_date) que devuelva el rating tal como era justo ANTES de esa fecha.
- Persiste los ratings/histórico en TENNIS/data/processed.

TESTS:
- Test de NO-fuga: el Elo "as-of" de un partido no cambia si añades partidos posteriores al dataset.
- Test de cordura: al final del histórico, los jugadores con mejor Elo deben ser nombres de primer nivel plausibles (enséñame el top 10 de cada género para que lo valide yo).

Al terminar: ejecuta el cálculo completo y muéstrame (a) el top 10 de Elo general por género, (b) un ejemplo de get_elo as-of para un jugador en dos fechas distintas, y (c) el resultado de los tests.
```

---

### PROMPT 4 — Scraper de Tennis Explorer (partidos del día + cuotas + slug)

```
Prompt 4 de 9. Fases 1-3 hechas; respeta TENNIS/AGENTS.md. Recordatorio: solo dentro de TENNIS/, no toques start.ps1 ni CS:GO, documenta todo, pregunta si dudas.

OBJETIVO: obtener, para una fecha dada, TODOS los partidos programados de ese día (ATP, WTA, Challenger, ITF) desde tennisexplorer.com, con sus cuotas.

REQUISITOS:
- Módulo en TENNIS/src que, dada una fecha (por defecto hoy), devuelva un DataFrame con: torneo, categoría/nivel, género (dedúcelo del torneo), jugador 1, jugador 2, cuota de cada jugador si está disponible, y estado del partido (programado, en juego, terminado, walkover, cancelado).
- MUY IMPORTANTE: para CADA jugador, además del texto visible (formato "Apellido I."), extrae el ENLACE a su ficha (el href / slug del jugador). Ese slug es la clave ESTABLE que usaremos para mapear en la fase 5; el texto visible NO es fiable. Si un jugador no tiene enlace, márcalo.
- Scraping responsable: User-Agent de navegador normal, un pequeño delay entre peticiones, y CACHEA el resultado por fecha en TENNIS/data/raw (no re-descargar la misma fecha). Respeta lo que diga su robots/términos; si detectas que bloquean, PÁRATE y dímelo, no fuerces.
- Maneja con elegancia: días sin partidos, partidos sin cuotas, y cambios de formato de la web (si el HTML no encaja con lo esperado, error claro, no datos inventados).
- Script en TENNIS/scripts que reciba la fecha por argumento.
- Un test que use un HTML de ejemplo guardado como fixture en TENNIS/tests (descarga una página real una vez, guárdala, y testea el parser contra ese archivo, para no depender de la red en los tests).

NO inventes la estructura del HTML: inspecciónala de verdad antes de escribir el parser. Si la estructura para Challenger/ITF difiere de la de ATP/WTA, trátalo explícitamente.

Al terminar, ejecútalo con una fecha reciente y enséñame las primeras filas, incluyendo los slugs y las cuotas, y dime cuántos partidos salieron por nivel.
```

---

### PROMPT 5 — Mapeo de jugadores (Tennis Explorer ↔ Sackmann)

```
Prompt 5 de 9. Fases 1-4 hechas; respeta TENNIS/AGENTS.md. Recordatorio: solo dentro de TENNIS/, no toques start.ps1 ni CS:GO, documenta todo, pregunta si dudas.

OBJETIVO: dado un partido del scraper (fase 4), resolver cada jugador a su player_id de Sackmann del género correcto. Esta es la parte con más casos límite; hazla robusta y honesta.

ESTRATEGIA (en este orden):
1. CLAVE PRINCIPAL = el slug de la ficha de Tennis Explorer. Mantén una tabla persistente slug -> player_id en TENNIS/data/processed que se rellena y cachea: un slug ya resuelto NUNCA se recalcula.
2. Resolución automática de slugs nuevos: normaliza nombres en AMBOS lados (minúsculas, sin acentos con unidecode, manejo explícito de apellidos compuestos y guiones) y empareja por apellido + inicial, PERO restringiendo los candidatos de Sackmann a: (a) el género correcto, y (b) jugadores ACTIVOS (con partidos en los últimos ~3 años). Usa el país/bandera como desempatador cuando Tennis Explorer lo dé.
3. Tabla de overrides manual en TENNIS/data/overrides.csv para lo que no se resuelva solo. Documéntala.

DEGRADACIÓN ELEGANTE (obligatorio): si un jugador no se resuelve (típico en ITF/Challenger, jugadores nuevos o con poco histórico), NO rompas y NO adivines: marca ese partido como "no mapeado" y regístralo en un CSV de no-resueltos en TENNIS/data/processed para revisión. El pipeline debe poder seguir con los partidos que sí se resolvieron.

TESTS (incluye casos reales):
- Caso fácil (jugador top con slug conocido).
- Apellido compuesto ("Bautista Agut R.", "Auger-Aliassime F.", "Van De Zandschulp B.").
- Nombre con acento/transliteración ("Cerundolo" vs "Cerúndolo").
- Colisión resuelta por país o por override.
- Un no-resuelto que se registra correctamente sin romper.

Al terminar, pásale la salida del scraper de una fecha reciente y repórtame el % resuelto automáticamente POR NIVEL (espera que ATP/WTA sea alto y Challenger/ITF más bajo) y la lista de no-resueltos.
```

---

### PROMPT 6 — Ingeniería de características (sin fugas)

```
Prompt 6 de 9. Fases 1-5 hechas; respeta TENNIS/AGENTS.md. Recordatorio: solo dentro de TENNIS/, no toques start.ps1 ni CS:GO, documenta todo, pregunta si dudas. Fase muy sensible a fugas: haz /plan primero.

OBJETIVO: construir el vector de features de un partido y el dataset de entrenamiento por género, TODO "as of" la fecha del partido (nada del futuro).

FEATURES (dadas dos jugadores, superficie, fecha y contexto del torneo). Constrúyelas como DIFERENCIAS (valor jugador A menos jugador B) donde tenga sentido:
- Elo general as-of y Elo de superficie as-of (usa el sistema de la fase 3).
- Forma reciente: % de victorias en los últimos N partidos y en los últimos M meses.
- Head-to-head as-of: global y en la superficie.
- Descanso: días desde el último partido de cada jugador.
- Ranking y rank_points as-of (diferencia).
- Edad, y la transformación "distancia a 30 años" (edad óptima ~28-32; úsala como en la literatura de Grand Slams).
- Contexto como features: tour_level, surface, best_of, round.
- Cuotas: probabilidad implícita de cada jugador a partir de la cuota, "de-vigada" (quita el margen de la casa normalizando las dos probabilidades para que sumen 1). Guarda por separado la probabilidad de mercado de-vigada, porque la usaremos SIEMPRE para comparar el modelo contra el mercado (columna de edge = prob_modelo - prob_mercado).

ANTI-FUGAS Y ORIENTACIÓN:
- Regla de oro: para un partido con fecha D, ninguna feature puede usar información con fecha >= D (incluidos Elo, forma, H2H, descanso, ranking).
- Para no filtrar el resultado por el orden ganador/perdedor: ALEATORIZA qué jugador es "A" y cuál "B" al construir cada fila, y define la etiqueta y = 1 si gana A, 0 si gana B, coherente con esa aleatorización. Documenta la semilla.
- Construye el dataset de entrenamiento aplicando estas features a todo el histórico, SEPARADO por género.

TESTS:
- Test de NO-fuga: las features de un partido no cambian si añades partidos posteriores al histórico.
- Test de balance: tras la aleatorización, la etiqueta y está ~50/50 y no correlaciona con "quién aparece primero".

Al terminar: genera y muéstrame el vector de features COMENTADO para un enfrentamiento de ejemplo, el tamaño del dataset por género, y el resultado de los tests.
```

---

### PROMPT 7 — Modelos, validación temporal y calibración

```
Prompt 7 de 9. Fases 1-6 hechas; respeta TENNIS/AGENTS.md. Recordatorio: solo dentro de TENNIS/, no toques start.ps1 ni CS:GO, documenta todo, pregunta si dudas. Fase crítica: haz /plan primero.

OBJETIVO: entrenar UN modelo por género, validarlo de forma temporal y honesta, y calibrar sus probabilidades. El objetivo no es solo acertar el ganador, sino que las PROBABILIDADES estén bien calibradas (si dice 80%, que gane ~80% de las veces).

MODELOS:
- Baseline: regresión logística sobre las features.
- Principal: gradient boosting (LightGBM). Uno por género (M y F).
- Baselines de referencia OBLIGATORIOS para comparar: "gana el favorito según ranking" y "gana el favorito según la cuota de-vigada". El modelo tiene que justificarse frente a estos.

VALIDACIÓN TEMPORAL (innegociable, aquí es donde se cuela el 90% falso):
- NADA de split aleatorio. Usa validación temporal tipo ventana expansiva: entrenas con el pasado y predices el siguiente tramo (por temporada o por torneo, en orden cronológico), repitiendo y acumulando.
- Reporta métricas GLOBALES y POR SEGMENTO (tour_level x superficie, y género): accuracy, log-loss, Brier y AUC. Quiero ver el desglose, no solo el número bonito global.

CALIBRACIÓN:
- Calibra las probabilidades (isotónica o Platt), ajustando el calibrador SOLO con datos del pasado y evaluándolo en el futuro (nada de calibrar con el test).
- Enséñame la curva de calibración/fiabilidad y el Brier antes y después de calibrar.
- Compara la probabilidad calibrada del modelo con la probabilidad de mercado de-vigada: ¿el modelo aporta algo sobre el mercado, y en qué segmentos?

ENTREGABLES:
- Guarda modelos y calibradores en TENNIS/models, versionados y documentados.
- Un script de REENTRENO reproducible en TENNIS/scripts (para reentrenar cuando lleguen datos nuevos), que reutilice el MISMO pipeline sin fugas.
- Un informe en TENNIS/docs/model_report.md con métricas por segmento, curvas de calibración, comparación con los baselines y con el mercado, y una sección honesta de LIMITACIONES.

Al terminar, enséñame el informe resumido: métricas por segmento, comparación con "favorito por cuota" y con el mercado, y si algún segmento sale con accuracy sospechosamente alto (>85%), audítalo por posible fuga y avísame.
```

---

### PROMPT 8 — Pipeline diario + lanzador + integración con start.ps1

```
Prompt 8 de 9. Fases 1-7 hechas; respeta TENNIS/AGENTS.md. Recordatorio: solo dentro de TENNIS/, no toques start.ps1 ni CS:GO, documenta todo, pregunta si dudas.

OBJETIVO: un pipeline que cada mañana produzca las predicciones del día uniendo todo lo anterior, y su integración (no invasiva) con mi start.ps1 existente.

PIPELINE (script principal en TENNIS/scripts, con una fecha opcional; por defecto hoy):
1. Scrapea los partidos del día (fase 4).
2. Mapea los jugadores (fase 5).
3. Calcula features as-of (fase 6).
4. Aplica el modelo del género correspondiente y CALIBRA las probabilidades (fase 7).
5. Produce una tabla con: torneo, nivel, superficie, jugador A, jugador B, prob. del modelo (calibrada) de cada jugador, prob. de mercado de-vigada, edge (modelo - mercado), y una columna de CONFIANZA/flag.
- Los partidos con algún jugador no mapeado o con histórico muy pobre se MARCAN como baja confianza; no rompen la ejecución ni se inventan probabilidades.
- Salida a consola (tabla legible) y a CSV en TENNIS/data/processed con timestamp.

LANZADOR:
- Crea TENNIS/run_tennis.ps1 que active el entorno virtual y ejecute el pipeline del día. Que sea autocontenido y ejecutable desde cualquier ruta.

INTEGRACIÓN CON start.ps1 (NO la hagas tú):
- NO edites mi start.ps1 (está en la raíz y orquesta CS:GO). En su lugar, escríbeme en TENNIS/docs/integracion.md la LÍNEA EXACTA que debo añadir yo al final de mi start.ps1 para llamar a TENNIS/run_tennis.ps1 después del scraping de CS:GO, con una nota de qué hace y cómo revertirla.

DOCUMENTACIÓN:
- Actualiza el README con el "cómo se usa" completo de principio a fin: instalar, actualizar histórico, reentrenar, y lanzar la predicción del día. Incluye cómo LEER el output con honestidad (qué significa el edge, por qué desconfiar de probabilidades extremas en niveles bajos).

Al terminar, ejecuta el pipeline para hoy y muéstrame la tabla de predicciones, y pégame la línea que debo añadir a start.ps1.
```

---

### PROMPT 9 — Auditoría anti-fugas y verificación end-to-end

```
Prompt 9 de 9. El proyecto está completo (fases 1-8); respeta TENNIS/AGENTS.md. Recordatorio: solo dentro de TENNIS/, no toques start.ps1 ni CS:GO, documenta todo, pregunta si dudas. Haz /plan primero.

OBJETIVO: auditar todo el proyecto buscando fugas de datos y comportamientos deshonestos, y dejar un informe de verificación. El sistema debe ser CIERTO, no aparentar serlo.

REVISA Y DOCUMENTA en TENNIS/docs/audit.md:
1. Fugas temporales: repasa cada feature y el pipeline de entrenamiento/validación y confirma, con evidencia, que ninguna información con fecha >= fecha-del-partido entra en la predicción de ese partido (Elo, forma, H2H, ranking, descanso, cuotas). Si encuentras una fuga, corrígela y reejecuta la validación.
2. Fuga por orientación: confirma que la aleatorización A/B es correcta y que la etiqueta no es predecible por el orden.
3. Fuga de calibración/selección de modelo: confirma que calibradores e hiperparámetros se eligieron solo con datos pasados, nunca con el tramo de test.
4. Honestidad del backtest: confirma que la validación es temporal (no aleatoria) y que las métricas por segmento del informe se reproducen al reejecutar. Si algún segmento da accuracy >85%, investígalo a fondo como sospecha de fuga.
5. Robustez operativa: fuerza casos límite (día sin partidos, jugador sin mapear, jugador sin histórico, cuotas ausentes, HTML inesperado) y confirma que el pipeline degrada con elegancia y nunca inventa datos.
6. Reproducibilidad: semillas fijadas y documentadas; reentreno reproducible.

Añade un test end-to-end en TENNIS/tests que ejecute el pipeline sobre una fecha con datos de fixture y verifique que la salida tiene la forma esperada y respeta los flags de confianza.

Al terminar, entrégame el informe de auditoría con: qué revisaste, qué encontraste, qué corregiste, y una valoración honesta final de en qué segmentos el sistema es fiable y en cuáles NO conviene fiarse. Si algo quedó dudoso, dímelo claramente en vez de maquillarlo.
```

---

## Nota final

Estos 9 prompts construyen el sistema completo y honesto. Cuando termines, tendrás en `TENNIS/` un pipeline que cada mañana (encadenado tras tu CS:GO) saca los partidos del día con probabilidad del modelo, probabilidad del mercado y el edge entre ambos, calibrado y con desglose de fiabilidad por segmento. Recuerda la expectativa realista de la Parte 1: apunta a ~70% global / ~80% en Grand Slams y a una buena calibración, y desconfía de cualquier número que parezca demasiado bueno.
