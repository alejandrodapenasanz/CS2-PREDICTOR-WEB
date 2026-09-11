# Proyecto CS2 + TENNIS — TODOS los prompts (consolidado v3)

## Cómo usarlos

- Codex con el modelo más potente disponible y razonamiento **`xhigh`**. Usa **`/plan`** en los marcados `[usa /plan]`.
- Pásalos **de uno en uno y en orden**. Tras cada uno: revisa el `git diff` y comprueba el resultado antes del siguiente.
- **Garantía clave:** cada prompt termina ejecutando el smoke/pipeline + las puertas y deja el proyecto **arrancable de punta a punta**. Si algo queda roto, se arregla o se revierte DENTRO de ese prompt.
- Reglas transversales (van dentro de cada prompt): respeta `AGENTS.md`; `predictions`/`observations`/`settlements` de tenis y el `prediction_ledger` de CS2 son datos **SAGRADOS** (solo lectura; cualquier añadido es migración ADITIVA o tabla lateral, nunca mutar filas ni tocar triggers); anti-fugas estricto; si algo no coincide con lo descrito, **PARA y pregunta**.

## Nota de expectativas (léela)

En CS2 (~65%) y tenis (~70,83% real) estás cerca del **techo pre-match**. Estas mejoras suben **accuracy donde se puede (odds, Elo fresco), y sobre todo calibración, robustez, frescura y usabilidad**; no llevan a un 90%. Lo que más "accuracy bruta" añade: odds en el modelo (ambos deportes) y, en tenis, refrescar el Elo congelado. Lo que quita "palos gordos": el rating sensible al roster en CS2.

## Mapa (orden óptimo por dependencias)

**Base**
1. Dependencias como fuente única + Python 3.13
2. Puertas de coherencia (AGENTS.md) + CI

**CS2 — modelo**
3. Champion/challenger + rollback + poda — **la "puerta" de CS2**
4. Reproducibilidad + banda de confianza — **la banda la usan 5 y 19**
5. Manejo de odds vía la puerta (la puerta elige arquitectura)
6. Ablación enhanced-info + ponderación por recencia
7. Medición de calibración por segmento (Elo/stage/LAN) + interacción
8. Rating sensible al roster (challenger por la puerta)
9. Gráfico CS2 desde `prediction_ledger`

**Higiene de datos (ambos deportes)**
10. Fetch de resultados con puerta temporal (2.5h) — CS2 + tenis

**TENNIS**
11. Arreglo del "0.0%" + accuracy/calibración live real *(independiente)*
12. Scraper de tenis: parser de robots + cliente HTTP consolidado
13. Elo fresco: overlay Tennis Explorer con handoff fijo (Elo general)
14. Catálogo de superficies (Elo de superficie)
15. `best_of`/`round` nulos en producción
16. Champion/challenger + rollback + poda en TENNIS — **la "puerta" de tenis (la usan 17 y 18)**
17. Odds en el modelo de tenis + fallback sin odds
18. Medición de calibración por segmento en tenis

**Web e integración**
19. Web: pestañas + gráfico "Modelo" con filtros + "Best opportunity"
20. Orquestador raíz CS2+TENNIS con `-Retrain`
21. Migración a `/BBDD` + backup consistente + detector de USB *(independiente; capstone)*






### PROMPT 5 — Manejo de odds vía la puerta (CS2)  [usa /plan]

```
Contexto (CS2): DECISIÓN de producto: la accuracy del pick es lo primordial. Metemos las odds en producción, PERO el modelo debe seguir funcionando MUY BIEN cuando no hay odds. Estado actual: Model A (producción, SIN odds) y Model B (solo opening odds, auto-gated ≥120 partidos). Las odds se scrapean de HLTV y se guardan. Las odds de APERTURA son pre-match (no hay fuga temporal). Ya existe una banda de confianza del estimado y una puerta champion/challenger (registry/latest.json/last_good). Respeta AGENTS.md. Si algo no coincide, PARA y pregunta.

REANCLAJE HONESTO (déjalo en el informe): incluso con odds, el techo realista ronda el ~70%. La rama SIN odds tiene un techo estructuralmente algo más bajo. Objetivo de la rama sin odds: LO MÁS CERCA POSIBLE, no paridad.

OBJETIVO: maximizar accuracy con odds y ser MUY robusto sin ellas, dejando que la PUERTA elija la arquitectura ganadora (no la fijamos a priori; hay tiempo de sobra para entrenar todas).

ARQUITECTURAS CANDIDATAS (compáralas por la puerta, mismo hold-out temporal; promociona la de mejor log-loss/Brier):
- CANDIDATA A — ROUTER DE DOS MODELOS: PRIMARIO (features COMPLETAS + odd de APERTURA de-vigada, normaliza a suma 1; solo opening odds); RESERVA (modelo DEDICADO sin odds, con su propio tuning/selección/calibración, reforzando Elo/Glicko/forma/fuerza-rival/ranking). Cada uno calibrado en su propio régimen.
- CANDIDATA B — MODELO ÚNICO: la odd de-vigada como feature NaN nativo de LightGBM + indicador "odds_available", entrenado con datos MIXTOS.
- (Si la ablación ya corrió, incluye sus variantes de filas.)

REQUISITOS COMUNES:
1. ROUTER: modo con odds solo si hay odds VÁLIDAS (dos cuotas coherentes, de-vig sensato); si no, modo sin odds. SIEMPRE sale una predicción.
2. RECUPERACIÓN DE ODDS antes de caer a sin-odds: si el scraper falló hoy pero teníamos la odd de APERTURA de un intento anterior, reutilízala.
3. LEDGER: MIGRACIÓN ADITIVA (columna nullable o tabla lateral): régimen (odds/no_odds), arquitectura, odd recuperada.
4. MEDICIÓN SEPARADA: accuracy/log-loss/Brier/CALIBRACIÓN de la porción CON odds y SIN odds por separado + fracción sin odds.
5. BENCHMARK: mantén Model B (solo-odds) como referencia de mercado (evaluación, no producción).
6. CONFIANZA: predicción sin odds genuinamente incierta -> BAJA CONFIANZA en salida y dashboard.

PUERTA: evalúa el sistema COMPLETO (router incluido) y promociona la arquitectura ganadora solo si mejora frente a la producción actual.

VERIFICACIÓN Y ESTADO EJECUTABLE:
- Tests del router, recuperación de odds, predicción válida sin odds, etiqueta de ledger, medición separada.
- Enséñame el ranking de candidatas y el desglose con-odds vs sin-odds (calibración incluida) y la fracción sin odds.
- Ejecuta el smoke/pipeline de punta a punta (incluye un caso sin odds) y las puertas; confirma que arranca.
```

---

### PROMPT 6 — Ablación enhanced-info + ponderación por recencia (CS2)  [usa /plan]

```
Contexto (CS2): hay muchos partidos. Decidir CON DATOS la composición del training, no a priori. El Elo del día del partido depende de TODO el historial previo -> warm-up cronológico completo aunque no todos entren al training. Respeta AGENTS.md (anti-fugas). Si algo no coincide, PARA y pregunta.

TAREAS:
1. WARM-UP de features sobre TODO el historial en orden cronológico (Elo/forma/H2H as-of), independientemente de qué filas entren al training. Detecta si las features hoy están CONGELADAS como columnas o se RECALCULAN: si se recalculan, el warm-up es OBLIGATORIO; si están congeladas, salvaguarda. Si dudas, PARA y pregunta.
2. CONSERVA el historial crudo (fechas + resultados); solo se EXCLUYEN partidos de las FILAS de training, nunca se borran.
3. AUSENCIA de enhanced-info: indicador "enhanced_available" + imputación (o nulos nativos), NO tirando filas en la variante "todos".
4. ABLACIÓN: (a) todos, (b) solo enhanced-info, (c) todos con pesos que decaen con la antigüedad. Compáralas en el MISMO hold-out reciente vía la puerta; promociona la mejor. (Sobre la arquitectura de producción del prompt de odds.)
5. SESGO: reporta si el enhanced-info está sesgado hacia tier alto/LAN/reciente y avísalo.

VERIFICACIÓN Y ESTADO EJECUTABLE:
- Warm-up correcto; excluir viejos no cambia features as-of de las conservadas; las tres variantes se comparan.
- Ejecuta el smoke/pipeline y las puertas; confirma que arranca. Dime qué variante gana.

Alcance: CS2 (salvo que confirmes TENNIS).
```

---

### PROMPT 7 — Medición de calibración por segmento + interacción (CS2)  [usa /plan]

```
Contexto (CS2): hipótesis: distintos tramos de Elo, stages (final/semi/cuartos...) y LAN/online podrían comportarse distinto. NO construimos 30-40 modelos por segmento a ciegas: fragmentar empeora, y el Elo ya está diseñado para que una misma diferencia signifique la misma probabilidad en todo el rango. Enfoque: MEDIR primero, tocar el modelo solo si los datos lo piden, por la puerta, sin perder el modelo actual. Respeta AGENTS.md. Si algo no coincide, PARA y pregunta.

CLAVE: mide la calibración por segmento sobre el HOLD-OUT TEMPORAL / BACKTEST (miles de partidos), y trackéala en vivo desde prediction_ledger según se acumula. Con la muestra live actual NO se decide por segmento.

TAREAS:
1. HERRAMIENTA DE MEDICIÓN: sobre el backtest, calibración (prob media predicha vs real) por (a) tramos de diferencia de Elo (define bins), (b) stage, (c) LAN/online. Por segmento: fiabilidad + desviación (ECE) + TAMAÑO DE MUESTRA. Replica en vivo desde prediction_ledger. N pequeño -> "sin muestra suficiente", no se interpreta.
2. ESCALERA (documéntala; no auto-ejecutes saltos): todos calibran bien -> nada. Un segmento se desvía -> features de INTERACCIÓN al modelo general (elo_diff x stage, elo_diff x tramo, LAN/online x elo_diff) y la puerta decide. Solo si un segmento es GRANDE y sigue desviándose -> modelo dedicado, también por la puerta.
3. IMPLEMENTA la vía de interacción como OPCIÓN lista, pero NO fuerces ninguna: la puerta decide. Modelo actual nunca se pierde.
4. INFORME honesto con N por segmento y aviso de que la muestra live aún no decide.

VERIFICACIÓN Y ESTADO EJECUTABLE:
- Enséñame la tabla de calibración por segmento del backtest (con N) y dime dónde hay desviación real vs ruido.
- Ejecuta el smoke/pipeline y las puertas; confirma que arranca.

Alcance: CS2.
```

---

### PROMPT 8 — Rating sensible al roster (CS2, challenger por la puerta)  [usa /plan]

```
Contexto (CS2): confirmado por auditoría — el Elo/Glicko del equipo está asociado a la IDENTIDAD de la organización y NO se resetea ni descuenta cuando cambian 2-3 jugadores. Existen señales de cambio de roster (alerta ROSTER_CHANGE_90D; diferencia entre alineación anunciada y la de los últimos 90d; entradas/salidas; antigüedad/tamaño del roster; riesgo de stand-in), pero NINGUNA corrige matemáticamente el estado Elo/Glicko. Consecuencia real: tras una reconstrucción profunda, el modelo conserva demasiado crédito histórico del equipo viejo -> palos gordos cuando un equipo reconstruido ya no es el de antes. La disciplina anti-fugas actual es sólida (ChronologicalState emite features ANTES de observar cada resultado; test_leakage_audit.py comprueba a 9 decimales que las features no cambian al añadir partidos posteriores; validación walk-forward sin shuffle). HAY QUE RESPETARLA. Si algo no coincide, PARA y pregunta.

OBJETIVO: un rating causal sensible al roster que reduzca esos palos, entrando como CHALLENGER por la puerta champion/challenger (no reemplazo directo).

TAREAS:
1. RATING SENSIBLE AL ROSTER: cuando el NÚCLEO del equipo cambia (usa la detección de cambio de roster ya existente para definir "cambio de núcleo"), en el SIGUIENTE partido conserva solo una FRACCIÓN del rating proporcional al núcleo mantenido y AUMENTA la incertidumbre (RD de Glicko) para que el sistema descuente crédito obsoleto y se recalibre rápido con los nuevos resultados. NO reset a lo bruto: descuento proporcional. Fórmula de fracción/decay documentada y configurable.
2. CAUSAL Y AS-OF: reconstruye online en orden cronológico igual que el rating actual (emitir antes de observar), de modo que pase el test_leakage_audit.py existente. Añade un test de que el rating ajustado por roster de un partido no cambia si añades partidos posteriores.
3. CHALLENGER POR LA PUERTA: no sustituyas el rating actual; añádelo como alternativa y deja que la puerta lo compare contra el rating actual en el walk-forward (log-loss/Brier). Solo promociona si mejora. El rating actual nunca se pierde.
4. NOTA honesta en el informe: el beneficio esperado son MENOS palos gordos en rosters reconstruidos (calibración/robustez), no un salto grande de accuracy media.

TESTS:
- Un cambio de núcleo dispara descuento proporcional + subida de RD; el rating vuelve a converger con nuevos resultados.
- Pasa el test de no-fuga (as-of estable).
- La puerta compara el rating-roster contra el actual y solo promociona si mejora.

VERIFICACIÓN Y ESTADO EJECUTABLE:
- Enséñame la comparación rating-roster vs rating actual por la puerta (log-loss/Brier + calibración), y algún ejemplo de equipo reconstruido donde cambie la predicción.
- Ejecuta el smoke/pipeline y las puertas; confirma que arranca. Si algo queda roto, arréglalo o revierte aquí.

Nota: NO fuerces encender la familia announced_lineups/stand-in: la puerta ya la evaluó y no mejoraba; esto es distinto (un defecto estructural del rating), por eso sí entra como challenger.
```

---

### PROMPT 9 — Gráfico CS2 desde el `prediction_ledger`  [usa /plan]

```
Contexto (CS2): web estática; WEB/build_web.py (etapa 9 de start.ps1, se regenera en CADA run) genera WEB/data.js (window.__CS2_PREDICTOR_DATA__) que consume WEB/index.html (JS vanilla). Hoy el gráfico muestra accuracy por franja horaria y se alimenta de favorite_accuracy_bands.json (BACKTEST), NO de aciertos reales. Los reales están en prediction_ledger (cs2_prediction_schema.sql ~línea 517): fila por partido con lo predicho y lo realizado (actual_team1_win, realized_log_loss/brier). prediction_ledger NO está en BLACKBOX (solo `predictions`) -> un restore lo perdería. Respeta AGENTS.md; el ledger es dato sagrado (solo lectura). Mantén los cambios ACOTADOS a la sección CS2 de build_web.py. Si algo no coincide, PARA y pregunta.

OBJETIVO: accuracy/calibración live desde el ledger real y preservar el ledger en backups. (Análogo CS2 del arreglo de tenis del prompt 11; ambos alimentan el gráfico unificado del prompt 19.)

TAREAS:
1. QUITA el gráfico por franja horaria.
2. AÑADE desde prediction_ledger (solo liquidadas): curva de calibración + accuracy en ventana móvil (últimos N). En build_web.py -> data.js -> index.html. Si el prompt de odds añadió la etiqueta de régimen, permite desglosar con-odds vs sin-odds. Expón los cálculos de forma reutilizable por el prompt de web (19).
3. HONESTIDAD/UX: sin datos, no muestres nada (ni 0% ni línea plana). Muestra SIEMPRE el nº de muestras por bucket; el backtest puede ir como referencia SECUNDARIA etiquetada.
4. PRESERVA EL LEDGER: añade prediction_ledger a SOURCE_OF_TRUTH_TABLES (o equivalente) para que BLACKBOX lo incluya; verifica backup->restore.

VERIFICACIÓN Y ESTADO EJECUTABLE:
- Fixture poblado -> calibración y ventana móvil correctas. Ledger vacío -> no muestra nada. backup->restore preserva el ledger.
- Ejecuta build_web + el smoke/pipeline y confirma que la web se genera y el proyecto arranca.
```

---

### PROMPT 10 — Fetch de resultados con puerta temporal (CS2 + tenis)

```
Contexto: para RECORTAR llamadas y ruido, solo se debe hacer fetch del resultado de un partido cuando es razonable que YA HAYA TERMINADO. Que un partido haya EMPEZADO (hora_inicio < ahora) NO es que haya TERMINADO. Los partidos de CS2 (Bo1/Bo3/Bo5) y de tenis varían en duración. Ambos proyectos tienen un backfill de resultados (CS2 desde HLTV; tenis desde Tennis Explorer, con la lógica _settle_observation por source_match_id). Respeta AGENTS.md; datos sagrados solo lectura. Si algo no coincide, PARA y pregunta.

OBJETIVO: una puerta temporal en el backfill de CADA deporte que evite fetches inútiles.

TAREAS:
1. PUERTA TEMPORAL: en el backfill de resultados de cada deporte, solo intenta hacer fetch/settle de un partido si ahora > hora_inicio_programada + margen. Margen = 2.5h (configurable). Si aún no ha pasado, NO hagas nada con ese partido en esta pasada.
2. Si tras la puerta la fuente sigue marcando el partido como NO terminal, no reintentes hasta la siguiente pasada (evita machaque).
3. Partidos que nunca resuelven (walkover/cancelado/aplazado): trátalos con elegancia, sin martillear.
4. NO cambies la lógica de settlement existente (join por source_match_id, validación de estado terminal); esto solo controla CUÁNDO se hace fetch.

TESTS:
- Un partido empezado hace 1h se SALTA; uno empezado hace >2.5h se intenta; uno no-terminal tras la puerta no se reintenta hasta la siguiente pasada.

VERIFICACIÓN Y ESTADO EJECUTABLE:
- Enséñame, en una pasada de prueba de cada deporte, cuántos fetches se ahorran por la puerta.
- Ejecuta el smoke/pipeline de ambos y confirma que arrancan. Si algo queda roto, arréglalo o revierte aquí.
```

---

### PROMPT 11 — TENNIS: arreglo del "0.0%" + accuracy/calibración live real  [independiente, lánzalo cuando quieras]

```
Contexto (TENNIS + WEB): la web muestra "fiabilidad 0.0%" en tenis, pero es un bug de presentación, NO accuracy cero. Diagnóstico confirmado:
- "fiabilidad" = assess_vector_confidence() (TENNIS/src/daily_pipeline/confidence.py:52): mide CALIDAD DE LOS INPUTS y devuelve nivel HIGH/MEDIUM/LOW/UNAVAILABLE. NO mide aciertos.
- build_web.py:1091 lee columnas confidence_score/reliability_score/reliability de predictions que NO existen -> JSON null.
- index.html:1134 hace Number(reliabilityData.score); Number(null) === 0 -> 0.0%.
- La web solo carga la última jornada oficial e ignora los 72 settlements liquidados del día 9.
- predictions es append-only y NO tiene columnas de resultado; el resultado y la liquidación viven en settlements (official_prediction_id, winner_slug, actual_outcome_a, settlement_id, settled_at_utc). Hay 72 predicciones liquidadas con 51 aciertos (70,83%), hoy no calculadas ni mostradas.
Respeta AGENTS.md. predictions y el triplete son datos SAGRADOS: solo lectura. Mantén los cambios ACOTADOS a la sección de tenis de build_web.py. Si algo no coincide, PARA y pregunta.

OBJETIVO: dejar de mentir con el 0.0% y mostrar dos métricas DISTINTAS y bien etiquetadas.

TAREAS:
1. SEPARA Y RENOMBRA: "Confianza del input" (assess_vector_confidence) como NIVEL (ALTA/MEDIA/BAJA/NO DISPONIBLE), nunca % (si null/UNAVAILABLE, "—" o el nivel, JAMÁS 0.0%); "Acierto real (partidos liquidados)" como métrica NUEVA desde settlements.
2. CALCULA accuracy/calibración live desde settlements (solo liquidadas): accuracy + curva de calibración con Brier/log-loss. En build_web.py -> data.js -> index.html. Expón el cálculo reutilizable por el prompt de web (19).
3. CARGA TODO EL HISTÓRICO LIQUIDADO, no solo la última jornada.
4. HONESTIDAD: muestra SIEMPRE el nº de partidos liquidados junto al %; con 0, "sin muestra", nunca 0.0%.

VERIFICACIÓN Y ESTADO EJECUTABLE:
- build_web contra la tennis.sqlite3 real (o fixture): "confianza del input" como nivel, "acierto real" 70,83% sobre 72 con calibración, y 0 liquidados como "sin muestra".
- Ejecuta el smoke/pipeline y confirma que la web se genera y el proyecto arranca.
```

---

### PROMPT 12 — TENNIS: scraper con parser de robots + cliente HTTP consolidado

```
Contexto (TENNIS, scraper): scraping HTML estático y sin evasión. Fuentes: Sackmann (CSV vía mirror GitHub autorizado, requests), Tennis Abstract Elo (2 páginas HTML públicas; rutas de jugador/JS bloqueadas y NO se tocan), Tennis Explorer (URL canónica de partidos del día, con cuotas), Match Charting. Tennis Explorer usa Scrapling como transporte HTTP (impersonate=None, stealthy_headers=False, http3=False) con un cortacircuitos que DETECTA WAF/Cloudflare y se detiene (no lo pasa). Rate-limit: 1s mínimo, un GET, lock por host, caché SHA-256. Respeta AGENTS.md. IMPORTANTE: NO construyas evasión (ni Selenium/headless para saltarte bloqueos, ni rotación de proxies, ni bypass de robots) y no debilites el cortacircuitos. Si algo no coincide, PARA y pregunta.

CONTEXTO DE DISEÑO: las features que MÁS mueven la aguja no dependen de páginas bloqueadas. El Elo se calcula desde resultados (elo.sqlite3), frescos a diario desde la fuente de partidos del día (permitida). Solo los agregados de saque van con retraso en los CSV, y son de MENOR valor. No hace falta tocar nada bloqueado.

OBJETIVO: robustez y frescura por vías legítimas + un CLIENTE HTTP consolidado reutilizable (lo usan 14 y 15).
TAREAS:
1. PARSER DE ROBOTS.TXT real: respeta robots POR DEFECTO (Disallow no se pide), cachéalo con TTL. Única excepción, explícita y aislada: la URL canónica de partidos del día de Tennis Explorer, bajo authorized_matches_endpoint_exception, y SOLO esa URL. Verifica que las rutas bloqueadas de Tennis Abstract nunca se solicitan.
2. FRESCURA legítima: Elo/forma/H2H/descanso/ranking computables desde resultados frescos (fuente de partidos del día). Documenta qué feature sale de qué fuente y con qué frescura.
3. CLIENTE HTTP CONSOLIDADO: rate-limiter + caché unificados, reutilizable por todas las fuentes (1s por host, lock por host, caché SHA-256, cortacircuitos WAF intacto). Config para subir el ritmo si hay API/allowlist.

VERIFICACIÓN Y ESTADO EJECUTABLE (sin red; fixtures):
- Disallow no se pide; la excepción autorizada SÍ (y solo esa); mínimo de 1s + caché; cortacircuitos intacto. Todas las fuentes por el cliente único.
- Ejecuta el smoke del scraper y confirma que arranca.

Nota: si un campo que necesites fresco SOLO existe en páginas bloqueadas, NO lo scrapees: anótalo como "pendiente de pedir a Sackmann (endpoint/allowlist)".
```

---

### PROMPT 13 — TENNIS: Elo fresco vía overlay con handoff fijo (Elo general)  [usa /plan]

```
Contexto (TENNIS): el Elo es ~85% de lo que predice, pero está CONGELADO (Sackmann ~70 días atrasado, último masculino 2026-06-01); los resultados frescos de Tennis Explorer HOY no realimentan el Elo (solo settlement). Refrescamos el Elo SIN corromperlo. Confirmado:
- Núcleo Elo genérico: MatchEvent (fecha, género, IDs, ganador/perdedor, superficie opcional, nivel, procedencia); events_from_dataframe() mapea columnas (types.py:47, events.py:324); admite surface=None (solo general).
- build_elo_database() (build.py:1110) acoplado a Sackmann; no hay adaptador BBDD->MatchEvent, ni manifiesto multifuente, ni incremental.
- Dedup actual: logical_key=(gender, source_record_hash) sobre 49 celdas Sackmann (types.py:112), solo dentro de cada bloque de fecha. Un registro TE tendría otro hash -> conectarlo directamente contaría DOS VECES.
- Identidad: puente (gender, TE slug)->Sackmann player_id en player_mapping.sqlite3 (store.py:37). ~59% de partidos del día con ambos mapeados (peor en ITF). observations guarda SLUGS; hay que cruzarlos.
- Resultados TE con observed_at_utc real -> NO aplican el embargo de 21 días de Sackmann (ese embargo es por tourney_date = inicio del torneo, temporal.py:38). Con el motor por días, un resultado de hoy se usa como pronto MAÑANA (nunca su mismo día).

DECISIÓN YA TOMADA — HANDOFF FIJO (no la reabras): congela un commit de Sackmann como BASE y acepta de TE SOLO partidos con fecha POSTERIOR a una fecha de corte FIJA. Épocas no solapadas -> ningún partido dos veces, SIN dedup. Asumimos un hueco ~3 semanas y que algún qualifying ITF que solo tiene TE no entre nunca. El re-baseline (commit más nuevo + mover el corte) es MANUAL documentado.

OBJETIVO: en cada -Retrain, reconstruir el Elo desde (base Sackmann congelada + overlay TE post-corte), actualizando el Elo GENERAL (superficie en el prompt siguiente; aquí surface=None).

TAREAS:
1. ADAPTADOR BBDD->MatchEvent: lee resultados TERMINALES y VÁLIDOS de TE, cruza slugs con player_mapping (ambos player_id del género), construye MatchEvent (procedencia="tennis_explorer", surface=None). Sin ambos mapeados -> OMITE y loguea.
2. HANDOFF: congela el commit Sackmann base + fecha de corte versionada; el overlay solo admite fecha > corte; documenta commit + corte + re-baseline manual.
3. CONSTRUCCIÓN MULTIFUENTE en orden cronológico, respetando disponibilidad (Sackmann embargo 21d; TE observed_at_utc, uso desde el día siguiente). Procedencia y corte en el FINGERPRINT.
4. FALLBACK ELEGANTE: si el overlay falla, cae a Sackmann-solo sin romper; avisa.
5. ANTI-DOBLE-CONTEO (defensa en profundidad): aserción + test de que ningún evento del overlay tiene fecha <= corte.

TESTS: partido TE post-corte con ambos mapeados actualiza el general; sin mapeo se omite; ningún evento <= corte; reconstrucción reproducible; overlay fallando -> Sackmann-solo; un resultado TE de hoy no altera predicciones de su mismo día.

VERIFICACIÓN Y ESTADO EJECUTABLE:
- -Retrain: cuántos eventos TE entraron, cuántos omitidos, antigüedad del Elo (debe bajar de ~70 días). Top de Elo antes/después.
- Ejecuta el smoke/pipeline y confirma que arranca.
```

---

### PROMPT 14 — TENNIS: catálogo de superficies (Elo de superficie)  [usa /plan]

```
Contexto (TENNIS): elo_surface_diff es la feature más importante en hombres (~45%). El build multifuente del Elo actualiza hoy solo el general. Confirmado: de 387 terminados, 187 con superficie directa, 211 con ambos mapeados, intersección 127; propagando por mismo tournament_href + edición sin conflictos -> ~136/387. De las 190 que faltarían: 170 son el bloque agregado "Futures 2026" sin enlace al torneo; 12 "Indoors" (no distingue hard/carpet); 8 aislados. Regla: superficie LEGÍTIMA vía catálogo tournament_href + año/edición -> surface desde la FICHA EXACTA del torneo, con procedencia. NO inferir por ciudad, nombre histórico o "Indoors". Usa el CLIENTE HTTP consolidado. Respeta AGENTS.md; datos sagrados solo lectura. Si algo no coincide, PARA y pregunta.

OBJETIVO: superficie fiable para el overlay -> actualizar también el Elo de superficie, sin inventar.

TAREAS:
1. CATÁLOGO tournament_href + año/edición -> surface desde la ficha exacta (por el cliente consolidado), con procedencia; caché persistente.
2. RESOLUCIÓN por evento, EN ORDEN, solo evidencia exacta: (a) superficie directa; (b) catálogo por tournament_href + edición idéntica; (c) propagación desde mismo tournament_href/edición sin conflictos. Si ninguna aplica, surface=None.
3. PROHIBIDO inferir por ciudad/nombre/"Indoors"/pistas históricas. Ambigüedad/conflicto -> None y loguea. No rellenar los 170 "Futures 2026" ni los 12 "Indoors".
4. Integra en el build multifuente: eventos con superficie fiable actualizan Elo de superficie; el resto, solo general. Catálogo + procedencia en el FINGERPRINT.

TESTS: superficie directa/por catálogo -> Elo de superficie; "Indoors"/agregado/ambiguo -> None y solo general; el catálogo nunca asigna por ciudad/nombre; reproducibilidad con el catálogo en el fingerprint.

VERIFICACIÓN Y ESTADO EJECUTABLE:
- Cuántos eventos pasaron a superficie fiable (esperado ~136/387) y cuántos siguen en None y por qué.
- Ejecuta el smoke/pipeline y confirma que arranca.
```

---

### PROMPT 15 — TENNIS: `best_of` y `round` nulos en producción

```
Contexto (TENNIS): en producción best_of y round llegan NULAS (el modelo se entrenó con ellas desde Sackmann, pero el mapeo TE -> features no las rellena en vivo). best_of importa (Grand Slam a 5 sets predice distinto que 3). Bug de mapeo, no de fuente. Usa el CLIENTE HTTP consolidado. Respeta AGENTS.md; datos sagrados solo lectura. Si algo no coincide, PARA y pregunta.

OBJETIVO: rellenar best_of y round en producción por vía legítima, con manejo honesto de la ausencia.

TAREAS:
1. Deriva best_of y round de la info de torneo/cuadro que Tennis Explorer sí expone (nivel, ronda), por el cliente consolidado. Si no se puede determinar con evidencia, deja el valor ausente y usa el indicador de ausencia que el modelo ya conoce (como ranking_missing/age_missing), no un default silencioso.
2. Verifica que la definición coincide con la de entrenamiento (mismos códigos de round, mismo best_of por nivel). Si hay discrepancia de codificación, PARA y pregunta.

TESTS: Grand Slam -> best_of=5; tour normal -> 3; indeterminado -> ausente con flag; codificación de round coincide con entrenamiento.

VERIFICACIÓN Y ESTADO EJECUTABLE:
- % de partidos con best_of y round rellenos vs ausentes (con flag).
- Ejecuta el smoke/pipeline y confirma que arranca.
```

---

### PROMPT 16 — TENNIS: champion/challenger + rollback + poda  [usa /plan]

```
Contexto (TENNIS): a diferencia de CS2, tenis NO tiene puerta champion/challenger. La necesitamos porque los prompts de odds y de segmentos de tenis dependen de ella, y para que el reentreno de tenis sea seguro. Replica el patrón de la puerta de CS2 (registry/latest.json/last_good) sobre el registro de modelos de TENNIS. Respeta AGENTS.md (anti-fugas: validación temporal, nada de shuffle). Si algo no coincide, PARA y pregunta.

OBJETIVO: puerta de promoción + rollback + poda para el modelo de tenis. (La usan los prompts 17 y 18, y el orquestador 20.)

TAREAS:
1. PUERTA DE PROMOCIÓN: tras entrenar el challenger de tenis, evalúalo contra el modelo de tenis vivo sobre el MISMO hold-out temporal sin fugas. Métrica principal log-loss, desempate Brier. Promociona solo si iguala/mejora por un margen mínimo documentado; si no, descarta, deja producción intacta y avisa con los números.
2. PUNTERO last_good de tenis; actualízalo al promocionar. ROLLBACK que lo restaure.
3. PODA del registro de modelos de tenis: conserva últimos N + vivo + last_good (nunca borres vivo ni last_good). Enséñame la keep-list antes de la primera poda.
4. Conecta al reentreno de tenis: manual y automático pasan por la puerta.

VERIFICACIÓN Y ESTADO EJECUTABLE:
- La puerta promociona un challenger mejor y rechaza uno peor sin tocar producción; rollback restaura; poda respeta vivo + last_good + N.
- Ejecuta el smoke/pipeline de tenis y confirma que arranca.
```

---

### PROMPT 17 — TENNIS: odds en el modelo + fallback sin odds  [usa /plan]

```
Contexto (TENNIS): DECISIÓN de producto: máxima accuracy. Hoy las cuotas de Tennis Explorer (probabilidad de-vigada) EXISTEN como feature disponible pero están EXCLUIDAS del modelo activo. Las metemos para subir accuracy, con fallback robusto para cuando Tennis Explorer NO traiga cuota (típico en ITF). Es el equivalente en tenis del prompt 5 de CS2. Ya existe la puerta champion/challenger de tenis (prompt 16). Las cuotas de Tennis Explorer para el partido del día son pre-match. Respeta AGENTS.md; el triplete predictions/observations/settlements es SAGRADO (cualquier etiqueta nueva va en tabla lateral o columna ADITIVA nullable, sin tocar filas ni triggers). Si algo no coincide, PARA y pregunta.

REANCLAJE HONESTO (déjalo en el informe): con odds el techo de tenis sube unos puntos (literatura: ~69-71% con odds vs ~64-68% sin), pero sigue habiendo techo (~70-80% en tour/Grand Slam). La rama sin odds tiene un techo algo más bajo; objetivo: lo más cerca posible, no paridad.

OBJETIVO: máxima accuracy con odds y robustez sin ellas, dejando que la PUERTA de tenis elija la arquitectura ganadora.

ARQUITECTURAS CANDIDATAS (compáralas por la puerta, mismo hold-out temporal; promociona la mejor en log-loss/Brier):
- A — ROUTER DE DOS MODELOS: PRIMARIO (features completas de tenis + probabilidad de-vigada de la cuota); RESERVA (modelo dedicado sin odds, reforzando Elo/superficie/forma/ranking). Cada uno calibrado en su régimen.
- B — MODELO ÚNICO: la probabilidad de-vigada como feature NaN nativo + indicador "odds_available", entrenado con datos MIXTOS.

REQUISITOS COMUNES:
1. ROUTER: modo con odds solo si hay cuota VÁLIDA; si no, modo sin odds. SIEMPRE sale una predicción (crítico: en ITF muchas veces no hay cuota).
2. LEDGER/almacenamiento: etiqueta por predicción (régimen odds/no_odds, arquitectura) en tabla lateral o columna ADITIVA nullable, respetando el triplete sagrado.
3. MEDICIÓN SEPARADA: accuracy/log-loss/Brier/calibración de la porción CON odds y SIN odds por separado + fracción sin odds (probablemente alta por ITF).
4. CALIBRACIÓN por régimen (la reserva se calibra solo con partidos sin odds).

PUERTA: evalúa el sistema completo (router incluido) y promociona solo si mejora frente al modelo de tenis actual.

VERIFICACIÓN Y ESTADO EJECUTABLE:
- Tests: router con/sin odds; predicción válida sin cuota (caso ITF); etiqueta de régimen; medición separada.
- Enséñame el ranking de candidatas y el desglose con-odds vs sin-odds (calibración incluida) + la fracción sin odds.
- Ejecuta el smoke/pipeline de tenis (incluye un caso sin odds) y las puertas; confirma que arranca y que un partido sin cuota predice.
```

---

### PROMPT 18 — TENNIS: medición de calibración por segmento  [usa /plan]

```
Contexto (TENNIS): equivalente del prompt 7 de CS2. Segmentos relevantes en tenis: superficie (hard/clay/grass), nivel de torneo (Grand Slam/Masters/ATP/Challenger/ITF), best_of (3 vs 5) y género. Queremos MEDIR dónde el modelo calibra peor y decidir con datos, por la puerta de tenis (prompt 16), sin perder el modelo actual. Respeta AGENTS.md (anti-fugas). Si algo no coincide, PARA y pregunta.

CLAVE: mide la calibración por segmento sobre el HOLD-OUT TEMPORAL / BACKTEST (donde hay volumen) y trackéala en vivo desde settlements según se acumula. Con la muestra live actual (72 partidos) NO se decide por segmento.

TAREAS:
1. HERRAMIENTA DE MEDICIÓN: sobre el backtest, calibración (prob media predicha vs real) por superficie, nivel de torneo, best_of y género. Por segmento: fiabilidad + desviación (ECE) + TAMAÑO DE MUESTRA. Replica en vivo desde settlements. N pequeño -> "sin muestra suficiente".
2. ESCALERA (documéntala; no auto-ejecutes): todos calibran bien -> nada. Un segmento se desvía -> features de INTERACCIÓN (p.ej. elo_surface_diff x nivel, best_of x elo_diff) y la puerta decide. Solo si un segmento es GRANDE y sigue desviándose -> modelo dedicado por segmento, por la puerta.
3. IMPLEMENTA la vía de interacción como OPCIÓN lista; NO fuerces ninguna: la puerta decide. Modelo actual nunca se pierde.
4. INFORME honesto con N por segmento y aviso de que la muestra live aún no decide.

VERIFICACIÓN Y ESTADO EJECUTABLE:
- Enséñame la tabla de calibración por segmento del backtest (con N) y dime dónde hay desviación real vs ruido.
- Ejecuta el smoke/pipeline de tenis y confirma que arranca.
```

---

### PROMPT 19 — Web: pestañas + gráfico "Modelo" con filtros + "Best opportunity"  [usa /plan]

```
Contexto (WEB): web estática; WEB/build_web.py (etapa 9, se regenera en CADA run) genera WEB/data.js (window.__CS2_PREDICTOR_DATA__) que consume WEB/index.html (JS vanilla). Ya existen: accuracy/calibración live de CS2 desde prediction_ledger (prompt 9) y de tenis desde settlements (prompt 11); ambos deportes producen predicciones diarias; existe una banda de confianza por predicción (prompt 4). Reutiliza esos cálculos, no los dupliques. Respeta AGENTS.md; datos sagrados solo lectura. Si algo no coincide, PARA y pregunta.

OBJETIVO: reorganizar la web en pestañas y añadir el gráfico "Modelo" filtrable y "Best opportunity".

TAREAS:
1. PESTAÑAS: Partidos | Best opportunity | CS2 | Tenis | Modelo | BBDD.
   - Partidos: TODOS los partidos del día (CS2 + tenis). CS2: solo CS2. Tenis: solo tenis.
   - Best opportunity: partidos donde el modelo está MÁS SEGURO. Ordénalos por probabilidad (más cercana a 100%/0%) Y banda de confianza ESTRECHA (seguro = alto + poca incertidumbre, no solo un número alto). Muestra la banda por partido. Ambos deportes, filtrable por deporte.
   - Modelo: el gráfico (abajo). BBDD: panel de estado del almacenamiento/backup (se completa con el prompt 21; de momento informativo).
2. GRÁFICO "MODELO" CON FILTROS: tiempo (1 semana / 1 mes / 3 meses / 6 meses / 1 año) x deporte (todos / CS2 / tenis) x subconjunto "best opportunity" (on/off). Accuracy en la ventana seleccionada. Fuente: accuracy live unificada de prediction_ledger (CS2) + settlements (tenis). Se actualiza en CADA start.ps1.
   - HONESTIDAD/UX: sin datos en la ventana -> NO muestres nada (ni 0% ni línea plana). Muestra SIEMPRE el %. Tooltip por día: "4/5 partidos, X% acierto". N pequeño -> avísalo.
3. build_web.py organizado, reutilizando lo de 9 y 11.

TESTS: las pestañas renderizan y filtran; el gráfico filtra por tiempo x deporte x best-opportunity; ventana sin datos -> nada; tooltip con nº de muestras + accuracy; Best opportunity ordena por confianza y muestra la banda.

VERIFICACIÓN Y ESTADO EJECUTABLE:
- build_web contra datos reales (o fixtures): pestañas, gráfico con un par de filtros, y Best opportunity.
- Ejecuta el smoke/pipeline y confirma que la web se genera y el proyecto arranca.
```

---

### PROMPT 20 — Orquestador raíz CS2+TENNIS con `-Retrain`  [usa /plan]

```
Contexto: hoy el start.ps1 de la RAÍZ solo llama a CS2\start.ps1 y propaga su exit code (líneas 1-2). TENNIS tiene su runner TENNIS/run_tennis.ps1 (manual, venv propio) y CS2 nunca lo lanza. Única relación: la etapa web de CS2 (etapa 9, build_web.py) lee TENNIS/BBDD/tennis.sqlite3 en solo lectura para el dashboard. Las puertas champion/challenger ya EXISTEN en ambos (CS2 prompt 3, TENNIS prompt 16). Respeta AGENTS.md. Si algo no coincide, PARA y pregunta.

OBJETIVO: un run conjunto cada mañana, seguro y con los dos dominios independientes en fallos.

TAREAS:
1. ORQUESTADOR RAÍZ: convierte el start.ps1 de la raíz en un orquestador que llame a CS2 y a TENNIS con try/catch POR PROYECTO y exit-code POR DOMINIO: que uno falle NO aborte el otro. Cada proyecto mantiene su abort-on-first-error por etapa. ORDEN: ejecuta TENNIS ANTES de la etapa web de CS2 (o haz que la etapa web tolere datos de tenis desfasados) para que el dashboard lea un tennis.sqlite3 fresco; justifica la elección.
2. FLAG -Retrain a nivel raíz: propaga el reentreno a AMBOS proyectos, y cada reentreno pasa OBLIGATORIAMENTE por SU puerta (ya existentes). Sin -Retrain: solo predecir + backfill.
3. Resumen final con estado y exit-code por dominio (CS2 y TENNIS por separado).

VERIFICACIÓN Y ESTADO EJECUTABLE:
- Simula un fallo en CS2 y demuestra que TENNIS corre igual (y viceversa), con exit-codes por dominio. Ejecuta la raíz con y sin -Retrain y demuestra que -Retrain pasa por las puertas de ambos. Demuestra el orden TENNIS -> etapa web de CS2.
- Ejecuta el orquestador de punta a punta y confirma que ambos dominios arrancan.
```

---

### PROMPT 21 — Migración a `/BBDD` + backup consistente + detector de USB  [usa /plan] [independiente; capstone]

```
Contexto: hoy las BBDD están separadas: CS2 usa BBDD/cs2.db (SQLite) + JSON auxiliares (master/matches.json) + BLACKBOX (SQLite gz); TENNIS usa varios SQLite (tennis.sqlite3, elo.sqlite3, player_mapping.sqlite3) + Parquet (datasets) + CSV (predicciones por día). Los MODELOS viven aparte (CS2 MODEL/artifacts/registry/, PIPELINE/runs/; y el registro de tenis). Respeta AGENTS.md; los datos sagrados (triplete de tenis, prediction_ledger de CS2) NO se pierden ni se mutan. Si algo no coincide, PARA y pregunta.

DECISIONES YA TOMADAS (no las reabras):
- Unificar TODAS las BBDD/datos bajo una única /BBDD en la raíz, MANTENIENDO la estructura interna de cada proyecto (p.ej. /BBDD/cs2/... y /BBDD/tennis/...).
- Lo MÁS LIGERA posible pero con TODA la info de datos: NADA sagrado se pierde.
- SIN MODELOS en /BBDD (los registros de modelos NO se mueven; se pueden reentrenar). No importa reentrenar.
- Backup semanal a USB: copia CONSISTENTE (no copia en caliente) + detector de USB con menú NUMERADO.

TAREAS:
1. MIGRACIÓN: mueve las BBDD y ficheros de datos de CS2 y TENNIS bajo una única /BBDD en la raíz, preservando la estructura interna. Actualiza TODAS las referencias de ruta en AMBOS proyectos, en build_web.py, en el orquestador raíz y en cualquier config. NO muevas modelos. Verifica que el pipeline completo corre contra /BBDD.
2. LIGERA PERO COMPLETA: /BBDD contiene TODA la info de datos; excluye modelos y artefactos intermedios podables. Documenta qué hay y qué se excluye (y que lo excluido es reentrenable/regenerable, NO datos perdidos).
3. BACKUP CONSISTENTE: script que produzca una copia ÍNTEGRA y verificada de los SQLite con el backup online de SQLite (.backup / API) o un dump — NUNCA copia en caliente del fichero vivo. Incluye los datos no-SQLite (Parquet/CSV/JSON). Verifica que la copia abre y pasa integrity_check.
4. DETECTOR DE USB: detecta SOLO unidades extraíbles (NUNCA C: ni el disco del sistema). Menú NUMERADO por etiqueta:
     USBs detectados:
       1) Terra (E:)
       2) KINGSTON (F:)
     Escribe el número del USB destino:
   El usuario teclea el número; confirma ("copiar a Terra (E:)?"); copia; VERIFICA que la copia en el USB abre y está íntegra. Sin USB, avisa y no hace nada. Número inválido -> rechaza y vuelve a preguntar.
5. RESTAURACIÓN PROBADA: procedimiento documentado y CON TEST de "restaurar en limpio y comprobar que todo está y abre".

TESTS: pipeline corre contra /BBDD; backup consistente + integrity_check + datos no-SQLite; detector lista SOLO extraíbles con menú numerado; número inválido rechazado; sin USB no hace nada; test de restauración en limpio pasa.

VERIFICACIÓN Y ESTADO EJECUTABLE:
- Ejecuta el pipeline tras la migración (arranca contra /BBDD). Ejecuta backup + restauración de prueba y enséñame que todo queda íntegro.
- Si algo queda roto, arréglalo o revierte aquí.
```

---

## Cierre

Orden por dependencias, todo dejando el proyecto arrancable:

- **1–2**: arreglo de raíz del yaml/parsel + puertas/CI.
- **3–9 (CS2)**: puerta, reproducibilidad+banda, odds, ablación, segmentos, **rating sensible al roster** (el que quita palos gordos), y gráfico honesto.
- **10**: fetch de resultados con puerta temporal (2.5h) en ambos deportes.
- **11–18 (TENNIS)**: 0.0% (independiente), scraper limpio + cliente consolidado, Elo fresco (handoff), superficie, best_of/round, **puerta propia de tenis**, **odds en el modelo con fallback**, y segmentos.
- **19**: web con pestañas + gráfico "Modelo" filtrable + Best opportunity.
- **20**: orquestador conjunto con `-Retrain` (las puertas ya existen en ambos).
- **21**: `/BBDD` + backup a USB verificado (independiente; adelántalo si quieres respaldo ya).

Qué sube accuracy de verdad: **odds en el modelo** (CS2 y ahora tenis) y **Elo fresco** en tenis. Qué quita palos: **rating-roster** en CS2. El resto sube calibración, robustez y usabilidad. Recuerda el techo (~70% pre-match en ambos): esto acerca al techo y lo hace fiable, no lo rebasa.

Puntos irreversibles a vigilar tras cada `git diff`: primera poda (CS2 y tenis: confirma keep-list), columnas nuevas en datos sagrados (siempre aditivas/laterales), y la migración a `/BBDD` (que el backup consistente y la restauración pasen antes de fiarte del USB). Si Codex ve que la estructura no encaja, que pare y lo ajustamos.
