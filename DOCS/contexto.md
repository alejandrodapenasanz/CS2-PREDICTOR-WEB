# Contexto técnico integral — CS2-Predictor

Fecha de auditoría: **14 de septiembre de 2026**. Idioma de operación: español;
los mensajes públicos de Telegram se generan en inglés.

Este es el punto de entrada técnico del repositorio: explica su arquitectura,
el recorrido de los datos, las reglas de modelado, la operación y el estado
privado portable. Integra íntegramente el antiguo `DOCS/VAULT.md` al final.
No contiene credenciales ni sustituye los contratos ejecutables, esquemas,
configuración, tests o [AGENTS.md](../AGENTS.md).

Las cifras de ejecución son una **fotografía fechada**, no contadores que este
Markdown actualice automáticamente. Para conocer el estado posterior, consultar
los manifiestos, decisiones de promoción, logs y dashboard generados en VAULT.
Una capacidad implementada, un experimento evaluado y una capacidad activada en
producción son tres cosas distintas. Este documento las distingue expresamente.

## Índice

1. [Orientación y estado verificado](#1-orientación-y-estado-verificado)
2. [Arquitectura, propiedad y rutas](#2-arquitectura-propiedad-y-rutas)
3. [Contratos irrenunciables](#3-contratos-irrenunciables)
4. [Arranque y orquestación reales](#4-arranque-y-orquestación-reales)
5. [CS2: adquisición y persistencia](#5-cs2-adquisición-y-persistencia)
6. [CS2: features, modelos, inferencia y promoción](#6-cs2-features-modelos-inferencia-y-promoción)
7. [CS2: experimentos y decisiones de producto](#7-cs2-experimentos-y-decisiones-de-producto)
8. [TENNIS: fuentes, identidades y adquisición](#8-tennis-fuentes-identidades-y-adquisición)
9. [TENNIS: Elo, features y modelos](#9-tennis-elo-features-y-modelos)
10. [TENNIS: predicciones oficiales y liquidaciones](#10-tennis-predicciones-oficiales-y-liquidaciones)
11. [WEB, Telegram y frescura](#11-web-telegram-y-frescura)
12. [Dependencias, pruebas y CI](#12-dependencias-pruebas-y-ci)
13. [Retención, recuperación y operación segura](#13-retención-recuperación-y-operación-segura)
14. [Limitaciones, deuda y mapa de mantenimiento](#14-limitaciones-deuda-y-mapa-de-mantenimiento)
15. [Contrato VAULT y evidencia de migración conservados](#15-contrato-vault-y-evidencia-de-migración-conservados)

## 1. Orientación y estado verificado

### 1.1 Qué es el proyecto

Sistema local de predicción prepartido para Counter-Strike 2 y tenis masculino
y femenino. Recoge fuentes externas, conserva observaciones y procedencia,
reconstruye features temporales, entrena candidatos, mantiene una producción
protegida y publica una web estática. CS2 publica además picks en Telegram.

No es un servicio de apuestas automáticas. Hay utilidades económicas y
simuladores de stake en CS2, pero no deben confundirse con calidad predictiva,
confianza del input ni una garantía de rentabilidad. El objetivo de producto es
acertar ganadores; la puerta de promoción decide por **log-loss y Brier**, no
por accuracy aislada.

### 1.2 Estado observado al terminar los arranques del 14/09

| Elemento | Estado verificado |
|---|---|
| CS2 vivo | `20260810_064827Z` |
| CS2 anterior / `last_good` | `20260803_064101Z` |
| TENNIS vivo | `d080b15c768228596b17f613a45b0757d790b193ce2380674fa1c4b73f71c1a5` |
| TENNIS `last_good` | `2320aad663fddd960fc08e78f5dc6723c278584eb419126ecd39ba098d59b594` |
| Migración VAULT | `migrated`, 852 entradas de inventario, 0 pendientes, 12 archivos imprescindibles presentes |
| Ledger CS2 | 1.385 filas; no equivale a 1.385 predicciones ya evaluadas |
| Triplete TENNIS | 6.507 predictions / 7.625 observations / 1.708 settlements |
| Web generada | 14/09, 12:30:26 Europe/Madrid; 88 partidos CS2 y 56 TENNIS |
| Predicciones TENNIS de esa web | 39 con probabilidad y 17 `not_predicted`, con motivos explícitos |

El run raíz iniciado a las 10:06 completó CS2 a las 11:55:16. Su etapa de tenis
esperó al mutex de otro arranque legítimo, y terminó a las 12:30:35 con código
cero. El arranque independiente de tenis terminó a las 12:27:46. No hubo dos
entrenamientos de tenis simultáneos.

La última puerta CS2 rechazó el candidato: N=1.015; vivo log-loss **0,634694** /
Brier **0,222387**, candidato **0,650394 / 0,226988**. La última puerta TENNIS
rechazó un empate: N=55.798, ambos **0,574738 / 0,196566**. Entrenamiento
completado no significa modelo sustituido.

La migración local está cerrada. Publicar el código actualizado en GitHub es una
operación independiente: la verificación de VAULT no hace `commit` ni `push`.
El estado privado se entrega por separado y nunca se sube al repositorio.

### 1.3 Cómo leer los resultados del proyecto

- **Backtest walk-forward / OOS**: predicciones fuera de muestra de sucesivos
  ajustes temporales. Describe una receta a través de distintos periodos.
- **Puerta**: comparación emparejada y fechada contra la producción, con su
  propio soporte, calibración y umbral anti-churn. No se mezcla con el OOS largo.
- **Live**: predicciones oficiales congeladas antes del partido, posteriormente
  liquidadas. Es la evidencia real de uso, no una reconstrucción retrospectiva.
- **Cartelera**: partidos que se pueden mostrar, incluidos los que todavía no
  permiten una predicción legítima. Su tamaño no es el tamaño de evaluación.

Las referencias históricas de CS2 de 64,40% (5.097/7.914) y otras cifras de
~65% corresponden a soportes/versiones concretos. No son una constante del
sistema. Tampoco 70,83% sobre 72 resultados de tenis describe automáticamente
el rendimiento sobre miles de partidos. Siempre acompañar una métrica con N,
periodo, artefacto, régimen y protocolo.

### 1.4 Convenciones de métricas

Para etiqueta binaria `y` y probabilidad `p` del mismo lado del encuentro:

```text
accuracy = aciertos del ganador elegido / N
log-loss = -media(y × ln(p) + (1-y) × ln(1-p))
Brier = media((p-y)²)
ECE = suma_por_bucket((N_bucket/N) × abs(media(p)_bucket - media(y)_bucket))
```

Menor log-loss/Brier/ECE es mejor; mayor accuracy es mejor. Los módulos recortan
probabilidades extremas para calcular log-loss de forma finita: conservar la
misma convención en una comparación. ECE depende del binning, de la orientación
(lado A/team1 frente a ganador elegido) y de N; no comparar dos ECE sin indicar
esa definición. Para diferencias entre candidatos usar las mismas filas y
analizar diferencias emparejadas, no dos conjuntos seleccionados por separado.

Un CI95 que contiene cero no demuestra una mejora ni equivalencia: deja la
hipótesis sin resolver con esa muestra. Explorar muchos segmentos incrementa el
riesgo de hallazgos casuales; una señal de un informe no autoriza sucesivos
ajustes sobre el mismo hold-out hasta lograr una promoción.

## 2. Arquitectura, propiedad y rutas

### 2.1 División física y lógica

```text
repositorio/
├── AGENTS.md, README.md, DOCS/contexto.md
├── start.ps1                       orquestador compartido
├── scripts/                        bootstrap, migración y auditoría VAULT
├── CS2/
│   ├── start.ps1, AGENTS.md
│   ├── BBDD/                       esquemas, ingesta, recuperación
│   ├── MODEL/cs2model/              features, estimadores, puerta, inferencia
│   ├── PIPELINE/                   adquisición coordinada y outputs
│   ├── SCRAPER/hltv-scraper-api/    componente HTTP aislado
│   ├── TESTS/, scripts/, DOCS/
│   └── requirements.txt, requirements.lock.txt
├── TENNIS/
│   ├── run_tennis.ps1, AGENTS.md
│   ├── src/                        dominio completo de tenis
│   ├── scripts/, config/, tests/, docs/
│   └── requirements.txt, requirements.lock.txt
├── TELEGRAM/                       publicación; entorno y tests propios
├── WEB/                            build_web.py, index.html, serve.py
├── .github/workflows/ci.yml
└── VAULT/                          privado; excluido de Git
    ├── CS2/                        BBDD, MODEL, PIPELINE, SCRAPER, .venv
    ├── TENNIS/                     BBDD, data, models, logs, .venv
    ├── TELEGRAM/                   .env, data, daily_report.txt, .venv
    ├── WEB/data.js                 exportación pública generada
    ├── runtime/                    cachés regenerables compartidas
    ├── archive/                    procedencias/conflictos heredados
    ├── migration.json
    └── verification/               auditorías y pruebas de reubicación
```

CS2 y TENNIS no se importan entre sí ni consultan las bases del otro. El scraper
HLTV tiene sus propias dependencias y entrega contratos de adquisición; el
modelo no importa su implementación. WEB puede consumir las exportaciones y
lecturas públicas acordadas de cada dominio, pero no calcula features ni
entrena. Telegram consume la salida publicada de CS2, no implementa el modelo.

### 2.2 Mapa de estado que no debe perderse

| Ruta desde la raíz | Propósito / protección |
|---|---|
| `VAULT/CS2/BBDD/cs2.db` | SQLite CS2; contiene `prediction_ledger` sagrado |
| `VAULT/CS2/BBDD/backups/` | Snapshots verificados y política de rotación |
| `VAULT/CS2/BBDD/BLACKBOX/` | Respaldo lógico de fuentes de verdad, incluido ledger |
| `VAULT/CS2/MODEL/artifacts/registry/` | Modelos versionados, `latest.json`, `last_good.json` |
| `VAULT/CS2/MODEL/artifacts/model.pkl` | Copia operativa del modelo que debe reconciliar con `latest` |
| `VAULT/CS2/MODEL/results/` | Evaluaciones y decisiones generadas; no confundir con informes históricos versionados |
| `VAULT/CS2/PIPELINE/master/`, `runs/`, `logs/` | Publicación canónica, procedencia de runs y diagnóstico |
| `VAULT/TENNIS/BBDD/tennis.sqlite3` | Operación, predicciones oficiales, observaciones y liquidaciones |
| `VAULT/TENNIS/data/raw/` | Fuentes originales, manifiestos, revisiones y capturas |
| `VAULT/TENNIS/data/processed/tennisratio.sqlite3` | Observaciones y estadísticas adquiridas de TennisRatio |
| `VAULT/TENNIS/data/processed/elo/elo.sqlite3` | Estado Elo reconstruible y evidencia temporal |
| `VAULT/TENNIS/models/phase7/` | `manifest.json` activo, `last_good.json`, `runs/`, decisiones |
| `VAULT/TELEGRAM/data/telegram_publish_state.sqlite3` | Idempotencia de envíos y partidos ya incluidos en daily report |
| `VAULT/TELEGRAM/.env` | Credenciales; nunca imprimirlas ni publicarlas |
| `VAULT/WEB/data.js` | Único payload de datos público del dashboard |

La configuración sigue en Git: por ejemplo `CS2/MODEL/config.yaml`,
`CS2/BBDD/backup_retention.json` y `TENNIS/config/model_promotion.json`.
No trasladar configuración versionada a VAULT para arreglar una ruta equivocada.

### 2.3 Resolución de rutas y portabilidad

Los entrypoints resuelven la raíz desde su propio archivo, no desde `cwd` ni
desde `C:\dev`. El bootstrap está en `scripts/vault_bootstrap.ps1`; las
utilidades `vault_paths.py`, `vault_migrate.py` y los resolvedores del dominio
mantienen separadas raíz de código y raíz de estado.

Los manifiestos históricos pueden contener rutas absolutas antiguas. Se
resuelven al leer únicamente mediante raíces de procedencia admitidas; **no se
reescribe el histórico** para sustituir cadenas. Se verifican hashes y cortes.
En tenis, `src/modeling/vault_compatibility.py` contiene compatibilidad estrecha
para inventarios de código revisados; no es permiso para ignorar cualquier
fingerprint incompatible.

Un venv copiado no es un intérprete portable. Los launchers pueden reconstruirlo
con CPython 3.13 y sus requisitos. Cookies cifradas por Windows pueden exigir
autenticación de nuevo en otra máquina. GitHub + VAULT no garantiza acceso a
fuentes externas ni elimina las políticas de Application Control del equipo.

## 3. Contratos irrenunciables

### 3.1 Anti-fugas: contrato exigido

Leer también [CS2/AGENTS.md](../CS2/AGENTS.md) y
[TENNIS/AGENTS.md](../TENNIS/AGENTS.md). Para un partido de día D:

1. La fecha efectiva **y** la fecha de disponibilidad de una feature deben ser
   estrictamente anteriores a D. Que el dato exista hoy no demuestra que se
   conociera entonces.
2. No inventar orden intradía cuando solo hay fecha civil. Emitir todas las
   features del día antes de observar sus resultados.
3. Excepción exclusivamente CS2 opening odds: captura UTC demostrable anterior
   al kickoff UTC publicado, primera captura válida, dos cuotas y de-vig. No
   habilita cuotas live/cierre ni otras familias capturadas durante D.
4. Reconstruir estados sobre **todo** el historial cronológico antes de excluir
   filas del entrenamiento. Filtrar training no borra ni olvida el warm-up.
5. Entrenamiento, selección, tuning, calibración y evaluación temporales; jamás
   shuffle/K-fold aleatorio para evaluar el modelo.
6. Transformaciones aprendidas solo sobre el bloque permitido del fold. No
   ajustar imputación, calibración o hiperparámetros con resultados del test.
7. Semilla 42 explícita. Añadir futuro no debe cambiar features históricas.

**Distinción importante entre norma y código legado:** en CS2,
`build_training_frame(..., freeze_civil_day=False)` todavía tiene una rama que,
si los timestamps son exactos, emite y observa secuencialmente dentro del día.
Con `freeze_civil_day=True` o fechas `date_only` sí congela el día completo.
Además, `enhanced_info.py::_strictly_before` compara la captura con kickoff para
todas sus fuentes, no exclusivamente opening odds. Esos contratos locales son
más permisivos que el `< D` civil de AGENTS. Esta auditoría documental **no los
corrige ni certifica que todas las rutas cumplan el contrato más estricto**.
Requieren revisión temporal específica antes de apoyar nuevas promociones en
ellos. Pasar un test de «añadir futuro» no prueba por sí solo el embargo civil.

### 3.2 Historiales sagrados

- CS2: `prediction_ledger` solo cambia mediante sus APIs operativas sancionadas
  para publicar antes del partido y completar su ciclo de vida. No SQL manual,
  backfills destructivos, ni reprobabilizar el pasado con el modelo actual.
- TENNIS: `predictions`, `observations`, `settlements` son append-only y están
  protegidas por triggers. Nunca `UPDATE`, `DELETE` ni retirar triggers. Datos
  nuevos mediante APIs o tablas laterales con trazabilidad.
- Trasladar una BBDD, restaurarla o corregir identidad no autoriza alterar estos
  historiales. Una pérdida de cobertura no se «arregla» inventando predicciones.

### 3.3 Seguridad operativa

No publicar VAULT, `.env`, tokens, cookies ni perfiles del navegador. No servir
la raíz del repositorio con un servidor HTTP genérico. No cargar pickles de
procedencia no confiable: deserializar puede ejecutar código.

Antes de mover/copiar estado: detener escritores y comprobar SQLite/WAL. Un
hash de archivo mientras otro proceso escribe no demuestra una copia coherente.
No aplicar `immutable=1` a una BBDD viva con WAL pendiente para fingir integridad.
No forzar ACL, propiedad o eliminación de directorios por una advertencia de
retención. La reparación excepcional de VAULT tuvo un alcance autorizado exacto.

## 4. Arranque y orquestación reales

### 4.1 Comando habitual y orden

Desde la raíz, con CPython 3.13 instalado:

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

El [launcher raíz](../start.ps1) coordina esta secuencia:

```text
CS2/start.ps1 ── éxito ──> TELEGRAM/run_telegram.ps1 ──> TENNIS/run_tennis.ps1
       │                       │                         │
       └ fallo: termina        └ fallo: registra,         └ exit final de tenis
                                 tenis continúa
```

Todos los argumentos originales van a CS2. Solo `-Retrain` se propaga también
a tenis. Telegram no recibe los argumentos de CS2. Si tenis falla, su código
prevalece; si tenis termina bien, se conserva un eventual fallo de Telegram.
Los switches canónicos `-DryRun` / `-WhatIf` evitan continuar a Telegram/tenis;
no asumir que abreviaturas arbitrarias tienen la misma detección en el wrapper.

Para mantenimiento de un solo componente, usar su entrypoint directamente.
En particular, **no ejecutar rollback de CS2 a través del wrapper raíz**:
un retorno cero de CS2 permitiría continuar a Telegram/tenis.

### 4.2 Flujo CS2

Fuentes de verdad del flujo: [CS2/start.ps1](../CS2/start.ps1),
`PIPELINE/pipeline.helpers.ps1` y `PIPELINE/pipeline.config.psd1`.
El número de etapas es dinámico; el comentario histórico de «etapa 9» no debe
usarse como contrato del contador actual.

1. Bootstrap VAULT, entornos separados modelo/scraper, certificados y guardas.
2. Auto-heal/restore BLACKBOX si procede e inicialización controlada de SQLite.
3. Scrape HLTV y recuperación de pendientes; `-SkipScrape` reutiliza un run.
4. Ingesta preentreno de hechos, cuotas, snapshots y assets.
5. Reparación de integridad core / reconstrucción point-in-time y health gate.
6. Decisión de reentreno por falta de modelo, `-Retrain` o nuevas etiquetas.
   Cualquier candidato pasa por promoción, no sustituye producción a ciegas.
7. Inferencia enriquecida, probabilidades operativas, odds y flags.
8. Análisis de calibración por contexto.
9. Ingesta final y exportación master compatible.
10. Drift causal, evaluación del ledger congelado y health gate operativo.
11. Retención/backups según política y regeneración de WEB.

Opciones relevantes: `-Retrain`, `-SkipScrape`, `-AllowOfflineFallback`,
`-NoDb`, `-MaxMatches`, `-SkipPlayerStats`, `-SkipTeamProfiles`,
`-SkipMatchAssets`, `-SkipAnalytics`, `-SkipRankings`, `-SkipSameDayRecovery`,
`-BackupBlackbox`, `-RestoreBlackbox`, `-RollbackModel`, `-DryRun`, `-Config`.
`-MaxMatches` es diagnóstico: no equivale a publicar un master completo.
Un fallback se marca; no es evidencia de adquisición fresca.

Códigos CS2 de `pipeline.config.psd1`: 1 genérico, 2 dependencias, 3 scrape,
4 ingesta previa, 5 entrenamiento/health previo, 6 enriquecimiento, 7 contexto,
8 ingesta final, 9 drift/health operativo, 10 BLACKBOX, 11 WEB.

### 4.3 Flujo TENNIS

El entrypoint real es [TENNIS/run_tennis.ps1](../TENNIS/run_tennis.ps1),
**no** `TENNIS/start.ps1`.

```powershell
.\TENNIS\run_tennis.ps1
.\TENNIS\run_tennis.ps1 -Retrain
.\TENNIS\run_tennis.ps1 -EnvironmentOnly
.\TENNIS\run_tennis.ps1 -UpdateOnly
```

El arranque completo valida su entorno, actualiza fuentes auxiliares y
TennisRatio/Tennis Abstract, conserva la base Sackmann congelada, publica los
mapeos válidos, reconstruye Elo/features, entrena **o reutiliza por fingerprint**,
ejecuta la puerta, produce predicciones, reconcilia resultados y regenera WEB.
`-Retrain` se mantiene compatible, pero el ciclo completo ya es el predeterminado.
Una fecha explícita con `-Date` sin `-Retrain` no equivale a un reentreno histórico
forzado; `-Date -Retrain` solicita ambos. Las guardas temporales siguen vigentes.

`-EnvironmentOnly` prepara/valida dependencias y termina. `-UpdateOnly` actualiza
fuentes, sin producir el ciclo de predicciones/entrenamiento/web; no confundirlo
con una ejecución completa. Las actualizaciones incompletas se señalan y el modo
completo puede continuar con evidencia válida previa bajo sus flags.

Root y launcher directo comparten mutex Windows por raíz de estado. Un segundo
arranque espera; Windows libera el mutex si el propietario cae. Los logs
incluyen `RUN_END status=... exit_code=...`: la ausencia de ese cierre después
de un apagado **no es éxito**.

### 4.4 Qué es automático y qué no

Cada arranque ejecuta el ciclo que corresponda. TennisRatio tiene control
idempotente diario UTC: una publicación válida del día puede reutilizarse.
Existe `TENNIS/scripts/manage_tennisratio_daily_task.ps1` para gestionar una
tarea programada, pero su existencia no demuestra que esté instalada en cada PC.
El repositorio no se ejecuta mágicamente si el ordenador está apagado o no hay
un scheduler activo. La frescura debe seguir avisando en ese caso.

## 5. CS2: adquisición y persistencia

### 5.1 HLTV y contratos de captura

El componente `CS2/SCRAPER/hltv-scraper-api/` recoge partidos, resultados,
equipos/jugadores, cuotas, Analytics, assets de mapas, alineaciones y rankings.
Entrega JSON/manifiestos con procedencia al pipeline. Los originales y su
momento de captura importan tanto como los valores normalizados.

La configuración de producción habilita Scrapling HTTP, impersonación Chrome y
recuperación de sesión con ventana visible cuando corresponde. Desde el cambio
del 14/09, un challenge/403 abre directamente `grab_cf.py` sobre la URL afectada;
no se abre primero Stealth. `HLTV_SOLVE_CLOUDFLARE=0` es el default y `1` lo
rehabilita explícitamente como alternativa legada. La renovación visible está
limitada a un intento por run; si no obtiene HTML válido, la URL queda pendiente.
Los 429 y errores de servidor sin challenge mantienen backoff. Sus guardas
incluyen intervalo mínimo 2,5 s, máximo 2.500 requests/run, cuarentena de URL
900 s, TTL de caché 240 s y límites de intentos/recuperaciones. Revisar el
configurado efectivo: variables ya definidas pueden prevalecer sobre defaults.

Un challenge, 403 o 429 no es una página de estadísticas vacía. Detectar antes
de parsear/publicar, respetar pausas y conservar pendientes. El código de
recuperación no garantiza que HLTV permita completar la adquisición. No
convertir un bloqueo externo en ceros ni en datos supuestamente observados.

### 5.2 Persistencia relacional

Esquema: [cs2_prediction_schema.sql](../CS2/BBDD/cs2_prediction_schema.sql).
La BBDD mantiene claves de entidades y tablas de observaciones; no reducirla
a una única tabla de predicciones.

| Grupo | Tablas importantes |
|---|---|
| Identidad | `teams`, `team_aliases`, `players`, `events`, `team_rosters` |
| Partidos/mapas | `matches`, `maps`, `veto`, `match_lineups` |
| Rondas | `map_round_sources`, `map_rounds` |
| Prepartido | `prematch_lineup_snapshots`, `player_stat_snapshots`, `odds` |
| Rankings | `team_ranking_snapshots`, `match_team_ranking_observations` |
| Analytics | `match_analytics_snapshots`, `match_analytics_map_stats`, `match_analytics_map_handicap` |
| Box scores | `map_player_stats`, `map_player_side_stats` |
| Estados/materializaciones | `ratings_history`, `match_features`, `predictions` |
| Evidencia operativa | `prediction_ledger`, `health_gate_runs`, `fetch_state`, `ingest_runs` |
| Crudos | `raw_results`, `raw_snapshots` |

Los box scores pueden actualizar estados para el futuro; nunca cuentan como
información enriquecida disponible antes de su propio partido. La existencia
de `match_features` o `ratings_history` no implica que el entrenamiento lea una
foto congelada: la ruta de modelado reconstruye estados cronológicamente.

### 5.3 Ledger y respaldo

`predictions` sirve a la operación; `prediction_ledger` conserva la predicción
prepartido evaluable y su ciclo. `upsert_prediction_ledger` y
`finalize_prediction_ledger` son vías del dominio con guardas de publicación y
congelación; no son una autorización para SQL externo.

El ledger conserva probabilidad, identificación del modelo, evidencia de la
predicción y, al liquidarse, `actual_team1_win`, `realized_log_loss` y
`realized_brier`. Las ampliaciones de incertidumbre y régimen son aditivas.
Los cálculos live deben usar sus filas evaluadas, no el último pronóstico de una
tabla mutable ni probabilidades rehechas con el modelo actual.

`BBDD/blackbox.py::SOURCE_OF_TRUTH_TABLES` incluye el ledger y las observaciones
de ranking. BLACKBOX permite preservar esa evidencia en backup/restore; no basta
respaldar solo `predictions`. Los modelos binarios se conservan aparte.

## 6. CS2: features, modelos, inferencia y promoción

### 6.1 Estado cronológico y familia de features

Código central: [features.py](../CS2/MODEL/cs2model/features.py), `dataio.py`,
`glicko2.py`, `reproducibility.py`, `config.py` y
[config.yaml](../CS2/MODEL/config.yaml).

`ChronologicalState` produce las features antes de observar el resultado. El
warm-up recorre el histórico completo; después se seleccionan filas para el
learner. Columnas diferenciales cambian de signo al invertir equipos; columnas
simétricas no. Mantener ese contrato de orientación en inferencia, evaluación y
calibración, no solo en el estimador base.

El núcleo incluye ratings, diferencias de fuerza, forma, H2H y contexto de
serie. Familias opcionales incluyen snapshots, rankings, roster, Analytics,
interacciones, ratings alternativos, odds y pistolas. Una columna disponible no
se enciende necesariamente: intervienen cobertura, selección temporal y puerta.

Valores relevantes de configuración:

| Parámetro | Valor versionado auditado |
|---|---:|
| Semilla | 42 |
| Warm-up | 10 semanas |
| Training mínimo | 800 filas |
| Semivida de forma | 120 días |
| Semivida de recencia | 365 días |
| Fracción final de calibración | 0,18 |
| Optuna | 8 trials; retuning cada 26 semanas |
| Selección: validación | 8 periodos; mínimo 80 filas |
| Selección: cobertura por familia en validación | 20 filas |
| Ganancia mínima de selección de familia | log-loss 0,0005 |
| Puerta: mínimo hold-out común | 100 partidos |
| Auto-retrain | 100 nuevas etiquetas respecto del corte de intento válido |

No mezclar estos parámetros con el harness de evaluación independiente, que
también define gaps, ventanas y calibradores. La receta exacta de un modelo es
su config/manifiesto, no todos los defaults que existan en el repositorio.

### 6.2 Estimadores y calibración

`model_zoo.py`, `calibration.py`, `calibration_suite.py`, `optuna_tuning.py` y
`evaluation.py` contienen candidatos y validación. Hay regresión logística,
LightGBM, CatBoost, XGBoost, Random Forest y referencias de rating. El inventario
de algoritmos no describe por sí solo los miembros efectivos del artefacto vivo.
Leer su metadata/manifest para saber pesos, columnas y calibrador.

Las calibraciones disponibles incluyen Platt, isotónica y Beta según la ruta.
La calibración transforma probabilidades, por lo que puntuar el mismo learner
con otro calibrador **no es evaluar el mismo sistema**. Las comparaciones de
calidad guardan la receta de calibración y los hashes de predicciones.

### 6.3 Opening odds, router y recuperación

Contrato en [odds.py](../CS2/MODEL/cs2model/odds.py) y
[ODDS_ARCHITECTURES.md](../CS2/DOCS/ODDS_ARCHITECTURES.md).
Se valida una pareja de cuotas decimales, no una cuota suelta ni una proxy de
probabilidad. Límites del módulo: cuota 1,001–100 y suma de implícitas
0,90–1,50. Para cuotas `o1`, `o2`:

```text
q1 = 1/o1; q2 = 1/o2
p_mercado_team1 = q1 / (q1 + q2)
p_mercado_team2 = q2 / (q1 + q2)
```

Se elige la primera captura válida anterior al kickoff fiable. No renovar el
timestamp de apertura por reutilizarla hoy. Un fallo de scraper no elimina una
captura anterior válida: se recupera antes de caer a `no_odds`. Si no puede
probarse causalidad/coherencia, no se usa como opening feature.

Candidatas implementadas: router de rama completa con odds + reserva dedicada
sin odds, y modelo mixto con NaN nativo/indicador `odds_available`. Calibración
y selección deben respetar el régimen. La evaluación ejerce la decisión de
rama fila a fila y reporta fracción sin odds, accuracy, log-loss, Brier y
calibración separados.

La producción auditada sigue identificada como `model_a_no_odds`; recoger odds
no equivale a activar el router como modelo vivo. Existe además postprocesado
operativo con mercado, descrito abajo. El nombre legado
`model_b_stats_plus_opening_odds` **no debe presentarse como un modelo puramente
solo-odds** sin inspeccionar su composición.

Etiquetas aditivas del contrato: `prediction_regime`,
`prediction_architecture`, `opening_odds_recovered`,
`opening_odds_captured_at_utc`. Si un artefacto ya incorpora odds, no mezclar el
mercado una segunda vez en la decisión operativa.

### 6.4 Probabilidad del modelo, confianza ajustada y banda

Son conceptos distintos:

| Campo/concepto | Significado |
|---|---|
| `model_prob_team1` | Probabilidad calibrada del artefacto para team1 |
| `decision_prob_team1` | Probabilidad operativa después de controles de fiabilidad/mercado |
| `decision_confidence` | `max(decision_prob_team1, 1 - decision_prob_team1)` |
| Fiabilidad/cobertura del input | Calidad y cantidad de evidencia, no probabilidad de ganar |
| `estimate_confidence_level` | Nivel de incertidumbre del estimado: high/medium/low |
| `P modelo del favorito ± x pp` | Probabilidad del modelo y dispersión estimada sobre esa probabilidad |

La decisión se calcula en `PIPELINE/enrich_predictions.py`, no en WEB. Sin
mercado se contrae una probabilidad con poca fiabilidad hacia 0,5. Con mercado,
`decision_probability_team1` combina modelo y probabilidad normalizada; puede
usar peso aprendido walk-forward o prior dinámico, acotado hasta 0,60 en la
ruta de política aprendida. Conserva origen y motivos. No llamar a ese resultado
la probabilidad pura del modelo.

`cs2model/uncertainty.py` implementa `ensemble_dispersion_history_v1`:

```text
coverage = clip(min(historial_team1, historial_team2) / 30, 0, 1)
s = desviación ponderada entre miembros calibrados del ensemble
penalización_historial = 0,10 × (1 - coverage)
penalización_miembro = 0,05 si hay menos de 2 miembros efectivos; si no, 0
half_width = min(0,25, sqrt(s² + penalización_historial² + penalización_miembro²))
```

Alta: cobertura ≥0,8, al menos 2 miembros y banda ≤0,05. Media: cobertura ≥0,4,
al menos 2 miembros y banda ≤0,10. En los demás casos, baja. Los extremos se
recortan a [0,1]. La banda es **incertidumbre sobre p**, no intervalo estadístico
riguroso ni rango de resultados del partido. La varianza Bernoulli `p(1-p)` ya
describe la aleatoriedad de un resultado con esa p; no es esta dispersión.
No reutilizar automáticamente la banda del modelo como banda del postprocesado.

### 6.5 Determinismo

La ruta de inferencia limita hilos de estimadores anidados y librerías
numéricas. Misma fila + mismo artefacto/config/input deben reproducir p dentro
de `1e-12` en los tests, incluidos procesos separados. No reescribe el pickle
para aplicar esos controles. Un cambio de probabilidad entre días puede deberse
a inputs/modelos distintos; comparar sus fingerprints antes de culpar a azar.
No se garantiza identidad bit a bit entre cualquier CPU o versión de librería.

### 6.6 Puerta champion/challenger

Implementación: [promotion.py](../CS2/MODEL/cs2model/promotion.py),
`artifacts.py`, `retrain_policy.py`, `MODEL/train.py`, `MODEL/manage_models.py`.

Para comparar una receta con el vivo, la puerta construye un challenger sombra
ajustado sobre el prefijo permitido por `live.metadata.date_max`; ambos se
evalúan sobre el mismo sufijo posterior. Se registran IDs, etiquetas, periodo,
N, cohort hash, prediction hashes y calibración. El artefacto final entrenado
con más historia no se puntúa sobre sus propias filas como si fueran OOS.
La validación corresponde a la receta/sombra, no a declarar fuera de muestra
el binario final que ya vio todo el histórico disponible.

Márgenes: log-loss 0,001; desempate Brier 0,0005. Una reducción diminuta que no
supera el criterio anti-churn no cambia producción. Accuracy se reporta, no
decide. N común <100 no es evidencia suficiente. Repetir la puntuación idéntica
tiene tolerancia `1e-12`.

Los casos iniciales sin modelo vivo tienen bootstrap validado; no son una vía
para saltarse el gate de un modelo existente. La promoción cambia punteros y
copia operativa de forma controlada, con hashes/locks; `last_good` conserva el
vivo saliente validado. El rollback verifica y restaura ese artefacto, no el
«último directorio que parece reciente».

El auto-retrain usa el máximo entre el corte del vivo y el último intento
terminal válido para contar etiquetas nuevas. Así no repite a diario un
candidato recién rechazado por los mismos datos. Un modelo rechazado puede
seguir apareciendo en informes, pero no es producción.

La auditoría de odds corrigió discrepancias de receta de calibración en la
evaluación sombra. Aun así, un OOS de 707 partidos y un hold-out posterior de la
puerta **no son intercambiables**. Consultar
`MODEL/audit_promotion_consistency.py` y los informes fechados, no comparar sus
log-loss omitiendo soporte y objeto evaluado.

## 7. CS2: experimentos y decisiones de producto

### 7.1 Enhanced-info y composición del training

`enhanced_available=1` significa al menos una fuente enriquecida verificable:
opening odds, snapshots de jugadores, rankings, Analytics, contexto o alineación
anunciada. Resultados y box scores no cuentan. `enhanced_info.py` registra
`enhanced_source_<fuente>` y evidencia de captura por fila. Véase la salvedad de
corte civil de §3: disponibilidad antes de kickoff no basta para todas las
familias bajo la norma vigente.

Las variantes son todos, solo enhanced y todos con pesos de recencia. Se
anota/filtra **después** de reconstruir estados sobre todo el historial.
El peso de recencia es `max(1e-3, 0.5 ** (age_days / half_life_days))`, con
semivida configurada; no elimina el pasado crudo.

Resultado histórico del 26/08: 1.061/10.414 filas enhanced (10,19%). Las fuentes
se solapan: odds 719, jugadores 684, rankings 758, Analytics 840, contexto 986,
alineación 737; no sumar esas cifras como filas independientes.

| Variante; hold-out común N=324, 11–26/08 | Accuracy | Log-loss | Brier |
|---|---:|---:|---:|
| Vivo | 62,04% | 0,648969 | 0,228952 |
| Todos | 62,35% | 0,648702 | 0,228807 |
| Solo enhanced | 60,80% | 0,667428 | 0,236703 |
| Recencia | 61,73% | 0,648004 | 0,228474 |

Ninguna superó el margen de promoción. Recencia obtuvo mejor log-loss de las
candidatas, no una promoción. Enhanced suele concentrarse en fechas recientes,
tier alto y mejor cobertura: medir distribución antes de generalizar.

### 7.2 Segmentos e interacciones

`segment_calibration.py` mide bins de diferencia Elo, stage y LAN/online con
N, ECE, gap y límites de incertidumbre. La escalera es medir → si hay evidencia,
interacción en el modelo general → solo si persiste una desviación en un grupo
grande, considerar modelo dedicado. No ejecutar automáticamente esa escalera.

`--segment-interactions` y variante mínima stage/swiss existen, apagadas si la
puerta no las acepta. La señal Swiss del backtest largo no se trasladó sin más
al hold-out posterior: informe de agosto, OOS Swiss N=214 y gap +6,87 pp con
CI95 [0,5;13,25] pp; hold-out Swiss N=87 y gap −0,60 pp con CI95
[−10,99;9,79] pp. El segundo no prueba desviación. Las variantes no obtuvieron
promoción. Elo 200–300, group y LAN siguen hipótesis cuando su CI incluye cero.

El live no habilita decisiones segmentadas antes de 1.000 resultados totales;
además exige muestra suficiente del segmento. Que un gráfico pueda mostrarse
no autoriza elegir arquitectura a partir de unos pocos aciertos.

### 7.3 Rating sensible al roster

`roster_rating.py` mantiene una alternativa al rating de identidad de la
organización. Detecta cambio de núcleo con alineaciones causales completas:
5 jugadores, mirada de 90 días y al menos 2 sustituciones para activar descuento.
La fórmula de crédito configurada usa fracción mantenida y piso 0,20:

```text
f = 0,20 + 0,80 × (jugadores del núcleo mantenidos / 5)
rating_ajustado = 1500 + f × (rating_anterior - 1500)
RD_ajustada² = f × RD_anterior² + (1-f) × 350²
```

No resetear a lo bruto ni inferir cambios a partir de una alineación futura.
Emitir el ajuste para el siguiente partido permitido antes de observar su
resultado; comprobar convergencia con nuevos resultados y estabilidad as-of.
Es familia candidata por la puerta, no permiso para forzar announced_lineups o
stand-in, ni afirmación de que el vivo ya la usa. El beneficio esperado es menor
crédito obsoleto/calibración más robusta, no un salto garantizado de accuracy.

### 7.4 Pistol rounds y fuerza del rival

Dos productos distintos: tablas agregadas HLTV `/stats/teams/pistols` y hechos
por ronda/mapa. Una URL con `endDate` histórico no demuestra que su respuesta
estuviera disponible aquel día. Guardar captura, periodo, mapa/lado y fuente;
no usar una descarga de hoy como observación de ayer.

`map_round_sources` / `map_rounds` conservan rondas y procedencia. Identificar
las pistolas de mitades reglamentarias, no tratar rondas de overtime como
pistolas. Medir conversiones de ronda 2 y breaks con su denominador; «ganar
pistola equivale a 3 rondas» no es un dato ni una regla del modelo.

`pistol_opponents.py` añade una estimación causal ajustada por fuerza rival.
Referencia auxiliar logística:

```text
P(gana CT) = sigmoid(intercepto_CT + efecto_mapa
                    + beta × (Elo_CT - Elo_T) / 400)
offset_equipo = clip(sum(y-p) / (sum(p×(1-p)) + 5), -1,5, 1,5)
```

Ventana 90 días, mínimo 100 pistolas/20 partidos para el auxiliar, `C=1`,
`max_iter=300`, semilla 42; mínimo 10 pistolas causales para el residual del
equipo. Las expectativas históricas se congelan antes de observar sus rondas.
Sin mapa/lado futuro conocido se promedia CT/T, no se introduce el veto real
posterior. Features: `pistol_opponent_edge_diff`, `pistol_opponent_available`,
`pistol_opponent_sample_min`.

Resultado exploratorio del 10/09, hold-out N=815, 11/08–09/09:

| Variante | Accuracy | Log-loss | Brier | ECE |
|---|---:|---:|---:|---:|
| Vivo | 63,19% | 0,628462 | 0,219508 | 0,028899 |
| Control | 62,82% | 0,632516 | 0,221495 | 0,058383 |
| Pistolas simples | 63,44% | 0,632087 | 0,221285 | 0,050165 |
| Pistolas ajustadas al rival | 63,19% | 0,632114 | 0,221357 | 0,052426 |

Todas rechazadas. Más información puede añadir ruido, sobreajuste o sesgo de
cobertura. Aquí solo 774/10.905 filas tenían cobertura causal (7,10%):
134/9.929 en training frente a 573/815 en hold-out. La mejora de accuracy de una
fila de la tabla no compensó peores probabilidades. El flujo dedicado es
`MODEL/run_pistol_ablation.py --opponent-adjusted`; no afirmar que todo reentreno
ordinario activa esta familia. La comprobación prospectiva requiere datos
nuevos y puerta separada, no explotar reiteradamente el mismo test.

Referencias: [PISTOL_ROUNDS.md](../CS2/DOCS/PISTOL_ROUNDS.md) y
[PISTOL_OPPONENTS.md](../CS2/DOCS/PISTOL_OPPONENTS.md).

### 7.5 Ranking HLTV y VRS

Los badges de la alineación y snapshots de ranking se almacenan en observaciones
aditivas con team ID, proveedor, ranking, edición/fecha, captura y procedencia.
No son Elo: un puesto ordinal no tiene la escala probabilística ni dinámica del
rating. No reemplazar el Elo por un puesto sin evaluación.

`match_rankings.py` y `test_match_rankings.py` cubren features causales y
ediciones compatibles. La familia puede examinarse en `-Retrain` si alcanza
200 partidos cerrados utilizables, mínimo 20 observaciones en la validación y
ganancia de selección 0,0005; después aún debe superar la puerta global.
Primera captura auditada el 12/09: 687 observaciones (435 HLTV / 252 VRS), pero
**0 filas históricas cerradas causalmente utilizables** en ese momento. Capturar
muchos badges hoy no permite insertarlos retroactivamente en training.

## 8. TENNIS: fuentes, identidades y adquisición

### 8.1 Fuentes: para qué sirve cada una

| Fuente | Uso implementado | No confundir con |
|---|---|---|
| TennisRatio | Principal: agenda, perfiles, resultados, 19 stats avanzadas y cuotas publicadas | Estadísticas ya activadas en el learner |
| Tennis Explorer | Fallback de partidos/resultados y evidencia de contexto; endpoint acotado | Acceso general sin restricciones a todo el sitio |
| Sackmann, mirror autorizado | Base canónica congelada de histórico, jugadores y entrenamiento | Feed diario actualizado por cada start |
| Tennis Abstract Elo | Informes HTML ATP/WTA de rating externo | Elo propio calculado por el proyecto |
| Tennis Abstract perfiles/historial | Adquisición autorizada separada, raw y cuarentena; prioridad jugadores de la cartelera | Reemplazo ya completado del histórico o features del vivo |
| Match Charting Project | CSV auxiliares con metadatos/puntos/stats | Universo completo ni IDs Sackmann garantizados |

El mirror Sackmann configurado es `Aneeshers/tennis-sackmann-archive`, commit
`83733587353df8a41f2fd4f516147d5aa83f5a8d`, corte 2026-06-02. El arranque normal
usa `--skip-sackmann`: mantener esa base es deliberado para que el overlay
multifuente no duplique resultados ni cambie silenciosamente el contrato.
Cambiar la base exige revisar commit, manifiesto, cutoff y reconstrucción,
validación temporal y promoción; no basta descargar un CSV nuevo.

### 8.2 Cliente HTTP y políticas por fuente

Núcleo compartido: [responsible_http.py](../TENNIS/src/responsible_http.py).
Consolida rate-limit mínimo 1 s por host, lock entre procesos por host, caché
con SHA-256 y estado de pausas. Cada fuente mantiene adaptador/parser y política
de acceso; un único coordinador HTTP no significa mismo parser ni mismas rutas.

Robots: caché TTL 24 h, grupos de agente, reglas Allow/Disallow, coincidencia
de ruta, comodines/ancla cuando aplica. Respuestas 401/403 deniegan, 404/410
permiten según la política del cliente; si no puede verificarse el permiso ni
hay caché válida, falla cerrado. Redirecciones también se validan.

La excepción `authorized_matches_endpoint_exception` de Tennis Explorer está
aislada a la URL canónica de partidos; no se extiende a perfiles u otras rutas.
Su cortacircuitos de WAF no se debilita por el hecho de usar Scrapling.
TennisRatio usa su adaptador Scrapling y controles de respuesta/robots; el HTML
de challenge no se guarda como lote válido.

Tennis Abstract tiene un conector y autorización declarada por el operador
posteriores al contrato histórico de «solo dos páginas Elo». La declaración no
es verificación independiente del permiso. Las opciones de acceso y sesiones
son específicas de esa fuente; no copiar automáticamente sus excepciones a
Tennis Explorer ni convertirlas en permiso global para saltarse restricciones.
Documentación actual de esa ruta:
[tennis_abstract_daily.md](../TENNIS/docs/tennis_abstract_daily.md).

429 de TennisRatio/Tennis Abstract: reintentos acotados, pausa persistente por
host y `Retry-After` respetado. El presupuesto de espera local es limitado; un
plazo del servidor más largo no se recorta para volver a pedir antes. La caché
válida puede servirse durante pausa y `--force` no borra la restricción. La
espera se realiza fuera del lock y en tramos que permiten registrar progreso.

La ruta de navegador TA existente tiene timeout, presupuesto de recuperación,
sesión propia y circuito ante fallos repetidos; no instala dependencias en
runtime ni escribe tokens en logs. Su existencia no prueba que todas las
páginas se hayan descargado ni que un WAF haya sido resuelto.

### 8.3 TennisRatio: publicación y estadísticas

Código `src/tennisratio/{client,parser,service,store,agenda_context}.py`.
Mantiene agenda, inventario de perfiles, fingerprints y último lote válido.
La actualización diaria selecciona participantes de agenda y perfiles cambiados
en inventario. En una primera carga o cambio masivo de `lastmod`, puede ser un
volumen grande; no necesariamente se ha ignorado la selección de hoy.

Ejemplo del 14/09: el inventario marcó 1.426 perfiles como actualizados, se
intentaron todos, 1.375 válidos y 51 fallidos. Esa fuente explica buena parte
del tiempo del run. Los perfiles previos válidos se conservan cuando procede.
Fracciones imposibles, porcentajes fuera de rango y columnas incompletas se
rechazan/cuarentenan; no sustituirlos por ceros o rellenar campos inventados.

`match_observations` y la vista `match_statistics_v1` guardan las 19 estadísticas
avanzadas: servicio, resto, aces, dobles faltas, break points, juegos y presión.
`features/tennisratio_stats.py` proporciona agregados causales y tests de futuro.
**Actualmente están fuera de `MODEL_FEATURE_COLUMNS`**. Capturarlas y conservarlas
es útil para construir cobertura futura; no afirmar que ya mejoran la accuracy.

Para una ablación válida, primero medir cuántos partidos históricos tienen
agregados disponibles antes de D para ambos jugadores. No reinterpretar una
descarga actual de resultados antiguos como disponibilidad histórica. Si la
cobertura es insuficiente, ese es el resultado y no se entrena un modelo
especializado ficticiamente fiable.

### 8.4 Identidades y procedencia

Sackmann aporta el identificador canónico; cada proveedor tiene IDs/slugs propios.
`src/player_mapping/` separa normalización, candidatos, revisión y publicación.
La coincidencia tiene que ser única, coherente con género y evidencia disponible.
Un nombre parecido no autoriza a fusionar dos jugadores.

Mapeos, capturas y manifiestos tienen fecha de disponibilidad. Un remapeo de hoy
no corrige retrospectivamente features anteriores. Los cruces inciertos quedan
en revisión y pueden impedir predicción. `identity_integrity.py` detecta
identidades problemáticas; un DOB actual que revela una edad imposible en una
fila antigua no permite filtrar silenciosamente ese pasado con información
futura. Separa diagnóstico actual y corrección causal documentada.

### 8.5 Qué falta para que TA sea histórico canónico

`tennis_abstract_selection.py` prioriza perfiles de los jugadores de la agenda;
`--full-inventory` es explícito. El proceso sigue peticiones secuenciales
controladas: priorizar un conjunto no significa obtener todas las páginas en
una sola solicitud. `tennis_abstract_store.py` conserva raw y evidencia en
`ta_rows`, `ta_snapshots`, `ta_sightings`, `ta_checks`, `ta_state`,
`ta_raw_documents`; mantiene versiones acotadas de documentos sin borrar hechos.

Las filas cortas, como 27 columnas frente al contrato de 44, quedan en
cuarentena por decisión del operador. El estado auditado conserva filas sin
`exact_match_date` ni `canonical_player_id` y `model_ready_rows=0`. No se
ejecuta JS de la fuente como código confiable para inventar un esquema.

Para integrarlas: validar esquema/semántica, resolver fecha exacta e identidad,
documentar disponibilidad, deduplicar contra el histórico/overlay, medir
cobertura causal y construir candidato por la puerta. Descargar más páginas no
resuelve automáticamente estos requisitos. El vivo continúa sin estas stats.

## 9. TENNIS: Elo, features y modelos

### 9.1 Elo propio y embargo del histórico

Módulos `src/elo/{events,engine,build,operational,store,service,parameters}.py`.
Estados separados por `(gender, player_id)`, general y superficies
hard/clay/grass/carpet. No mezclar ATP/WTA ni copiar ratings de Tennis Abstract.

```text
Rating inicial = 1500
E_A = 1 / (1 + 10 ** ((R_B - R_A) / 400))
K_i = 250 / (n_i + 5) ** 0,4
R_i_nuevo = R_i + K_i × (resultado_i - E_i)
Elo_superficie_combinado = 0,5 × Elo_general + 0,5 × Elo_superficie
```

Si no hay superficie fiable, se actualiza solo el general. No inferir superficie
por ciudad, tradición del torneo ni «Indoors». No aplicar por defecto más peso
a un Grand Slam, margen de sets o inactividad: cambiar esas fórmulas sería otro
experimento.

El histórico usa `tourney_date`, que no es necesariamente el día real de cada
ronda. Se aplica disponibilidad conservadora `source_date + 21 días`, y el uso
exige `< D`: primera elegibilidad D+22. Se congelan estados de cada día de
disponibilidad para acumular actualizaciones sin inventar orden. Los 21 días
son una hipótesis conservadora explícita, no prueba de publicación exacta de
cada partido ni garantía para cualquier torneo excepcionalmente largo.

Excluir Bye/W/O y eventos sin resultado deportivo utilizable; RET/DEF/ABD/ABN
con resultado jugado válido no se eliminan para seleccionar solo partidos
«bonitos». Deduplicación exacta con procedencia, no por `match_num` aislado.
La huella de la fila cruda/campos, commit y origen permite reconstruirla.

### 9.2 Overlay multifuente y catálogo de superficie

La base congelada y eventos posteriores al corte se combinan bajo el contrato
`multisource-general-elo-v6`. Prioridad TennisRatio y fallback Tennis Explorer;
clave de reconciliación por género, fecha y pareja canónica no ordenada.
Fecha efectiva y disponibilidad anteriores a D; no doble contabilizar un
resultado visto por ambas fuentes ni volver a incorporar el tramo ya persistido.

`surface_catalog.py` y sus consumidores resuelven superficie en este orden:

1. Evidencia directa del partido capturada antes de D.
2. Catálogo de **torneo y edición idénticos**, con procedencia anterior a D.
3. Propagación dentro de esa misma edición, sin conflicto.
4. Ninguna prueba: `surface=None`, motivo registrado; solo Elo general.

El catálogo se alimenta de cartelera TennisRatio y ficha exacta de torneo cuando
la fuente autorizada la proporciona. Un agregado «Futures», solo «Indoors», una
ciudad o un nombre histórico no identifican superficie. Conflictos no se
resuelven por mayoría silenciosa. Catálogo y evidencia participan del fingerprint.

Una superficie capturada durante D no se usa para predecir D aunque parezca un
dato obvio. Puede servir para el futuro. Tenerla publicada/capturada antes de D
sí permite completar `elo_surface_diff` legítimamente y mejorar confianza.

Elo se publica en staging/replace controlado, con manifiestos e inventarios de
código. El servicio evita duplicar eventos anteriores a su máximo ya persistido.
Forma y H2H conservan sus propios estados; actualizar Elo no actualiza por magia
todas las features.

### 9.3 Features y contrato exacto del learner

Código en `src/features/`; salidas Parquet/Zstd con manifiesto activo, runs por
fingerprint, archivos `training_M.parquet` / `training_F.parquet` y auditoría de
conflictos de ranking. Los datos crudos no se destruyen al producir estas filas.

Las 43 columnas declaradas de `MODEL_FEATURE_COLUMNS` son:

```text
surface, tour_level, tour_level_raw, best_of, round,
elo_general_diff, elo_surface_diff, elo_surface_raw_diff,
elo_general_matches_diff, elo_surface_matches_diff,
elo_cold_start_a, elo_cold_start_b,
recent_n_win_rate_diff, recent_n_matches_a, recent_n_matches_b,
recent_months_win_rate_diff, recent_months_matches_a, recent_months_matches_b,
h2h_global_balance, h2h_global_matches, h2h_surface_balance, h2h_surface_matches,
rest_days_a, rest_days_b, rest_days_diff,
rank_diff, rank_points_diff, ranking_missing_a, ranking_missing_b,
ranking_age_days_a, ranking_age_days_b,
ranking_conflict_dates_skipped_a, ranking_conflict_dates_skipped_b,
age_diff, age_distance_30_diff, age_missing_a, age_missing_b,
odds_a, odds_b, market_probability_a, market_probability_b,
market_overround, market_margin
```

El perfil vivo `sports_only` excluye las seis últimas de mercado: 37 entradas.
No contiene las 19 stats avanzadas TennisRatio ni métricas de perfiles TA.
Los parámetros, orden/orientación, inventario de código y procedencias se
verifican: no añadir columnas solo a inferencia ni ignorar una incompatibilidad
de feature manifest para conseguir que desaparezca un warning.

Rankings y edad tienen indicadores explícitos; conflictos de ranking se
registran. Ausencia no equivale a rango/edad cero. Los preprocessors y categorías
se ajustan en los bloques de training permitidos; la aplicación diaria reutiliza
el bundle, no vuelve a ajustar transformaciones sobre la cartelera.

### 9.4 best_of y round en producción

Contrato `schedule_match_format_v1` en `features/match_format.py` y
`tennisratio/agenda_context.py`. Reutiliza HTML archivado validando hash y
manteniendo la fecha original, no simula que una captura nueva es antigua.

- Grand Slam masculino, cuadro principal: `best_of=5`.
- Grand Slam femenino: `best_of=3`; «Grand Slam = 5» sin género es incorrecto.
- ATP/WTA/Challenger/ITF ordinario moderno con evidencia válida: `3`.
- Qualy masculina de Slam, Next Gen, exhibiciones u otros formatos ambiguos:
  no generalizar; desconocido cuando no existe evidencia suficiente.
- Códigos compartidos con entrenamiento: `F`, `SF`, `QF`, `R128`, `R64`, `R32`,
  `R16`, `RR`, `Q1`, `Q2`, `Q3` cuando la fuente permite identificarlos.
- «First round» sin tamaño de cuadro y «Qualification» sin número no permiten
  inventar el código exacto. Ausente con `best_of_missing` / `round_missing`.

La captura debe ser anterior a D. El modelo aplica su tratamiento aprendido de
ausentes / categoría `__MISSING__`; los flags de salida no implican que se haya
añadido una nueva columna al modelo sin reentrenarlo.

### 9.5 Entrenamiento y puerta TENNIS

Módulos `src/modeling/`; script [retrain_models.py](../TENNIS/scripts/retrain_models.py).
La familia viva es LightGBM con Platt por género; baselines y candidatos se
evalúan temporalmente. El corte de entrenamiento usa disponibilidad de resultados
estrictamente anterior a `training_as_of_date`.

La puerta [promotion.py](../TENNIS/src/modeling/promotion.py) compara el último
año completo OOF común de ambos runs. Para el test 2025: estimador hasta 2023,
calibración 2024 y test 2025. La clave es `(gender, record_id)` y deben coincidir
fecha, fecha de disponibilidad, nivel, superficie, etiqueta y temporada de test.
La columna comparada es `lightgbm_platt`.

**Precisión metodológica:** compara las predicciones OOF persistidas y
verificadas de las recetas; no ejecuta el modelo final entrenado con todo el
histórico sobre ese mismo histórico y lo llama hold-out. La prueba de repetición
de métricas verifica igualdad sobre ese artefacto de evaluación/cohorte; no
debe venderse como una nueva predicción OOS del pickle final.

Config [model_promotion.json](../TENNIS/config/model_promotion.json): log-loss
0,001, Brier 0,0005, tolerancia `1e-12`, mínimo 120 filas, una temporada común.
Se registran periodo, N, fingerprints, hashes de cohort/probabilidades,
calibración, métricas globales, género y segmentos. Si no concuerdan los
conjuntos, no seleccionar silenciosamente un subconjunto favorable.

`training.py` escribe candidatos inmutables; no publica directamente el activo.
La conmutación normal pasa por la puerta, que verifica manifiestos y el puntero.
Bootstrap validado y rollback son operaciones diferenciadas, no bypass de la
comparación de un challenger contra un vivo existente.

Activo en `models/phase7/manifest.json`; anterior en `last_good.json`; bundles
en `runs/<fingerprint>/`; decisiones bajo `promotion/decisions/`. El bundle
incluye estimador, preprocesado, calibración y contratos. Una ejecución idéntica
puede reutilizar el fingerprint: no necesita fabricar otra copia ni promover.

Última evaluación fechada: 55.798 partidos, 2025-01-06 a 2025-12-29; accuracy
69,1889%, log-loss 0,574738, Brier 0,196566 tanto vivo como challenger.
Rechazo por empate sin mejora. No usar esa accuracy como accuracy live.

## 10. TENNIS: predicciones oficiales y liquidaciones

### 10.1 Recorrido de una predicción

`src/daily_pipeline/` construye contexto causal y confianza;
`src/modeling/service.py` carga/verifica el bundle;
`src/operations/daily.py` orquesta persistencia por APIs del dominio.

Una fila necesita, entre otras guardas, identidades utilizables, modelo válido,
corte de entrenamiento compatible y posibilidad de publicar estrictamente
antes del inicio UTC. Mostrar un partido en agenda no satisface por sí solo
esas condiciones. Si faltan inputs blandos se puede predecir con LOW; si falla
una condición dura se publica `not_predicted`, probabilidad/ganador ausentes y
motivos, nunca 50% inventado ni una predicción retrospectiva.

No se interpreta una hora local ambigua como kickoff UTC exacto. Un torneo de
género contradictorio tampoco se reconcilia silenciosamente. Entre flags
observados: `prediction_not_strictly_before_scheduled_start`,
`feature_tour_gender_conflict`, `player_unmapped`, campos de feature ausente y
falta de fingerprint/probabilidad del modelo.

La confianza del input no es el porcentaje de ganar. Incluye cobertura de Elo
general/superficie, rankings, historia y vigencia de fuentes. Referencias de
muestra: 20 partidos generales y 10 de superficie; ausencias/contexto pueden
rebajar a LOW. `UNAVAILABLE` indica que no hay predicción utilizable, no certeza
baja de una probabilidad existente.

### 10.2 Tablas y ciclo append-only

Esquema: [operations/schema.py](../TENNIS/src/operations/schema.py).

| Tabla | Responsabilidad |
|---|---|
| `runs`, `snapshots`, `run_snapshots` | Ejecución y procedencias publicadas |
| `matches` | Identidad operativa del evento |
| `predictions` | Predicciones/estados registrados, protegidos por triggers |
| `official_predictions` | Selección de la predicción oficial válida congelada |
| `player_statistics` | Evidencia lateral de jugadores/inputs |
| `observations`, `run_observations` | Resultados observados y procedencia |
| `settlements` | Liquidación trazable de la predicción oficial |
| `conflicts`, `review_queue` | Ambigüedad/conflictos que no se resuelven inventando |

`OperationsStore` y reconciliadores del dominio son las vías de escritura.
Varias ejecuciones pueden registrar evidencia del mismo evento, pero no se
elige después la predicción que más acertó. Las restricciones de oficialidad,
idempotencia y triggers preservan la primera predicción oficial admisible.

### 10.3 Mapeo de resultados

`operations/result_sources.py`, `stored_results.py`, `identifiers.py` y
`validation.py` resuelven resultados de TennisRatio/Explorer. Se requiere
identidad coherente, fecha/género/pareja, marcador terminal y ganador consistente;
el resultado debe haberse observado después de la predicción. Coincidencias
ambiguas van a revisión, no a una liquidación aproximada.

Se reutilizan snapshots almacenados idempotentemente y existe recuperación de
fechas pendientes. El planificador consulta una fecha anterior pendiente por
ciclo, priorizando según historial de intentos; registra también intentos sin
resultados para no atascarse eternamente en una fecha. No significa recuperar
todo el backlog con una sola ejecución.

Un resultado descargado puede no liquidar todavía por falta de mapeo unívoco.
Ese es el primer sitio que revisar si aumenta la agenda pero el gráfico de
liquidados no cambia. No modificar el triplete a mano para que suba el contador.

## 11. WEB, Telegram y frescura

### 11.1 Contrato de publicación WEB

[WEB/build_web.py](../WEB/build_web.py) genera `VAULT/WEB/data.js`, asignando
`window.CS2_PREDICTOR_DATA`. [WEB/index.html](../WEB/index.html) lo consume con
JavaScript vanilla; no carga pickles ni entrena en el navegador.

El payload contiene, entre otros, `generatedAt`, `runDir`, `manifest`,
`masterManifest`, `model`, `database`, `matches`, `tennis`, `freshness` y
`maintenance`. La sección tenis incluye `latest_run`, `health`, `matches`,
`settled_performance`, `tables` y sus tiempos/fuentes. Los contadores se leen
del contrato publicado, no del número de tarjetas actualmente visibles.

El builder se ejecuta desde CS2 y vuelve a ejecutarse al terminar tenis. Si
tenis no llega al final, la web puede tener generación reciente con estado de
tenis anterior: comprobar timestamps **por deporte y por fuente**, no solo el
mtime de `data.js`.

«Modelo no cargado» es un estado de metadata/disponibilidad expuesto por el
builder/UI, no que el navegador tenga que descargar el modelo. Revisar el
artefacto, puntero, hashes, último run y razones de `not_predicted` antes de
ocultarlo con una etiqueta de éxito.

### 11.2 Rendimiento real y filtros

CS2: gráfico live desde filas evaluadas del `prediction_ledger`, calibración por
bucket con N y accuracy móvil sobre los últimos N resultados. Puede separar
régimen odds/no_odds si hay etiqueta. `favorite_accuracy_bands.json` es backtest
secundario, no fuente de aciertos reales ni sustituto del ledger vacío.

TENNIS: `settled_performance` usa todas las predicciones oficiales liquidadas
admisibles, no solo partidos de hoy. Los filtros combinables son género
Todos/M/F y probabilidad del **ganador pronosticado** 50–60, 60–70, 70–80,
80–90, 90–100. Actualizan conjuntamente N, aciertos, accuracy, Brier, log-loss
y curva. No usar p del jugador A para clasificar un pick que favorece a B.

El builder genera el soporte/variantes del contrato y la UI selecciona la
combinación; cada regeneración debe incorporar nuevas liquidaciones. El orden
por cabecera «Probabilidad» de la tabla de partidos es una interacción distinta
del filtro del dashboard de rendimiento.

Con N=0 se muestra ausencia de datos, no 0% ni línea plana. Cada bucket informa
su muestra. La diagonal es calibración ideal; una curva de pocos partidos no
demuestra mejora. Predicción original y resultado observado se muestran por
separado, sin sustituir el ganador previsto por el ganador real.

### 11.3 Best Opportunity y tasa histórica de derrota

En la web una tarjeta requiere `opportunity_eligible=true`, identidades no TBD,
fiabilidad mínima y **confianza ajustada ≥65%**, no solo `model_prob ≥65%`.
Los controles del usuario pueden endurecer el umbral, no bajarlo de ese piso;
la fiabilidad tiene piso 45% en el filtro. Además se aplican fechas/orden por
oportunidad. No afirmar que cualquier favorito al 65% ya es oportunidad.

La tarjeta muestra probabilidad calibrada del favorito y su banda, separadas de
la confianza ajustada utilizada para filtrar. La tasa histórica de upset aquí
es la **proporción de derrotas de ese equipo cuando una predicción histórica
congelada lo marcó favorito ≥51%**. Incluye partidos con y sin odds, usando
IDs HLTV estables y solo evidencia anterior al partido mostrado.

No es `1-p` del encuentro actual. No usa todos los resultados crudos si no
existe predicción causal histórica que acredite favoritismo. La ventana por
defecto es todo el histórico admisible, aunque sobreviva un nombre legado
`favorite_upset_90d`. Sin muestra devuelve ausencia; N pequeño se advierte.
Nunca ejecutar el modelo actual sobre el pasado para completar esa tasa.

### 11.4 Telegram y daily_report

Contrato [TELEGRAM/AGENTS.md](../TELEGRAM/AGENTS.md) y
[README](../TELEGRAM/README.md); módulos de fuente, modelos, formateo, reporte,
estado, cliente, publisher y configuración. Entorno propio, estado en VAULT.

- Canal de publicación configurado: `@cs2DailyPicks`; bot
  `@CS2PredictorPublisherBot`. Los nombres no son credenciales.
- Primer mensaje: todas las Best Opportunities de ese día.
- Segundo mensaje: los demás partidos de ese día. Fechas Europe/Madrid.
- Filtro Telegram: elegibilidad upstream y `decision_confidence > 0.65`
  **antes de redondear**. Exactamente 65% va a otros partidos. La web usa `>=`;
  esa diferencia existe hoy y no debe ocultarse en documentación.
- Confianza de la decisión a partir de `estimate_confidence_level`; si falta,
  ausencia explícita, no derivar HIGH únicamente del porcentaje del pick.
- `daily_report.txt`: oportunidades no publicadas antes, de ese día y futuros,
  con identidad/canonical URL, ganador y fecha; sin repetir partidos ya
  registrados en SQLite. No limitarlo solo al día ni solo a la mejor tarjeta.

Los dos mensajes se preparan y validan antes del envío, incluido límite de
4.096 caracteres. No asumir paginación automática. Reporte UTF-8 atómico tras
confirmar ambos envíos/no-op. Una respuesta ambigua de Telegram exige
reconciliación; no reintentar a ciegas duplicando posts. Reruns confirmados son
idempotentes; cambios de payload ya publicado se tratan como conflicto.

El dry-run de Telegram no envía HTTP, no abre su SQLite, no escribe reporte ni
lee credenciales. Su `.env` queda dentro de VAULT. No copiar un token real a
este documento, tests, comandos, capturas ni logs.

### 11.5 Frescura visible

`CS2/PIPELINE/freshness.py` y `TENNIS/src/freshness.py` producen estado por
fuente: última publicación válida, último intento, antigüedad, umbral y causa.
Fuentes diarias tienen umbral de referencia de 24 h; Sackmann usa 1.080 h
(45 días), separado y configurable. Configuración en
`CS2/PIPELINE/freshness.config.json` y `TENNIS/freshness.config.json`. Consultar
los valores vigentes; no aplicar el umbral diario indiscriminadamente a un
histórico congelado.

Mensajes distintos para «falló la actualización» y «el pipeline lleva X tiempo
sin ejecutarse». Servir caché/reutilizar un lote no renueva su edad. Fallback y
datos obsoletos se señalan en predicción/resumen/dashboard y rebajan confianza
cuando corresponde, sin reescribir la predicción oficial histórica.

El reloj de la UI puede recalcular antigüedad aunque no haya nueva generación.
Un dashboard abierto dos días no debe seguir presentando su lote como recién
adquirido. Esto es observabilidad, no permiso para fabricar un refresh correcto.

## 12. Dependencias, pruebas y CI

### 12.1 Entornos reproducibles por componente

Runtime CPython **3.13**. CS2, TENNIS, scraper y Telegram tienen manifiestos
directos propios; los tres primeros mantienen lock. Declarar cada dependencia
directa en su `requirements.txt`, no confiar en transitivas. La ruta CI instala
primero desde ese manifiesto para comprobar arranque/imports y después valida
el lock. Los launchers pueden aplicar su lock para reconstruir entornos.

Dependencias relevantes: CS2 usa numpy/pandas/scipy, sklearn, LightGBM, XGBoost,
CatBoost, Optuna, SHAP/Numba, YAML y herramientas; TENNIS usa numpy/pandas,
sklearn, LightGBM, PyArrow, BeautifulSoup/lxml, Scrapling/curl-cffi, requests y
herramientas. No unificar venvs para ahorrar espacio: hay versiones deliberadas
distintas y artefactos serializados con requisitos concretos.

Restricciones Windows documentadas en los manifiestos:

- CS2 conserva `scikit-learn==1.8.0` por los dos modelos protegidos. Actualizar
  sklearn no es una migración de modelo válida por sí sola.
- Numba de CS2 está acotado a la rama 0,65.1 porque el binario 0,66 resultó
  bloqueado por Application Control en el host.
- TENNIS fija PyArrow 23,0.1: la rama 25 falló importando soporte Parquet bajo
  Application Control. Un `import pyarrow` no basta: probar `pyarrow.parquet`.
- TENNIS usa mypy 1,10.1 por wheel Python pura; binarios mypyc fueron bloqueados.
- Scrapling TENNIS está declarado con extras fetchers y versión 0,4.12.

No resolver una DLL bloqueada apagando seguridad del sistema ni ocultando
ImportError. Provisionar la versión declarada validada y probar el entrypoint
real con ese intérprete.

### 12.2 Puertas locales

Rutas de ejecución principales, desde la raíz con entornos ya preparados:

```powershell
& .\VAULT\CS2\.venv\Scripts\python.exe .\CS2\scripts\quality_gate.py
& .\VAULT\TENNIS\.venv\Scripts\python.exe .\TENNIS\scripts\run_quality_gates.py
```

Cubren lint, formato, mypy, pytest, imports y smoke según el componente. Los
scripts fijan su directorio interno cuando lo necesitan. Para scraper/Telegram,
consultar sus manifiestos y entrypoints de pruebas usando sus propios Python;
no ejecutar sus tests con un venv de otro dominio que casualmente importe todo.

La cobertura de imports comprueba que los módulos externos tienen dependencia
directa declarada y ejercita imports de arranque. Quitar una dependencia de la
declaración debe fallar aunque todavía esté instalada transitivamente.
Los smokes usan fixtures deterministas: no acreditan permisos de fuentes online.

Los gates de formato/tipado son progresivos. CS2 y TENNIS conservan baseline o
ámbito seleccionado para código legado; «puertas verdes» no significa todo el
repositorio bajo tipado ultraestricto. Deuda en
[CS2/QUALITY_PENDING.md](../CS2/QUALITY_PENDING.md) y
[TENNIS/docs/quality_gates.md](../TENNIS/docs/quality_gates.md).

### 12.3 Pruebas para encontrar un contrato

| Contrato | Pruebas de referencia |
|---|---|
| CS2 as-of | `test_leakage_audit.py`, `test_odds_point_in_time.py`, `test_training_methodology.py` |
| CS2 puerta/consistencia | `test_model_promotion.py`, `test_training_promotion_wiring.py`, `test_odds_router.py` |
| CS2 nuevas familias | `test_enhanced_info_ablation.py`, `test_segment_interaction_ablation.py`, `test_roster_sensitive_rating.py`, `test_pistol_opponents.py`, `test_match_rankings.py` |
| CS2 inferencia/UX | `test_inference_determinism.py`, `test_estimate_uncertainty.py`, `test_best_opportunity.py`, `test_web_model_accuracy_chart.py` |
| Retención CS2 | `test_retention.py`, `test_backup_retention.py`, `test_blackbox_disaster.py` |
| TENNIS HTTP | `test_responsible_http.py`, `test_tennisratio_rate_limit.py`, `test_tennis_explorer_transport.py`, tests `test_tennis_abstract_*` |
| TENNIS temporal/contexto | `test_temporal_embargo.py`, `test_surface_catalog.py`, `test_match_format.py`, `test_operational_elo_overlay.py`, `test_tennisratio_stats_features.py` |
| TENNIS modelos | `test_model_promotion.py`, `test_model_retention.py`, `test_modeling_splits.py`, `test_model_artifacts.py` |
| TENNIS operación/web | `test_operations_store.py`, `test_operations_daily.py`, `test_phase9_end_to_end.py`, `test_web_tennis_performance.py` |
| VAULT | `scripts/test_vault.py`, `scripts/test_vault_web.py`; tests `test_vault_*` de cada componente |
| Logging/arranque | `test_launcher_logging.py`, tests de bootstrap e imports de cada dominio |

Los nombres CS2 son relativos a `CS2/TESTS/`; los de tenis a `TENNIS/tests/`.
Una regresión temporal debe comprobar disponibilidad, no solo orden deportivo.

### 12.4 CI y última evidencia local

[ci.yml](../.github/workflows/ci.yml) corre push/PR a main/dev/pre-dev con Python
3.13. Jobs separados: `vault-portability` en Ubuntu/Windows y puertas de CS2,
scraper y TENNIS en Ubuntu, cada uno con sus requisitos. La prueba VAULT/WEB usa
fixtures, no requiere publicar datos privados. Telegram tiene puertas locales;
el workflow auditado no tiene job dedicado de Telegram.

Evidencia local conservada del cierre VAULT: CS2 371 tests, 17 subtests y
4 skips; TENNIS 632 tests y 264 subtests; Telegram 69; scraper 23 propios y
35 de integración; migración 12 pasados y 1 skip por permisos de symlink; WEB
público 3. Lint, formato, tipos, imports y smokes de ese cierre pasaron.
No son resultados de CI remoto ni una nueva ejecución de todos los gates por
escribir este Markdown. No anunciar CI verde hasta ver la ejecución publicada.

## 13. Retención, recuperación y operación segura

### 13.1 Qué significa «actual y anterior» realmente

El objetivo de peso es conservar operación y rollback, **no borrar evidencia
sagrada, procedencia necesaria ni inputs únicos porque sean antiguos**. Hay
políticas por clase de artefacto, no un `Remove-Item` global «salvo dos carpetas».

| Clase | Política / matiz |
|---|---|
| Modelos CS2 | `registry_keep=0` adicional: proteger vivo + last_good; guardas de rutas/hashes |
| Runs CS2 | `pipeline_runs_keep=2`, además de los referenciados por master/BBDD/procedencia |
| Backups SQLite CS2 | `keep=1` snapshot gestionado además de la BBDD viva; crear/validar antes de podar |
| Logs CS2 | Dos runs operativos; previews separados para no desplazarlos |
| Modelos TENNIS | Últimos 2 **más** champion y last_good; puede ser más de dos en total |
| Features TENNIS | Activas y generación previa verificadas, sin borrar inputs canónicos |
| Raw/caché fuentes | Rotaciones propias; no equiparar caché con observaciones históricas únicas |
| Secretos/estado Telegram | Persistentes; no borrar SQLite para «liberar espacio» |

En particular, la política TENNIS no es «solo dos modelos en total». Su primera
poda requiere confirmación de keep-list. La migración a VAULT no concede esa
confirmación ni prueba que cada almacén tenga ya solo dos generaciones. Si hay
una aprobación pendiente, los candidatos pueden acumularse hasta resolverla
mediante el mecanismo sancionado.

### 13.2 Inspección y mantenimiento de modelos

Comandos no destructivos de diagnóstico:

```powershell
& .\VAULT\CS2\.venv\Scripts\python.exe .\CS2\MODEL\manage_models.py status
& .\VAULT\CS2\.venv\Scripts\python.exe .\CS2\MODEL\manage_models.py prune --scope all
& .\VAULT\TENNIS\.venv\Scripts\python.exe .\TENNIS\scripts\retrain_models.py --retention-preview
```

Estos previews no autorizan borrar. Revisar keep-list, protegidos, token y
estado actual. Una confirmación es para el inventario/política exactos; no
reutilizar un token tras cambios. Los fallos de borrado dejan residuo y aviso
sin sustituir un buen modelo ni esconder el problema. La UI/log de mantenimiento
debe conservar pendientes; no escalar a tomar propiedad automáticamente.

Rollback **cambia producción**; ejecutar solo con intención explícita:

```powershell
.\CS2\start.ps1 -RollbackModel
& .\VAULT\TENNIS\.venv\Scripts\python.exe .\TENNIS\scripts\retrain_models.py --rollback
```

Después comprobar punteros, hashes, carga real y predicción fixture. No editar
`latest.json`, `manifest.json` o `last_good.json` a mano para lograr una promoción.

### 13.3 Backups, residuos y restauración

`CS2/BBDD/manage_backups.py` es la vía de poda de snapshots. Valida copia
autocontenida, `quick_check`, ledger y conteos; si otra copia contiene evidencia
que se perdería o no puede comprobarse, bloquea. La base viva y sus WAL/SHM,
BLACKBOX, symlinks/reparse points y nombres no sancionados no son candidatos.
Un backup nuevo se crea con la API SQLite, se verifica y solo entonces habilita
la rotación aprobada. La política está en Git, no junto a la BBDD en VAULT.

`scripts/repair_vault_residue_acl.ps1` fue una reparación excepcional con lista
literal y procesos detenidos. No es un paso habitual del pipeline. El
challenger bloqueado `20260824_125852Z` ya fue eliminado tras verificar rechazo
y protección del vivo/anterior; su decisión permanece en `VAULT/verification/`.
No quedan sus bytes en el archivo histórico por el hecho de conservar su JSON.

VAULT es estado **activo**, no una segunda copia independiente. Para recuperarse
de fallo de disco hace falta otro soporte. Copiar solo `cs2.db` ignorando WAL,
modelos, manifiestos, raw y estado de tenis/Telegram no reproduce el proyecto.

### 13.4 Comprobación de VAULT

Con procesos detenidos, lectura/auditoría de integridad:

```powershell
py -3.13 .\scripts\vault_audit.py
git ls-files VAULT
git ls-files --others --ignored --exclude-standard --directory
```

La primera comprueba integridad SQLite, tablas sagradas y hashes; rehúsa un WAL
activo. La segunda debe estar vacía. La tercera mostró solo `VAULT/` en la
migración terminada; no basta por sí sola para probar que todo archivo necesario
esté presente. Contrastar `migration.json`, sus imprescindibles y los manifiestos.

`scripts/verify_vault_portability.py` hace una prueba real de reubicación con
código/modelos y comparación de inferencia. Crea artefactos de verificación;
no es solo una consulta. No representa una copia completa de la adquisición ni
una instalación validada en otro PC. Leer sus opciones antes de repetirla.

### 13.5 Runbook de incidencias frecuentes

| Síntoma | Comprobación inicial; no hacer |
|---|---|
| Start parece parado | Último log, etapa, PID, espera HTTP y mutex; no lanzar tres entrenos en paralelo |
| TENNIS sin ganador | Flags duros, mapeo, kickoff y modelo válido; no fabricar p=50% ni predecir después del inicio |
| Gráfico liquidado no sube | `settlements` oficiales, matching/review queue y `generatedAt`; no modificar filas sagradas |
| Fuente «actualizada» pero datos viejos | Fecha de publicación válida vs intento/caché y fallback; no renovar captura por leer caché |
| Parquet / DLL bloqueada | Python exacto, lock y boot `pyarrow.parquet`; no desactivar Application Control |
| Warning sklearn al cargar | Versión de serialización/pin y hash del modelo; no regrabar el pickle sin validación |
| Acceso denegado a metadata | Registrar entrada inaccesible, preservar punteros, inspeccionar ACL/proceso; no borrado forzado |
| 429 / WAF | Estado de pausa, Retry-After y último lote válido; no parsear el challenge como datos |
| Entrenó pero sigue modelo antiguo | Decisión del gate: un rechazo/empate correcto mantiene producción |
| Web local enseña estado viejo | Tiempo por deporte, archivo servido y recarga; no asumir que servir otro directorio usa el data.js nuevo |
| Apagado a mitad de run | Ausencia de RUN_END, temporales/manifiestos; no promover candidatos incompletos |

## 14. Limitaciones, deuda y mapa de mantenimiento

### 14.1 Qué no está resuelto por tener el pipeline en verde

1. **Embargo CS2:** revisar las ramas intradía y enhanced-info descritas en §3.
   Esta documentación no certifica cierre de esa discrepancia normativa.
2. **TA no model-ready:** faltan contrato completo, fecha exacta, mapeo canónico
   y cobertura causal antes de sustituir Sackmann o entrar al learner.
3. **TennisRatio stats fuera del modelo:** el constructor existe; falta demostrar
   cobertura histórica suficiente y beneficio OOS por la puerta. La hipótesis de
   ayuda en coinflips exige entrenar con todo y evaluar por tramos, no fragmentar
   training con un subconjunto minúsculo.
4. **Fuentes fallidas/parciales:** 51 perfiles TennisRatio inválidos en el lote
   documentado, pausas TA, resultados sin identidad/superficie. No ocultarlos.
5. **Base Sackmann congelada:** overlay actualiza estado operativo; no equivale
   a disponer de un histórico canónico nuevo completo para entrenamiento.
6. **Cobertura prospectiva de nuevas familias CS2:** pistolas/rankings necesitan
   capturas válidas anteriores al partido; no rescatar cobertura haciendo fuga.
7. **Retención no uniforme:** protecciones y aprobaciones pueden conservar más
   de dos objetos. No prometer «exactamente dos de todo».
8. **QA progresiva:** formatos/tipos legado aún tienen límites explícitos; CI
   remoto necesita la versión publicada y su ejecución, no solo evidencia local.
9. **Documentación histórica:** algunos README de fases hablan de rutas antiguas,
   TA solo-Elo o refresh Sackmann automático. Para operación actual prevalecen
   código/config y contratos vigentes; informes anteriores son evidencia fechada.

El techo de accuracy alrededor de ~70% que se ha usado en discusiones de producto
es una expectativa orientativa, no cota matemática ni promesa. Sin odds cabe
esperar menos información; el objetivo es acercarse lo posible, no imponer
paridad artificial. Una mejora pequeña, nula o negativa de una nueva feature
es un resultado válido si está medida honestamente.

### 14.2 Dónde cambiar cada cosa

| Necesidad | Dueño / punto de entrada |
|---|---|
| Rutas/portabilidad | `scripts/vault_*`, resolvedores del dominio; no reescritura de históricos |
| Secuencia entre deportes | `start.ps1` raíz; lógica interna en launcher del componente |
| Parseo HLTV | scraper aislado y fixtures; persistencia en `CS2/BBDD/` |
| Feature/estimador CS2 | `CS2/MODEL/cs2model/`, config, tests temporales, gate |
| Confianza operativa CS2 | `CS2/PIPELINE/enrich_predictions.py`, contrato de salida, tests UX |
| Fuentes TENNIS | adaptadores bajo `TENNIS/src/`, cliente responsable y fixtures |
| IDs TENNIS | `player_mapping`, `identity_integrity`; revisión, no SQL ad hoc |
| Superficie/formato TENNIS | `surface_catalog`, `features/match_format`, `agenda_context` |
| Elo/features TENNIS | `elo/`, `features/`, parámetros e inventarios; rebuild verificado |
| Modelo/promoción TENNIS | `modeling/`, config y script `retrain_models.py` |
| Liquidación TENNIS | `operations/`; APIs append-only |
| Métricas/gráficos/filtros | contratos de lectura + `WEB/build_web.py`, `WEB/index.html` |
| Posts/reporte Telegram | `TELEGRAM/`; fuente publicada CS2 y su SQLite de idempotencia |
| Dependencias | `requirements.txt` + lock del componente, boot real CPython 3.13 |

### 14.3 Documentación técnica complementaria

Este contexto centraliza la visión vigente, pero no copia miles de líneas de
SQL ni reemplaza los módulos. Puntos de profundización:

- [CS2/README.md](../CS2/README.md), [PROJECT.md](../CS2/PROJECT.md),
  [EVALUATION.md](../CS2/DOCS/EVALUATION.md),
  [ESTIMATE_UNCERTAINTY.md](../CS2/DOCS/ESTIMATE_UNCERTAINTY.md),
  [BACKUP_RETENTION.md](../CS2/DOCS/BACKUP_RETENTION.md).
- [TENNIS/README.md](../TENNIS/README.md),
  [data_dictionary.md](../TENNIS/docs/data_dictionary.md),
  [elo.md](../TENNIS/docs/elo.md), [features.md](../TENNIS/docs/features.md),
  [model_promotion.md](../TENNIS/docs/model_promotion.md),
  [operations_database.md](../TENNIS/docs/operations_database.md),
  [match_format.md](../TENNIS/docs/match_format.md),
  [http_acquisition.md](../TENNIS/docs/http_acquisition.md).
- [TELEGRAM/README.md](../TELEGRAM/README.md) y los tests enumerados en §12.

Al cambiar un contrato observable, actualizar esta referencia, documentación
local y tests pertinentes. Una tabla de resultados se añade con fecha y soporte;
no se reemplaza el resultado histórico para aparentar una mejora.

### 14.4 Checklist de cierre de cambios

**Arquitecto:** dueño correcto, sin imports cruzados ni dominio en WEB/launcher;
contrato explícito, solución mínima y documentación/pruebas coherentes.

**Revisor anti-fugas:** fecha efectiva y disponible, corte civil/excepción odds,
warm-up completo, estados antes del resultado, splits/calibración temporales,
seed 42, invariancia ante futuro, sin tocar historiales sagrados.

**Revisor de dependencias:** import directo declarado en el componente, lock
sincronizado si cambió, Python 3.13, pip check, arranque real, lint/formato/tipos,
tests y smoke. No presentar gates antiguos como recién ejecutados.

Para esta consolidación documental no se han cambiado modelos, fuentes, BBDD,
probabilidades, políticas de promoción ni dependencias. Se conserva a
continuación todo el contenido del documento VAULT anterior, con sus fechas.

## 15. Contrato VAULT y evidencia de migración conservados

El texto siguiente integra el antiguo `DOCS/VAULT.md`. Se conserva su contenido
completo; solo se rebaja el nivel de sus encabezados para incluirlo aquí. Sus
resultados fechados son evidencia de esas ejecuciones, no comprobaciones online
continuas ni autorización para repetir eliminaciones.

### Estado privado portable

El código se obtiene de GitHub. `VAULT/`, a su lado, contiene exclusivamente
estado privado y artefactos regenerables. No debe publicarse en Git ni
compartirse públicamente: contiene las credenciales de Telegram y sesiones.

```
proyecto/
  CS2/ TENNIS/ TELEGRAM/ WEB/   código y configuración versionados
  VAULT/
    CS2/                       BBDD, MODEL, PIPELINE, SCRAPER y entornos
    TENNIS/                    BBDD, data, models, logs y entornos
    TELEGRAM/                  .env, data, daily_report.txt y entorno
    WEB/                       data.js generado
    archive/                   temporales heredados sin uso operativo
```

Las rutas se calculan desde el código, nunca desde el directorio de trabajo
ni desde una letra de unidad fija. Los modelos y registros sagrados se
trasladan íntegros: cambiar la ubicación no autoriza recalcular predicciones,
reescribir settlements ni modificar snapshots históricos.

Los entornos Python son regenerables, no un respaldo portable del intérprete.
Los lanzadores verifican Python 3.13, requisitos y arranque y los reconstruyen
automáticamente cuando no sirven en la nueva ubicación. Las sesiones cifradas
del navegador pueden requerir reautenticación en otro equipo.

Una copia de VAULT debe hacerse con el pipeline detenido. VAULT es el estado
activo, no sustituye una copia de seguridad en otro dispositivo. Los backups
internos y la retención de modelo vivo/anterior siguen siendo de cada dominio.

La migración debe registrar cada archivo y su SHA-256, rechazar conflictos,
conservar lo ilegible como pendiente y comprobar las BBDD y los modelos antes
de anunciar que la restauración está lista. El migrador no fuerza propiedad ni
ACL; la reparación excepcional autorizada se describe más abajo.

#### Uso automático

Con CPython 3.13 instalado, coloca el código de esta versión y la carpeta
`VAULT` juntos y ejecuta `powershell -ExecutionPolicy Bypass -File .\start.ps1`.
No hay que cambiar rutas ni activar entornos manualmente. Los entrypoints de
CS2, tenis y Telegram usan el mismo bootstrap. La primera preparación de un
entorno necesita acceso a los repositorios de dependencias; las fuentes externas
y sus bloqueos siguen siendo independientes de la portabilidad.

La web generada está en `VAULT/WEB/data.js`. Abre `WEB/index.html` localmente o
ejecuta `python WEB/serve.py` y visita `http://127.0.0.1:8000/`. Este servidor
solo expone los recursos de WEB y el data.js público, no el resto de VAULT.

#### Evidencia de la migración (2026-09-12)

- `VAULT/migration.json`: inventario y hashes, con recuperación ante una
  interrupción. Los conflictos no se sobrescriben; la falta de un archivo
  imprescindible bloquea el arranque en vez de crear una BBDD vacía.
- `VAULT/verification/before.json` y `after.json`: idénticos antes y después
  del traslado. Integridad SQLite correcta; ledger CS2 de 1.305 filas y
  triplete de tenis de 6.395/7.602/1.685 filas preservados, incluidos triggers.
  Una ejecución operativa posterior puede añadir resultados por las APIs normales.
- `VAULT/verification/portability.json`: copia a otra raíz con espacios,
  carga de modelos y predicciones idénticas en procesos separados. Se copiaron
  los modelos CS2 vivo/anterior y el bundle activo de tenis. Es una prueba local
  de reubicación, no una simulación de otra máquina ni una segunda copia completa
  de los 10 GB de TennisRatio.
- Las referencias absolutas antiguas se resuelven al leer solo desde raíces
  de procedencia registradas. No se reescriben manifiestos ni BBDD históricas.
  Los hashes de modelos y los cortes temporales siguen verificándose.

El 13/09 el operador autorizó reparar permisos de los residuos, incluida elevación
de Windows. `scripts/repair_vault_residue_acl.ps1` restringe la operación a una
lista literal, exige procesos detenidos y rechaza enlaces y versiones protegidas.
Su modo predeterminado es solo preview; no mueve ni borra archivos. La ejecución
elevada reparó únicamente esos residuos y quedó registrada en
`VAULT/verification/residue_acl_repair.json`. El uso normal de `start.ps1` no
requiere ejecutarlo como administrador.

La migración posterior terminó con `status=migrated`, sin pendientes. Las cuatro
features antiguas se conservaron con hashes, sin reemplazar `features_active`.
Los conflictos no críticos conservan ambas copias en `archive/legacy_collisions`.
Se retiraron directorios vacíos y dos marcadores regenerables idénticos a sus
copias archivadas. `before_acl_finish.json` y `after_acl_finish.json` son
idénticos: SQLite íntegra, ledger CS2 de 1.370 filas y triplete de tenis de
6.395/7.602/1.685 filas, con sus mismos hashes y triggers.

Tras validar rechazo, rutas, ausencia de enlaces y punteros, se eliminó únicamente
el challenger bloqueado `20260824_125852Z` (66.399.125 bytes). Su decisión queda
en `VAULT/verification/rejected_20260824_125852Z_decision.json`. Los hashes del
vivo `20260810_064827Z` y `last_good` `20260803_064101Z` no cambiaron. El binario
rechazado eliminado no se conserva en VAULT.

#### Corrección del fallo de tenis y revisión de logs (13/09/2026)

Los arranques de tenis del 12/09 a las 16:47 y del 13/09 a las 10:52 fallaban en
`retrain_models.py`. Se reprodujo la causa: `_parameter_payload` intentaba hacer
`feature_source.path.relative_to(PROJECT_ROOT)` sobre una ruta que ahora está
en `VAULT/TENNIS`. Se corrigió usando `STATE_ROOT` para ese origen y se probó
un entrenamiento real de fixture en VAULT. Se conserva la puerta temporal;
no se activa ningún modelo mediante atajos.

Además, el transcript de PowerShell omitía los diagnósticos de Python. El helper
`Invoke-TennisPython` ahora registra stdout/stderr sin búfer, conserva el código
nativo, conserva los acentos en UTF-8 y no confunde una advertencia de stderr
con un fallo. Los tests verifican
éxito y fallo reales mediante PowerShell. CS2 aplica el mismo tratamiento a
stdout/stderr: un INFO de Optuna ya no aparece como `NativeCommandError`, y un
exit no cero sigue deteniendo su etapa. Las advertencias no se descartan al
terminar correctamente. Los previews de CS2 se separan en
`logs/preview` para no desplazar los dos logs de ejecuciones operativas.

Los launchers de tenis comparten un mutex por raíz VAULT. Un segundo arranque
espera sin duplicar adquisición, entrenamiento ni publicación; Windows libera
el bloqueo si el dueño termina inesperadamente. Se prueban ambos escenarios.
Se corrigió además el permiso de escritura de la caché heredada de pytest CS2
para la cuenta del operador, sin cambiar permisos de modelos ni BBDD.

No todas las advertencias son errores de programación: el 429 de Tennis
Abstract conserva Retry-After y el progreso; una página HLTV bloqueada queda
pendiente sin inventar su contenido; la alerta de drift sigue visible. No se
silencian para simular una adquisición completa o una mejora del modelo.
En el último lote publicado de TennisRatio (13/09), 51 perfiles tienen un
error de esquema: 41 con fracciones imposibles (numerador mayor que denominador),
3 con fracciones de juegos inválidas y 7 con porcentajes fuera de rango. Es una
advertencia de calidad de fuente, no una pérdida de archivos por VAULT; se
conserva el último perfil validado cuando existe, sin inventar estadísticas.

#### Revisión de roles

Puertas locales verificadas, incluidos los rankings añadidos después: CS2
(371 pruebas y 17 subtests; 4 omitidas por sus condiciones), tenis (632 pruebas y 264 subtests),
Telegram (69), scraper (23 propias y 35 de integración), migración (12 pasadas;
una prueba de creación de symlink requiere permisos no disponibles en esta cuenta) y límite
público de WEB (3). Los smokes, lint, formato, tipos e imports de los componentes
pasaron. CI incluye además las pruebas de migración/WEB sin VAULT en Linux y
Windows; su ejecución remota requiere publicar esta versión del código.

Estas cifras no sustituyen el resultado de la ejecución online: un bloqueo
de las fuentes o un fallo de Telegram se informa como tal, no como éxito.

Se ejecutó el `start.ps1` raíz real a las 10:48 del 12 de septiembre. Tras la
adquisición HLTV y la ingesta, falló a las 13:08: el backup buscaba su política
versionada en `VAULT/CS2/BBDD/backup_retention.json`. Se corrigió para leer
`CS2/BBDD/backup_retention.json`, conservando la configuración particular de
fixtures/directorios alternativos, y se añadió una regresión. Esa ejecución
NO llegó a Telegram/tenis. En la revisión del 13/09 se identificó además el
fallo de rutas del entrenamiento de tenis descrito arriba.

El arranque del 13/09 a las 11:46 completó un challenger de CS2, rechazado sobre
990 filas: log-loss vivo 0,632479 vs challenger 0,649095; Brier 0,221402 vs
0,226414. Esa ejecución se interrumpió antes de llegar a tenis. La repetición de
las 18:25 reutilizó el lote HLTV descargado correctamente por la mañana
(`-SkipScrape`): CS2 terminó a las 18:31, conservó producción, no repitió ese
reentreno, rotó su backup y regeneró WEB. Tenis avanzó hasta el entrenamiento,
pero esa ejecución se interrumpió sin `RUN_END`; no se considera completada ni
se publican sus candidatos incompletos. El 14/09 a las 10:22 se lanzó de nuevo
el entrypoint real de tenis, con log persistente y protegido frente a arranques
simultáneos. Terminó a las 12:27:46 (hora de Madrid), `status=ok exit_code=0`.
El arranque raíz de las 10:06 completó CS2 a las 11:55:16; su etapa de tenis
esperó el mutex y terminó a las 12:30:35, también con código cero. Reutilizó el
lote y el entrenamiento ya calculados; no ejecutó ambos entrenamientos de tenis
simultáneamente.

#### Cierre verificado (14/09/2026)

- Migración: `migrated`, 852 entradas del inventario, cero pendientes y los 12
  archivos imprescindibles presentes. `git ls-files --others --ignored
  --exclude-standard --directory` devuelve únicamente `VAULT/`; no queda
  estado privado ignorado dentro de los componentes. VAULT no contiene archivos
  versionados en Git.
- `VAULT/verification/final_20260914.json`: integridad SQLite correcta en CS2,
  tenis y Telegram. Ledger CS2: 1.385 filas; triplete tenis:
  6.507/7.625/1.708. Son aumentos del ciclo operativo, no una reescritura de la
  migración. El prefijo anterior de cada tabla sagrada de tenis conserva su hash
  y sus triggers no cambiaron.
- Se repitió `verify_vault_portability.py` después de estos arranques:
  copia a otra ruta con espacios y probabilidades/fingerprints exactamente
  iguales. Esta prueba copia modelos y código; no duplica toda la BBDD de
  adquisición ni acredita una instalación en otro ordenador.
- CS2 rechazó su challenger: N=1.015, log-loss 0,634694 vivo vs 0,650394
  candidato; Brier 0,222387 vs 0,226988. Tenis rechazó el empate sin mejora:
  N=55.798, log-loss 0,574738 y Brier 0,196566 en ambos. Siguen activos
  CS2 `20260810_064827Z` y tenis `d080b15c…`, con los modelos anteriores
  protegidos. Entrenar no significa sustituir producción.
- WEB se regeneró a las 12:30:26: 88 partidos CS2 y 56 de tenis. En tenis hay
  39 predicciones con probabilidad; las otras 17 siguen sin predicción bajo sus
  guardas y flags explícitos (incluidos partidos cuyo inicio ya había pasado).
  No se crean predicciones retrospectivas para completar la cartelera.

La migración local está terminada. Para que una descarga de GitHub use este
contrato, primero debe publicarse el código actualizado; estos cambios locales
no se han enviado automáticamente al remoto. VAULT se copia por separado,
con los procesos detenidos, y nunca se publica en GitHub.

- Arquitectura: cada dominio conserva sus datos y entornos; los launchers
  coordinan rutas/preparación, sin incorporar lógica de predicción.
- Anti-fugas: este cambio solo reubica archivos; no cambia features, splits,
  semilla, resultados ni probabilidades históricas. La prueba de reubicación
  verifica igualdad exacta de inferencia y se mantienen las pruebas temporales.
- Dependencias: no se añadieron paquetes; requisitos y locks siguen separados
  por componente. Los entornos se verifican con Python 3.13, pip check, imports
  reales y las puertas existentes.
