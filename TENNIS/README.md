# Predictor profesional de partidos de tenis

## Objetivo

Construir por fases un sistema profesional, modular y testeable para estimar probabilidades de resultados de partidos de tenis. El diseño debe preservar el orden temporal de la información: para predecir un partido con fecha `D`, únicamente se podrá usar información anterior a `D`.

Las nueve fases están implementadas y la auditoría final está cerrada en
[`docs/audit.md`](docs/audit.md). La identidad vigente se consulta siempre en
`models/phase7/manifest.json`; no se fija en este README porque solo la puerta
champion/challenger puede cambiarla. El proyecto puede descubrir, actualizar,
validar y cargar el histórico de singles de Jeff Sackmann separado por género.
También versiona Match Charting Project como fuente auxiliar y captura dos
referencias Elo públicas con acceso secuencial y cortacircuitos. Además,
calcula y persiste un
Elo propio general y por superficie con consultas históricas estrictamente
anteriores a cada fecha. El Elo general se reconstruye desde una base Sackmann
congelada y resultados operativos posteriores a un corte fijo. TennisRatio
aporta ahora la cartelera y un sidecar diario de perfiles, rankings y
resultados. Sus resultados se concilian por
identidad canónica incluso con cobertura parcial segura. Tennis Explorer queda
como respaldo y puede cruzar una agenda de otra fuente por fecha, género y el
par no ordenado de IDs Sackmann; un cruce ambiguo nunca se liquida. La fase 5 resuelve esas
identidades web a IDs Sackmann mediante un mapping persistente, auditable y con
degradación honesta ante ambigüedades. La fase 6 construye además un vector
causal común y dos datasets de entrenamiento Parquet, con orientación A/B
estable, rankings en cuarentena cuando la fuente es contradictoria y un
allowlist explícito de features. Entrena y calibra temporalmente un LightGBM
por género y ya dispone de un pipeline diario que conserva todos los partidos,
publica probabilidades calibradas y compara de forma honesta con el mercado.
La auditoría final añade inferencia simétrica A/B, un diagnóstico reproducible
de IDs incompatibles para bloquear o degradar la inferencia actual —nunca para
filtrar retrospectivamente el histórico—, pruebas end-to-end y una base SQLite
append-only que conserva predicciones oficiales, estadísticas prepartido y
resultados conciliados por slug estable. La web solo lee esa evidencia oficial.
Las predicciones y settlements de la BBDD operativa nunca retroalimentan el
modelo. Solo observaciones terminales de fuente, leídas sin mutarlas y con
mapping/fechas causales, pueden alimentar el Elo mediante el handoff fijo.

Todas las fuentes remotas usan el
[cliente HTTP responsable consolidado](docs/http_acquisition.md): robots por
defecto, lock y rate-limit por host, caché SHA-256 y una única excepción exacta
para la cartelera canónica de Tennis Explorer.

## Estructura

```text
TENNIS/
├── .venv/                 # Entorno virtual local (ignorado por Git)
├── BBDD/
│   └── tennis.sqlite3     # Predicciones, stats y resultados append-only
├── data/
│   ├── raw/
│   │   ├── atp/           # CSV masculinos conservando la ruta del mirror
│   │   ├── wta/           # CSV femeninos conservando la ruta del mirror
│   │   ├── sackmann_manifest.json
│   │   ├── sackmann_manifests/          # Un manifiesto por commit
│   │   ├── tennis_MatchChartingProject/ # Fuente auxiliar independiente
│   │   ├── tennis_abstract/elo/         # HTML Elo + metadata temporal
│   │   └── tennis_explorer/             # Carteleras, metadata y control de red
│   ├── processed/
│   │   ├── elo/           # SQLite Elo generado, ignorado por Git
│   │   ├── features_active/ # Puntero y runs inmutables de features
│   │   ├── predictions/   # CSV diarios timestamped, ignorados por Git
│   │   ├── player_mapping.sqlite3 # Caché y auditoría de identidades
│   │   └── unresolved_players.csv # Cola actual de revisión manual
│   └── overrides.csv      # Correcciones manuales slug → player_id
├── docs/
│   ├── data_dictionary.md # Esquema, tipos, códigos y cautelas temporales
│   ├── elo.md             # Fórmula, parámetros y garantía anti-fugas
│   ├── player_mapping.md  # Resolución, persistencia y límites de fase 5
│   ├── phase2_report.md   # Recuentos y auditoría de la ejecución real
│   ├── phase3_report.md   # Runs, exclusiones, top Elo y validación final
│   ├── phase4_report.md   # Cobertura y validación del snapshot diario real
│   ├── phase5_report.md   # Cobertura real y pendientes de identidad
│   ├── phase6_report.md   # Recuentos, balance y auditoría del dataset real
│   ├── phase8_report.md   # Ejecución real, cobertura y flags diarios
│   ├── daily_pipeline.md  # Contrato causal, confianza y output diario
│   ├── audit.md           # Evidencia y valoración final anti-fugas
│   ├── operations_database.md # Esquema, conciliación y límites de la BBDD
│   ├── integracion.md     # Orquestación conjunta desde el start.ps1 raíz
│   ├── features.md        # Fórmulas y contrato causal de las features
│   ├── tennis_explorer.md # Contrato, límites y política del scraper diario
│   └── source_freshness.md
├── models/                # Runs versionados, modelos y calibradores
├── scripts/
│   ├── check_scrapling_release.py
│   ├── build_elo.py       # Construcción reproducible del histórico Elo
│   ├── build_features.py  # Dataset causal Parquet separado por género
│   ├── daily_predictions.py # Predicción, consola y CSV del día
│   ├── ingest_sackmann.py # Ingesta y resumen completo del histórico
│   ├── manage_tennisratio_daily_task.ps1 # Scheduler diario Install/Status/Remove
│   ├── map_tennis_explorer_players.py # Mapping diario a IDs Sackmann
│   ├── reconcile_results.py # Replay append-only de resultados almacenados
│   ├── scrape_tennis_explorer.py
│   ├── update_sources.py  # Actualizador coordinado de fuentes
│   └── update_tennisratio.py # Actualización diaria idempotente de TennisRatio
├── src/
│   ├── config.py          # Rutas del proyecto centralizadas y portables
│   ├── data_loaders.py    # Loaders tipados de partidos y jugadores
│   ├── elo/               # Motor, persistencia, servicio y orquestación Elo
│   ├── features/          # Estado causal, rankings, vector, spool y dataset
│   ├── daily_pipeline/    # Contexto dirigido, confianza y orquestador
│   ├── modeling/          # Entrenamiento, evaluación y servicio de inferencia
│   ├── operations/        # SQLite operativo, reglas y orquestación diaria
│   ├── player_mapping/    # Matcher causal, caché, overrides y revisión
│   ├── match_charting_download.py
│   ├── sackmann_download.py
│   ├── scrapling_release.py
│   ├── tennis_explorer/   # Parser, cliente y transporte Scrapling estático
│   └── tennis_abstract_elo.py
├── tests/
│   ├── fixtures/          # Muestras mínimas con el esquema real
│   ├── test_data_loaders.py
│   ├── test_elo_engine.py
│   ├── test_elo_pipeline.py
│   ├── test_elo_store.py
│   ├── test_match_charting_download.py
│   ├── test_player_mapping_candidates.py
│   ├── test_player_mapping_pipeline.py
│   ├── test_player_mapping_review.py
│   ├── test_player_mapping_store.py
│   ├── test_sackmann_download.py
│   ├── test_scrapling_release.py
│   ├── test_tennis_explorer_client.py
│   ├── test_tennis_explorer_parser.py
│   ├── test_tennis_explorer_transport.py
│   └── test_tennis_abstract_elo.py
├── .gitignore             # Exclusiones específicas del proyecto
├── AGENTS.md              # Reglas permanentes de trabajo
├── README.md              # Descripción, uso, arquitectura y roadmap
├── run_tennis.ps1         # Lanzador y verificador autocontenido
├── requirements.txt       # Rangos directos para mantener el lock
└── requirements.lock.txt  # Entorno operativo completo y fijado
```

Los directorios que aún no contienen código o artefactos conservan un archivo `.gitkeep` para que Git pueda versionar la estructura.

Los datos de `data/raw`, la base Elo de `data/processed/elo`, los Parquet y
manifiestos generados de `data/processed/features_active`, la caché SQLite de mappings
y la cola de no resueltos se ignoran en Git. Sus `.gitkeep`, la tabla manual
`data/overrides.csv`, el código, los tests y la documentación sí se versionan.

## Entorno reproducible (CPython 3.13)

El runtime obligatorio es **CPython 3.13**. `TENNIS/.venv` es derivado,
desechable y no se copia entre rutas o máquinas. El lanzador verifica
implementación, versión, `sys.prefix`, paquetes instalados, `pip check`, la
carga real de los binarios científicos/ML y un sello SHA-256 del lock.

Los manifiestos tienen funciones distintas:

- `requirements.txt` declara solo dependencias directas con rangos mantenibles.
- `requirements.lock.txt` contiene el cierre completo, pins exactos y hashes.

`pyarrow` permanece acotado a `23.0.1`: su wheel de CPython 3.13 carga
`pyarrow.parquet` bajo Windows Application Control en esta máquina. La rama
`25.0.x` fue rechazada al cargar el `parquet.dll` no firmado. El smoke del
lanzador importa expresamente `pyarrow.parquet`, no solo el paquete base, para
que una instalación incapaz de leer los Parquet reales nunca se promocione.
Una subida exige repetir ese smoke antes de regenerar y promover el lock.

`mypy` queda acotado a `1.10.1`, cuya wheel universal es Python pura. Las
wheels mypyc de CPython 3.13 probadas fueron rechazadas por la misma política.
El smoke importa `mypy.main` para comprobar la ruta que usa la puerta de tipos,
no solo el namespace vacío del paquete.

Producción instala **solo** `requirements.lock.txt` con `--no-deps`,
`--require-hashes` y `--only-binary=:all:`. Si falta el entorno o difiere el
lock/sello, se construye primero `.venv.build`; solo tras validar pins,
`pip check` e imports binarios se promueve a `.venv`. El entorno anterior queda recuperable como
`.venv.previous`, y un fallo durante la promoción restaura el anterior.

No es necesario activar el entorno para ejecutar el pipeline. Para una sesión
interactiva, después de que el launcher lo haya validado:

```powershell
.\TENNIS\.venv\Scripts\Activate.ps1
```

### Regenerar el lock

El lock se regenera deliberadamente bajo CPython 3.13 con un entorno de
herramientas separado; nunca desde el venv operativo:

```powershell
py -3.13 -m venv .\TENNIS\.lock-tools
.\TENNIS\.lock-tools\Scripts\python.exe -m pip install "pip-tools==7.5.2"
.\TENNIS\.lock-tools\Scripts\python.exe -m piptools compile --resolver=backtracking --generate-hashes --allow-unsafe --strip-extras --no-emit-index-url --output-file .\TENNIS\requirements.lock.txt --pip-args="--only-binary=:all:" .\TENNIS\requirements.txt
```

Después se valida en un entorno limpio con la suite completa y se reentrenan
ambos modelos. Un lock nuevo no autoriza reutilizar artefactos serializados por
otra versión de Python o de las librerías ML.
## Uso completo de principio a fin

### Instalación inicial

1. Instala CPython 3.13 x64 y verifica el Python Launcher:

```powershell
py -3.13 -c "import platform, sys; print(platform.python_implementation(), sys.version)"
```

2. Desde la raíz del repositorio ejecuta el lanzador:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\TENNIS\run_tennis.ps1 -EnvironmentOnly
```

Ese modo no ejecuta predicciones, BBDD ni WEB. El primer arranque construye
`.venv.build`, instala exclusivamente el lock con
hashes y wheels, valida el entorno y luego lo promociona a `.venv`. Si ya
existía un entorno operativo, queda en `.venv.previous`; antes de cada nueva
promoción se elimina solo ese `.venv.previous`, de modo que se conservan el
actual y una versión anterior. El launcher actual no crea directorios
`.venv.failed.*` ni elimina automáticamente archivos legacy anteriores a este
contrato. Si CPython 3.13 no
está disponible o falla la validación del staging, el entorno activo no se
sustituye.

Para comprobar manualmente el entorno promovido:

```powershell
.\TENNIS\.venv\Scripts\python.exe --version
.\TENNIS\.venv\Scripts\python.exe -m pip check
```

No instales `requirements.txt` en producción: sus rangos sirven para regenerar
el lock, no para ejecutar el predictor.
### Actualizar y reentrenar cuando cambie el histórico

El orden es obligatorio porque cada artefacto fija el fingerprint del anterior:

```powershell
python scripts\update_sources.py
python scripts\update_tennisratio.py
python scripts\audit_identities.py
python scripts\build_elo.py
python scripts\build_features.py
python scripts\retrain_models.py
```

El lanzador ejecuta ese mismo orden y después la predicción diaria en todos
los arranques operativos completos:

```powershell
.\TENNIS\run_tennis.ps1
.\TENNIS\run_tennis.ps1 -Retrain
.\TENNIS\run_tennis.ps1 -Retrain -Date 2026-07-30
```

`-Retrain` se conserva como alias compatible para el wrapper raíz. TENNIS ya
actualiza las fuentes aprobadas, regenera la cuarentena de identidades, Elo,
features y modelos también sin ese flag. Este ciclo no incorpora predicciones
ni resultados operativos al dataset: la BBDD se reserva para seguimiento
prospectivo hasta que exista una política de incorporación de labels auditada
por separado.

`update_sources.py` consulta los commits remotos y conserva lo ya verificado.
Elo, features y modelos son idempotentes por fingerprint. Si el commit
Sackmann no cambió, no hay que forzar una reconstrucción. Si sí cambió, deben
completarse los cuatro pasos antes de volver a predecir; el pipeline diario
rechaza mezclar raw nuevo con artefactos antiguos.

Match Charting Project y el Elo público de Tennis Abstract se actualizan y
versionan como fuentes auxiliares, pero no se concatenan silenciosamente al
dataset de entrenamiento. El predictor activo sigue usando el histórico
Sackmann auditado.

Todo arranque operativo completo, con o sin `-Retrain`, actualiza primero las
fuentes históricas y después ejecuta `scripts/update_tennisratio.py` una sola
vez. El actualizador usa la fecha civil real de la máquina, conserva su último
snapshot válido y realiza como máximo una actualización por día. Un dato
observado el día `D` conserva ese instante de
disponibilidad y solo puede ser elegible desde `D+1`, nunca para una predicción
del mismo día ni para reconstruir el pasado. La cartelera actual se conserva
para estado y presentación, mientras superficie, nivel, cuotas e identidades
del modelo se leen de una vista publicada estrictamente antes de `D`; si no
existe, el partido sigue visible pero queda sin probabilidad. El inicio es una
puerta operativa separada: se conserva el `data-utc` exacto de la agenda actual
y solo se infiere si `prediction_as_of_utc < scheduled_start_utc`. Un TBD queda
visible sin probabilidad; no se inventa una zona horaria.

El adaptador usa el transporte Scrapling con una identidad TLS fija de Chrome
y cabeceras stealth coherentes con esa identidad, sin proxies, navegador ni
redirects, y una allowlist de
`GET` limitada a `robots.txt`, agendas, sitemap y perfiles HTML
públicos de TennisRatio; nunca llama a `/api/`. Las reglas wildcard de query
se evalúan completas: `/*?q=` no se confunde con una prohibición global `/`. Cada
batch conserva URL, SHA-256, primera disponibilidad y la atribución
TennisRatio.com / CC BY-NC 4.0. Las cuotas y agregados actuales de las fichas
son evidencia de auditoría, no features históricas. La SQLite lateral vive en
`data/processed/tennisratio.sqlite3`; no sustituye ni modifica la BBDD sagrada.

Ante un **HTTP 429 sin desafío**, TennisRatio respeta `Retry-After` y aplica
backoff por host: tres intentos como máximo, esperas base de 30/60 s y un
presupuesto de espera de 90 s por llamada. Una espera mayor se registra para
una ejecución posterior; nunca se reintenta antes de lo pedido por el servidor.
La pausa persiste entre procesos. Si no se recupera, se detiene el lote y se
conserva el último publicado; los WAF reales siguen deteniendo la descarga.
No requiere flags en `start.ps1` ni `run_tennis.ps1`. Configuración y límites:
[recuperación HTTP 429](docs/http_acquisition.md#recuperación-http-429-de-tennisratio-06092026).

Esta automatización forma parte de `run_tennis.ps1` y, por tanto, también del
`start.ps1` raíz que lo coordina. El modo `-UpdateOnly`
valida el entorno, ejecuta exactamente una vez el actualizador idempotente y
termina antes del reentreno, la BBDD operativa y la WEB. Permite programar la
descarga sin duplicar el pipeline:

```powershell
.\TENNIS\run_tennis.ps1 -UpdateOnly
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\TENNIS\scripts\manage_tennisratio_daily_task.ps1 -Mode Install
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\TENNIS\scripts\manage_tennisratio_daily_task.ps1 -Mode Status
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\TENNIS\scripts\manage_tennisratio_daily_task.ps1 -Mode Remove
```

El instalador registra o reemplaza una única tarea
`CS2-Predictor-TennisRatio-Daily`, por defecto a las 06:00 hora local, con
`StartWhenAvailable`, requisito de red, bloqueo de instancias simultáneas y
tres reintentos separados 30 minutos cuando la ejecución falla. Admite hasta
12 horas por ejecución para no cortar el inventario inicial. No guarda
credenciales: usa el token interactivo del usuario que la instala. Por
ello recupera una ejecución perdida cuando ese usuario vuelve a iniciar sesión,
pero no corre sin una sesión suya. `-DailyTime HH:mm` cambia la hora. El script
se entrega sin registrar la tarea automáticamente; la instalación es una
decisión local y reversible mediante `-Mode Remove`.

### Predicción cotidiana

Desde cualquier ruta de PowerShell:

```powershell
& 'C:\ruta\al\repositorio\TENNIS\run_tennis.ps1'
```

Para ejecutar únicamente tenis desde la raíz del repositorio basta con:

```powershell
.\TENNIS\run_tennis.ps1
```

El lanzador raíz ya ejecuta ambos deportes en orden. Sin argumentos, TENNIS
actualiza sus fuentes y entrena o reutiliza por fingerprint sus dos modelos
antes de predecir. `-Retrain` sigue siendo necesario para solicitar además el
reentreno de CS2 y se acepta en TENNIS por compatibilidad:

```powershell
.\start.ps1
.\start.ps1 -Retrain
```

Los flags distintos de `-Retrain` pertenecen a CS2 y no se reenvían a tenis.
`-DryRun` y `-WhatIf` omiten tenis para mantener el carácter no mutante de la
comprobación.

Si Windows bloquea scripts por la política local, puede ejecutarse una sola
vez sin cambiar la configuración permanente:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\TENNIS\run_tennis.ps1
```

Sin argumentos usa hoy; para una fecha explícita:

```powershell
.\TENNIS\run_tennis.ps1 -Date 2026-07-30
```

Una fecha explícita puede producir un CSV diagnóstico si el modelo fue
entrenado con un corte estrictamente anterior. La BBDD solo la convierte en
predicción oficial si captura y predicción tienen fecha civil no posterior a
la jornada, el snapshot aún dice `scheduled`, existe un inicio UTC publicado y
`prediction_as_of_utc < scheduled_start_utc`, y no existe una observación
previa. Igualdad, postinicio y TBD quedan visibles pero no predichos/oficiales.
Por tanto, ejecutar hoy una fecha pasada nunca crea rendimiento retrospectivo:
el replay se conserva, pero queda no oficial.

La ejecución diaria completa siempre recorre actualización, Elo, features y
entrenamiento antes de inferir. Cada fase es idempotente: si sus entradas no
cambian reutiliza el fingerprint existente. Un run nuevo queda challenger y
solo sustituye al activo si mejora el mismo hold-out temporal por los márgenes
de la puerta. La retención conserva últimos N + champion + `last_good`.
`-UpdateOnly` es la única ruta diaria estrecha que
actualiza TennisRatio sin entrenar, predecir ni abrir la BBDD operativa. Cada
ejecución completa registra la cartelera en `BBDD/tennis.sqlite3`, reprocesa
primero los snapshots de resultados ya guardados y consulta después una
jornada anterior pendiente. El replay es idempotente y solo añade filas por la
API operativa; no actualiza ni borra el triplete sagrado.

Una ejecución diagnóstica con `-Date YYYY-MM-DD` no reemplaza producción con
un corte histórico: reutiliza el modelo activo. Solo la combinación explícita
`-Retrain -Date YYYY-MM-DD` entrena con ese corte causal.

Cada arranque real escribe un transcript en
`TENNIS/logs/run_tennis_<UTC>_<pid>.log`, también cuando se invoca a través del
`start.ps1` raíz. El log conserva stdout y stderr y termina con `RUN_END`, el
estado y el código exacto de salida. Al cerrarse conserva solo el transcript
actual y el anterior. Los logs `CS2/PIPELINE/logs/start_*.log`
solo cubren el proceso hijo de CS2 y no deben usarse para diagnosticar un fallo
posterior de TENNIS.

## Ingesta histórica Sackmann

### Fuente y licencia

Los repositorios originales `JeffSackmann/tennis_atp` y
`JeffSackmann/tennis_wta` dejaron de estar accesibles públicamente en julio de
2026. Esta fase usa, con autorización expresa, el mirror archivado
[`Aneeshers/tennis-sackmann-archive`](https://github.com/Aneeshers/tennis-sackmann-archive).
El mirror declara snapshots originales ATP/WTA de junio de 2026, conserva sus
README y distribuye los datos bajo **CC BY-NC-SA 4.0**.

La licencia exige atribuir a Jeff Sackmann / Tennis Abstract, limita el uso a
fines no comerciales y obliga a mantener la misma licencia en redistribuciones.
El mirror es un snapshot, no una fuente viva: los archivos de 2026 son parciales.

### Alcance descargado

El inventario se consulta en GitHub antes de cada ejecución y se valida contra
las familias realmente inspeccionadas:

- ATP principal: `atp_matches_YYYY.csv`, desde 1968.
- ATP qualifying y Challenger: `atp_matches_qual_chall_YYYY.csv`, desde 1978.
- ATP Futures/ITF: `atp_matches_futures_YYYY.csv`, desde 1991.
- WTA principal: `wta_matches_YYYY.csv`, desde 1968.
- WTA qualifying/ITF: `wta_matches_qual_itf_YYYY.csv`, desde 1968.
- Maestros `atp_players.csv` y `wta_players.csv`.
- Todos los CSV de rankings disponibles en ambos directorios.

Se excluyen de forma intencionada los dobles y `atp_matches_amateur.csv`, que no
pertenecen al alcance de singles ATP/Challenger/ITF y WTA/ITF definido.

### Ejecución

Desde `TENNIS/`, con el entorno virtual activado:

```powershell
python scripts\ingest_sackmann.py
```

La descarga es idempotente e incremental. Un archivo existente y verificado se
conserva. Si el commit remoto cambia, solo se descargan blobs nuevos o
modificados. Una alteración local no reconocida detiene el proceso; para
repararla de forma explícita:

```powershell
python scripts\ingest_sackmann.py --force
```

Para descargar y auditar únicamente los archivos, sin construir el DataFrame
completo en memoria:

```powershell
python scripts\ingest_sackmann.py --download-only
```

`--force` no redescarga archivos correctos. Cada ejecución resuelve un único
commit, utiliza URLs fijadas a ese SHA, escribe primero archivos `.part`,
verifica tamaño y SHA de blob Git, y solo entonces reemplaza el destino. El
manifiesto activo registra commit, rutas, tamaños y hashes; cada revisión queda
además archivada en `data/raw/sackmann_manifests/<commit>.json`.

Si una ruta previamente gestionada desaparece del inventario remoto, la
actualización se detiene sin borrar nada. Esto evita que una eliminación o
reestructuración aguas arriba pase inadvertida.

### Uso programático

```python
from src.data_loaders import load_matches, load_players

all_matches = load_matches()
mens_matches = load_matches("M")
womens_players = load_players("F")
```

`load_matches()` conserva las 49 columnas reales del CSV y añade:

- `gender`: `M` o `F`;
- `tour_level`: copia textual exacta de `tourney_level`;
- `source_family`: familia de procedencia;
- `source_file`: CSV anual de procedencia;
- `source_anomaly`: marca una excepción fuente aprobada.

No se colapsan códigos históricos ni categorías ITF numéricas. Las fechas
`tourney_date` se convierten estrictamente a `datetime64[ns]`. El detalle
completo está en [`docs/data_dictionary.md`](docs/data_dictionary.md).

La única anomalía numérica de los 1.749.872 partidos es
`draw_size="exho"` en `Rondebosch Exho`. Solo esa celda se carga como nula y
queda marcada; el CSV crudo no se modifica.

## Actualización de fuentes

La actualización coordinada preparada para el pipeline diario se ejecuta desde
`TENNIS/` con:

```powershell
python scripts\update_sources.py
```

Puede omitirse una fuente para diagnóstico:

```powershell
python scripts\update_sources.py --skip-tennis-abstract
```

Este script no instala tareas del sistema ni procesos en segundo plano. El
lanzador diario de fase 8 tampoco lo ejecuta implícitamente.

### Match Charting Project

Los CSV públicos de
[`JeffSackmann/tennis_MatchChartingProject`](https://github.com/JeffSackmann/tennis_MatchChartingProject)
se descargan a un directorio independiente, fijados a un commit y con
manifiestos activo/versionados. La ejecución auditada incorporó 42 archivos del
commit `2c59eef194967e688b69e73df344184a06322cd8`.

Esta fuente no sustituye al histórico Sackmann. Es una muestra voluntaria y
selectiva, no aporta IDs Sackmann y `Player 1` identifica a quien sirvió
primero, no al ganador. No se concatena con `load_matches()`; sus estadísticas
solo podrán enlazarse mediante un mapping auditable específico. La fase 5
resuelve los slugs de Tennis Explorer, no concatena Match Charting Project.

Los dos CSV de metadatos contienen algunas filas parciales y ocho IDs
duplicados en total. No se han corregido ni fusionado; el detalle exacto está en
[`docs/source_freshness.md`](docs/source_freshness.md).

### Tennis Abstract Elo

Desde el 05/09/2026 el transporte Elo también es Scrapling, a través del
cliente común. La autorización expresa del operador permite además un
[conector y auditor del histórico](docs/http_acquisition.md)
en `scripts/audit_tennis_abstract.py` (véase la sección de auditoría del
documento HTTP). El parser se limita a datos literales de rutas inspeccionadas;
no evalúa como código los históricos descargados.
Desde el 06/09/2026, `scripts/update_tennis_abstract.py` adquiere también
históricos/estadísticas de jugadores **automáticamente** desde `run_tennis.ps1`
y, por tanto, desde el `start.ps1` raíz. La tarea diaria existente que invoca
`run_tennis.ps1 -UpdateOnly` incorpora esta adquisición, sin reentrenar/publicar.
Desde el 09/09, primero se actualiza TennisRatio y después se adquieren **solo
los participantes de la cartelera de hoy UTC** (o de `-Date`). El inventario de
enlaces exactos de los informes Elo ATP/WTA sirve para resolver los perfiles,
no para descargarlos todos. Se deduplican los participantes y se exige nombre
completo normalizado + género únicos; nombres abreviados, ambiguos o fuera del
inventario se informan, sin crear slugs ni mappings canónicos de modelo.
Cada ficha válida se guarda una vez por día UTC. Si falta cartelera no se
sustituye por un barrido completo. Se conserva todo el histórico ya adquirido.

La [migración es por fases](docs/tennis_abstract_daily.md): Sackmann se conserva
como archivo de entrenamiento. Los nuevos datos se guardan en
`data/processed/tennis_abstract.sqlite3`, **separados del modelo** hasta validar
fechas exactas, IDs y cobertura. No se ha sustituido todavía el overlay de
TennisRatio/Explorer ni se ha promovido un modelo. Se respetan los 429 con
reintentos acotados. Conforme a la autorización actual de `AGENTS.md`, un
challenge permite una recuperación con **Chrome visible mediante Scrapling**,
con perfil persistente dedicado y límite total de 90 s. Si el bloqueo persiste,
se conserva el progreso y se avisa; nunca se ignora un `Retry-After` del servidor.
Las filas cortas se guardan en cuarentena, sin completar campos, y se continúa
con las válidas. Un esquema no reconocido sigue deteniendo la adquisición.

Desde el 08/09, un timeout **de navegación de una ficha**, sin evidencia WAF ni
plazo del servidor, aplaza solo ese jugador durante 30 minutos y continúa con
los demás. Tres timeouts consecutivos detienen el lote 15 minutos para no
insistir ante una posible caída. Una descarga correcta reinicia esa racha. Los
plazos se guardan en SQLite entre ejecuciones; un timeout no borra el último
snapshot válido. Timeouts del solver/arranque y fallos sin clasificar conservan
la parada prudente, igual que los bloqueos WAF y robots. `--status` muestra las
fichas aplazadas en `deferred_player_timeouts`.

La recuperación es automática en el cliente de fichas. Para solicitar un
reintento explícito de un fallo **local** desde la raíz:

```powershell
.\TENNIS\.venv\Scripts\python.exe .\TENNIS\scripts\update_tennis_abstract.py --browser-recovery
```

El barrido de todo el inventario requiere ahora `--full-inventory` explícito.
`--date YYYY-MM-DD` selecciona otra fecha visible de la cartelera, por ejemplo
mañana para preparar capturas previas al partido. No cambia la fecha real de
captura: con la regla `< D`, descargar hoy no habilita esa captura para predecir
hoy. La integración de las matrices TA al modelo continúa pendiente.

Para consultar el progreso guardado, incluso mientras descarga, sin red ni
escrituras:

```powershell
.\TENNIS\.venv\Scripts\python.exe .\TENNIS\scripts\update_tennis_abstract.py --status
```

`stored_inventory_players` cuenta los jugadores del inventario ya adquiridos;
`never_acquired`, los que todavía no se han capturado; `pending_today` incluye
también fichas antiguas pendientes de refresco **de todo el catálogo**, no solo
de la selección diaria. `last_agenda_selection` muestra fecha, participantes
seleccionados y nombres sin resolver. `pause` muestra el motivo y plazo de una
pausa. El comando no afirma si hay un proceso activo: lee progreso
confirmado en SQLite, no sustituye la supervisión del proceso.

No necesita Selenium ni nuevas dependencias Python: utiliza
`scrapling[fetchers]==0.4.12` y Chrome instalado. El perfil del navegador queda
en `data/raw/_browser_profiles/tennis_abstract/`, fuera de Git. Este cambio no
convierte las estadísticas adquiridas en entradas del modelo todavía.

El adaptador específico **de Elo** sigue limitado a los informes públicos
[ATP](https://www.tennisabstract.com/reports/atp_elo_ratings.html) y
[WTA](https://www.tennisabstract.com/reports/wta_elo_ratings.html). El módulo:

- deja la frecuencia de ejecución en manos del operador y no impone un cupo
  diario artificial;
- realiza como máximo un GET secuencial por género en cada invocación;
- usa `ETag` y `Last-Modified` en peticiones condicionales;
- no sigue redirecciones ni reintenta fallos;
- se detiene ante el primer error;
- guarda HTML, metadata, cabeceras, SHA-256 y `retrieved_at_utc`;
- valida estrictamente la tabla antes de activarla;
- nunca consulta `/jsfrags/`, `/jsmatches/` ni `/jsplayers/`.

La ejecución inicial guardó 548 filas ATP y 544 WTA, ambas con fecha Elo
2026-07-27. Las respuestas condicionales evitan volver a descargar contenido
sin cambios. Un error HTTP/red/esquema abre un cortacircuitos persistente de 24
horas, pero una ejecución correcta no bloquea la siguiente.

Uso programático de un snapshot ya validado:

```python
from src.tennis_abstract_elo import load_latest_elo

mens_external_elo = load_latest_elo("M")
womens_external_elo = load_latest_elo("F")
```

Estos ratings son un benchmark externo. No reemplazan el Elo propio de la fase
3 ni se aplican retroactivamente a fechas anteriores a su adquisición. La
política completa está en
[`docs/source_freshness.md`](docs/source_freshness.md).

## Sistema Elo propio

La fase 3 implementa un Elo causal en `src/elo/` con responsabilidades
separadas:

- `parameters.py` valida y versiona la configuración matemática;
- `events.py` convierte las filas canónicas en eventos con procedencia y hash;
- `engine.py` congela cada bloque efectivo, permite previsualizar sin mutación,
  calcula deltas y mantiene universos independientes `M` y `F`;
- `store.py` persiste ejecuciones y estados históricos en SQLite;
- `service.py` expone las consultas temporales `get_elo()` y `get_elos()`;
- `build.py` verifica el manifiesto y los blobs fuente y coordina una
  construcción reproducible.

Cada jugador mantiene un Elo general y uno puro para `Hard`, `Clay`, `Grass` y
`Carpet`. Los valores por defecto configurables son rating inicial `1500`,
escala logística `400`, `K(n) = 250 / (n + 5) ** 0.4` y peso de superficie
`0.5`. El rating efectivo en una superficie combina Elo general y Elo puro con
ese peso; sin superficie, coincide con el general. La fórmula, los criterios de
elegibilidad y el bloqueo por fecha se especifican en
[`docs/elo.md`](docs/elo.md).

La versión activa `multisource-general-elo-v6` calcula la identidad de una
copia Sackmann exacta sobre las 49 cadenas crudas del CSV antes de tipar fechas
o identificadores.
Así, dos filas que solo se vuelven iguales después de normalizarlas no se
colapsan silenciosamente. También conserva resultados oficiales `RET`, `DEF`,
`ABD` y `ABN`, excluye solo partidos no iniciados (`W/O`, `Walkover`, `BYE`) y
no usa la cuarentena DOB como selector histórico.

Como `tourney_date` es normalmente el inicio aproximado del torneo, la época
Sackmann de Elo v6 aplica el contrato versionado
`sackmann-tourney-start-embargo-v1`:

```text
result_available_date = tourney_date + 21 días
resultado utilizable en D ⇔ result_available_date < D
```

La igualdad se excluye; un resultado fuente `T` puede afectar por primera vez
a `T+22`. La DOB del maestro actual solo genera una lista diagnóstica para
bloquear o degradar identidades incompatibles en inferencia actual. No elimina
filas de Elo, features, backtest ni reentreno.

### Construcción y artefacto

Desde `TENNIS/`, con el histórico Sackmann y su manifiesto ya ingeridos:

```powershell
python scripts\build_elo.py
```

El comando usa por defecto `config/elo_handoff.json`: verifica el commit base
congelado, incorpora TennisRatio post-corte y usa Tennis Explorer solo donde
Ratio no cubra la misma identidad deportiva. `--as-of-date YYYY-MM-DD` fija el
corte causal reproducible y `--disable-operational-overlay` queda reservado a
diagnóstico. El procedimiento de re-baseline manual está en
[`docs/elo.md`](docs/elo.md#re-baseline-manual).

Los resultados operativos con superficie literal actualizan también el Elo de
superficie. Si falta, solo se propaga evidencia capturada de la misma edición
exacta. El catálogo persistente conserva procedencia y entra en el fingerprint;
no deduce por ciudad, historial, `Indoors` ni bloques agregados.

Por defecto construye ambos géneros. `--gender M` y `--gender F` limitan el
universo, mientras que `--force` reconstruye aunque el fingerprint activo
coincida. También existen `--manifest`, `--raw-dir`, `--database` y
`--chunksize` para tests o diagnósticos. Una reconstrucción parcial no puede
reemplazar una base existente; en ese caso se debe usar `--gender all` o una
ruta `--database` independiente.

El artefacto generado es `data/processed/elo/elo.sqlite3`, ignorado por Git. La
base conserva runs y parámetros, checkpoints por fecha, el historial ancho de
ratings y un puntero activo por género. Solo un run completo puede activarse;
un fallo durante la construcción no sustituye al run anterior. Se usa un único
escritor y journal `DELETE`, sin WAL, para evitar ficheros laterales en el
directorio sincronizado. El esquema completo está en
[`docs/data_dictionary.md`](docs/data_dictionary.md).

### Consulta temporal

```python
from datetime import date

from src.elo import EloQuery, get_elo, get_elos

single = get_elo(
    gender="M",
    player_id=104925,
    surface="Clay",
    as_of_date=date(2024, 6, 1),
)

batch = get_elos(
    queries=(
        EloQuery("M", 104925, "Clay", date(2024, 6, 1)),
        EloQuery("F", 201594, None, date(2024, 6, 1)),
    )
)
```

Ambas APIs consultan exclusivamente el último estado efectivo que cumpla
`state_date < as_of_date`; el estado de la propia fecha `D` nunca es visible en
una consulta para `D`. Ese `state_date` es la disponibilidad embargada, mientras
que la fecha Sackmann original se conserva como procedencia. `as_of_date` debe
ser un `datetime.date` estricto.
`surface` admite `Hard`, `Clay`, `Grass`, `Carpet` o `None` sin distinguir
capitalización. Si no hay historia anterior, se devuelve el rating inicial del
run con `is_cold_start=True`. `get_elos()` conserva el orden de entrada y
resuelve cada consulta contra el run completo solicitado o el activo de su
género.

## Cartelera diaria de Tennis Explorer

La fase 4 obtiene en una sola página diaria los singles masculinos ATP,
Challenger e ITF y los singles femeninos WTA e ITF. Excluye dobles y UTR, y
conserva para cada jugador tanto el texto visible como el `href` y el `slug`
estable de su ficha. Las cuotas ausentes permanecen nulas. La misma página
contiene un catálogo semanal con superficie explícita: se une por
`tournament_href` sin emitir otra petición. Una superficie ausente o una
cabecera Futures agregada permanece nula; no se deduce por calendario.

El responsable del proyecto ha comunicado autorización de Tennis Explorer para
este uso académico, no comercial y de baja frecuencia. Como excepción
estrecha basada en esa autorización reportada, el cliente no solicita
`robots.txt` para la única URL diaria canónica
`/matches/?day=DD&month=MM&type=all&year=YYYY`.
Así, una fecha nueva genera un solo GET y una fecha cacheada genera cero. La
excepción no se presenta como una verificación independiente ni se extiende a
perfiles, detalles, endpoints internos u otras rutas.

El transporte HTTP estático usa `scrapling[fetchers]==0.4.12` con una identidad
y cabeceras fijas. No utiliza `StealthyFetcher`, `DynamicFetcher`, adaptación
de selectores, impersonación, proxies, rotación, navegadores ni resolución de
desafíos. El cliente HTTP común serializa por host y garantiza al menos `1,0`
segundo entre GET; no hay reintentos ni redirecciones automáticas.
La captura válida queda cacheada con URL, instante UTC, tamaño y SHA-256. El
pin de Scrapling solo se actualiza tras revisar una release y ejecutar los
tests; nunca sigue automáticamente la rama `main`.

El comprobador de mantenimiento realiza un único GET a la release estable más
reciente de GitHub y no modifica el entorno ni los archivos:

```powershell
python scripts\check_scrapling_release.py
```

Devuelve `0` si el pin coincide, `1` si existe una release posterior y `2` si
la comprobación no es concluyente. Puede invocarse como comprobación de
mantenimiento antes de promover una actualización validada; nunca modifica
automáticamente el entorno.

Un `403`, `429` o una firma inequívoca de desafío/WAF detiene la ejecución y
abre `data/raw/tennis_explorer/policy/circuit_breaker.json`. Mientras exista,
las fechas ya cacheadas siguen disponibles, pero cualquier fecha no cacheada
se detiene antes de acceder a la red. El circuito no caduca ni se cierra
automáticamente: solo se elimina manualmente después de coordinar y revisar el
bloqueo.

Desde `TENNIS/`, la fecha es opcional:

```powershell
python scripts\scrape_tennis_explorer.py
python scripts\scrape_tennis_explorer.py --date 2026-07-30
```

El estado se interpreta conservadoramente. `scheduled`, `finished`,
`in_progress`, `walkover` y `cancelled` solo se asignan cuando existe evidencia
directa suficiente; los casos que la tabla diaria no permite distinguir usan
`unknown` y explican el motivo en `status_evidence`. El widget externo de
EnetScores no se consulta.

Tennis Explorer agrupa los Futures masculinos como `Futures 2026`, sin sede ni
enlace de torneo. El scraper conserva ese texto, asigna `tour_level="ITF"` y
deja `tournament_href` nulo. No se inventa una sede ni se hacen peticiones por
partido. El contrato completo, los selectores inspeccionados y la decisión de
limitar Scrapling a transporte HTTP estático están documentados en
[`docs/tennis_explorer.md`](docs/tennis_explorer.md).

## Mapeo de jugadores

La fase 5 enlaza los slugs de la cartelera con jugadores Sackmann usando
`(gender, slug)` como identidad persistente. Cada universo de género se
resuelve por separado y una clave ya guardada no se vuelve a calcular
automáticamente.

Para un slug nuevo en un partido de fecha `D`, el matcher:

1. normaliza con `Unidecode` el apellido completo y la inicial en ambas
   fuentes;
2. restringe los candidatos al género correcto y a quienes jugaron en
   cualquier nivel dentro de `[D - 3 años naturales, D)`;
3. acepta solo una coincidencia exacta y única.

No se aplica fuzzy matching ni se adivina a partir del ranking, nivel o slug.
Las colisiones y ausencias se conservan como no resueltas sin detener el resto
del lote. La página diaria actual no publica el país de cada participante y
las banderas del encabezado pertenecen al torneo; por ello no se usan como
desempate ni se solicitan perfiles individuales.

Las asociaciones vigentes y su auditoría append-only viven en
`data/processed/player_mapping.sqlite3`. Una resolución automática es
inmutable; solo una decisión explícita en `data/overrides.csv`, con el esquema
exacto `gender,slug,player_id,reason`, puede insertarla o corregirla
manualmente. Los pendientes actuales se agregan de forma idempotente en
`data/processed/unresolved_players.csv` y desaparecen de la cola cuando se
resuelven.

Desde `TENNIS/`, se mapea una fecha con `--date`; al omitirlo se usa hoy:

```powershell
python scripts\map_tennis_explorer_players.py
python scripts\map_tennis_explorer_players.py --date 2026-07-30
```

El DataFrame resultante conserva todas las filas y añade IDs y métodos de
resolución por participante, además de `mapping_status`, que exige ambos IDs
para marcar un partido como mapeado. El contrato, los límites y el flujo de
revisión manual se detallan en
[`docs/player_mapping.md`](docs/player_mapping.md), y los esquemas persistidos
en [`docs/data_dictionary.md`](docs/data_dictionary.md).

## Features causales y datasets de entrenamiento

La fase 6 recorre los registros fuente por `tourney_date`, que Sackmann usa
habitualmente como inicio aproximado del torneo. Cada fila conserva
`result_available_date=tourney_date+21 días`. Para construir un partido de
fecha `D`, previsualiza el vector sin mutar el estado y solo incorpora resultados
cuya `result_available_date < D`; la igualdad sigue excluida. Así no inventa un
orden entre rondas ni presenta como pasado una ronda que quizá aún no se había
disputado. El esquema activo es `tennis-features-v2`.

El vector incluye, con orientación A menos B:

- Elo general, Elo puro de superficie y Elo efectivo combinado;
- forma en los últimos 10 partidos y 3 meses naturales, con contadores;
- H2H global y por superficie, descanso, ranking y puntos;
- edad decimal y `abs(edad - 30)`;
- nivel raw y canónico, superficie, `best_of` y ronda;
- cuotas, probabilidades implícitas y mercado de-vigado cuando existe una
  observación temporalmente válida.

La orientación histórica se decide independientemente para cada fila mediante
SHA-256 con semilla `42`; por ello, reordenar el histórico o añadir partidos
futuros no modifica A/B ni `y` de partidos anteriores. Las columnas de
procedencia, IDs y objetivo se conservan para auditoría, pero
`MODEL_FEATURE_COLUMNS` las excluye del futuro entrenamiento.

Los rankings aplican siempre `ranking_date < D`. Los 599 conflictos ATP y 226
WTA encontrados para una misma clave jugador-fecha se omiten completos; la
consulta retrocede al snapshot limpio anterior y expone cuántas fechas
conflictivas saltó. El inventario íntegro queda en
`data/processed/features_active/runs/<fingerprint>/ranking_conflicts.csv`.

La cuarentena DOB se publica como diagnóstico operativo actual, pero no
selecciona ejemplos históricos. Sus fechas proceden del maestro presente y no
son evidencia causal disponible en el pasado. Una clave incompatible se marca
como no disponible o baja confianza en la predicción actual; sus filas no se
eliminan retrospectivamente de Elo, features, validación o reentreno.

Sackmann no contiene cuotas históricas. Por tanto, `odds_*`,
`market_probability_*`, `model_probability_a` y `edge` permanecen nulas en los
Parquet históricos. El constructor de un partido diario sí admite dos cuotas
decimales, pero exige que su timestamp de adquisición sea estrictamente
anterior al instante de predicción. `edge` se poblará cuando la fase 7 aporte
`model_probability_a`.

Desde `TENNIS/`:

```powershell
python scripts\build_features.py
python scripts\build_features.py --gender M `
  --output-dir data\processed\features_active\diagnostic_M --force
```

La ejecución es idempotente mediante un fingerprint de fuentes, parámetros,
esquema y fórmulas. Publica:

- `data/processed/features_active/manifest.json`: puntero activo atómico;
- `data/processed/features_active/runs/<fingerprint>/training_M.parquet`;
- `data/processed/features_active/runs/<fingerprint>/training_F.parquet`;
- `data/processed/features_active/runs/<fingerprint>/manifest.json`: hashes,
  parámetros y auditorías inmutables;
- `data/processed/features_active/runs/<fingerprint>/ranking_conflicts.csv`:
  inventario de observaciones conflictivas en cuarentena.

El cálculo usa un workspace separado para SQLite y CSV intermedios. Solo el
subdirectorio publicable se mueve a `runs/<fingerprint>` en una operación; el
puntero cambia después de verificar hashes y tamaños. Si esa activación falla,
el run anterior sigue íntegro y el nuevo puede reactivarse sin recalcular.
Tras una activación verificada, la retención automática conserva exactamente
el run activo y la generación anterior más reciente; nunca quedan más de dos
directorios bajo `features_active/runs/`. La keep-list se recalcula antes de
cada borrado y se contrasta con un sello SHA-256 de todos los manifiestos. La
poda rechaza enlaces o junctions en el almacén, el puntero y los árboles de run,
así como cualquier preview cuyo puntero, keep-list o inventario haya quedado
obsoleto.

El contrato matemático y temporal completo está en
[`docs/features.md`](docs/features.md); el resultado reproducible de la
ejecución real está en [`docs/phase6_report.md`](docs/phase6_report.md).

## Modelos y calibración temporal

La fase 7 entrena por separado una regresión logística de referencia y un
LightGBM principal para hombres y mujeres. No existe split aleatorio. El
backtest acumulado de 2016–2025 usa, para cada temporada `Y`:

```text
train: temporadas <= Y-2 con result_available_date < inicio de calibración
calibración: Y-1 con result_available_date < inicio del test Y
test: Y
```

La imputación, el escalado, las categorías y el modelo se ajustan solo con
`train`. El calibrador Platt recibe exclusivamente probabilidades generadas
para `Y-1` y queda congelado antes de predecir `Y`. Los modelos finales usan
solo filas con `result_available_date < training_as_of_date`; sus calibradores
se ajustan con predicciones OOF temporales que cumplen el mismo corte, nunca
con probabilidades in-sample.

Las métricas anteriores a Elo v5/features v2 quedaron superseded y no deben
citarse como rendimiento actual. El run activo posterior a la reconstrucción
publica las cifras definitivas globales y por segmento en
[`docs/model_report.md`](docs/model_report.md), y el cierre independiente las
contrasta en [`docs/audit.md`](docs/audit.md).

El mercado no es evaluable todavía en el histórico Sackmann porque no contiene
cuotas con timestamp causal. El perfil `auto` utiliza `sports_only` y
`market_enhanced` falla expresamente hasta que exista histórico causal; no se
imputan cuotas. El evaluador ya está preparado para comparar modelo y mercado
sobre soporte común cuando se acumulen observaciones.

Desde `TENNIS/`:

```powershell
.venv\Scripts\python.exe scripts\retrain_models.py
```

Los modelos, calibradores, predicciones OOF, curvas y métricas se versionan en
`models/phase7/runs/<fingerprint>/`, con tamaño y SHA-256 en el manifiesto. Una
segunda ejecución idéntica verifica y reutiliza el run sin saltarse la puerta.
Log-loss decide y Brier desempata; accuracy se informa pero no promociona.
El contrato completo
está en [`docs/modeling.md`](docs/modeling.md) y las métricas reales en
[`docs/model_report.md`](docs/model_report.md).

## Pipeline diario e interpretación del output

El lanzador `run_tennis.ps1` funciona desde cualquier directorio, activa el
entorno virtual del proyecto y ejecuta `scripts/daily_predictions.py`. Solo
llama al modelo para partidos `scheduled` con dos IDs resueltos y un inicio UTC
publicado estrictamente posterior al instante de inferencia. Como snapshot
histórico de fase 8, la ejecución del 30 de julio de 2026 conservó 313 filas,
produjo 204 predicciones calibradas de 228 partidos programados y dejó 109
filas sin probabilidad por estado terminal, estado desconocido o mapping
incompleto; esas cifras no describen el cierre actual.

La ejecución final de fase 9 publicó 429 filas: 130 partidos programados, 83
predicciones calibradas y 346 filas degradadas sin inventar probabilidades. Las
83 predicciones oficiales se registraron en la BBDD y se publicaron en la web.

Jugador A es siempre el primer jugador visible y B el segundo. El CSV incluye
`model_probability_a/b`, `market_probability_a/b` de-vigadas,
`market_comparison_status` y `edge_a/b`. Bajo la regla estricta del proyecto,
el edge solo se publica si la captura de mercado está acreditada en una fecha
anterior al partido. Las cuotas obtenidas la misma mañana pueden mostrarse como
contexto, pero quedan `prestart_unverified` y con edge nulo: no se finge que
existían antes de `D`. Cuando sea evaluable, un edge positivo significa
únicamente que el modelo asigna más probabilidad que la casa; no garantiza
rentabilidad ni constituye una recomendación.

El modelo activo es `sports_only`: las cuotas no fueron feature durante el
entrenamiento porque el histórico tiene cobertura de mercado 0 %. Las cuotas
del día sirven para comparar, no para afirmar que el modelo ya ha demostrado
ventaja sobre el mercado. Esa conclusión solo será posible después de acumular
predicciones y cuotas realmente prospectivas.

`confidence` y `confidence_flags` deben leerse antes que el edge:

- `UNAVAILABLE`: no hay probabilidad; nunca se rellena por aproximación.
- `LOW`: hay probabilidad, pero falla algún criterio fuerte de cobertura o
  frescura.
- `MEDIUM`: inputs utilizables con advertencias de contexto o muestra.
- `HIGH`: todos los umbrales documentados están cubiertos.

Los niveles ITF y Challenger merecen especial cautela: jugadores nuevos,
mappings pendientes, rankings ausentes y poca muestra por superficie producen
probabilidades más frágiles. Una cifra extrema no eleva la confianza. En el
run auditado del 30 de julio todas las predicciones quedaron `LOW`, porque el
histórico/modelo terminaba 58–59 días antes y los rankings 52 días antes. Los
Futures masculinos además carecen de superficie en la fuente. Es una
advertencia real, no un error que deba ocultarse.

La salida completa se publica atómicamente en:

```text
data/processed/predictions/predictions_YYYY-MM-DD_<timestamp UTC>.csv
```

La misma ejecución guarda evidencia append-only en `BBDD/tennis.sqlite3`.
Para cada partido se conserva la primera predicción válida como oficial; una
repetición posterior no puede reemplazarla. Cada mañana se elige una sola fecha
anterior aún pendiente, se guarda el snapshot final y se liquida únicamente si
`winner_slug` coincide con uno de los dos slugs oficiales y la observación es
estrictamente posterior a la predicción. El corte del modelo es obligatorio y
anterior a la fecha del partido; capturas posteriores a la jornada o con una
observación ya conocida nunca son oficiales. Tampoco lo son una inferencia a
la hora de inicio o posterior ni un horario TBD sin zona. Walkovers, estados
parciales, HTML ambiguo o slugs contradictorios se registran o envían a revisión
sin inventar ganador. `WEB/build_web.py` abre esta base en modo solo lectura y
la pestaña Tenis muestra ganador previsto, probabilidad, **confianza del
input** como nivel cualitativo y, cuando existe, ganador real como dato
separado. El rendimiento observado se publica aparte como **acierto real
(partidos liquidados)**, agregado sobre todo el historico oficial liquidado,
con `n`, accuracy, Brier, log-loss y curva de calibracion. Una muestra vacia se
muestra como no calculada, nunca como `0.0 %`; menos de 100 partidos se marca
como provisional. El panel expone además el instante de la última liquidación
incorporada, y la cabecera `Probabilidad` ordena la cartelera en ambos sentidos
manteniendo siempre al final las filas que no tienen predicción. El bloque de
rendimiento permite combinar género (`Todos`, `M`, `F`) con confianza del
ganador previsto (`50-60`, `60-70`, `70-80`, `80-90`, `90-100 %`); cada cambio
recalcula conjuntamente N, aciertos, accuracy, Brier, log-loss, estado y curva
de calibración. Los límites son semiabiertos para no duplicar partidos, salvo
el último tramo, que incluye el 100 %.

La cartelera web corresponde siempre al ultimo run diario completo que haya
persistido partidos. Si ese run no pudo producir predicciones oficiales por el
corte causal `< D`, la web muestra la jornada actual con cada partido marcado
como `Sin prediccion`; nunca conserva silenciosamente una jornada oficial mas
antigua como si fuera la ejecucion vigente. Las metricas historicas siguen
calculandose exclusivamente con predicciones oficiales liquidadas.

La consola muestra como máximo 50 filas ordenadas por edge absoluto; el CSV
siempre conserva la cartelera completa, los timestamps, fingerprints, cortes
de datos y motivos de confianza. El contrato detallado está en
[`docs/daily_pipeline.md`](docs/daily_pipeline.md). El orden del lanzador raíz,
la propagación de `-Retrain` y los códigos de salida están documentados en
[`docs/integracion.md`](docs/integracion.md).

## Tests

Desde `TENNIS/`:

```powershell
.\.venv\Scripts\python.exe scripts\run_quality_gates.py
```

La puerta conjunta ejecuta Ruff (lint y ratchet de formato), Mypy gradual,
la suite completa con pytest, cobertura de imports directos y el smoke
determinista de los entrypoints reales. El contrato, los comandos aislados y
la deuda inicial pendiente están en
[`docs/quality_gates.md`](docs/quality_gates.md).

Los tests usan fixtures locales y transportes HTTP simulados. Además de las
pruebas de ingesta de la fase 2, `test_elo_engine.py`,
`test_elo_pipeline.py` y `test_elo_store.py` comprueban la fórmula configurable,
aislamiento por género y superficie, bloqueo por fecha, exclusiones y
deduplicación raw exacta, determinismo, validación de manifiesto y blobs, estabilidad
histórica ante datos futuros, transacciones y activación SQLite, cold start y
las consultas estrictas individual y vectorizada. Las pruebas de Tennis
Explorer parsean la captura HTML real sin red y cubren alcance, niveles,
slugs, cuotas, estados conservadores, cambios de esquema, caché, transporte
Scrapling estático, cabeceras fijas, ausencia de `robots.txt`, integridad,
exclusión mutua de procesos, pausa previa, redirecciones, parada ante
403/429/WAF y persistencia del cortacircuitos.
`test_player_mapping_candidates.py`, `test_player_mapping_pipeline.py`,
`test_player_mapping_store.py` y `test_player_mapping_review.py` cubren
apellidos compuestos, acentos, aislamiento por género, ventana activa causal,
colisiones honestas, nombres no parseables, overrides, caché automática
inmutable, auditoría SQLite y cola de no resueltos idempotente.
Los tests `test_feature_*.py` y `test_build_features_script.py` cubren el
freeze diario, estabilidad ante datos futuros, ventanas naturales, H2H,
descanso, rankings estrictos con cuarentena, edad, niveles auditados, de-vig,
timestamps, orientación estable y balanceada, esquema/allowlist, spool
cronológico, Parquet, publicación idempotente, embargo estricto de labels y
ausencia de roles ganador/perdedor entre las columnas persistidas.
Los tests `test_model_*.py` y `test_modeling_*.py` cubren contrato de
allowlist, perfiles con/sin mercado, preprocesamiento ajustado solo con train,
aislamiento de género, regresión logística, LightGBM, baselines, folds
expansivos, Platt, métricas segmentadas, soporte común, estabilidad al añadir
futuro, publicación inmutable, hashes, carga segura e idempotencia. La suite
`test_daily_pipeline.py` añade reconstrucción dirigida, no-fuga al añadir
futuro, orientación A/B, bloqueo del modelo no causal, degradación por estado
o mapping, confianza y publicación CSV. `test_daily_documentation.py` fija el
lanzador y la línea no invasiva de integración. La suite final, ejecutada tras
reconstruir Elo v5, features v2 y modelos v3, obtuvo 329/329 tests correctos; la
evidencia consolidada está en `docs/audit.md`.

Los recuentos de la ejecución auditada de la fase 2 están en
[`docs/phase2_report.md`](docs/phase2_report.md).
Los runs, exclusiones, top 10 y consultas de la ejecución Elo completa están en
[`docs/phase3_report.md`](docs/phase3_report.md).
La cobertura, los estados, las cuotas y la validación de la captura diaria real
están en [`docs/phase4_report.md`](docs/phase4_report.md).
La cobertura del mapping real, la lista íntegra de pendientes y la advertencia
de frescura están en [`docs/phase5_report.md`](docs/phase5_report.md).
Los tamaños, balance, conflictos, vector de ejemplo y auditoría de la fase 6
están en [`docs/phase6_report.md`](docs/phase6_report.md).
Las métricas temporales, curvas de fiabilidad, segmentos y limitaciones de la
fase 7 están en [`docs/model_report.md`](docs/model_report.md).
El contrato, confianza, publicación y limitaciones diarias están en
[`docs/daily_pipeline.md`](docs/daily_pipeline.md).
La ejecución real, cobertura y flags de fase 8 están en
[`docs/phase8_report.md`](docs/phase8_report.md).

## Principios permanentes de diseño

- El alcance máximo incluye hombres (`ATP`, `Challenger` e `ITF`) y mujeres (`WTA` e `ITF`).
- Existen dos universos de rating Elo separados por género; hombres y mujeres nunca se mezclan. Dentro de cada género, todos los niveles comparten el mismo pool.
- Existe un modelo por género, dos en total. El nivel del torneo, la superficie, el formato `best-of` y la ronda son features, no modelos separados.
- La evaluación y la calibración se realizan también por segmento de nivel de torneo y superficie, no únicamente de forma global.
- El contrato reserva las cuotas como feature para un futuro perfil con cobertura histórica. El perfil activo `sports_only` no las usa; una comparación solo se publica cuando la captura de mercado cumple el corte causal documentado.
- Toda generación de ratings, features, particiones, validaciones y predicciones cumple la regla anti-fugas: para una fecha `D`, solo consulta información estrictamente anterior a `D`.

## Roadmap

1. **Andamiaje — completada:** estructura, reglas permanentes, documentación, dependencias, entorno virtual y configuración de rutas.
2. **Ingesta y vigencia de fuentes — completada:** histórico Sackmann reproducible, actualizaciones incrementales, Match Charting separado y snapshots Elo externos con límite temporal.
3. **Sistema Elo por superficie y género — completada:** motor Elo v5, ratings generales y por superficie, embargo causal, persistencia SQLite y API histórica estricta.
4. **Scraper Tennis Explorer — completada:** cartelera diaria de singles, slugs estables, cuotas, estados conservadores, caché íntegra y acceso de baja frecuencia.
5. **Mapeo de jugadores — completada:** resolución exacta y causal de slugs, caché SQLite auditable, overrides y cola honesta de no resueltos.
6. **Features sin fugas — completada:** esquema v2, embargo de labels, vector causal común, orientación A/B estable, rankings temporales, Parquet separados por género y publicación generacional.
7. **Modelos, validación temporal y calibración — completada:** LightGBM y baseline logística por género, backtest expansivo, Platt causal, métricas por segmento y artefactos versionados.
8. **Pipeline diario e integración — completada:** scraper y mapping encadenados, contexto causal dirigido, inferencia calibrada, edge, confianza, CSV timestamped, lanzador autocontenido y orquestación conjunta desde `start.ps1` con `-Retrain` para ambos deportes.
9. **Auditoría anti-fugas — completada:** revisión causal integral, corrección de fecha fuente, identidades y retiros, simetría A/B, auditoría de segmentos sospechosos, casos límite, E2E, BBDD prospectiva y verificación reproducible; el cierre está registrado en `docs/audit.md`.

## Formato y ronda en producción

La cartelera de TennisRatio conserva nivel y ronda exactos para rellenar
`best_of`/`round` solo con captura y publicación anteriores a D. Se mantienen
los códigos Sackmann y los flags de ausencia; no se cambia el modelo activo.
El resumen diario muestra cobertura e impacto en confianza. Véase
[contrato, limitaciones y auditoría reproducible](docs/match_format.md).

La descarga con Scrapling distingue el script pasivo JSD de un bloqueo real:
una cartelera HTTP 200 válida ya no se rechaza solo por incluir ese script.
Se mantienen robots, pausas y los cortacircuitos WAF. Detalles y pruebas en
[adquisición HTTP](docs/http_acquisition.md#corrección-del-falso-positivo-jsd-2026-09-05).
