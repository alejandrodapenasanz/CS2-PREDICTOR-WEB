# Predictor profesional de partidos de tenis

## Objetivo

Construir por fases un sistema profesional, modular y testeable para estimar probabilidades de resultados de partidos de tenis. El diseño debe preservar el orden temporal de la información: para predecir un partido con fecha `D`, únicamente se podrá usar información anterior a `D`.

Las fases 1 a 8 están implementadas. El proyecto puede descubrir, actualizar,
validar y cargar el histórico de singles de Jeff Sackmann separado por género.
También versiona Match Charting Project como fuente auxiliar y captura dos
referencias Elo públicas con una cuota estricta. Además, calcula y persiste un
Elo propio general y por superficie con consultas históricas estrictamente
anteriores a cada fecha, y obtiene la cartelera diaria de Tennis Explorer con
identidades web, cuotas, trazabilidad y caché. La fase 5 resuelve esas
identidades web a IDs Sackmann mediante un mapping persistente, auditable y con
degradación honesta ante ambigüedades. La fase 6 construye además un vector
causal común y dos datasets de entrenamiento Parquet, con orientación A/B
estable, rankings en cuarentena cuando la fuente es contradictoria y un
allowlist explícito de features. Entrena y calibra temporalmente un LightGBM
por género y ya dispone de un pipeline diario que conserva todos los partidos,
publica probabilidades calibradas y compara de forma honesta con el mercado.

## Estructura

```text
TENNIS/
├── .venv/                 # Entorno virtual local (ignorado por Git)
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
│   │   ├── features/      # Parquet por género, manifiesto y cuarentenas
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
│   ├── integracion.md     # Línea no invasiva para el start.ps1 raíz
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
│   ├── map_tennis_explorer_players.py # Mapping diario a IDs Sackmann
│   ├── scrape_tennis_explorer.py
│   └── update_sources.py  # Actualizador coordinado de fuentes
├── src/
│   ├── config.py          # Rutas del proyecto centralizadas y portables
│   ├── data_loaders.py    # Loaders tipados de partidos y jugadores
│   ├── elo/               # Motor, persistencia, servicio y orquestación Elo
│   ├── features/          # Estado causal, rankings, vector, spool y dataset
│   ├── daily_pipeline/    # Contexto dirigido, confianza y orquestador
│   ├── modeling/          # Entrenamiento, evaluación y servicio de inferencia
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
├── run_tennis.ps1         # Lanzador autocontenido desde cualquier ruta
└── requirements.txt       # Dependencias de Python
```

Los directorios que aún no contienen código o artefactos conservan un archivo `.gitkeep` para que Git pueda versionar la estructura.

Los datos de `data/raw`, la base Elo de `data/processed/elo`, los Parquet y
manifiestos generados de `data/processed/features`, la caché SQLite de mappings
y la cola de no resueltos se ignoran en Git. Sus `.gitkeep`, la tabla manual
`data/overrides.csv`, el código, los tests y la documentación sí se versionan.

## Entorno virtual

El entorno local se encuentra en `TENNIS/.venv`.

Desde la raíz del repositorio, se activa en PowerShell con:

```powershell
.\TENNIS\.venv\Scripts\Activate.ps1
```

Si la terminal ya está situada dentro de `TENNIS/`:

```powershell
.\.venv\Scripts\Activate.ps1
```

En macOS o Linux, el entorno debe crearse en esa máquina y se activa desde la raíz del repositorio con:

```bash
source TENNIS/.venv/bin/activate
```

Las dependencias declaradas se instalan, con el entorno activado, mediante:

```powershell
python -m pip install -r TENNIS\requirements.txt
```

Para salir del entorno:

```text
deactivate
```

## Uso completo de principio a fin

### Instalación inicial

Desde la raíz del repositorio:

```powershell
.\TENNIS\.venv\Scripts\Activate.ps1
python -m pip install -r .\TENNIS\requirements.txt
Set-Location .\TENNIS
```

Si `.venv` todavía no existe en esa máquina, créalo primero:

```powershell
python -m venv .\TENNIS\.venv
```

### Actualizar y reentrenar cuando cambie el histórico

El orden es obligatorio porque cada artefacto fija el fingerprint del anterior:

```powershell
python scripts\update_sources.py
python scripts\build_elo.py
python scripts\build_features.py
python scripts\retrain_models.py
```

`update_sources.py` consulta los commits remotos y conserva lo ya verificado.
Elo, features y modelos son idempotentes por fingerprint. Si el commit
Sackmann no cambió, no hay que forzar una reconstrucción. Si sí cambió, deben
completarse los cuatro pasos antes de volver a predecir; el pipeline diario
rechaza mezclar raw nuevo con artefactos antiguos.

Match Charting Project y el Elo público de Tennis Abstract se actualizan y
versionan como fuentes auxiliares, pero no se concatenan silenciosamente al
dataset de entrenamiento. El predictor activo sigue usando el histórico
Sackmann auditado.

### Predicción cotidiana

Desde cualquier ruta de PowerShell:

```powershell
& 'C:\ruta\al\repositorio\TENNIS\run_tennis.ps1'
```

Desde la raíz del repositorio basta con:

```powershell
.\TENNIS\run_tennis.ps1
```

Si Windows bloquea scripts por la política local, puede ejecutarse una sola
vez sin cambiar la configuración permanente:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\TENNIS\run_tennis.ps1
```

Sin argumentos usa hoy; para una fecha explícita:

```powershell
.\TENNIS\run_tennis.ps1 --date 2026-07-30
```

Una fecha histórica solo se predice si el modelo fue entrenado con un corte
estrictamente anterior. Si no, las filas se conservan como
`UNAVAILABLE/model_not_causal_for_date`, evitando una evaluación retrospectiva
con fuga.

La ejecución diaria no reentrena: hacerlo cada mañana sería costoso e
innecesario si la fuente no cambió. La rutina recomendada es ejecutar solo
`run_tennis.ps1` cada día y repetir el bloque de actualización/reentreno cuando
`update_sources.py` detecte un commit histórico nuevo.

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

La ingesta web se limita a los informes públicos
[ATP](https://www.tennisabstract.com/reports/atp_elo_ratings.html) y
[WTA](https://www.tennisabstract.com/reports/wta_elo_ratings.html). El módulo:

- permite como máximo dos GET en una ventana deslizante de 24 horas;
- usa `ETag` y `Last-Modified` en peticiones condicionales;
- no sigue redirecciones ni reintenta fallos;
- se detiene ante el primer error;
- guarda HTML, metadata, cabeceras, SHA-256 y `retrieved_at_utc`;
- valida estrictamente la tabla antes de activarla;
- nunca consulta `/jsfrags/`, `/jsmatches/` ni `/jsplayers/`.

La ejecución inicial guardó 548 filas ATP y 544 WTA, ambas con fecha Elo
2026-07-27. Una segunda ejecución inmediata realizó cero GET por el límite de
24 horas.

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
- `engine.py` congela el estado por `tourney_date`, calcula deltas y mantiene
  universos independientes `M` y `F`;
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

La versión activa `sackmann-elo-v2` calcula la identidad de una copia exacta
sobre las 49 cadenas crudas del CSV antes de tipar fechas o identificadores.
Así, dos filas que solo se vuelven iguales después de normalizarlas no se
colapsan silenciosamente.

### Construcción y artefacto

Desde `TENNIS/`, con el histórico Sackmann y su manifiesto ya ingeridos:

```powershell
python scripts\build_elo.py
```

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

Ambas APIs consultan exclusivamente el último estado que cumpla
`state_date < as_of_date`; el estado de la propia fecha `D` nunca es visible en
una consulta para `D`. `as_of_date` debe ser un `datetime.date` estricto.
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
desafíos. Un lock local serializa el acceso, se aplica una pausa de `1,0`
segundo antes del único GET y no hay reintentos ni redirecciones automáticas.
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

La fase 6 recorre los 1.749.872 registros fuente en bloques completos de
`tourney_date`. Para una fecha `D`, primero calcula todas las filas contra el
estado que solo contiene fechas `< D` y después incorpora los resultados de
`D`. Como Sackmann usa habitualmente la fecha de inicio del torneo, no se
inventa un orden entre rondas de la misma fecha: ninguna ronda de ese bloque
alimenta otra. Es una decisión deliberadamente conservadora contra fugas.

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
`data/processed/features/ranking_conflicts.csv`.

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
  --output-dir data\processed\features\diagnostic_M --force
```

La ejecución es idempotente mediante un fingerprint de fuentes, parámetros,
esquema y fórmulas. Publica:

- `data/processed/features/training_M.parquet`: 929.149 filas;
- `data/processed/features/training_F.parquet`: 751.126 filas;
- `data/processed/features/manifest.json`: hashes, parámetros y auditorías;
- `data/processed/features/ranking_conflicts.csv`: 1.650 observaciones
  conflictivas en cuarentena.

El contrato matemático y temporal completo está en
[`docs/features.md`](docs/features.md); el resultado reproducible de la
ejecución real está en [`docs/phase6_report.md`](docs/phase6_report.md).

## Modelos y calibración temporal

La fase 7 entrena por separado una regresión logística de referencia y un
LightGBM principal para hombres y mujeres. No existe split aleatorio. El
backtest acumulado de 2016–2025 usa, para cada temporada `Y`:

```text
train <= Y-2
calibración = Y-1
test = Y
```

La imputación, el escalado, las categorías y el modelo se ajustan solo con
`train`. El calibrador Platt recibe exclusivamente probabilidades generadas
para `Y-1` y queda congelado antes de predecir `Y`. Los modelos finales usan
todo el histórico causal disponible; sus calibradores se ajustan con
predicciones OOF temporales, nunca con probabilidades in-sample.

El run activo produjo:

| Género | Predicciones OOF | Accuracy LightGBM calibrado | Log-loss | Brier | AUC |
|---|---:|---:|---:|---:|---:|
| M | 266.980 | 0,7025 | 0,563036 | 0,191690 | 0,7780 |
| F | 262.315 | 0,7176 | 0,543717 | 0,183847 | 0,7966 |

En soporte idéntico de ranking, LightGBM alcanza 0,6904 M y 0,6946 F frente a
0,6544 y 0,6484 del favorito por ranking. No apareció ningún segmento con
accuracy superior al 85 %.

El mercado no es evaluable todavía: las 1.680.275 filas históricas tienen
cobertura de cuotas 0 %. El perfil `auto` utiliza `sports_only` y
`market_enhanced` falla expresamente hasta que exista histórico causal; no se
imputan cuotas. El evaluador ya está preparado para comparar modelo y mercado
sobre soporte común cuando se acumulen observaciones.

Desde `TENNIS/`:

```powershell
.venv\Scripts\python.exe scripts\retrain_models.py
```

Los modelos, calibradores, predicciones OOF, curvas y métricas se versionan en
`models/phase7/runs/<fingerprint>/`, con tamaño y SHA-256 en el manifiesto. Una
segunda ejecución idéntica verifica y reutiliza el run. El contrato completo
está en [`docs/modeling.md`](docs/modeling.md) y las métricas reales en
[`docs/model_report.md`](docs/model_report.md).

## Pipeline diario e interpretación del output

El lanzador `run_tennis.ps1` funciona desde cualquier directorio, activa el
entorno virtual del proyecto y ejecuta `scripts/daily_predictions.py`. El
pipeline conserva las 313 filas del snapshot real de referencia; solo llama al
modelo para partidos `scheduled` con dos IDs resueltos. En la ejecución del 30
de julio de 2026 produjo 204 predicciones calibradas de 228 partidos
programados y dejó 109 filas totales sin probabilidad por estado terminal,
estado desconocido o mapping incompleto.

Jugador A es siempre el primer jugador visible y B el segundo. El CSV incluye
`model_probability_a/b`, `market_probability_a/b` de-vigadas y
`edge_a/b = P_modelo - P_mercado`. Un edge positivo significa únicamente que
el modelo asigna más probabilidad que la casa después de normalizar el margen;
no garantiza rentabilidad, no incorpora el precio necesario para decidir una
apuesta y no debe interpretarse como recomendación.

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

La consola muestra como máximo 50 filas ordenadas por edge absoluto; el CSV
siempre conserva la cartelera completa, los timestamps, fingerprints, cortes
de datos y motivos de confianza. El contrato detallado está en
[`docs/daily_pipeline.md`](docs/daily_pipeline.md). La línea exacta que el
usuario puede añadir a `start.ps1`, sin que este proyecto lo modifique, está en
[`docs/integracion.md`](docs/integracion.md).

## Tests

Desde `TENNIS/`:

```powershell
python -m unittest discover -s tests -v
```

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
cronológico, Parquet, publicación idempotente y ausencia de roles
ganador/perdedor entre las columnas persistidas.
Los tests `test_model_*.py` y `test_modeling_*.py` cubren contrato de
allowlist, perfiles con/sin mercado, preprocesamiento ajustado solo con train,
aislamiento de género, regresión logística, LightGBM, baselines, folds
expansivos, Platt, métricas segmentadas, soporte común, estabilidad al añadir
futuro, publicación inmutable, hashes, carga segura e idempotencia. La suite
`test_daily_pipeline.py` añade reconstrucción dirigida, no-fuga al añadir
futuro, orientación A/B, bloqueo del modelo no causal, degradación por estado
o mapping, confianza y publicación CSV. `test_daily_documentation.py` fija el
lanzador y la línea no invasiva de integración. La suite completa de cierre de
fase 8 obtuvo `258/258`.

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
- El contrato reserva las cuotas como feature para un futuro perfil con cobertura histórica. El perfil activo `sports_only` no las usa todavía, pero siempre compara su probabilidad con el mercado cuando hay dos cuotas.
- Toda generación de ratings, features, particiones, validaciones y predicciones cumple la regla anti-fugas: para una fecha `D`, solo consulta información estrictamente anterior a `D`.

## Roadmap

1. **Andamiaje — completada:** estructura, reglas permanentes, documentación, dependencias, entorno virtual y configuración de rutas.
2. **Ingesta y vigencia de fuentes — completada:** histórico Sackmann reproducible, actualizaciones incrementales, Match Charting separado y snapshots Elo externos con límite temporal.
3. **Sistema Elo por superficie y género — completada:** motor causal, ratings generales y por superficie, persistencia SQLite y API histórica estricta.
4. **Scraper Tennis Explorer — completada:** cartelera diaria de singles, slugs estables, cuotas, estados conservadores, caché íntegra y acceso de baja frecuencia.
5. **Mapeo de jugadores — completada:** resolución exacta y causal de slugs, caché SQLite auditable, overrides y cola honesta de no resueltos.
6. **Features sin fugas — completada:** vector causal común, orientación A/B estable, rankings temporales, Parquet separados por género y auditoría de publicación.
7. **Modelos, validación temporal y calibración — completada:** LightGBM y baseline logística por género, backtest expansivo, Platt causal, métricas por segmento y artefactos versionados.
8. **Pipeline diario e integración — completada:** scraper y mapping encadenados, contexto causal dirigido, inferencia calibrada, edge, confianza, CSV timestamped, lanzador autocontenido y guía no invasiva para `start.ps1`.
9. **Auditoría anti-fugas:** revisión integral y pruebas específicas para demostrar el cumplimiento temporal.
