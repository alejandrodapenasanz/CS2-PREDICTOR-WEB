# Tennis Explorer: partidos diarios y cuotas

## Propósito y alcance

La fase 4 obtiene la cartelera diaria de **singles** publicada por Tennis
Explorer y conserva las identidades web necesarias para el mapeo de jugadores
de la fase 5. El alcance admitido es:

- hombres: ATP, Challenger e ITF;
- mujeres: WTA e ITF.

Los dobles y los eventos UTR quedan fuera. El parser no
convierte una competición desconocida en una categoría conocida para aumentar
artificialmente la cobertura.

La fuente diaria es una sola página con `type=all`:

```text
https://www.tennisexplorer.com/matches/?day=DD&month=MM&type=all&year=YYYY
```

Aunque la página pueda mostrar días adyacentes, solo se procesa la tabla cuyo
encabezado coincide exactamente con la fecha solicitada.

## Autorización y límites de uso

El responsable del proyecto ha comunicado que obtuvo autorización expresa de
Tennis Explorer y Tennis Abstract para realizar este uso académico y no
comercial, incluido el almacenamiento en una base de datos. Esa autorización
es información aportada por el usuario: el proyecto no afirma haberla
verificado de manera independiente ni amplía su alcance.

El scraper se limita a la página pública de partidos de Tennis Explorer. No
consulta perfiles individuales, endpoints internos ni el widget externo de
resultados de EnetScores. Los datos y artefactos obtenidos no deben usarse con
fines comerciales.

El archivo `robots.txt` se revisó manualmente el **2026-07-30**. En esa revisión
no se prohibía `/matches/`; sí figuraban como no permitidas `/redirect/`,
`/terms-of-use/` y `/contact/`, rutas que el scraper no solicita.

Para reducir el tráfico, la autorización académica/no comercial reportada se
aplica como una excepción estrecha: el cliente no solicita `robots.txt`
únicamente cuando accede a la URL diaria canónica
`/matches/?day=DD&month=MM&type=all&year=YYYY`. Esto reduce una fecha nueva a
un solo GET. La excepción no autoriza ninguna otra ruta; si la URL necesaria
cambia, el proceso debe detenerse para revisar el alcance antes de acceder.

## Política de acceso de bajo ruido

La prioridad es reducir solicitudes y detenerse ante una defensa del sitio, no
sortearla:

1. Se consulta primero la caché local. Una fecha correctamente almacenada
   provoca **cero** peticiones, incluso si el cortacircuitos está abierto.
2. Para una fecha no cacheada se comprueba el cortacircuitos persistente antes
   de crear una sesión o acceder a la red.
3. Si el circuito está cerrado, se espera `1,0` segundo y se realiza como
   máximo un GET a la página canónica `type=all`. No se solicita
   `robots.txt` bajo la excepción limitada descrita arriba.
4. Un lock local por host impide que dos procesos emitan tráfico concurrente.
   Un lock abandonado no se elimina automáticamente: exige revisión manual.
5. Scrapling se utiliza solo como transporte HTTP estático, con una identidad,
   cabeceras y configuración fijas y auditables.
6. No hay reintentos ni redirecciones automáticas. Cualquier estado distinto
   de `200`, una redirección, un contenido inesperado o una firma inequívoca de
   Cloudflare/WAF detiene la ejecución. Un HTML ajeno al esquema tampoco se
   cachea.
7. No se usan navegadores, adaptación, sigilo, impersonación, proxies,
   rotación de identidad, cookies para superar controles ni resolución
   automática de desafíos.

Estas medidas reducen el riesgo de bloqueo, pero no prometen que un tercero no
pueda bloquear una petición. Si ocurre, el operador debe parar y revisar la
situación; no debe aumentar la frecuencia ni intentar evadir el control.

### Uso acotado de Scrapling

El proyecto fija `scrapling[fetchers]==0.4.12`, según la
[instalación oficial de Scrapling](https://github.com/D4Vinci/Scrapling#installation),
y usa exclusivamente su transporte HTTP estático. Un adaptador propio impone la
identidad y las cabeceras fijas del proyecto, desactiva las opciones de
impersonación y expone únicamente el único `GET` sin redirecciones que necesita
el cliente.

No se utilizan `StealthyFetcher`, `StealthySession`, `DynamicFetcher`,
`DynamicSession`, navegadores, fingerprints, `stealthy_headers`, proxies,
rotación, resolución de Cloudflare/CAPTCHA ni funciones similares. Tampoco se
usa el parser adaptativo de Scrapling: el HTML cacheado continúa
interpretándose mediante Beautiful Soup/lxml y los selectores estrictos
documentados. Por ello un cambio estructural genera un error en vez de
relocalizar elementos por similitud.

La instalación no ejecuta `scrapling install` ni descarga navegadores. La
[licencia BSD-3-Clause de Scrapling](https://github.com/D4Vinci/Scrapling/blob/main/LICENSE)
no sustituye la autorización necesaria para acceder a Tennis Explorer; son
asuntos independientes.

La versión no sigue automáticamente la rama `main` del repositorio. Una nueva
release puede detectarse en el pipeline de mantenimiento, pero actualizar el
pin requiere revisar el cambio y ejecutar los tests antes de incorporarlo. Así
se mantiene una dependencia reproducible sin ejecutar código remoto no
validado.

La comprobación ya disponible se ejecuta desde `TENNIS/`:

```powershell
python scripts\check_scrapling_release.py
```

Consulta una sola vez `D4Vinci/Scrapling/releases/latest`, no sigue
redirecciones ni `main`, y no instala ni modifica nada. Sus códigos son `0`
cuando coincide el pin, `1` cuando existe una release posterior y `2` cuando
GitHub no permite concluir el estado de forma segura.

## Caché y trazabilidad

El contenido queda bajo `data/raw/tennis_explorer/`:

```text
data/raw/tennis_explorer/
├── policy/
│   ├── network.lock
│   └── circuit_breaker.json
└── daily/
    └── YYYY/
        ├── YYYY-MM-DD.html
        └── YYYY-MM-DD.metadata.json
```

La metadata registra la URL fuente, fecha solicitada, instante UTC de
adquisición, tamaño, SHA-256 del contenido, transporte y fundamento de acceso.
Los bytes se descargan a un archivo temporal, se valida su tipo y estructura,
y solo entonces se activa la caché mediante un reemplazo atómico.

No existe opción `--force`: una captura válida no se vuelve a descargar. Si el
HTML o su metadata faltan, no coinciden o fallan la verificación de integridad,
la ejecución se detiene y no sustituye silenciosamente el artefacto. Esta
decisión hace la ingesta idempotente y conserva evidencia reproducible de lo
que realmente estuvo disponible.

### Snapshots append-only de resultados

La captura matinal sigue siendo inmutable. Para observar cómo evolucionan los
estados sin destruir esa evidencia, `refresh_daily_results()` publica cada
refresco bajo un namespace distinto:

```text
data/raw/tennis_explorer/results/
└── YYYY/
    └── YYYY-MM-DD/
        ├── YYYYMMDDTHHMMSSffffffZ_<sha12>.html
        └── YYYYMMDDTHHMMSSffffffZ_<sha12>.metadata.json
```

Cada llamada realiza exactamente un GET, con el mismo delay, lock,
cortacircuitos, URL canónica, validación de esquema y ausencia de reintentos
que la captura diaria. No hay un contador lógico que limite el número de
ejecuciones: el proceso que orquesta los refrescos decide cuándo hacen falta.
La API tampoco interpreta varias llamadas como permiso para solaparlas o
eludir una defensa del sitio.

Un snapshot se escribe solo después de parsear el documento completo. Nunca
sustituye un archivo existente ni toca `daily/`. El objeto
`TennisExplorerResultSnapshot` devuelve el DataFrame y también SHA, instante
UTC y rutas de HTML/metadata; así una fecha sin partidos conserva procedencia
a nivel de snapshot aunque no tenga filas.

Este módulo todavía no concilia resultados en SQLite. Esa capa debe consumir
los snapshots append-only, mantener predicciones inmutables y tratar como
conflicto cualquier resultado terminal contradictorio.

### Cortacircuitos persistente

`policy/circuit_breaker.json` se crea únicamente al recibir un `403`, un `429`
o una firma inequívoca de desafío/WAF. Registra la evidencia necesaria para
auditar por qué se abrió. Desde ese momento:

- las fechas ya cacheadas continúan cargándose sin red;
- toda fecha no cacheada se detiene antes de crear una sesión HTTP;
- no existe caducidad, espera seguida de reintento ni reapertura automática.

El archivo solo debe borrarse manualmente después de coordinar con el operador
del sitio y revisar la causa. El cliente no lo elimina para «probar de nuevo».
Un error HTTP ordinario distinto de esas señales también detiene la ejecución,
pero no se presenta como bloqueo WAF sin evidencia.

## Estructura HTML inspeccionada

El parser se diseñó después de inspeccionar una captura real, no a partir de
selectores supuestos. La estructura relevante observada fue:

- cada bloque diario usa una tabla cuya clase es exactamente `result`,
  precedida por un `ul.tabs` cuya fecha normalizada coincide con
  `DD.MM.YYYY`;
- la barra lateral también contiene una tabla `result`, pero no tiene ese
  encabezado de fecha y se ignora;
- un torneo comienza en `tr.head.flags`; su celda `td.t-name` contiene el
  nombre y, cuando existe, el enlace;
- `span.type-men2` identifica singles masculino y `span.type-women2`, singles
  femenino;
- `span.type-men4`, `span.type-women4`, enlaces con `?type=double` y jugadores
  bajo `/doubles-team/` pertenecen a dobles y se excluyen;
- un partido ocupa dos filas consecutivas: la primera tiene un identificador
  como `r137` y la segunda usa el mismo identificador con sufijo `b`, por
  ejemplo `r137b`;
- el enlace estable del jugador aparece en `td.t-name a` y sigue la ruta
  `/player/<slug>/`;
- las dos cuotas están en las columnas H/A de la primera fila, con
  `rowspan="2"`. El orden de columna determina a qué jugador pertenece cada
  cuota; la clase visual `coursew` no se interpreta como identidad ni como
  resultado;
- el bloque lateral `This week's tournaments` publica la superficie en
  `td.s-color span[title]` y enlaza el mismo `tournament_href`; se usa solo
  ese enlace exacto para unir `Hard`, `Clay`, `Grass` o `Carpet`, sin otra
  petición;
- el texto «Live streams» puede aparecer anidado en la celda horaria y es un
  enlace de retransmisión, no evidencia de que el partido esté en juego.

Un cambio incompatible en estas relaciones provoca un error de esquema claro.
No se devuelven filas parcialmente reinterpretadas.

## Identidades de jugadores

Para cada participante se conserva:

- el texto visible del ancla, sin mezclar la cabeza de serie que aparezca fuera
  de ella;
- el `href` relativo completo;
- el `slug`, extraído únicamente de una ruta válida `/player/<slug>/`;
- una marca booleana que indica si existía un enlace.

Ejemplos reales inspeccionados:

| Texto visible | `href` | `slug` |
|---|---|---|
| `Michelsen A.` | `/player/michelsen-a98bb/` | `michelsen-a98bb` |
| `Mannarino A.` | `/player/mannarino-a7108/` | `mannarino-a7108` |
| `Kuzuhara B.` | `/player/kuzuhara/` | `kuzuhara` |

Si un jugador no tiene enlace, se conserva el nombre visible, los campos
`href` y `slug` quedan nulos y la marca booleana vale `False`. Si existe un
enlace presentado como jugador pero su ruta no cumple el patrón, se considera
un cambio de esquema y no se inventa un slug.

## Género y nivel del torneo

`gender` se deriva exclusivamente del marcador de singles:

| Marcador HTML | `gender` |
|---|---|
| `type-men2` | `M` |
| `type-women2` | `F` |

`tour_level` usa solo cuatro valores en esta fuente diaria: `ATP`, `WTA`,
`Challenger` e `ITF`. La clasificación se aplica de lo específico a lo
general:

1. un nombre o enlace con marcador ITF se clasifica como `ITF`;
2. `Futures 2026` se clasifica como `ITF` masculino;
3. un nombre o enlace con marcador Challenger se clasifica como
   `Challenger`;
4. los restantes singles con ruta `/atp-men/` se clasifican como `ATP`;
5. los restantes singles con ruta `/wta-women/` se clasifican como `WTA`.

El usuario confirmó expresamente la última convención: una competición
`wta-women` sin marcador ITF se trata como WTA. Esto permite incluir eventos
como WTA 125, pero la página diaria no ofrece de forma uniforme el subtipo y el
parser no inventa `WTA 125`, `WTA 250`, `ATP 500`, etc.

### Limitación de Futures masculinos

En la captura inspeccionada, los Futures masculinos aparecen bajo el
encabezado agregado `Futures 2026`, sin enlace ni sede específica. El resultado
conserva exactamente:

```text
tour_level = "ITF"
tournament = "Futures 2026"
tournament_href = null
surface = null
```

No se realizan peticiones por partido para intentar reconstruir la sede y no
se asigna un torneo o una superficie inventados.

## Superficie

La superficie se obtiene únicamente del catálogo incluido en el mismo HTML y
se une por el `tournament_href` exacto. Una celda vacía, un torneo ausente del
catálogo o una cabecera sin enlace produce `surface=null`. El rótulo observado
`Indoors` describe el recinto, no el material: también produce
`surface=null` y el diario añade baja confianza, sin asumir `Hard`. Cualquier
otro título nuevo, o dos valores contradictorios para el mismo enlace, se
trata como cambio de esquema.

En la captura real del 30 de julio de 2026 hay superficie para 180 de 313
partidos: cobertura completa de ATP, WTA, Challenger e ITF femenino. Los 133
Futures masculinos permanecen nulos porque la página los agrega sin href.

## Cuotas

`player_1_odds` y `player_2_odds` son números decimales anulables. Se asignan
por la posición H/A de la tabla, en el mismo orden que las dos filas de
jugadores. Una celda vacía produce un valor nulo; no se sustituye por `1.0`,
cero ni una estimación.

Las cuotas representan lo publicado en el instante
`retrieved_at_utc`. Ese instante debe conservarse cuando se integren como
feature: una captura posterior nunca demuestra que la cuota estuviera
disponible antes de una predicción histórica.

## Contrato conservador de estados

Los valores normalizados son `scheduled`, `in_progress`, `finished`,
`walkover`, `cancelled` y `unknown`. Cada fila incluye `status_evidence`, que
explica qué evidencia directa permitió la clasificación.

| `status` | Evidencia exigida |
|---|---|
| `walkover` | Texto explícito `W.O.`, `W/O` o equivalente directo del partido. |
| `cancelled` | Texto explícito `Cancelled`/`Canceled`. |
| `in_progress` | Indicador directo `Live`/`Interrupted`; nunca sets parciales por sí solos ni el enlace «Live streams». |
| `finished` | Texto final explícito o uno de los resultados terminales documentados acompañado de sets. |
| `scheduled` | Texto programado explícito, o una hora prevista con resultado y sets vacíos. |
| `unknown` | La estructura existe, pero no permite distinguir con seguridad uno de los estados anteriores. |

La prioridad se aplica en ese orden para no confundir un walkover o una
cancelación con un resultado ordinario. Si hay señales contradictorias, se
usa `unknown` o se lanza un error de esquema según la incompatibilidad; nunca
se fuerza la etiqueta más conveniente.

Los códigos exactos de `status_evidence` son:

| Estado normalizado | `status_evidence` posible |
|---|---|
| `walkover` | `explicit_walkover` |
| `cancelled` | `explicit_cancelled` |
| `in_progress` | `explicit_live_or_interrupted` |
| `finished` | `explicit_finished`, `terminal_result_with_set_scores` |
| `scheduled` | `explicit_scheduled`, `scheduled_time_with_empty_result_and_scores` |
| `unknown` | `undetermined_no_start_time`, `nonterminal_or_scoreless_result`, `partial_terminal_result`, `set_scores_without_explicit_live_state`, `unrecognised_status_text` |

Tennis Explorer muestra estados más ricos en su vista de resultados en vivo,
pero parte de esa información procede del widget externo
`widget.enetscores.com`. Ese widget **no se consulta**: no forma parte de la
autorización documentada y no es necesario para el contrato conservador
confirmado por el usuario.

## Identificador fuente y resultado observable

`source_match_id` procede exclusivamente del parámetro decimal de
`/match-detail/?id=NUMERO`. Se conserva como texto opaco y, junto con la
fuente `tennis_explorer`, es la clave natural para reconciliar snapshots. El
ID de fila DOM (`r10`), el orden de los jugadores y una combinación de
nombres/torneo no son claves estables.

El parser conserva `player_1_sets_won`, `player_2_sets_won` y `sets_score`
cuando las dos celdas de resultado son numéricas. Un `1-0` parcial puede por
tanto aparecer como marcador observado, pero no como ganador.

`winner_slug` y `winner_side` solo se publican si se cumplen simultáneamente:

1. `status == "finished"`;
2. hay scores de sets visibles;
3. el par de sets es un final convencional `2-0`, `2-1`, `3-0`, `3-1` o
   `3-2`, en cualquier orientación;
4. el lado ganador tiene un slug estable.

No se usa la primera fila, la clase visual `coursew`, la cuota ni el favorito.
Un walkover, cancelación, marcador parcial, retirada no caracterizada o final
sin slug mantiene ganador nulo. Antes de soportar retiradas o ganadores por
walkover debe guardarse e inspeccionarse un HTML real que aporte evidencia de
qué lado ganó.

`result_evidence` usa uno de estos valores:

| Valor | Interpretación |
|---|---|
| `winner_from_terminal_sets_and_slug` | Ganador demostrable por sets terminales y slug. |
| `winner_not_inferred_nonterminal_or_special_status` | Estado pendiente, desconocido, walkover o cancelación. |
| `winner_not_inferred_without_terminal_sets` | `finished` explícito sin un par terminal demostrable. |
| `winner_not_inferred_missing_winner_slug` | Final convencional, pero sin identidad web estable del ganador. |

En el fixture real hay 76 finales liquidables: 64 con `2-0` y 12 con `2-1`.
Los ocho `1-0` observados permanecen sin ganador.

## DataFrame diario

El parser puro produce 27 columnas deportivas. `get_daily_matches()` y
`refresh_daily_results()` añaden otras tres columnas de trazabilidad, por lo
que su DataFrame público tiene 30 columnas:

| Columna | Tipo pandas | Descripción |
|---|---|---|
| `match_date` | `datetime64[ns]` | Fecha diaria solicitada. |
| `tournament` | `string` | Texto del encabezado de torneo tal como se muestra. |
| `tournament_href` | `string` anulable | Enlace relativo del torneo, si existe. |
| `tour_level` | `string` | `ATP`, `WTA`, `Challenger` o `ITF`. |
| `gender` | `string` | `M` o `F`. |
| `surface` | `string` anulable | `Hard`, `Clay`, `Grass` o `Carpet`, unido por href desde el catálogo del mismo HTML. |
| `scheduled_time` | `string` anulable | Hora visible; no se convierte a UTC sin una zona horaria inequívoca. |
| `player_1_name`, `player_2_name` | `string` | Textos visibles de los jugadores. |
| `player_1_href`, `player_2_href` | `string` anulable | Rutas relativas de las fichas. |
| `player_1_slug`, `player_2_slug` | `string` anulable | Claves estables extraídas de las rutas. |
| `status` | `string` | Estado conservador normalizado. |
| `status_evidence` | `string` | Evidencia estructural o textual usada. |
| `player_1_sets_won`, `player_2_sets_won` | `Int64` anulable | Contadores numéricos observados, aunque sean parciales. |
| `sets_score` | `string` anulable | Par observado con orientación de la página, por ejemplo `2-1`. |
| `winner_side` | `string` anulable | `player_1` o `player_2` solo para un final convencional liquidable. |
| `winner_slug` | `string` anulable | Slug estable del ganador demostrable. |
| `result_evidence` | `string` | Motivo exacto por el que se publicó o no un ganador. |
| `source_match_id` | `string` | ID opaco extraído del enlace de detalle. |
| `match_detail_href` | `string` | Enlace relativo obligatorio al detalle del partido. |
| `player_1_has_link`, `player_2_has_link` | `boolean` | Existencia de enlace de ficha. |
| `player_1_odds`, `player_2_odds` | `Float64` | Cuotas decimales, anulables. |
| `source_url` | `string` | URL diaria exacta usada. |
| `retrieved_at_utc` | `datetime64[ns, UTC]` | Instante real de adquisición. |
| `snapshot_sha256` | `string` | SHA-256 de los bytes HTML validados. |

Una tabla diaria válida sin partidos dentro del alcance devuelve un DataFrame
vacío con este mismo esquema y tipos. Si la página omite o cambia la tabla
esperada, se produce un error claro en vez de asumir silenciosamente que el día
estaba vacío. Un día con partidos sin cuotas conserva esas filas con cuotas
nulas.

## Ejecución

Desde `TENNIS/`, con el entorno virtual activado:

```powershell
python scripts\scrape_tennis_explorer.py --date 2026-07-30
```

Si se omite `--date`, se utiliza la fecha local de ejecución:

```powershell
python scripts\scrape_tennis_explorer.py
```

El script muestra las filas obtenidas y un resumen por nivel. No ofrece una
opción para forzar una nueva descarga de una fecha ya cacheada.

El refresco append-only se invoca desde el orquestador Python, no mediante
`--force` sobre la caché diaria:

```python
from datetime import date

from src.tennis_explorer import refresh_daily_results

snapshot = refresh_daily_results(date(2026, 7, 30))
print(snapshot.html_path)
print(snapshot.matches[["source_match_id", "winner_slug"]])
```

Cada invocación anterior es una observación nueva y emite una sola petición.
El llamador debe decidir si necesita refrescar; la API no repite
automáticamente, pero tampoco impone un límite arbitrario de dos ejecuciones.

## Fixture y tests sin red

El parser se prueba contra la captura real:

```text
tests/fixtures/tennis_explorer/matches_2026-07-30_all.html
```

Procede de la URL diaria `type=all` del **2026-07-30**, ocupa `928.679` bytes y
su SHA-256 es:

```text
2910901e884a468c0f0ae3ef55d0032bb44748bc6026752c878f63075f9758ec
```

El fixture se descargó una sola vez. Los tests del parser son locales: no
dependen de Tennis Explorer ni generan tráfico. También se comprueban los
errores de esquema, cuotas o enlaces ausentes, caché íntegra, transporte
Scrapling estático con identidad fija, una sola petición sin `robots.txt`,
ausencia de redirecciones/reintentos y apertura persistente del cortacircuitos
ante 403/429/WAF.
