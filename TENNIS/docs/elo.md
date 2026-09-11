# Sistema Elo de tenis

## Objetivo y alcance

Esta fase construye ratings propios desde dos épocas no solapadas: una base
Sackmann congelada y resultados operativos posteriores al corte fijo. Match
Charting Project y el Elo publicado por Tennis Abstract continúan siendo
fuentes auxiliares: no intervienen en este cálculo ni se aplican
retroactivamente.

El contrato activo es `multisource-general-elo-v6`. Mantiene el contrato
temporal de `sackmann-elo-v5` para la base y añade un handoff operativo
versionado. El commit base es
`83733587353df8a41f2fd4f516147d5aa83f5a8d` y el corte fijo es
`2026-06-02`; ambos viven en `config/elo_handoff.json`.

Existen dos universos completamente independientes:

- `M`: ATP, qualifying, Challenger, ITF, competiciones por equipos y Juegos
  Olímpicos masculinos.
- `F`: WTA, qualifying, ITF, competiciones por equipos y Juegos Olímpicos
  femeninos.

Un identificador solo es válido junto con su género. Ningún resultado masculino
puede modificar, inicializar ni servir de rival para un rating femenino, ni a
la inversa. Dentro de cada género, todos los niveles admitidos comparten el
mismo pool y tienen el mismo peso.

Se excluyen los niveles `E` (exhibición) y `J` (juniors). No se aplican
multiplicadores por nivel, ronda, formato o importancia del torneo.

## Fórmula

Todos los jugadores comienzan con:

```text
R0 = 1500
```

Para dos jugadores `i` y `j`, la expectativa prepartido de `i` es:

```text
E_i = 1 / (1 + 10 ** ((R_j - R_i) / 400))
```

El factor de actualización es individual y decrece con la experiencia:

```text
K_i(n_i) = 250 / (n_i + 5) ** 0.4
```

`n_i` cuenta exclusivamente partidos elegibles estrictamente anteriores al
partido que se valora. La actualización es:

```text
R_i' = R_i + K_i(n_i) * (S_i - E_i)
```

donde `S_i = 1` para el ganador y `S_i = 0` para el perdedor. Cada jugador usa
su propio `K`; por tanto, cuando sus contadores difieren, el intercambio no
tiene por qué ser de suma cero.

Los valores `1500`, `400`, `250`, `5` y `0.4` son parámetros explícitos,
validados y versionados. La formulación y los valores del K proceden del
enfoque de FiveThirtyEight para tenis:

- [How We're Forecasting The 2016 U.S. Open](https://fivethirtyeight.com/features/how-were-forecasting-the-2016-us-open/)
- [Serena Williams And The Difference Between All-Time Great And Greatest Of All Time](https://fivethirtyeight.com/features/serena-williams-and-the-difference-between-all-time-great-and-greatest-of-all-time/)

## Rating general y ratings por superficie

Cada jugador mantiene cinco estados:

- un Elo general, actualizado con todo partido elegible;
- un Elo puro de `Hard`;
- un Elo puro de `Clay`;
- un Elo puro de `Grass`;
- un Elo puro de `Carpet`.

El contador del Elo general incluye todos los partidos elegibles. Cada contador
de superficie incluye solo partidos de esa superficie. Un partido con
superficie nula actualiza el Elo general, pero no ningún Elo de superficie. Las
variantes de capitalización se normalizan; cualquier superficie no reconocida
detiene el cálculo.

Para una predicción en superficie `s`, el rating combinado es:

```text
R_combinado(s) = 0.5 * R_general + 0.5 * R_superficie(s)
```

La mezcla 50/50 está documentada por Jeff Sackmann para dura, tierra y hierba:

- [An Introduction to Tennis Elo](https://www.tennisabstract.com/blog/2019/12/03/an-introduction-to-tennis-elo/)
- [Unpredictable Bounces, Predictable Results](https://www.tennisabstract.com/blog/2017/06/23/unpredictable-bounces-predictable-results/)

`Carpet` sigue la misma regla como extensión explícita del proyecto. El peso
`0.5` es configurable. Si no se solicita superficie, la consulta devuelve el
Elo general como rating efectivo.

## Regla temporal, embargo y bloqueo por fecha

La columna `tourney_date` de Sackmann representa normalmente el comienzo
aproximado del torneo, no el día real de cada ronda. Tratarla como fecha exacta
del partido permitiría que rondas aún no disputadas apareciesen como pasado. No
se usa `round`, `match_num`, el orden del CSV ni el nombre del archivo para
inventar esa cronología.

La política compartida `sackmann-tourney-start-embargo-v1`, centralizada en
`src/temporal.py`, define por defecto:

```text
source_date = tourney_date
result_available_date = source_date + 21 días
resultado_utilizable(as_of_date) ⇔ result_available_date < as_of_date
```

La desigualdad es estricta. Un resultado con `tourney_date=D` sigue excluido en
`D+21` y puede influir por primera vez en una predicción de `D+22`. El Elo v6
persiste el estado con la fecha efectiva de disponibilidad y conserva
`source_date` como procedencia; `get_elo(..., as_of_date=D)` solo recupera un
estado efectivo que cumpla `state_date < D`.

Para cada género y fecha efectiva de disponibilidad:

1. se congela el estado previo al bloque;
2. todos los resultados que se vuelven disponibles ese día calculan expectativa,
   K y delta desde el mismo estado congelado;
3. los deltas de cada jugador se agregan de forma determinista;
4. ratings y contadores se actualizan al cerrar el bloque.

Durante la construcción de features de un partido fuente `D`, el motor hace una
previsualización sin mutar el estado y solo aplica resultados anteriores cuyo
`result_available_date < D`. Así se cumple el mismo contrato en entrenamiento y
serving.

El embargo de 21 días es una salvaguarda conservadora y versionada, no una
afirmación de que todos los torneos terminen dentro de ese plazo. Un evento
excepcionalmente largo o reprogramado podría superar la ventana; sin fechas
reales por partido, esa incertidumbre residual no se puede eliminar y debe
figurar como limitación del backtest.

## Handoff fijo y resultados operativos

La base Sackmann se verifica contra el manifiesto inmutable
`data/raw/sackmann_manifests/83733587353df8a41f2fd4f516147d5aa83f5a8d.json`.
Ningún arranque normal actualiza esa base. Solo entran al overlay partidos con
fecha deportiva estrictamente posterior a `2026-06-02`; por diseño no se
intenta deduplicar entre las dos épocas.

TennisRatio es la fuente operativa principal y Tennis Explorer el respaldo.
Ambas exigen resultado terminal válido y los dos IDs Sackmann resueltos. La
clave común entre fuentes es `(gender, effective_date, min(player_id),
max(player_id))`: ante solapamiento gana TennisRatio y Explorer se suprime. Los
casos sin mapping completo o con resultado contradictorio se omiten sin
adivinar y se publican en
`data/processed/elo/operational_omissions.json`.

Para un resultado operativo se conserva la fecha deportiva y el instante real
de primera observación. El evento se procesa por su `available_date` y solo si
ambas fechas son `< D`; un resultado terminado u observado en `D` puede influir
por primera vez en `D+1`. Si el resultado publica literalmente `Hard`, `Clay`,
`Grass` o `Carpet`, el evento actualiza también el Elo de esa superficie. Si la
superficie directa falta, solo se admite la del mismo torneo y edición exactos
capturada como máximo en el instante en que el resultado estuvo disponible.
`Indoors`, agregados como `Futures`, conflictos y nombres parecidos permanecen
nulos y actualizan solo el Elo general.

El catálogo derivado se publica en `data/processed/surface_catalog.json`. Cada
evidencia conserva fuente, partido, edición, captura UTC, URL y SHA-256. El
fingerprint del run incluye el fingerprint completo del catálogo, además del
commit base, corte, precedencia y hashes exactos de los eventos seleccionados.
Las SQLite fuente se abren en modo de solo lectura; sus snapshots proceden del
cliente HTTP consolidado y el catálogo no realiza peticiones adicionales.

Si una SQLite operativa no puede leerse o validarse, el constructor publica un
aviso y ejecuta el camino Sackmann-only en vez de dejar una base parcial. La
base se sigue reemplazando atómicamente. En serving, el overlay en memoria solo
continúa Elo con observaciones posteriores al `max_event_date` persistido; no
reaplica eventos ya incorporados. Forma/H2H se reconstruyen por separado y sí
reciben una vez todos los resultados post-handoff que sean causales.

### Re-baseline manual

El corte nunca se mueve de forma automática. Para crear una base nueva hay que:

1. actualizar Sackmann manualmente y auditar el nuevo snapshot;
2. conservar su manifiesto versionado por commit;
3. elegir un corte que cubra toda la base y comprobar el posible hueco;
4. cambiar juntos `base.source_commit`, `base.manifest_path`, `cutoff_date` en
   `config/elo_handoff.json` y `SACKMANN_BASE_COMMIT` en `src/config.py`;
5. reconstruir Elo, features y challenger, dejando que la puerta de promoción
   decida el modelo activo.

No se mezclan dos commits Sackmann ni se mueve solo el corte.

## Partidos elegibles y auditoría

No actualizan ratings ni contadores:

- niveles `E` y `J`;
- marcadores con `W/O`, `Walkover` o `BYE`, porque no acreditan un partido
  iniciado con resultado deportivo;
- filas con el mismo identificador como ganador y perdedor;
- copias exactas de una fila ya procesada.

`RET`, `DEF`, `ABD` y `ABN` sí conservan el ganador oficial y alimentan el
Elo. Excluirlos usando una señal conocida únicamente después de comenzar el
partido produciría un universo retrospectivo más limpio que el disponible al
predecir. La auditoría de fase 9 corrigió la exclusión histórica de estos
resultados; los recuentos definitivos pertenecen al run Elo v6 regenerado, no a
los informes de fases anteriores.

Los CSV crudos nunca se modifican. Cada exclusión se contabiliza por motivo.
Un marcador ausente, desconocido o dañado no invalida por sí solo el resultado:
si existen ganador y perdedor distintos y no se trata de un partido no
iniciado, el partido es elegible porque Elo solo necesita el resultado.

La cuarentena de identidad se genera mediante `scripts/audit_identities.py`.
Una clave `(gender, player_id)` se marca si el histórico y la fecha de
nacimiento del maestro actual implican un partido antes de cumplir diez años.
Es una señal útil de reutilización o colisión del ID, pero la DOB maestra actual
es metadato retrospectivo: no acredita cuándo se conoció el conflicto.

Por ello, la cuarentena **no selecciona ni elimina ninguna fila histórica** de
Elo, features, backtest o reentreno. `first_match_date` se conserva únicamente
como dato diagnóstico y nunca como una supuesta «primera evidencia» causal. El
contrato persistido es
`diagnostic_current_inference_only_no_historical_selection`, con la exclusión
histórica desactivada.

La lista completa sí se usa de forma conservadora en inferencia operativa
actual: como la incompatibilidad ya es conocida hoy, un futuro participante
afectado se bloquea o degrada a baja confianza en lugar de recibir una
probabilidad inventada. El CSV y su manifiesto fijan commit fuente, SHA-256,
motivo y contrato de uso. Esta decisión evita reescribir el pasado con
biografía adquirida después, pero no repara la posible contaminación histórica
de ratings bajo un ID reutilizado; esa limitación permanece explícita.

Las colisiones entre posibles claves deportivas se conservan cuando las filas
no son idénticas: `match_num` y combinaciones de torneo/ronda/jugadores no son
claves fiables. En la época Sackmann de `multisource-general-elo-v6`, la
igualdad exacta se decide mediante un SHA-256 de las 49 cadenas fuente leídas
del CSV antes de tipar o normalizar
fechas, IDs y textos. Por ejemplo, los IDs crudos `1` y `01` no se consideran
la misma fila aunque ambos sean convertibles al entero `1`. La procedencia
reproducible de cada fila es:

```text
(source_commit, source_path, source_row_number)
```

`source_row_number` cuenta filas de datos desde uno y no incluye la cabecera.

## Persistencia y consultas

El histórico se guarda en:

```text
data/processed/elo/elo.sqlite3
```

La base generada no se versiona en Git. Sí se versionan el código, el esquema y
esta metodología. Cada ejecución registra el commit fuente, los blobs
verificados, los parámetros, la versión del algoritmo y un fingerprint
determinista.

Los parámetros de cada run conservan además `artifact_code_inventory` con
ruta, tamaño y SHA-256 de `ELO_CODE_PATHS`. Al resolver un run completo o
consultar ratings, el almacén recalcula ese inventario contra el código actual.
Si falta o difiere, la consulta falla cerrada y exige reconstruir Elo; nunca se
sirven ratings calculados por fórmulas distintas a las cargadas en runtime.

Una ejecución incompleta nunca sustituye a la activa. La base se construye
mediante staging y solo se publica después de completar y validar el cálculo.
SQLite se usa con un único escritor y sin modo WAL para mantener un único
artefacto publicable y no depender de ficheros laterales de una ruta concreta.

La API pública es:

```python
get_elo(
    *,
    player_id: int,
    gender: Literal["M", "F"],
    surface: str | None,
    as_of_date: date,
) -> EloSnapshot
```

La respuesta contiene el Elo general, el Elo puro de la superficie, el
combinado, contadores, fecha efectiva, commit y ejecución. Si todavía no existe
historia estrictamente anterior, devuelve el estado inicial de 1500. Para
generar features masivamente existe una consulta vectorizada equivalente.

## Decisiones no incluidas

Esta versión no incorpora:

- penalización o regresión por inactividad;
- margen de victoria, sets o juegos;
- ponderación por nivel, ronda o `best_of`;
- ratings externos de Tennis Abstract;
- estadísticas del propio partido;
- orden inferido dentro de una misma `tourney_date`.

Estas decisiones evitan parámetros no especificados y mantienen una base
reproducible para que cualquier extensión posterior se valide temporalmente.
