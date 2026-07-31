# Sistema Elo de tenis

## Objetivo y alcance

Esta fase construye ratings propios a partir del histórico canónico de partidos
de singles ingerido en la fase 2. Match Charting Project y el Elo publicado por
Tennis Abstract continúan siendo fuentes auxiliares: no intervienen en este
cálculo ni se aplican retroactivamente.

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

## Regla temporal y bloqueo por fecha

La columna `tourney_date` de Sackmann suele representar el comienzo aproximado
del torneo. No demuestra el día real de cada ronda. En consecuencia, no se usa
`round`, `match_num`, el orden del CSV ni el nombre del archivo para inventar
una cronología intratorneo.

Para cada género y fecha `D`:

1. se congela el estado existente al comenzar `D`;
2. todos los partidos fechados `D` calculan expectativa, K y delta desde ese
   mismo estado congelado;
3. los deltas de cada jugador se suman de forma determinista;
4. ratings y contadores se actualizan únicamente después de cerrar todo el
   bloque `D`.

Así, una consulta para `D` solo puede observar estados cuya fecha sea
estrictamente menor:

```text
state_date < as_of_date
```

El coste deliberado de esta garantía es que una ronda temprana no actualiza el
rating utilizado por otra ronda del mismo torneo cuando ambas comparten
`tourney_date`.

## Partidos elegibles y auditoría

No actualizan ratings ni contadores:

- niveles `E` y `J`;
- marcadores con `W/O`, `Walkover`, `BYE`, `RET`, `DEF`, `ABD` o `ABN`;
- filas con el mismo identificador como ganador y perdedor;
- copias exactas de una fila ya procesada.

Los CSV crudos nunca se modifican. Cada exclusión se contabiliza por motivo.
Un marcador ausente, desconocido o dañado no invalida por sí solo el resultado:
si existen ganador y perdedor distintos y no hay un estado explícito de
abandono, el partido es elegible porque Elo solo necesita el resultado.

Las colisiones entre posibles claves deportivas se conservan cuando las filas
no son idénticas: `match_num` y combinaciones de torneo/ronda/jugadores no son
claves fiables. En `sackmann-elo-v2`, la igualdad exacta se decide mediante un
SHA-256 de las 49 cadenas fuente leídas del CSV antes de tipar o normalizar
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

Una ejecución incompleta nunca sustituye a la activa. La base se construye
mediante staging y solo se publica después de completar y validar el cálculo.
SQLite se usa con un único escritor y sin modo WAL para evitar ficheros
laterales problemáticos dentro de un directorio sincronizado por OneDrive.

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
