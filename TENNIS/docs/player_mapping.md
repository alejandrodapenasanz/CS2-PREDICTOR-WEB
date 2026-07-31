# Mapeo de identidades Tennis Explorer → Sackmann

## Objetivo y unidad de identidad

La fase 5 enlaza cada participante de la cartelera diaria de Tennis Explorer
con el `player_id` de Jeff Sackmann del universo correcto. La identidad
persistente es:

```text
(gender, slug)
```

El `slug` procede del `href` de la ficha de Tennis Explorer. No se usa el texto
visible como identificador estable y tampoco se considera que un `player_id`
sea único sin su género: los universos `M` y `F` permanecen separados en todas
las búsquedas, claves y tablas.

El pipeline conserva todos los partidos de entrada. Una identidad que no puede
resolverse con evidencia unívoca se deja nula, el partido se marca como no
mapeado y la incidencia pasa a revisión; nunca se elige un candidato para
forzar la cobertura.

## Orden de resolución

Para cada lado del partido se aplica este orden:

1. Un override manual vigente para la clave exacta `(gender, slug)`.
2. Una asociación ya guardada en `data/processed/player_mapping.sqlite3`.
3. Para un slug nuevo, una coincidencia automática exacta contra jugadores
   Sackmann activos del mismo género.
4. Si falta el enlace, el nombre no cumple el formato esperado, no hay
   candidatos o hay más de uno, la identidad queda sin resolver.

Una asociación automática ya persistida no se recalcula. Solo una fila
explícita de `data/overrides.csv`, con motivo no vacío, puede insertar o
corregir manualmente una asociación; esa intervención queda auditada.

## Normalización y coincidencia exacta

El nombre abreviado de Tennis Explorer debe tener el formato
`Apellido compuesto I.`. Todos los componentes anteriores a la inicial se
tratan como apellido, por lo que se preservan casos como `Bautista Agut R.`,
`Auger-Aliassime F.` y `Van De Zandschulp B.`.

Ambas fuentes se convierten a una misma `NameKey`:

```text
(apellido_normalizado, inicial_del_nombre)
```

La normalización usa `Unidecode`, `casefold`, convierte guiones y demás
puntuación en espacios y colapsa espacios repetidos. Así elimina diferencias
tipográficas previsibles, como `Cerúndolo` frente a `Cerundolo`, pero no aplica
distancia difusa, no completa nombres y no reordena apellidos. En Sackmann se
normaliza `name_last` completo y se toma la primera letra normalizada de
`name_first`.

El matcher automático acepta una identidad únicamente cuando la clave exacta
produce un solo candidato. Cero candidatos y cualquier colisión se registran
como no resueltos. No se usan ranking, nivel del torneo, parecido del slug ni
recencia para romper empates.

## Candidatos activos y garantía temporal

Para mapear un partido de fecha `D`, un candidato automático debe haber jugado
algún partido del mismo género y de cualquier nivel en la ventana:

```text
D - 3 años naturales <= tourney_date < D
```

El extremo inicial es inclusivo y `D` es exclusivo. ATP, Challenger e ITF
alimentan juntos el índice masculino; WTA e ITF alimentan juntos el femenino.
Un partido de la propia fecha o del futuro nunca puede hacer que un jugador
sea candidato para `D`.

El mapeo identifica personas y no es una estadística deportiva. Aun así, la
creación automática de una asociación nueva respeta el corte anterior para no
usar actividad futura como evidencia. Las fases de features y modelo deberán
aplicar además su propia regla estricta: ratings, rankings, estadísticas y
cuotas solo son utilizables si estaban disponibles antes del instante de
predicción.

## País y límites de la fuente diaria

La página diaria inspeccionada de Tennis Explorer no publica el país del
jugador en la fila del partido. Las banderas de los encabezados describen la
sede del torneo y no son nacionalidades de los participantes.

Por tanto, la implementación actual no usa país como desempate y no solicita
fichas individuales para obtenerlo. Esto evita atribuir la bandera equivocada
y evita cientos de peticiones adicionales. El `ioc` de Sackmann se conserva
como dato auditable del candidato y del mapping, pero no resuelve colisiones
sin una observación equivalente y fiable en Tennis Explorer. Si la página
diaria incorporase ese dato en el futuro, habrá que inspeccionar y versionar
primero el nuevo contrato HTML.

## Overrides manuales

`data/overrides.csv` es un CSV UTF-8 con esta cabecera exacta y en este orden:

```csv
gender,slug,player_id,reason
```

| Columna | Contrato |
|---|---|
| `gender` | `M` o `F`; forma parte de la clave. |
| `slug` | Segmento de la ficha de Tennis Explorer, sin `/`, query ni fragmento. |
| `player_id` | Entero positivo del maestro Sackmann del mismo género. |
| `reason` | Justificación manual no vacía y revisable. |

No se permiten claves duplicadas. El pipeline valida el esquema completo y el
destino antes de aplicar una decisión manual. El override tiene precedencia
sobre la caché porque es el único mecanismo autorizado para corregirla.

## Persistencia SQLite y auditoría

`data/processed/player_mapping.sqlite3` se crea al primer uso. Utiliza journal
SQLite `DELETE` y contiene:

- `player_mappings`: estado vigente, con clave primaria `(gender, slug)`,
  identidad Sackmann, nombres observados, `ioc`, método de resolución, primera
  fecha de resolución y timestamp UTC de actualización.
- `mapping_audit`: historial append-only de inserciones automáticas,
  inserciones manuales y correcciones manuales, incluyendo valores anteriores
  y posteriores, motivo y timestamp.

Una inserción automática sobre una clave existente devuelve el registro
guardado sin modificarlo ni añadir otro evento. Una corrección manual conserva
la primera fecha de resolución y genera un evento de auditoría incluso cuando
formaliza de nuevo la identidad ya vigente. Los triggers de SQLite impiden
actualizar o borrar filas del historial de auditoría.

## Cola de no resueltos

`data/processed/unresolved_players.csv` representa la cola pendiente actual,
no un log de eventos sin depurar. Se actualiza bajo un lock local y mediante
reemplazo atómico. Su clave es `(gender, slug)`; cuando la fuente no aporta
slug, se usa como sustituto interno el apellido normalizado y la inicial. Si
además el nombre visible no puede interpretarse, se usa
`unparsed:<texto visible>` solo como clave interna de deduplicación: el apellido
y la inicial quedan vacíos y no se fabrica una identidad.

La cola guarda el nombre visible, clave normalizada, motivo, candidatos
encontrados, primera y última fecha observadas, conjunto de fechas, nivel y
torneo de la última observación. Repetir la misma identidad en la misma fecha
es idempotente: `occurrences` cuenta fechas observadas distintas, no el número
de filas de partidos. Cuando una clave se resuelve, se elimina de la cola en la
misma actualización.

Los motivos generados por el orquestador son:

| `reason` | Significado |
|---|---|
| `missing_player_slug` | La fila no contiene enlace estable para ese jugador. |
| `invalid_visible_name` | El texto no puede convertirse a `Apellido I.`. |
| `no_active_candidate` | No existe una clave exacta activa en la ventana causal. |
| `ambiguous_active_candidates` | La clave exacta corresponde a más de un jugador activo. |

## DataFrame resultante

El mapeo conserva todas las columnas de la cartelera y añade, para cada lado,
el ID Sackmann anulable y el método que explica su procedencia. También añade
un estado de partido que solo es `mapped` cuando ambos participantes tienen
identidad; de lo contrario es `unmapped`. Los nombres, tipos y valores
canónicos están especificados en
[`data_dictionary.md`](data_dictionary.md).

Los métodos persistidos son `automatic_name` y `manual_override`. Una lectura
de caché conserva el método de origen; no se etiqueta como una identidad
distinta. Una identidad sin evidencia suficiente no recibe `player_id`.

## Uso programático

La API pública recibe una única cartelera diaria por llamada:

```python
from datetime import date

from src.player_mapping import (
    load_unresolved_players,
    resolve_scraped_matches,
    summarize_mapping_coverage,
)
from src.tennis_explorer import get_daily_matches

selected_date = date(2026, 7, 30)
scraped = get_daily_matches(match_date=selected_date)
mapped = resolve_scraped_matches(scraped, as_of_date=selected_date)
coverage = summarize_mapping_coverage(mapped)
pending = load_unresolved_players()
```

`as_of_date` es opcional, pero si se proporciona debe coincidir exactamente
con la única fecha de `match_date` presente en el lote. Las rutas del maestro,
SQLite, overrides y cola se pueden inyectar, igual que el maestro y los índices
de candidatos en tests.

El resumen de cobertura agrupa por `gender` y `tour_level` y expone:

| Columna | Significado |
|---|---|
| `matches` | Partidos conservados en el segmento. |
| `player_slots` | Participantes observados, siempre dos por partido. |
| `automatic_player_slots` | Participantes cuyo método de origen es `automatic_name`. |
| `automatic_player_slot_pct` | Porcentaje automático sobre participantes. |
| `mapped_matches` | Partidos con ambos IDs, sea cual sea el método válido. |
| `mapped_match_pct` | Porcentaje de partidos con ambos IDs. |
| `automatic_matches` | Partidos cuyos dos lados proceden de `automatic_name`. |
| `automatic_match_pct` | Porcentaje de partidos completamente automáticos. |

## Ejecución

Desde `TENNIS/`, con el entorno virtual activado, la fecha puede omitirse para
usar el día local actual:

```powershell
python scripts\map_tennis_explorer_players.py
python scripts\map_tennis_explorer_players.py --date YYYY-MM-DD
```

La fecha selecciona la cartelera de la fase 4. Si su snapshot ya está
cacheado, no se vuelve a descargar. La ejecución imprime la cobertura por
género y nivel, separando cobertura de participantes y partidos completos, y
mantiene la caché de identidades y la cola de revisión.

Para tests o diagnósticos, el CLI permite sustituir explícitamente las rutas
con `--raw-dir`, `--database`, `--overrides` y `--unresolved`.

Para revisar una colisión:

1. comprobar el slug y los `candidate_player_ids` de
   `data/processed/unresolved_players.csv`;
2. verificar la identidad en las fuentes;
3. añadir una fila motivada a `data/overrides.csv`;
4. volver a ejecutar la misma fecha.

No se debe editar directamente la base SQLite ni convertir un pendiente en
mapping por intuición.

## Comportamiento ante errores

Una identidad ordinariamente ambigua o ausente no detiene el lote. Un nombre
visible fuera del formato inspeccionado se registra con
`reason="invalid_visible_name"` y tampoco detiene la ejecución. En cambio, un
cambio en las columnas requeridas, un CSV de overrides inválido, una caché
inconsistente, un ID de género incorrecto o la ausencia de archivos Sackmann
necesarios se tratan como errores explícitos. Esta diferencia permite continuar
con los partidos mapeables sin ocultar corrupción o cambios de contrato.

Los tests de fase 5 cubren normalización, apellidos compuestos y acentos,
aislamiento por género, ventana temporal exclusiva, colisiones, overrides,
inmutabilidad automática, auditoría SQLite y actualización idempotente de la
cola sin depender de la red.
