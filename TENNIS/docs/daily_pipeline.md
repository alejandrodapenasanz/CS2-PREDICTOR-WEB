# Pipeline diario de predicción

## Contrato operativo

`scripts/daily_predictions.py` encadena, en este orden:

1. carga o captura una única cartelera diaria mediante el cliente cacheado;
2. resuelve slugs a IDs Sackmann sin eliminar los no resueltos;
3. reconstruye forma, H2H y descanso solo con resultados cuya
   `result_available_date=source_date+21 días` cumple
   `result_available_date < D`;
4. consulta Elo, ranking y edad as-of `D`;
5. aplica el LightGBM del género correcto y su calibrador Platt;
6. quita el margen de las dos cuotas y compara modelo contra mercado;
7. publica todas las filas, predichas o no, en un CSV timestamped;
8. registra partido, predicción y estadísticas prepartido en
   `BBDD/tennis.sqlite3`;
9. observa y concilia una sola jornada anterior pendiente, si existe.

Jugador A es siempre `player_1` de Tennis Explorer y B es siempre `player_2`.
La probabilidad complementaria se calcula como `P(B) = 1 - P(A)`. Los edges
son `P_modelo - P_mercado` para cada lado y, cuando hay dos cuotas válidas,
suman cero salvo redondeo.

Solo recibe probabilidad una fila que cumpla simultáneamente:

- `status == "scheduled"` en el snapshot;
- ambos jugadores están mapeados;
- existe un modelo íntegro del mismo género;
- `training_available_max_date = training_max_date + 21 días`;
- `training_available_max_date < D`.

Los partidos terminados, en juego, cancelados, walkover, de estado desconocido
o no mapeados permanecen en el CSV con probabilidades nulas. La restricción del
corte del modelo evita una fuga que las features por sí solas no impedirían al
solicitar una fecha histórica.

## Contexto causal e integridad

Los Parquet de fase 6 se verifican por tamaño y SHA-256. Para cada género se
leen únicamente las columnas necesarias y las filas de los jugadores del día
cuyo `result_available_date < D`. `y` permite recuperar el ganador real pese a
la orientación A/B aleatoria del entrenamiento. Los resultados se aplican en
bloques completos de disponibilidad, igual que en fase 6; `match_date < D` por
sí solo nunca habilita un resultado.

Antes de predecir se exige que coincidan:

- fingerprint del dataset en modelo y manifiesto de features;
- número de filas y fecha máxima de entrenamiento;
- commit Sackmann de raw, features y Elo;
- versión y fecha máxima del run Elo.

Un cambio de fuente requiere reconstruir Elo, features y modelos. No se mezcla
silenciosamente un ranking o maestro nuevo con un modelo antiguo.

La misma página diaria incluye un catálogo semanal con superficie explícita,
que se une por `tournament_href` sin hacer otra petición. En el snapshot real
del 30 de julio de 2026 cubre 180 de 313 partidos. Los 133 Futures masculinos
están agregados sin href ni superficie y permanecen nulos. La tabla no aporta
de forma fiable ronda ni `best_of`; ambos permanecen nulos y el preprocesador
entrenado los imputa.

## Mercado y modelo actual

El modelo activo tiene perfil `sports_only`, porque Sackmann no contiene
cuotas históricas y la cobertura de fase 7 fue 0 %. Por tanto, las cuotas
diarias no entran actualmente al LightGBM. Dos cuotas válidas se transforman y
conservan como `market_probability_*` para presentación. Si fueron capturadas
el propio día, su estado es `prestart_unverified` y `edge_*` queda nulo. El edge
solo se calcula cuando `market_comparison_status=strictly_pre_date`.

Esto permite una comparación descriptiva, pero todavía no demuestra que el
modelo supere al mercado: esa afirmación exige acumular cuotas prospectivas y
resultados, y evaluarlos después sobre soporte temporal común.

## Confianza

La confianza depende de inputs, nunca de lo extrema que sea la probabilidad.
Los umbrales configurables de `src/daily_pipeline/confidence.py` son:

- mínimo 20 partidos de Elo general por jugador;
- mínimo 10 partidos en la superficie por jugador;
- máximo 14 días de antigüedad para histórico y fuente de rankings;
- máximo 14 días para el snapshot de ranking individual.

Superficie ausente, histórico obsoleto, ranking ausente o menos de 20 partidos
generan `LOW`. Ausencia de ronda/`best_of`, poca muestra por superficie, edad
ausente o mercado ausente generan advertencias de calidad. `UNAVAILABLE`
significa que no se calculó ninguna probabilidad y siempre lleva una razón.
`confidence_flags` contiene todas las razones separadas por `|`.

## Salida

Cada ejecución publica:

```text
data/processed/predictions/
└── predictions_YYYY-MM-DD_YYYYMMDDThhmmssffffffZ.csv
```

El CSV conserva procedencia y timestamp de captura, estado, identidades,
mapping, cuotas, probabilidades raw y calibrada, mercado de-vigado, ambos
edges, confianza, cortes de fuentes y fingerprints. La escritura es atómica y
nunca sobrescribe una publicación con el mismo timestamp.

Si el reloj de Windows devuelve el mismo tick para la captura y la inferencia
aunque la descarga ya haya terminado, el pipeline asigna a la inferencia el
microtick lógico siguiente. La BBDD mantiene la desigualdad estricta
`source_retrieved_at_utc < prediction_as_of_utc`; no relaja el corte.

La consola ordena las filas predichas por magnitud absoluta del edge y muestra
como máximo 50; el CSV siempre contiene la cartelera completa.

El contrato append-only, la selección de la primera predicción oficial, la
rotación de fechas pendientes y el uso semanal de `-Retrain` se documentan en
[`operations_database.md`](operations_database.md).

Un CSV histórico puede generarse para diagnóstico, pero la BBDD nunca lo
promueve retrospectivamente: para ser oficial, captura y predicción deben
tener fecha civil no posterior a la jornada, el snapshot debe seguir
`scheduled`, el corte del modelo debe ser estrictamente anterior y no puede
existir una observación previa. Un settlement exige además
`observed_at_utc > prediction_as_of_utc`.

## Contrato causal vigente de fase 9

Sackmann publica `tourney_date` como inicio del torneo, no la fecha exacta de
cada ronda. El pipeline aplica por ello el contrato conservador
`result_available_date = match_date + 21 dias` y solo reconstruye historia con
`result_available_date < D`. El bundle debe acreditar asimismo
`training_available_max_date < D` y que esa fecha disponible es exactamente
`training_max_date + 21 dias`.

El run Elo puede cubrir fechas disponibles posteriores al maximo incluido en
features; debe alcanzar al menos ese maximo y mantener el mismo contrato y
fingerprint. Esta asimetria evita rechazar un checkpoint valido por un ultimo
bloque raw no elegible.

Dos cuotas validas siempre se transforman y conservan para presentacion. Si el
snapshot es del propio dia, `market_comparison_status=prestart_unverified` y
los edges quedan nulos. Solo `strictly_pre_date` habilita una comparacion
causal y `edge = P_modelo - P_mercado`.
