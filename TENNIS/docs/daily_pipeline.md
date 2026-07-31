# Pipeline diario de predicción

## Contrato operativo

`scripts/daily_predictions.py` encadena, en este orden:

1. carga o captura una única cartelera diaria mediante el cliente cacheado;
2. resuelve slugs a IDs Sackmann sin eliminar los no resueltos;
3. reconstruye forma, H2H y descanso estrictamente con partidos de fecha `< D`;
4. consulta Elo, ranking y edad as-of `D`;
5. aplica el LightGBM del género correcto y su calibrador Platt;
6. quita el margen de las dos cuotas y compara modelo contra mercado;
7. publica todas las filas, predichas o no, en un CSV timestamped.

Jugador A es siempre `player_1` de Tennis Explorer y B es siempre `player_2`.
La probabilidad complementaria se calcula como `P(B) = 1 - P(A)`. Los edges
son `P_modelo - P_mercado` para cada lado y, cuando hay dos cuotas válidas,
suman cero salvo redondeo.

Solo recibe probabilidad una fila que cumpla simultáneamente:

- `status == "scheduled"` en el snapshot;
- ambos jugadores están mapeados;
- existe un modelo íntegro del mismo género;
- `training_max_date < D`.

Los partidos terminados, en juego, cancelados, walkover, de estado desconocido
o no mapeados permanecen en el CSV con probabilidades nulas. La restricción del
corte del modelo evita una fuga que las features por sí solas no impedirían al
solicitar una fecha histórica.

## Contexto causal e integridad

Los Parquet de fase 6 se verifican por tamaño y SHA-256. Para cada género se
leen únicamente las columnas de resultado y las filas `< D` donde participa
alguno de los jugadores del día. `y` permite recuperar el ganador real pese a
la orientación A/B aleatoria del entrenamiento. Los resultados se aplican en
bloques diarios completos, igual que en fase 6.

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
diarias no entran actualmente al LightGBM. Se transforman a probabilidad
de-vigada y se usan solo para `market_probability_*` y `edge_*`.

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

La consola ordena las filas predichas por magnitud absoluta del edge y muestra
como máximo 50; el CSV siempre contiene la cartelera completa.

