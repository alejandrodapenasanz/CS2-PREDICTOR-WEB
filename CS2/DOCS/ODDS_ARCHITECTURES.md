# Arquitecturas de producción con opening odds

## Objetivo y límite honesto

La prioridad de producto es el acierto del pick, sin dejar de producir una
probabilidad cuando no hay cuotas. Incluso incorporando mercado, una accuracy
realista ronda aproximadamente el 70 % y no constituye una garantía. La rama
sin odds tiene un techo estructuralmente inferior: se busca acercarla al máximo,
no prometer paridad.

## Contrato temporal

Una cuota solo es apta si existen dos decimales coherentes, el inicio UTC es
exacto y su primera captura verificable cumple
`captured_at_utc < kickoff_utc`. Se normaliza quitando el margen para que las dos
probabilidades sumen uno. Cierre, live, capturas tardías, proxies y partidos con
hora `date_only` quedan fuera. Si el scrape actual falla, se recupera la primera
apertura válida guardada por un intento anterior; nunca se sustituye por una
línea posterior. Sin evidencia suficiente se selecciona `no_odds`.

## Candidatas

- `router_two_models`: un LightGBM primario ajustado y calibrado únicamente con
  filas que tienen opening odds; cuando faltan, usa el modelo estadístico sin
  odds, que conserva su propia selección, tuning y calibración.
- `single_mixed_lgbm`: un LightGBM calibrado sobre filas mixtas. Las tres
  magnitudes de mercado son `NaN` nativo cuando faltan y `odds_available=0`
  identifica el régimen.

El benchmark histórico `model_b_stats_plus_opening_odds` se conserva como
referencia de mercado y no es por sí solo una arquitectura productiva. El
umbral `feature_thresholds.opening_odds` de `MODEL/config.yaml` vale 120 filas
causales antes de evaluar/ajustar la rama con odds.

## Selección y promoción

Las dos candidatas se predicen sobre los mismos folds walk-forward. Se publican
accuracy, log loss, Brier y calibración por separado para `odds` y `no_odds`,
además de la fracción sin odds. El ranking usa log loss y Brier como desempate.
Este resultado es **prequential**: cada fold ajusta un objeto nuevo con los datos
anteriores a ese fold. No es la métrica del pickle final ni la del shadow fijo de
la puerta. El informe lo etiqueta expresamente y registra modelo de reserva,
fracción de calibración y ponderación por recencia.

La evaluación prequential usa los mismos parámetros productivos: cola de
calibración `training.final_calibration_fraction`, pesos de recencia
`training.recency_half_life_days` y la familia de reserva elegida en el prefijo
permitido. Antes de esta corrección, el diagnóstico del router usaba por defecto
una cola 0,20 sin pesos para la rama con odds, mientras el shadow usaba 0,18 y
half-life de 365 días. Eso hacía que dos objetos distintos apareciesen bajo una
misma etiqueta y volvía engañosa la comparación directa de sus log loss.

Cuando existe incumbente, la receta de arquitectura se congela usando solo el
prefijo anterior a su `date_max`; el ranking posterior queda como diagnóstico.

La ganadora se integra en el challenger completo y atraviesa la puerta común
contra producción sobre exactamente el mismo hold-out temporal. La puerta ajusta
un solo shadow hasta el corte del vivo y lo mantiene congelado durante todo el
sufijo; vivo y challenger reciben las mismas identidades, etiquetas y features
point-in-time. La decisión conserva periodo, fracción con/sin odds, métricas por
régimen y dos huellas: cohorte y probabilidades emparejadas. Además vuelve a
puntuar el mismo shadow sobre las mismas filas y exige igualdad de probabilidades,
log loss y Brier con tolerancia absoluta `1e-12`.

Por construcción, el backtest prequential y la puerta fija pueden tener cifras
legítimamente distintas. Solo la puerta puede promover. Un rechazo o una muestra
insuficiente no mueve `latest.json`, `last_good.json` ni el pickle runtime.

## Inferencia y trazabilidad

El artefacto contiene la ruta completa, por lo que siempre devuelve una
predicción. Si la apertura ya forma parte del modelo, la capa de decisión no la
mezcla una segunda vez. Una predicción `no_odds` con baja cobertura, banda amplia
o probabilidad cercana al 50 % se etiqueta con confianza baja.

`predictions` y el `prediction_ledger` reciben, por la vía sancionada, cuatro
columnas aditivas nullable: `prediction_regime`, `prediction_architecture`,
`opening_odds_recovered` y `opening_odds_captured_at_utc`. No se reescribe ninguna
fila histórica. El dashboard muestra la ruta de cada partido y el ranking
producido en `MODEL/results/odds_architecture_eval.json`.

La banda `±x` sigue midiendo incertidumbre sobre el propio estimado `p`, no la
varianza Bernoulli del resultado ni un intervalo de confianza riguroso.
