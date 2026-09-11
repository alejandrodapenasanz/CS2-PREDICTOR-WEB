# Incertidumbre del estimado de probabilidad

Cada predicción de producción publica una banda simétrica alrededor de
`model_prob_team1`. La banda mide incertidumbre sobre el propio estimado `p`;
no mide la variabilidad del resultado de un partido y no es un intervalo de
confianza o de predicción riguroso.

Para un único partido, la incertidumbre Bernoulli del resultado ya está
contenida en el porcentaje: su varianza es `p(1-p)`. No se suma esa varianza a
la banda. La banda responde a otra pregunta: cuánto puede variar nuestro
estimado de `p` por desacuerdo del modelo y cobertura limitada.

## Método versionado

`ensemble_dispersion_history_v1` usa solo información disponible antes del
partido:

1. Desviación estándar ponderada de las probabilidades calibradas de los
   miembros del ensemble. Cada miembro se simetriza primero con
   `0.5 * (p_j(A,B) + 1 - p_j(B,A))`, de modo que la dispersión rodea
   exactamente el mismo estimado independiente del orden que se publica.
2. Cobertura de historia, definida como
   `min(partidos_previos_equipo_1, partidos_previos_equipo_2) / 30`, acotada a
   `[0, 1]`.
3. Penalización máxima de `0.10 * (1 - cobertura)`.
4. Penalización de `0.05` si hay menos de dos miembros con peso efectivo.

Las tres magnitudes se combinan en cuadratura. La semibanda se limita a `0.25`
y los extremos se recortan a `[0, 1]`. El nivel de confianza es:

- `high`: cobertura `>= 0.80`, al menos dos miembros y semibanda `<= 0.05`;
- `medium`: cobertura `>= 0.40`, al menos dos miembros y semibanda `<= 0.10`;
- `low`: cualquier otro caso.

Son umbrales operativos documentados, no cuantiles aprendidos. Correlación
entre miembros, sesgo común, drift, errores de calibración y datos no observados
pueden hacer que la banda sea demasiado estrecha. Las métricas walk-forward de
log-loss, Brier y ECE siguen siendo la autoridad para validar calibración.

## Contrato de datos

El JSON de predicción publica:

- `ensemble_disagreement`;
- `estimate_band_half_width`, `estimate_band_lower_team1` y
  `estimate_band_upper_team1`;
- `estimate_confidence_level` (`low`, `medium`, `high`);
- `estimate_history_coverage` y `estimate_effective_members`;
- `estimate_uncertainty_method`.

`predictions` y `prediction_ledger` conservan como columnas escalares nullable
la dispersión, la semibanda, el nivel y la cobertura. La migración es aditiva:
las filas históricas no se rellenan ni se reescriben. Una fila abierta solo se
actualiza por `BBDD.ingest.upsert_prediction_ledger`; una fila congelada o
evaluada permanece inmutable.

El dashboard muestra la banda alrededor de la probabilidad pura del modelo,
incluida la vista Best Opportunity. No rodea `decision_prob_team1` ni las odds.
El contrato queda disponible para una política posterior de odds/selección,
pero los campos nuevos no cambian el ranking ni calculan stake. La recomendación
Kelly heredada permanece fuera de este cambio y sigue leyendo exclusivamente su
campo histórico `model_epistemic_std`; no consume `ensemble_disagreement` ni
ningún `estimate_*`. Por tanto esta versión no incorpora la banda al bankroll.

## Determinismo de inferencia

Al cargar un artefacto para servir, la ruta de inferencia fuerza a uno todos los
parámetros de paralelismo reconocidos (`n_jobs`, `nthread`, `num_threads` y
`thread_count`) sin cambiar la receta de entrenamiento. La misma fila debe dar
la misma probabilidad dentro de una tolerancia absoluta de `1e-12`, también al
puntuar el mismo artefacto en procesos separados. La semilla global sigue
siendo 42. Cambios entre días con inputs diferentes no contradicen este
contrato: el determinismo se prueba manteniendo artefacto y `match_features`
idénticos.
