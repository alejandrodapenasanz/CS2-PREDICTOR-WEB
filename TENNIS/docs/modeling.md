# Modelado, validación temporal y calibración

## Objetivo y alcance

La fase 7 entrena dos universos completamente independientes:

- un modelo masculino sobre ATP, Challenger e ITF;
- un modelo femenino sobre WTA, Challenger e ITF.

En cada universo se ajusta una regresión logística de referencia y un
LightGBM principal. El artefacto desplegable es LightGBM seguido de un
calibrador Platt. No se crean modelos distintos por nivel, superficie o ronda:
estas variables forman parte del vector y la evaluación se desglosa por
`tour_level × surface`.

La etiqueta es `y=1` cuando gana el jugador orientado como A. La orientación
aleatoria estable de la fase 6 impide que el modelo aprenda los roles
ganador/perdedor de la fuente.

## Contrato de features

El único origen autorizado de nombres es
`manifest.model_feature_columns` de la fase 6. Debe coincidir exactamente, en
nombres y orden, con `src.features.MODEL_FEATURE_COLUMNS`; una columna nueva o
ausente detiene el reentreno.

El snapshot activo declara `historical_odds_available=false`. Por ello,
`profile=auto` se resuelve de forma explícita como `sports_only`: usa las 37
features deportivas y excluye las seis columnas derivadas de cuotas. El perfil
`market_enhanced` se rechaza mientras no exista al menos una pareja histórica
de probabilidades de mercado válida. Nunca se rellena una cuota ficticia.

Quedan fuera de la matriz, aunque se conserven para auditoría:

- `record_id`, IDs y nombres;
- `gender` y `match_date`;
- procedencia y orden fuente;
- `y`;
- `model_probability_a` y `edge`.

El preprocesamiento está dentro del pipeline de cada estimador, por lo que se
ajusta de nuevo con el `train` de cada fold:

- numéricas: mediana del train e indicador adicional de ausencia;
- numéricas en logística: estandarización aprendida en train;
- `surface`, `tour_level`, `tour_level_raw` y `round`: ausencia explícita,
  one-hot y `handle_unknown="ignore"`;
- ningún estadístico o vocabulario se calcula con calibración o test.

Las columnas contractuales totalmente nulas se conservan en el transformador
de bajo nivel mediante `keep_empty_features=True`. No obstante, el perfil
actual excluye las seis columnas de mercado antes de llegar al transformador,
evitando un esquema vacío e inestable.

## Estimadores y parámetros

Los parámetros forman parte del fingerprint y viven en
`src/modeling/parameters.py`.

### Regresión logística

```text
C = 1,0
solver = lbfgs
max_iter = 2.000
random_seed = 42
```

Se utiliza regularización L2 predeterminada de scikit-learn. Esta baseline
recibe exactamente el mismo perfil deportivo que LightGBM.

### LightGBM

```text
objective = binary
n_estimators = 500
learning_rate = 0,05
num_leaves = 31
max_depth = -1
min_child_samples = 50
subsample = 0,90
subsample_freq = 1
colsample_bytree = 0,90
reg_alpha = 0,0
reg_lambda = 1,0
random_seed = 42
deterministic = true
force_col_wise = true
```

Son valores conservadores fijados antes de observar los tests. No se hizo una
búsqueda de hiperparámetros ni early stopping usando la temporada reservada
para calibración o test.

## Validación expansiva

La evaluación principal comprende las temporadas completas 2016–2025. Para
cada temporada `Y`:

```text
train       = fechas de temporadas <= Y-2
calibración = temporada Y-1
test        = temporada Y
```

Los tres conjuntos son disjuntos y cumplen:

```text
max(train_date) < min(calibration_date)
max(calibration_date) < min(test_date)
```

Se repite el proceso para cada `Y` y se acumulan únicamente sus predicciones
de test. Así se obtienen 266.980 predicciones OOF masculinas y 262.315
femeninas. El fold de test 2021 utiliza 2020 para calibrar; se conserva porque
el protocolo se fijó antes de ver los resultados y todavía contiene 8.987
partidos M y 10.362 F.

## Calibración Platt

Para una probabilidad cruda `p`, primero se recorta exclusivamente por
estabilidad numérica:

```text
p_clip = clip(p, 1e-6, 1 - 1e-6)
z = log(p_clip / (1 - p_clip))
```

Se ajusta una regresión logística univariable:

```text
p_calibrada = sigmoid(a * z + b)
```

El calibrador usa `C=1.000.000`, `max_iter=1.000` y semilla 42, aproximando el
Platt clásico sin una regularización material.

En cada fold, `a` y `b` se estiman solo con las predicciones de `Y-1`
producidas por el modelo entrenado hasta `Y-2`; luego se congelan para `Y`.
Las curvas y métricas del informe utilizan estas probabilidades realmente
futuras.

Para despliegue:

1. se ajusta el modelo base final con todo el histórico causal disponible;
2. el calibrador final se ajusta con las probabilidades crudas OOF temporales
   de 2016–2025, nunca con predicciones in-sample del modelo final.

## Baselines

### Favorito por ranking

Si ambos rankings existen y son distintos, gana el jugador con menor número
de ranking. Empates y ausencias no se fuerzan. Como es una decisión y no una
probabilidad, solo se reporta accuracy y cobertura.

### Ranking probabilístico

Una regresión logística univariable aprende en cada `train` la relación entre
`rank_diff` y `P(A gana)`. Permite calcular log-loss, Brier y AUC sin asignar
arbitrariamente probabilidades 0/1 al favorito determinista.

### Mercado

El baseline es `market_probability_a`, previamente de-vigada en fase 6. Solo
es evaluable cuando existen las dos cuotas. En el histórico actual su
cobertura es 0 %, por lo que todas sus métricas son nulas y el informe muestra
`NO EVALUABLE`.

Además de la cobertura nativa, el evaluador crea:

- `ranking_common_support`: todos los métodos sobre las mismas filas donde
  existe favorito de ranking;
- `market_common_support`: modelo y mercado sobre las mismas filas con cuota,
  cuando las haya.

Esto evita comparar accuracies calculadas sobre poblaciones distintas.

## Métricas

Para etiquetas `y_i` y probabilidades `p_i`:

```text
accuracy = media(1[(p_i >= 0,5) == y_i])

log-loss =
  -media(y_i * log(p_i) + (1-y_i) * log(1-p_i))

Brier = media((p_i - y_i)^2)
```

AUC mide la capacidad de ordenar positivos frente a negativos. En un segmento
con una sola clase queda nula y se marca como no definida. Todas las tablas
incluyen `n_total`, `n_evaluated` y cobertura. Las curvas de fiabilidad usan
diez bins por cuantiles y muestran probabilidad media frente a tasa observada.

Se audita automáticamente todo segmento con accuracy estrictamente superior al
85 %, sin excluir muestras pequeñas. La auditoría revisa soporte, IDs OOF,
precedencia temporal y ausencia de columnas prohibidas.

## Artefactos y reentreno

Desde `TENNIS/`:

```powershell
.venv\Scripts\python.exe scripts\retrain_models.py
```

El default infiere la última temporada cerrada común a ambos géneros. Puede
fijarse explícitamente:

```powershell
.venv\Scripts\python.exe scripts\retrain_models.py `
  --first-test-season 2016 `
  --last-test-season 2025 `
  --n-jobs 4
```

Cada run vive en:

```text
models/phase7/runs/<fingerprint>/
├── evaluation/
│   ├── oof_M.parquet
│   ├── oof_F.parquet
│   ├── metrics.csv
│   ├── folds.csv
│   ├── reliability_bins.csv
│   ├── suspicious_segments.csv
│   └── calibration_{M,F}.png
├── models/{M,F}/
│   ├── deployment_bundle.joblib
│   ├── lightgbm.joblib
│   ├── lightgbm_platt.joblib
│   ├── logistic.joblib
│   ├── logistic_platt.joblib
│   └── ranking_probability.joblib
├── report/model_report.md
└── manifest.json
```

El fingerprint incorpora los Parquet y su manifiesto, parámetros, perfil,
versiones de librerías, código del paquete y script de reentreno. El run se
publica desde staging, no se sobrescribe y todos sus archivos quedan
inventariados con tamaño y SHA-256. Una segunda ejecución idéntica verifica y
reutiliza el run.

`src.modeling.service.load_active_deployment_model()` verifica todos los hashes
antes de deserializar el bundle y devuelve la probabilidad cruda, la calibrada
y, cuando existe mercado, `edge = p_modelo - p_mercado`.

Las métricas reales y las limitaciones están en
[`model_report.md`](model_report.md).
