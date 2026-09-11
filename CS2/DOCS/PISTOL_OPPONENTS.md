# Pistols ajustadas por fuerza del rival

Extensión optativa de [pistols y conversiones](PISTOL_ROUNDS.md). La pregunta es
si ganar más pistols **de lo esperable contra esos rivales** mejora la predicción
del ganador del partido. No se asume que una pistol valga tres rondas ni que una
relación entre Elo y pistols aporte información nueva al modelo de partidos.

## Dos niveles causales

`MODEL/cs2model/pistol_opponents.py` reconstruye el Elo general con la actualización
existente y congela el estado al inicio de cada día. Recorre todo el histórico,
no solo los partidos con rondas. Cruza cada pistol por ID nativo de partido,
equipos y fecha exacta. Solo cuenta una observación por mapa/mitad, no las dos
orientaciones del mismo resultado; las prórrogas no aportan pistols.

1. Un modelo logístico auxiliar aprende la relación entre diferencia de Elo y
   resultado de la pistol. Usa una ventana de 90 días y requiere 100 pistols de
   al menos 20 partidos. Se ajusta una vez por día con hechos cuyo día de juego
   **y captura original** son anteriores al día predicho. Semilla 42, `C=1`,
   máximo 300 iteraciones. No hay splits aleatorios.
2. Para una ronda histórica, congela la expectativa calculada con ese modelo
   anterior al resultado. Cuando su captura pasa a estar disponible, incorpora
   la diferencia entre resultado y expectativa al historial del equipo. Nunca
   calcula ese residuo con un auxiliar entrenado sobre la propia ronda.

La orientación CT y el mapa de una ronda pasada permiten evaluar su dificultad:

```text
p_CT = sigmoid(intercepto_CT + efecto_mapa + beta * (Elo_CT - Elo_T) / 400)
```

El mapa/lado real solo contextualiza una observación histórica ya disponible.
No se introduce el veto ni el lado inicial del partido objetivo como feature.
El diagnóstico de predicción de pistols tampoco usa su mapa o lado futuro:
promedia las dos orientaciones CT/T con el intercepto global, sin efecto del mapa
objetivo. Esto es una aproximación neutral, no una predicción condicionada al
mapa ni una marginalización aprendida de su distribución futura.

El ajuste de cada equipo usa una actualización logística regularizada:

```text
offset = clip(sum(resultado - expectativa) /
              (sum(expectativa * (1 - expectativa)) + 5), -1.5, 1.5)
```

El término 5 equivale a la información de 20 ensayos neutrales Bernoulli de
probabilidad 0,5. Evita sobrevalorar unas pocas victorias. Ganar el 50% contra
oponentes que dejaban una expectativa del 30% recibe más crédito que ganar el
50% cuando se esperaba el 70%. También cuentan las derrotas inesperadas.

Para un enfrentamiento futuro se añade `offset_A - offset_B` a la fuerza
logística del Elo y se promedian CT/T. Si alguno no reúne 10 pistols con
expectativas causales, el ajuste es cero y se emite disponibilidad falsa. Siempre
se obtiene una salida finita. Intercambiar A/B complementa la probabilidad.

Solo tres columnas nuevas entran en el challenger de partidos:

- `pistol_opponent_edge_diff`: probabilidad ajustada menos probabilidad basada
  únicamente en Elo; es la señal incremental, no una segunda copia del Elo.
- `pistol_opponent_available`: cobertura suficiente en ambos equipos.
- `pistol_opponent_sample_min`: mínimo de observaciones causales entre equipos.

Las demás columnas, expectativas y tamaños se conservan como diagnóstico. El
modelo de partidos se entrena con **todas** las filas, incluidas las no cubiertas.

## Disponibilidad y reproducción

- Se exige `played_day < D` y `original_capture_day < D`.
- Una descarga actual con fechas antiguas en la URL no se retrofecha.
- Un desacuerdo entre fecha del mapa y fecha canónica del partido se excluye y
  contabiliza. No se reescribe la base para forzar el cruce.
- Todo el día D usa el mismo estado, incluso con horas de partido precisas.
- El primer bloque sin auxiliar entrenado no inventa expectativas históricas;
  puede entrenar futuros auxiliares, pero no rellenar residuos anteriores.
- Añadir resultados futuros o una captura tardía no cambia features anteriores.
- La huella incluye fuente original, eventos, Elo previo y versión/política del
  cálculo. No se escribe en `prediction_ledger` ni se modifica su historia.

La opción `build_training_frame(..., freeze_civil_day=True)` aplica también el
corte diario a las features generales de este experimento. El comportamiento
legacy intradía para partidos con hora exacta **no se cambia globalmente**. Un
artefacto nuevo identificado por `pistol_opponent_version` reconstruye en vivo
el mismo estado diario desde la base canónica; un fallo de ese contrato se
propaga y no queda oculto por un predictor de reserva.

## Experimento y puerta

Desde `CS2/`:

```powershell
.\.venv\Scripts\python.exe MODEL/run_pistol_ablation.py --opponent-adjusted --coverage-only
.\.venv\Scripts\python.exe MODEL/run_pistol_ablation.py --opponent-adjusted
```

Se comparan el vivo, un control reentrenado sin nuevas features, tasas simples y
el ajuste por rival. Misma arquitectura, pesos, método de calibración, prefijo
de entrenamiento, filas del hold-out y tratamiento de ausentes. La calibración
separa días completos y utiliza la selección por cobertura documentada en el
experimento de pistols original; no se escoge mirando resultados del challenger.

El evaluador y la puerta usan el promedio simétrico A/B–B/A de producción,
incluida su calibración. La prueba de integración contrasta este evaluador con
la ruta real de inferencia y un artefacto serializado, tolerancia `1e-12`. Se
registra el hash del conjunto y se verifica igualdad entre candidatos. El Elo
diario y esta puntuación simétrica explican por qué la referencia puede diferir
del experimento anterior, que utilizaba puntuación direccional: no son una
promoción ni un cambio del modelo guardado.

El periodo hasta **2026-09-09** ya se examinó antes de concretar esta hipótesis.
Se etiqueta exploratorio. `--promote` exige además una puerta favorable en un
conjunto nuevo, posterior a esa fecha, con al menos el mínimo configurado
(`promotion.min_common_holdout_rows`, actualmente 100). No basta con que mejore
el conjunto reutilizado. Se conserva la puerta compartida por log-loss y Brier,
sus márgenes anti-churn y las validaciones de salud/serialización. Accuracy y
ECE se reportan, pero no autorizan una promoción. No se reajustan hiperparámetros
en respuesta a los resultados de esta prueba.

Los CI95 usan bootstrap **emparejado por día**, 2.000 réplicas y semilla 42; no
suponen independencia de las pistols de un mismo partido. Son intervalos
exploratorios, sin corrección por multiplicidad. Una pendiente positiva de Elo
no demuestra que el residuo por equipo sea estable ni que mejore el ganador del
partido. La cobertura de mapas detallados puede sesgar la muestra.

La ejecución normal sigue recogiendo las rondas mediante el flujo existente.
No entrena este experimento automáticamente ni activa sus columnas en el modelo
actual. Una ejecución rechazada no crea copias de modelos en el registro.

Informe y resultados: `MODEL/results/pistol_opponents/REPORT.md` y
`ablation.json` (métricas, intervalos, cobertura, decisiones y ejemplos).

El contrato de nombres de columnas reside en el módulo ligero `pistols.py`.
La ingesta desde el entorno aislado del scraper no carga el estimador auxiliar
ni exige NumPy/scikit-learn para importar esas columnas. Una regresión en
subproceso bloquea expresamente esos imports durante el arranque de BBDD;
los cálculos, cortes temporales y artefactos de producción no cambian.
