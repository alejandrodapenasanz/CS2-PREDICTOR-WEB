# Determinismo de inferencia

La ruta productiva carga `MODEL/artifacts/model.pkl` mediante
`cs2model.artifacts.load_artifact`. Al deserializar, el runtime fija a `1` los
parámetros de paralelismo soportados (`n_jobs`, `thread_count`, `num_threads` y
`nthread`) en el estimador y en los estimadores ajustados que viven dentro de
pipelines o calibradores. También limita a un hilo los runtimes OpenMP/BLAS
mediante variables de entorno. Esta política solo modifica el objeto en memoria:
no reescribe el pickle ni cambia la receta con la que se entrenó.

Dos predicciones de la misma fila, con el mismo artefacto y las mismas features,
deben diferir como máximo `1e-12` en valor absoluto. Normalmente son idénticas bit
a bit en un mismo proceso; la tolerancia cubre el último ruido de coma flotante
entre procesos, versiones nativas o CPU compatibles. Una diferencia mayor se
considera una regresión y debe bloquear la publicación.

La regresión está en `TESTS/test_inference_determinism.py`: verifica el camino de
carga con un estimador originalmente configurado con `n_jobs=-1`, repite la misma
fila en el proceso actual y vuelve a puntuarla desde dos intérpretes nuevos.
