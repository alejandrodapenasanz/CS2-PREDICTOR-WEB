# Pistols ajustadas por rival — 2026-09-10

Estado: comparación terminada a las 08:48 del 2026-09-10. Ninguna variante supera
la puerta frente al vivo; producción no se ha sustituido.

## Datos y diseño

- Vivo: `20260810_064827Z`; `last_good`: `20260803_064101Z`.
- Arquitectura vigente sin odds, con sus componentes y pesos; calibración sigmoid.
- Reconstrucción completa: 10.905 partidos; prefijo hasta 2026-08-10: 10.090.
- Ajuste base: 9.929 filas hasta 2026-08-03; calibración: 161 filas de
  2026-08-04 a 2026-08-10. Todos los cortes por días completos.
- Hold-out común: 815 partidos, 2026-08-11 a 2026-09-09. Es exploratorio:
  ya se había examinado este periodo antes de concretar el ajuste por rival.
- Hay **0 partidos nuevos** posteriores a 2026-09-09; se requieren al menos
  100 y una puerta favorable adicional antes de poder promocionar.
- Mismo Elo congelado al inicio de D, mismas filas, ausentes y promedio A/B–B/A
  de producción para vivo y challengers. No comparar directamente con los
  números anteriores que utilizaban scoring direccional y estado intradía.

La puerta compartida usa epsilon log-loss 0,001 y epsilon Brier 0,0005. Accuracy
y ECE son métricas reportadas, no el criterio de promoción. Fórmula y contrato:
[PISTOL_OPPONENTS.md](../../../DOCS/PISTOL_OPPONENTS.md).

## Cobertura causal

3.158 mapas con pistols en la fuente; 6.064 pistols admitidas para el cruce por
rival y 252 excluidas por conflicto de fecha/disponibilidad (126 mapas). Algunos
ejemplos cruzan la medianoche: la captura del mapa 232280 indica
2026-07-01T22:00Z y el partido 2395573 figura como 2026-07-02T00:00Z. No se ha
corregido ni sobrescrito la fecha de la base para recuperar esas observaciones.

| Bloque de partidos | Total | Con ajuste disponible | Cobertura |
| --- | ---: | ---: | ---: |
| Histórico completo | 10.905 | 774 | 7,10% |
| Ajuste base del modelo | 9.929 | 134 | 1,35% |
| Calibración | 161 | 67 | 41,61% |
| Hold-out común | 815 | 573 | 70,31% |

Las 242 filas del hold-out sin cobertura no se eliminan. La escasez en el
entrenamiento y el cambio de cobertura limitan mucho la generalización. La
calibración de una sola semana también puede ser inestable. No se han acortado
ventanas ni retocado hiperparámetros para mejorar los resultados observados.

## ¿Predice mejor quién gana la pistol?

Evaluación secuencial con estado previo a cada día: 3.092 pistols de 669 partidos
y 1.546 mapas, entre 2026-08-11 y 2026-09-06. Una fila por pistol. No usa el mapa
ni el lado inicial real del objetivo.

| Predictor de pistol | Accuracy | Log-loss | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| Solo relación Elo→pistol | 51,84% | 0,692531 | 0,249694 | 0,012346 |
| Elo + rendimiento residual por equipo | 50,36% | 0,702850 | 0,254639 | 0,055441 |
| Tasas simples | 49,90% | 0,704329 | 0,255427 | 0,059317 |
| Probabilidad constante 50% | 50,94% | 0,693147 | 0,250000 | 0,009379 |

El 50,94% de la constante depende del desempate por orientación de ID; no es
capacidad predictiva. Elo por sí solo mejora muy poco su log-loss.

Para el ajuste residual frente a solo Elo, CI95 emparejado por día (2.000 réplicas,
semilla 42): accuracy **−2,80 a −0,41 pp**; delta log-loss **+0,00595 a +0,01447**;
delta Brier **+0,00288 a +0,00688**. En esta muestra, el ajuste probado empeora.
Son intervalos exploratorios y sin corrección por comparar variantes.

El auxiliar ajustado a 2026-09-10 usa 6.064 pistols de 1.317 partidos y estima
`beta=0,60952` por 400 puntos de Elo. Sin conocer mapa/lado, su probabilidad es
57,56% con diferencia +200 y 64,78% con +400. Es una relación estimada en esta
muestra, **no** una garantía de acertar esas pistols ni evidencia de mejora en
el ganador del partido. El Elo ya está presente en ese modelo.

## Comparación del ganador del partido

| Modelo | Accuracy | Log-loss | Brier | ECE | Puerta |
| --- | ---: | ---: | ---: | ---: | --- |
| Vivo | 63,19% | 0,628462 | 0,219508 | 0,028899 | Se conserva |
| Control reentrenado | 62,82% | 0,632516 | 0,221495 | 0,058383 | Rechazado |
| Tasas simples de pistols | 63,44% | 0,632087 | 0,221285 | 0,050165 | Rechazado |
| Pistols ajustadas por rival | 63,19% | 0,632114 | 0,221357 | 0,052426 | Rechazado |

La variante ajustada empata en accuracy con el vivo, pero empeora log-loss
en +0,003653 y Brier en +0,001850. Frente al control mejora muy poco: delta
log-loss −0,000402, con CI95 [−0,001395; +0,000655], que incluye cero. No hay
evidencia de mejora atribuible al ajuste por rival en el ganador del partido.

En los 573 partidos con cobertura, el vivo acierta 61,08% y la variante ajustada
61,43%, pero el log-loss empeora de 0,634222 a 0,636205. En los 242 sin cobertura,
accuracy 68,18% frente a 67,36% y log-loss 0,614822 frente a 0,622428. Estos grupos
se definen por cobertura del ajuste por rival, igual para todas las variantes.

Ejemplos ilustrativos, no prueba de beneficio: METANOIA Wolves–MEIA NOITE
(2026-08-15), probabilidad de team1 49,75%→52,22%, cambio acertado;
Bushido Wildcats–Endless Journey (2026-08-19), 51,87%→48,99%, cambio fallado.
Los ejemplos incluyen el efecto del reentreno; no se atribuye todo el cambio
a una sola feature. El informe JSON contiene todos los picks que cambian.

La puerta usa el mismo conjunto, SHA-256
`fe35ad58588c2890965bd7ab4f421fd4ae6878ee6fb452a5407e7aa370136005`.
Al repetir cada artefacto sobre las mismas filas, diferencia máxima de
probabilidad 2,22e−16, inferior a la tolerancia 1e−12. No se ha promocionado nada,
ni se han creado carpetas de challengers rechazados. La recogida causal continúa;
estas features permanecen fuera del modelo activo.

## Verificación

- 344 tests de la puerta CS2 y 33 pruebas adicionales del parser/pipeline:
  **377 aprobados**, 4 omitidos por condiciones existentes; 17 subtests.
- Ruff lint/formato, mypy (24 módulos), cobertura de imports y `pip check`: OK.
- Smoke aislado: `smoke_pipeline=ok`, 160 filas, semilla 42, dataset
  `3a45dacdaa5d`; boots CLI de entrenamiento, gestión y pipeline: OK.
- Pruebas nuevas: rivales fuertes/débiles, regularización con pocos datos,
  congelación diaria, adición de futuro, captura tardía, ausencia, simetría,
  igualdad evaluación/inferencia/serialización y arranque sin histórico legacy.
- No se ha ejecutado el scraping de red ni la publicación raíz de `start.ps1`;
  la prueba de punta a punta usa capturas/fixtures offline y no publica apuestas.
- Base: `integrity_check=ok`, sin errores de claves foráneas. Ledger: 1.187 filas,
  SHA-256 `b5d77d0c5344ff90dc7f1e9178edfc7246070ed97e9d121b6fa962648a4af23f`,
  idéntico en las comprobaciones de la investigación anteriores al nuevo run
  operativo del usuario. Modelo vivo y punteros mantienen sus hashes.
- Sin dependencias nuevas ni copias adicionales en el registro.

Revisión de arquitectura: cálculo en MODEL y adaptador en PIPELINE, sin imports
de scraper ni cambios en TENNIS/WEB/TELEGRAM. Revisión anti-fugas: corte civil y
captura original, entrenamiento/calibración temporales, semilla 42 y ledger
intacto. Revisión de dependencias: manifiestos sin cambios, imports declarados
y puertas locales verificadas con CPython 3.13.

## Ejecución operativa posterior

El usuario ejecutó `start.ps1` a las 08:49, después de finalizar este experimento.
Su ingesta incorporó información nueva: no se mezclan esos datos con la tabla
cerrada de 815 partidos. Al comenzar la reparación de su error de arranque, el
ledger seguía teniendo 1.187 filas, con hash
`857b68a92f94f14542978cd376bc596885863c2d5b8bf6c6ad0c1c61390632ba`.
La reparación no reescribe esa historia. Se corrige por separado el escaneo de
carpetas inaccesibles y se alinea scikit-learn con los modelos guardados.
La reparación posterior pasa 383 pruebas y smoke; véase
[informe de arranque](../../../DOCS/START_REPAIR_20260910.md).
