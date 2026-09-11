# Experimento de pistols y conversiones — 2026-09-09

## Veredicto

**No se promociona ninguna variante.** El modelo vivo conserva mejor log-loss y
Brier. Este resultado no descarta el valor potencial de las pistols: prueba que
las dos familias evaluadas, con esta cobertura causal y arquitectura, no justifican
sustituir producción. No se siguen probando ventanas contra este mismo hold-out
hasta obtener una mejora aparente.

## Evidencia recuperada

Se revisaron 3.191 capturas HTML existentes. Se incorporaron 3.163 capturas válidas
y 67.634 filas de rondas a tablas laterales nuevas. Al deduplicar capturas por mapa,
el constructor usa **3.158 mapas MR12 distintos**. No se modificaron las tablas de
partidos, mapas, equipos, snapshots originales ni el ledger.

Quedaron fuera 17 capturas sin correspondencia inequívoca de mapa y 11 con IDs del
partido/equipos contradictorios con la base. Los originales siguen conservados;
el detalle está en [ingestion.json](ingestion.json). No se corrigieron identidades
por nombre ni se inventaron timestamps.

La consulta agregada de HLTV sugerida por el usuario no se usó para rellenar el
pasado: filtrar fechas en una descarga actual no demuestra disponibilidad histórica.
Se usaron exclusivamente las capturas originales verificadas. No se ha verificado
que el parámetro Top200 equivalga a todos los equipos.

## Diseño y cobertura

Se construyeron cronológicamente las features de **10.905 partidos**, antes de
aplicar cualquier máscara. Ventana de estadísticas: 90 días; disponibilidad y
fecha del mapa estrictamente anteriores al día predicho; mínimo 10 pistols por
equipo, con suavizado fijo de 20 observaciones a media 0,5.

| Bloque | N total | N con historial suficiente de ambos equipos |
| --- | ---: | ---: |
| Historial completo | 10.905 | 825 |
| Prefijo hasta 2026-08-10 | 10.090 | 247 |
| Ajuste base hasta 2026-08-03 | 9.929 | 169 |
| Calibración 2026-08-04 a 2026-08-10 | 161 | 78 |
| Hold-out 2026-08-11 a 2026-09-09 | 815 | 578 |

El hold-out tiene 70,92% de cobertura y 29,08% sin cobertura suficiente. La
limitación es el ajuste base: solo 169 partidos aportan una diferencia informativa
de pistols. Las demás filas se conservan, con indicador de ausencia/cobertura.

La fracción de calibración habitual dejaba **cero** filas con cobertura en el
ajuste; 14 días dejaban 78; 7 días dejaban 169. Se eligió esta última partición
solo por cobertura, antes de ajustar o evaluar challengers. Las tres variantes
comparten ese corte y calibración sigmoid. Una calibración de solo 161 partidos
es una limitación, no una garantía de generalización.

Se conserva la arquitectura vigente `model_a_no_odds` y los pesos de sus
componentes. El vivo es el artefacto real `20260810_064827Z`, no un proxy ni un
reentreno. El control permite separar la aportación de las columnas nuevas del
efecto de volver a ajustar/calibrar el modelo.

## Mismo hold-out: resultados globales

| Variante | Aciertos/N | Accuracy | Log-loss | Brier | ECE10 | Puerta |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Vivo | 517/815 | 63,44% | 0,628280 | 0,219458 | 0,031769 | Se conserva |
| Control reentrenado, sin pistols | 514/815 | 63,07% | 0,631940 | 0,221275 | 0,057320 | Rechazado |
| Pistols | 517/815 | 63,44% | 0,631381 | 0,220988 | 0,056033 | Rechazado |
| Pistols + conversiones/recuperación | 510/815 | 62,58% | 0,640006 | 0,224441 | 0,054451 | Rechazado |

Los márgenes compartidos son 0,001 de log-loss y 0,0005 de Brier. Pistols mejora
levemente al control (−0,000558 de log-loss; +0,368 pp de accuracy), pero no supera
al vivo. Los CI95 frente al control incluyen cero:

- Delta log-loss: [−0,002346; +0,001177].
- Delta accuracy: [−0,568; +1,294] puntos porcentuales.
- Delta Brier: [−0,001085; +0,000482].

No hay evidencia estadísticamente sólida de mejora. Son intervalos exploratorios
de bootstrap emparejado por día, 2.000 réplicas, semilla 42, sin corrección por
comparaciones múltiples. Nunca se usa ese remuestreo para entrenar o validar.

## Desglose por cobertura del hold-out

| Cobertura | Variante | N | Accuracy | Log-loss | Brier | ECE10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Suficiente | Vivo | 578 | 61,25% | 0,636319 | 0,223353 | 0,050576 |
| Suficiente | Pistols | 578 | 61,94% | 0,637264 | 0,224249 | 0,074490 |
| Suficiente | Pistols + conversiones | 578 | 60,38% | 0,652061 | 0,230112 | 0,071143 |
| Insuficiente | Vivo | 237 | 68,78% | 0,608674 | 0,209959 | 0,030681 |
| Insuficiente | Pistols | 237 | 67,09% | 0,617035 | 0,213036 | 0,064656 |
| Insuficiente | Pistols + conversiones | 237 | 67,93% | 0,610604 | 0,210610 | 0,034610 |

Incluso donde hay historial, el pequeño aumento de accuracy de pistols (+0,692 pp)
no mejora log-loss. Su CI95 de accuracy frente al vivo es [−0,684; +1,945] pp.
La diferencia de dificultad entre filas con/sin cobertura impide interpretar
sus accuracies como un efecto causal de disponer de estadísticas.

Ejemplos auditables, probabilidades del primer equipo:

| Partido | Fecha | Vivo | Control | Pistols | Resultado real |
| --- | --- | ---: | ---: | ---: | --- |
| Honvéd–Inner Circle Academy | 2026-08-23 | 51,68% | 51,73% | 48,18% | Ganó Inner Circle: corrige el pick |
| Bushido Wildcats–ex-RUSTEC | 2026-08-31 | 53,33% | 50,94% | 48,68% | Ganó Bushido: empeora el pick |

En total, pistols cambia 30 picks frente al vivo: corrige 15 y estropea otros 15.
La variante completa cambia 55: corrige 24 y estropea 31. No se seleccionan solo
los ejemplos favorables.

## ¿Una pistol equivale a tres rondas?

En la ventana causal disponible al 2026-09-09:

- Ganar la segunda ronda tras ganar la pistol: **81,87%**, N=6.189.
- Ganar la tercera tras ganar la pistol: **58,73%**, N=6.118.
- Ganar las tres primeras: **53,91%**, N=6.118.
- Media de rondas ganadas de las tres primeras, tras ganar la pistol:
  **2,4045**, N=6.118 secuencias completas.

Son estadísticas descriptivas de la muestra archivada, no un efecto causal de
2,4 rondas adicionales ni una garantía de 3–0. Los mapas que terminan antes no
aportan rondas inexistentes al denominador.

## Estado ejecutable y conservación

- Ingestión ordinaria conectada: las nuevas capturas de mapas de `start.ps1`
  alimentan las tablas laterales; no se añade otra descarga ni un segundo scraper.
- Features calculadas por el mismo lector as-of en entrenamiento e inferencia;
  **siguen fuera del artefacto productivo** porque la puerta las rechazó.
- Misma cohorte en las tres puertas:
  `fe35ad58588c2890965bd7ab4f421fd4ae6878ee6fb452a5407e7aa370136005`.
- Repetir el scoring del mismo artefacto/cohorte pasa con tolerancia `1e-12`:
  máxima diferencia de probabilidad `3,33e-16`, de métricas `2,78e-17`.
- Puertas locales: Ruff lint/formato, mypy, cobertura de imports y boot en verde;
  335 tests pasados, 4 omitidos por la suite, más 33 tests adicionales de parsers
  e ingestión. Smoke determinista en verde. `pip check` sin conflictos.
- Se probó el camino offline captura → base → features → predicción → puerta,
  incluyendo ausencia de historial, captura futura y backup/restauración. No se
  lanzó un `start.ps1` completo con red/publicación durante esta prueba.
- Modelo vivo `20260810_064827Z`; `last_good` `20260803_064101Z`, ambos intactos.
  No se guardaron copias de modelos rechazados.
- Ledger: 1.187 filas, mismo SHA-256 antes/después:
  `b5d77d0c5344ff90dc7f1e9178edfc7246070ed97e9d121b6fa962648a4af23f`.

Revisión de arquitectura: adquisición, persistencia y modelado separados; sin
cambios en TENNIS, WEB ni TELEGRAM. Revisión anti-fugas: doble fecha, días enteros,
warm-up completo e invariancia ante capturas/partidos futuros. Dependencias:
ninguna nueva; manifiestos sin cambios, Python 3.13 y puertas verificadas.

Detalle numérico: [ablation.json](ablation.json). Método y comandos:
[DOCS/PISTOL_ROUNDS.md](../../../DOCS/PISTOL_ROUNDS.md).
