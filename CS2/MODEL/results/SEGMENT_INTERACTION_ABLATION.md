# Ablación de interacciones por segmento

Ejecución: 2026-08-26. Semilla: 42. Artefacto vivo:
`20260810_064827Z`, entrenado hasta `2026-08-10`, arquitectura
`model_a_no_odds`.

## Cohorte y comparabilidad

Las tres sombras se ajustaron únicamente con el prefijo hasta el corte del
vivo. Vivo, control refit, bloque completo y variante mínima se evaluaron sobre
los mismos 324 partidos del `2026-08-11` al `2026-08-26`. El hash común es
`c93e6bddfeac5270b34b512b4ecfa78f913e6133b095a46b1ac29db3c75b5830`.

El runner comprueba con tolerancia `1e-12` que el log-loss y Brier calculados
fuera de la puerta coinciden con los usados por ella, y que cada comparación
usa el mismo vivo y el mismo hash de cohorte. Accuracy y ECE son informativos;
la promoción decide por log-loss y, dentro de epsilon, por Brier.

## Resultado global

| Variante | Accuracy | Log-loss | Brier | ECE10 | Puerta |
|---|---:|---:|---:|---:|---|
| Vivo | 62,04 % | 0,648969 | 0,228952 | 0,034577 | Referencia |
| Control refit sin interacciones | 61,73 % | 0,648685 | 0,228764 | 0,032650 | Rechazado por margen |
| Interacciones completas | 61,42 % | 0,650475 | 0,229520 | 0,042333 | Rechazado: regresión |
| Solo `elo_diff_x_stage_swiss` | 61,73 % | **0,648004** | **0,228474** | 0,042044 | Rechazado por margen |

La variante mínima es el mejor challenger, pero su mejora queda dentro del
margen anti-churn: `Δ log-loss=-0,000965` frente al mínimo `-0,001`, y
`Δ Brier=-0,000478` frente al mínimo `-0,0005`. No se creó artefacto ni se
movieron los punteros: `latest=20260810_064827Z` y
`last_good=20260803_064101Z`. La familia completa queda apagada por defecto.

## Calibración swiss

La referencia del prompt (7.914 OOS, swiss N=188, gap `+7,08 pp`) precedía a la
ejecución diaria del 26 de agosto. Tras ella hay 7.978 OOS y la señal larga
sigue siendo sólida: swiss N=214, predicha 58,08 %, real 64,95 %, gap
`+6,87 pp`, CI95 `[+0,50, +13,25] pp`, ECE10 0,093857.

Para medir el efecto antes/después se usa solo el hold-out común de la puerta:

| Variante | N swiss | Predicha | Real | Gap | CI95 | ECE10 |
|---|---:|---:|---:|---:|---:|---:|
| Vivo | 87 | 55,77 % | 55,17 % | -0,60 pp | [-10,99; +9,79] pp | 0,079498 |
| Interacciones completas | 87 | 55,95 % | 55,17 % | -0,78 pp | [-11,19; +9,63] pp | 0,083820 |
| Interacción mínima swiss | 87 | 55,84 % | 55,17 % | -0,67 pp | [-11,05; +9,70] pp | 0,093067 |

El gap largo no puede compararse como un antes directo contra estas sombras:
son periodos y objetos de evaluación distintos. En el hold-out independiente,
la señal no se repite y N=87 está por debajo del mínimo 100; ninguna variante
cierra una desviación demostrada y ambas empeoran ECE swiss.

## Hipótesis y seguimiento live

En la muestra larga actual, Elo 200–300 (N=293, CI95 del gap
`[-2,27; +6,47] pp`), stage group (N=339, `[-6,79; +2,89] pp`) y LAN
(N=259, `[-6,08; +4,50] pp`) siguen siendo hipótesis: sus intervalos del gap
medio incluyen cero. Elo 300+ tiene N=26 y sigue sin muestra suficiente.

El ledger tiene 550 predicciones evaluadas. Las decisiones por segmentos live
siguen deshabilitadas hasta 1.000; alcanzar ese umbral solo habilita revisión,
no una promoción automática.

El detalle ejecutable, métricas por régimen y fingerprints está en
`MODEL/results/segment_interaction_ablation/segment_interaction_ablation.json`.
