# Player Snapshot Ablation

Generado: 2026-07-13T06:16:57.359657Z

## Diseno

Comparacion walk-forward con bloques cronologicos expansivos. Ambos brazos usan exactamente los mismos partidos aptos, cortes y filas de entrenamiento; solo cambia la inclusion de las columnas de snapshots de jugador.

- Cerrados totales: 9548
- Aptos point-in-time: 117
- Evaluados en comun fuera de muestra: 57 (2026-07-07 a 2026-07-12)
- Logistic regularizada; 57 columnas base + 18 de jugadores.

## Resultado

| Brazo | Accuracy | Log loss | Brier | n |
|---|---:|---:|---:|---:|
| Base sin jugadores | 56.140% | 0.8706 | 0.3017 | 57 |
| Base + snapshots | 54.386% | 1.2253 | 0.3307 | 57 |
| Delta jugador - base | -1.754% | +0.3547 | +0.0290 | |

## Incertidumbre

- Bootstrap pareado 95% del delta de accuracy: [-15.789%, +10.526%].
- Bootstrap pareado 95% del delta de log loss: [+0.0094, +0.7697].
- McNemar exacto: jugador acierta solo=7, base acierta solo=8, p=1.0000.

No se cambia produccion con este resultado: la muestra es exploratoria y el umbral operativo sigue siendo 200 partidos cerrados con snapshots point-in-time aptos.
