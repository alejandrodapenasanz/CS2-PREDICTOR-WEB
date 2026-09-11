# Informe de modelos — fase 7

## Resultado reproducible

Se entrenó un LightGBM principal y una regresión logística de referencia para cada género. La evaluación es exclusivamente temporal y las probabilidades se calibran con Platt.

```text
model_fingerprint: d080b15c768228596b17f613a45b0757d790b193ce2380674fa1c4b73f71c1a5
feature_fingerprint: 58c7bc022b2be6a4a6f5c63179fc04ff65fe363c08d54424bf5ec65b2107f69a
source_commit: 83733587353df8a41f2fd4f516147d5aa83f5a8d
feature_profile: sports_only
test_seasons: 2016-2025
fold Y: train <= Y-2; calibración = Y-1; test = Y
```

| Género | Filas fuente | Filas entrenamiento final | Excluidas no disponibles | Última fecha fuente | Disponibilidad máxima | Corte de reentreno | Predicciones OOF | Folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M | 959020 | 959020 | 0 | 2026-06-01 | 2026-06-22 | 2026-08-31 | 277270 | 10 |
| F | 769835 | 769835 | 0 | 2026-06-02 | 2026-06-23 | 2026-08-31 | 270604 | 10 |

Los hiperparámetros fueron fijados antes de observar los bloques de test. No se hizo selección ni early stopping con calibración/test.

## Resultados globales — hombres

| Método | N | Cobertura | Accuracy | Log-loss | Brier | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Logística sin calibrar | 277.270 | 100,00 % | 0,6886 | 0,580802 | 0,198974 | 0,7597 |
| Logística + Platt | 277.270 | 100,00 % | 0,6886 | 0,580451 | 0,198861 | 0,7599 |
| LightGBM sin calibrar | 277.270 | 100,00 % | 0,6908 | 0,577575 | 0,197668 | 0,7631 |
| LightGBM + Platt | 277.270 | 100,00 % | 0,6908 | 0,577382 | 0,197600 | 0,7632 |
| Ranking probabilístico | 251.296 | 90,63 % | 0,6489 | 0,634632 | 0,221509 | 0,6981 |
| Mercado de-vigado | 0 | 0,00 % | — | — | — | — |
| Favorito por ranking | 251.199 | 90,60 % | 0,6490 | — | — | — |

### Efecto de la calibración

| Modelo | Brier antes | Brier después | Δ Brier | Log-loss antes | Log-loss después | Δ log-loss |
| --- | --- | --- | --- | --- | --- | --- |
| Regresión logística | 0.19897417935617784 | 0.19886060276747064 | -0.00011357658870719822 | 0.580801768208925 | 0.5804513918645158 | -0.00035037634440926446 |
| LightGBM | 0.19766792272381253 | 0.19759985768729382 | -6.806503651871076e-05 | 0.5775753077412382 | 0.5773820898907125 | -0.0001932178505257287 |

![Curva de calibración M](assets/model_calibration_M.png)

Un delta negativo significa mejora. La curva usa diez bins por cuantiles y acumula únicamente predicciones OOF.

### Comparación justa con ranking

| Método | N común | Accuracy | Log-loss | Brier | AUC |
| --- | --- | --- | --- | --- | --- |
| Logística + Platt | 251.199 | 0,6770 | 0,594329 | 0,204657 | 0,7444 |
| LightGBM + Platt | 251.199 | 0,6790 | 0,591737 | 0,203581 | 0,7474 |
| Ranking probabilístico | 251.199 | 0,6490 | 0,634610 | 0,221498 | 0,6981 |
| Favorito por ranking | 251.199 | 0,6490 | — | — | — |

Esta tabla contiene exactamente los partidos donde el favorito por ranking es observable; no penaliza rankings ausentes.

### LightGBM calibrado por nivel × superficie

| Nivel | Superficie | N | Accuracy | Log-loss | Brier | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| ATP Tour | Clay | 9.074 | 0,6466 | 0,621424 | 0,216628 | 0,7079 |
| ATP Tour | Grass | 2.423 | 0,6508 | 0,631665 | 0,220482 | 0,6989 |
| ATP Tour | Hard | 16.998 | 0,6614 | 0,608045 | 0,210917 | 0,7258 |
| Challenger | Carpet | 98 | 0,6531 | 0,645191 | 0,227482 | 0,6621 |
| Challenger | Clay | 37.245 | 0,6611 | 0,605558 | 0,210194 | 0,7268 |
| Challenger | Grass | 1.413 | 0,6561 | 0,614778 | 0,214258 | 0,7141 |
| Challenger | Hard | 40.621 | 0,6595 | 0,609210 | 0,211279 | 0,7249 |
| Grand Slam | Clay | 2.383 | 0,7067 | 0,560359 | 0,190684 | 0,7818 |
| Grand Slam | Grass | 2.145 | 0,6853 | 0,584743 | 0,200768 | 0,7549 |
| Grand Slam | Hard | 4.658 | 0,6900 | 0,579889 | 0,198885 | 0,7596 |
| ITF | Carpet | 2.793 | 0,6839 | 0,587924 | 0,201863 | 0,7524 |
| ITF | Clay | 76.166 | 0,7178 | 0,548527 | 0,185189 | 0,7939 |
| ITF | Grass | 611 | 0,7349 | 0,534659 | 0,178532 | 0,8089 |
| ITF | Hard | 78.495 | 0,7067 | 0,563125 | 0,191128 | 0,7799 |
| Other | Clay | 63 | 0,8095 | 0,475995 | 0,156259 | 0,8591 |
| Team | Carpet | 19 | 0,7895 | 0,553521 | 0,168455 | 0,8222 |
| Team | Clay | 543 | 0,7274 | 0,525285 | 0,177252 | 0,8120 |
| Team | Grass | 44 | 0,6818 | 0,572874 | 0,196402 | 0,7663 |
| Team | Hard | 1.425 | 0,7326 | 0,513508 | 0,173562 | 0,8189 |
| Team | Unknown | 53 | 0,6604 | 0,555849 | 0,188889 | 0,8036 |

## Resultados globales — mujeres

| Método | N | Cobertura | Accuracy | Log-loss | Brier | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Logística sin calibrar | 270.604 | 100,00 % | 0,7018 | 0,569609 | 0,192731 | 0,7757 |
| Logística + Platt | 270.604 | 100,00 % | 0,7018 | 0,567630 | 0,192738 | 0,7757 |
| LightGBM sin calibrar | 270.604 | 100,00 % | 0,7039 | 0,561440 | 0,191031 | 0,7797 |
| LightGBM + Platt | 270.604 | 100,00 % | 0,7039 | 0,561391 | 0,191027 | 0,7797 |
| Ranking probabilístico | 189.768 | 70,13 % | 0,6433 | 0,640181 | 0,224293 | 0,6877 |
| Mercado de-vigado | 0 | 0,00 % | — | — | — | — |
| Favorito por ranking | 189.623 | 70,07 % | 0,6435 | — | — | — |

### Efecto de la calibración

| Modelo | Brier antes | Brier después | Δ Brier | Log-loss antes | Log-loss después | Δ log-loss |
| --- | --- | --- | --- | --- | --- | --- |
| Regresión logística | 0.19273064993462882 | 0.19273750944498597 | 6.8595103571433835e-06 | 0.56960867952034 | 0.567630280805884 | -0.001978398714456042 |
| LightGBM | 0.19103125879159671 | 0.19102731914317875 | -3.939648417966568e-06 | 0.5614400357046458 | 0.5613907876882823 | -4.9248016363523384e-05 |

![Curva de calibración F](assets/model_calibration_F.png)

Un delta negativo significa mejora. La curva usa diez bins por cuantiles y acumula únicamente predicciones OOF.

### Comparación justa con ranking

| Método | N común | Accuracy | Log-loss | Brier | AUC |
| --- | --- | --- | --- | --- | --- |
| Logística + Platt | 189.623 | 0,6812 | 0,591996 | 0,203065 | 0,7488 |
| LightGBM + Platt | 189.623 | 0,6820 | 0,587635 | 0,201987 | 0,7516 |
| Ranking probabilístico | 189.623 | 0,6435 | 0,640140 | 0,224274 | 0,6878 |
| Favorito por ranking | 189.623 | 0,6435 | — | — | — |

Esta tabla contiene exactamente los partidos donde el favorito por ranking es observable; no penaliza rankings ausentes.

### LightGBM calibrado por nivel × superficie

| Nivel | Superficie | N | Accuracy | Log-loss | Brier | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Challenger | Clay | 3.670 | 0,6689 | 0,591719 | 0,204528 | 0,7429 |
| Challenger | Grass | 221 | 0,6516 | 0,624986 | 0,217388 | 0,7113 |
| Challenger | Hard | 3.167 | 0,6618 | 0,603152 | 0,209548 | 0,7283 |
| Grand Slam | Clay | 2.240 | 0,7031 | 0,577076 | 0,196984 | 0,7662 |
| Grand Slam | Grass | 2.066 | 0,6946 | 0,594001 | 0,203897 | 0,7484 |
| Grand Slam | Hard | 4.572 | 0,6918 | 0,587954 | 0,201585 | 0,7539 |
| ITF | Carpet | 5.379 | 0,7122 | 0,556638 | 0,188368 | 0,7871 |
| ITF | Clay | 105.335 | 0,7121 | 0,551317 | 0,186780 | 0,7899 |
| ITF | Grass | 1.622 | 0,6794 | 0,589796 | 0,203570 | 0,7466 |
| ITF | Hard | 111.696 | 0,7090 | 0,555935 | 0,188652 | 0,7855 |
| Other | Clay | 64 | 0,6406 | 0,607005 | 0,211683 | 0,7041 |
| Other | Hard | 127 | 0,6535 | 0,629900 | 0,218931 | 0,7085 |
| Team | Clay | 647 | 0,7450 | 0,497817 | 0,165591 | 0,8369 |
| Team | Grass | 4 | 0,5000 | 1,138026 | 0,353000 | 0,7500 |
| Team | Hard | 1.264 | 0,7326 | 0,513845 | 0,172810 | 0,8208 |
| Team | Unknown | 93 | 0,7419 | 0,484827 | 0,164536 | 0,8310 |
| WTA Tour | Clay | 7.283 | 0,6628 | 0,605818 | 0,209733 | 0,7294 |
| WTA Tour | Grass | 2.432 | 0,6521 | 0,615725 | 0,214318 | 0,7156 |
| WTA Tour | Hard | 18.722 | 0,6649 | 0,604505 | 0,209612 | 0,7293 |

## Comparación con el mercado

| Género | Filas OOF | Con mercado | Cobertura | Estado |
| --- | --- | --- | --- | --- |
| M | 277270 | 0 | 0,00 % | NO EVALUABLE — cobertura histórica 0 % |
| F | 270604 | 0 | 0,00 % | NO EVALUABLE — cobertura histórica 0 % |

Sackmann no contiene cuotas históricas y la fase 6 declara `historical_odds_available=false`. Por ello, el favorito por cuota y el valor añadido frente al mercado son **NO EVALUABLES** en este run. No se imputó ninguna cuota. El evaluador ya genera la comparación sobre soporte común cuando existan observaciones de-vigadas capturadas antes del partido.

## Auditoría de accuracy superior al 85 %

| Género | test_season | Método | Nivel | Superficie | N | Accuracy | N<200 | Conclusión |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M | ALL | Logística + Platt | Team | Carpet | 17 | 0,8824 | sí | muestra pequeña; no se detectó fuga estructural |
| M | ALL | LightGBM + Platt | Team | Carpet | 17 | 0,8824 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2019 | Logística sin calibrar | Team | Grass | 3 | 1,0000 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2021 | Logística sin calibrar | Team | Grass | 6 | 1,0000 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2022 | Logística sin calibrar | Team | Grass | 7 | 0,8571 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2024 | Logística sin calibrar | Team | Grass | 3 | 1,0000 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2019 | Logística + Platt | Team | Grass | 3 | 1,0000 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2021 | Logística + Platt | Team | Grass | 6 | 1,0000 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2022 | Logística + Platt | Team | Grass | 7 | 0,8571 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2024 | Logística + Platt | Team | Grass | 3 | 1,0000 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2019 | LightGBM sin calibrar | Team | Grass | 3 | 1,0000 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2021 | LightGBM sin calibrar | Team | Grass | 6 | 1,0000 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2022 | LightGBM sin calibrar | Team | Grass | 7 | 0,8571 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2024 | LightGBM sin calibrar | Team | Grass | 3 | 1,0000 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2019 | LightGBM + Platt | Team | Grass | 3 | 1,0000 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2021 | LightGBM + Platt | Team | Grass | 6 | 1,0000 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2022 | LightGBM + Platt | Team | Grass | 7 | 0,8571 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2024 | LightGBM + Platt | Team | Grass | 3 | 1,0000 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2019 | Ranking probabilístico | Team | Grass | 3 | 1,0000 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2021 | Ranking probabilístico | Team | Grass | 5 | 1,0000 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2023 | Ranking probabilístico | Team | Grass | 2 | 1,0000 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2024 | Ranking probabilístico | Team | Grass | 2 | 1,0000 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2019 | Favorito por ranking | Team | Grass | 3 | 1,0000 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2021 | Favorito por ranking | Team | Grass | 5 | 1,0000 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2023 | Favorito por ranking | Team | Grass | 2 | 1,0000 | sí | muestra pequeña; no se detectó fuga estructural |
| M | 2024 | Favorito por ranking | Team | Grass | 2 | 1,0000 | sí | muestra pequeña; no se detectó fuga estructural |
| F | 2020 | Logística sin calibrar | Team | Clay | 38 | 0,8684 | sí | muestra pequeña; no se detectó fuga estructural |
| F | 2020 | Logística + Platt | Team | Clay | 38 | 0,8684 | sí | muestra pequeña; no se detectó fuga estructural |
| F | 2020 | LightGBM sin calibrar | Team | Clay | 38 | 0,8947 | sí | muestra pequeña; no se detectó fuga estructural |
| F | 2020 | LightGBM + Platt | Team | Clay | 38 | 0,8947 | sí | muestra pequeña; no se detectó fuga estructural |

Todos los casos se conservaron. La auditoría comprobó IDs OOF únicos, precedencia temporal y ausencia de objetivo, identidades o postdicciones entre las features.

## Limitaciones

- No existen cuotas históricas causales: todavía no puede demostrarse ventaja sobre el mercado ni entrenarse el perfil `market_enhanced`.
- `tourney_date` suele ser el inicio del torneo, no la fecha real del partido. El sistema aplica un embargo conservador de 21 días y exige disponibilidad estrictamente anterior al corte. Esto evita la fuga conocida entre torneos solapados, pero no demuestra causalidad perfecta para una excepción que dure más de 21 días; esos casos no deben interpretarse como evidencia fuerte.
- El fold de test 2021 se calibra con la temporada atípica 2020. Se conserva porque mantiene el protocolo y tiene miles de casos.
- La cobertura de rankings, especialmente femenina, varía con el tiempo. Las comparaciones usan soporte común.
- Hay colisiones históricas de IDs/fechas de nacimiento. La DOB del maestro actual no se usa como filtro retroactivo porque sería información futura; las claves se bloquean en operación actual, pero puede quedar contaminación histórica residual. Los nulos y rankings antiguos se señalan sin inventar datos.
- 2026 es parcial y no forma parte del backtest principal. Sí entra al estimador final solo cuando su fecha de disponibilidad es estrictamente anterior al corte de reentreno.
- Los hiperparámetros son conservadores y fijos. La fase no presenta un barrido de hiperparámetros como si fuera evidencia independiente.

## Artefactos detallados

Las métricas completas, predicciones OOF, folds, calibradores y modelos están en `../models/phase7/runs/d080b15c768228596b17f613a45b0757d790b193ce2380674fa1c4b73f71c1a5`. `metrics.csv` incluye todos los modelos y segmentos, además de las poblaciones de soporte común.
