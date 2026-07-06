# Context Calibration Diagnostic

Generado: 2026-07-06T13:34:06Z  
Muestras cerradas con prediccion pre-partido: **90**  
Muestras con contexto HLTV: **90**  

## Resultado

- Estado: **insufficient_sample**
- Decision: No activar calibracion contextual en produccion.
- Motivo: Solo hay 90 muestras con contexto; minimo recomendado 200.

## Global

| Metrica | Modelo |
|---|---:|
| n | 90 |
| accuracy | 0.5111 |
| log_loss | 0.7509 |
| brier | 0.2746 |
| ece_10 | 0.2114 |
| avg_confidence | 0.6668 |

## Partidos Activos Por LAN/Online

| Hora | Partido | Env | Stage | Favorito | Conf | HLTV |
|---|---|---|---|---|---:|---|
| 2026-07-06 18:00 | Fluxo vs paiN Academy | online | Round of 16 | Fluxo | 0.6545 | [link](https://www.hltv.org/matches/2395588/fluxo-vs-pain-academy-cct-2026-south-america-series-3) |
| 2026-07-06 21:00 | Fake do Biru vs VEXA | online | Round of 16 | Fake do Biru | 0.7605 | [link](https://www.hltv.org/matches/2395589/fake-do-biru-vs-vexa-cct-2026-south-america-series-3) |
| 2026-07-07 00:00 | Keyd Stars vs GameHunters | online | Round of 16 | Keyd Stars | 0.6911 | [link](https://www.hltv.org/matches/2395590/keyd-stars-vs-gamehunters-cct-2026-south-america-series-3) |
| 2026-07-07 10:00 | BASEMENT BOYS vs The Last Resort | online | Group B opening match | The Last Resort | 0.5139 | [link](https://www.hltv.org/matches/2395520/basement-boys-vs-the-last-resort-european-pro-league-series-8) |
| 2026-07-07 12:30 | PsychoFace vs Enjoy | online | Group B opening match | Enjoy | 0.5811 | [link](https://www.hltv.org/matches/2395521/psychoface-vs-enjoy-european-pro-league-series-8) |
| 2026-07-07 15:00 | Isurus vs Yawara | online | Round of 16 | Yawara | 0.5713 | [link](https://www.hltv.org/matches/2395591/isurus-vs-yawara-cct-2026-south-america-series-3) |
| 2026-07-07 18:00 | LP vs BESTIA Academy | online | Round of 16 | LP | 0.7878 | [link](https://www.hltv.org/matches/2395592/lp-vs-bestia-academy-cct-2026-south-america-series-3) |
| 2026-07-07 21:00 | UNO MILLE vs Patins da Ferrari | online | Round of 16 | UNO MILLE | 0.5253 | [link](https://www.hltv.org/matches/2395593/uno-mille-vs-patins-da-ferrari-cct-2026-south-america-series-3) |
| 2026-07-08 00:00 | Bounty Hunters vs MIBR Academy | online | Round of 16 | Bounty Hunters | 0.8136 | [link](https://www.hltv.org/matches/2395594/bounty-hunters-vs-mibr-academy-cct-2026-south-america-series-3) |
| 2026-07-08 10:00 | Atreides vs eternal premium | online | Group C opening match | Atreides | 0.6408 | [link](https://www.hltv.org/matches/2395522/atreides-vs-eternal-premium-european-pro-league-series-8) |
| 2026-07-08 12:30 | SPARTA vs ENCE | online | Group C opening match | SPARTA | 0.5831 | [link](https://www.hltv.org/matches/2395523/sparta-vs-ence-european-pro-league-series-8) |
| 2026-07-09 10:00 | TYLOO vs 9z | lan | Quarter-final | 9z | 0.6447 | [link](https://www.hltv.org/matches/2395486/tyloo-vs-9z-xse-pro-league-guangzhou-2026) |
| 2026-07-09 10:00 | GenOne vs BRUTE | online | Group D opening match | GenOne | 0.6691 | [link](https://www.hltv.org/matches/2395524/genone-vs-brute-european-pro-league-series-8) |
| 2026-07-09 12:30 | RUSTEC vs Honvéd | online | Group D opening match | RUSTEC | 0.5188 | [link](https://www.hltv.org/matches/2395525/rustec-vs-honvd-european-pro-league-series-8) |
| 2026-07-09 13:40 | Alliance vs Nemesis | lan | Quarter-final | Nemesis | 0.5894 | [link](https://www.hltv.org/matches/2395487/alliance-vs-nemesis-xse-pro-league-guangzhou-2026) |
| 2026-07-10 02:00 | regain vs Club 333 | online | Grand final | regain | 0.6082 | [link](https://www.hltv.org/matches/2395626/regain-vs-club-333-crossfire-season-5) |
| 2026-07-10 10:00 | PARIVISION vs BIG | lan | Quarter-final | PARIVISION | 0.7095 | [link](https://www.hltv.org/matches/2395488/parivision-vs-big-xse-pro-league-guangzhou-2026) |
| 2026-07-10 13:40 | FaZe vs BetBoom | lan | Quarter-final | BetBoom | 0.7535 | [link](https://www.hltv.org/matches/2395489/faze-vs-betboom-xse-pro-league-guangzhou-2026) |

## Calibrador Contextual Diagnostico

| Metrica | Base | Contexto | Delta |
|---|---:|---:|---:|
| log_loss | 0.7844 | 0.7424 | -0.0419 |
| brier | 0.2895 | 0.2654 | -0.0241 |
| accuracy | 0.4800 | 0.5200 | 0.0400 |
| ece_10 | 0.2274 | 0.1133 | -0.1141 |

Eval folds: 50 con min_train=40. Diagnostico. No activar en produccion sin 200+ muestras y grupos con 30+ partidos.

## environment

| Grupo | n | acc | log_loss | brier | fav_acc | conf_gap |
|---|---:|---:|---:|---:|---:|---:|
| online | 65 | 0.4615 | 0.7944 | 0.2942 | 0.4615 | 0.2098 |
| lan | 25 | 0.6400 | 0.6378 | 0.2237 | 0.6400 | 0.0151 |

## stage

| Grupo | n | acc | log_loss | brier | fav_acc | conf_gap |
|---|---:|---:|---:|---:|---:|---:|
| swiss | 42 | 0.5714 | 0.7080 | 0.2555 | 0.5714 | 0.1099 |
| group | 13 | 0.5385 | 0.7277 | 0.2599 | 0.5385 | 0.0934 |
| final | 10 | 0.3000 | 0.7411 | 0.2731 | 0.3000 | 0.3059 |
| semi | 8 | 0.7500 | 0.6618 | 0.2334 | 0.7500 | -0.0471 |
| quarter | 7 | 0.5714 | 0.8182 | 0.3026 | 0.5714 | 0.0885 |
| lower_bracket | 6 | 0.3333 | 0.8855 | 0.3362 | 0.3333 | 0.3584 |
| league | 4 | 0.0000 | 1.1589 | 0.4679 | 0.0000 | 0.6822 |

## high_stakes

| Grupo | n | acc | log_loss | brier | fav_acc | conf_gap |
|---|---:|---:|---:|---:|---:|---:|
| yes | 60 | 0.5333 | 0.7537 | 0.2747 | 0.5333 | 0.1459 |
| no | 30 | 0.4667 | 0.7451 | 0.2744 | 0.4667 | 0.1753 |

## incentive_label

| Grupo | n | acc | log_loss | brier | fav_acc | conf_gap |
|---|---:|---:|---:|---:|---:|---:|
| playoff_or_bracket | 31 | 0.4839 | 0.7660 | 0.2818 | 0.4839 | 0.1759 |
| group_or_swiss | 26 | 0.5000 | 0.7496 | 0.2762 | 0.5000 | 0.1478 |
| elimination_match | 16 | 0.6875 | 0.6800 | 0.2346 | 0.6875 | 0.0019 |
| winner_advances | 13 | 0.4615 | 0.8152 | 0.3073 | 0.4615 | 0.2517 |
| opening_match | 2 | 0.5000 | 0.5526 | 0.1848 | 0.5000 | 0.1266 |
| winners_match | 2 | 0.0000 | 0.8793 | 0.3412 | 0.0000 | 0.5820 |

## winner_advances

| Grupo | n | acc | log_loss | brier | fav_acc | conf_gap |
|---|---:|---:|---:|---:|---:|---:|
| no | 72 | 0.4861 | 0.7705 | 0.2829 | 0.4861 | 0.1670 |
| yes | 18 | 0.6111 | 0.6725 | 0.2416 | 0.6111 | 0.1104 |

## loser_eliminated

| Grupo | n | acc | log_loss | brier | fav_acc | conf_gap |
|---|---:|---:|---:|---:|---:|---:|
| no | 74 | 0.4730 | 0.7662 | 0.2833 | 0.4730 | 0.1890 |
| yes | 16 | 0.6875 | 0.6800 | 0.2346 | 0.6875 | 0.0019 |

## format

| Grupo | n | acc | log_loss | brier | fav_acc | conf_gap |
|---|---:|---:|---:|---:|---:|---:|
| bo3 | 73 | 0.5068 | 0.7628 | 0.2799 | 0.5068 | 0.1657 |
| bo1 | 15 | 0.5333 | 0.7223 | 0.2617 | 0.5333 | 0.1068 |
| bo5 | 2 | 0.5000 | 0.5285 | 0.1767 | 0.5000 | 0.1563 |
