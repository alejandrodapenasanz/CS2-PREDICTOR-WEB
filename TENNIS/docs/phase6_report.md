# Informe de la fase 6: features causales

> **El informe original queda sustituido por este cierre de fase 9.** El run
> anterior (`tennis-features-v1`) usaba disponibilidad inmediata por
> `tourney_date`. Tampoco debe usarse la antigua ruta plana
> `data/processed/features`.

## Contrato activo

- Esquema `tennis-features-v2`.
- Publicación generacional e inmutable en
  `data/processed/features_active/runs/<fingerprint>/`, con puntero atómico en
  `data/processed/features_active/manifest.json`.
- `result_available_date=match_date+21 días`; una fila solo puede alimentar un
  estado, fold o reentreno cuando `result_available_date < as_of_date`.
- `RET`, `DEF`, `ABD` y `ABN` se incluyen tras el embargo. `W/O`, `Walkover` y
  `BYE` se excluyen por no acreditar un partido iniciado.
- La cuarentena DOB es diagnóstica para inferencia actual y nunca selecciona
  filas históricas.
- La orientación A/B usa un hash estable con semilla `42`; `y=1` si gana A y
  `y=0` si gana B. El rol original ganador/perdedor no es feature.
- La BBDD operativa no se concatena ni alimenta automáticamente el dataset.

El detalle de las features, allowlist, ranking, mercado y fórmulas está en
[`features.md`](features.md).

## Artefacto publicado

| Campo | Resultado verificado |
|---|---|
| Commit Sackmann | `83733587353df8a41f2fd4f516147d5aa83f5a8d` |
| Fingerprint v2 | `bfe639b5e0f818f3eb4bc58cae0167d2a95486c3211885719553df5d165b1f93` |
| Creación UTC | 2026-08-09 13:29:10 |
| Elo `M` | `elo-m-837d7bd8d6c48ae3f17f3e9c`; fingerprint `837d7bd8d6c48ae3f17f3e9c0beb953d79a4bd0da4c0568f7326261914e14fe9` |
| Elo `F` | `elo-f-ec4cffd280c630cab1de03b8`; fingerprint `ec4cffd280c630cab1de03b8ed276a16b8a63b823111f1e7b6bfdfcb75bc2072` |
| Filas `M`; `y=1` | 959.020; 478.892 (0,4993555922) |
| Filas `F`; `y=1` | 769.835; 385.246 (0,5004267148) |
| `match_date` `M` / `F` | 1967-12-28–2026-06-01 / 1967-12-25–2026-06-02 |
| `result_available_date` `M` / `F` | 1968-01-18–2026-06-22 / 1968-01-15–2026-06-23 |
| Exclusiones `M` | 4.188 por estado; 7 autopartidos |
| Exclusiones `F` | 11.882 duplicados; 4.639 por estado; 289 por nivel; 12 autopartidos |
| Conflictos de ranking | 1.650 observaciones en `ranking_conflicts.csv` |
| Parquet `M` | 158.768.102 bytes; SHA-256 `228eaa15cd555d922ea1504e8910a45dc1a5b49a84b2be8b706181a28929b09b` |
| Parquet `F` | 125.764.370 bytes; SHA-256 `9d1fe2312f0d7b3fafc42d60cde8c09038c0d8cacfc10ce6533dc18c3c39de66` |
| Manifiesto inmutable | 35.896 bytes; SHA-256 `ef017c3e0fa5a64ea65cdb74378cba6dfc585f407d63fa5f0ed31a7993fb8538` |

### Distribución nivel × superficie

`—` identifica superficie ausente en la fuente; no se imputa.

| Género | Nivel | Carpet | Clay | Grass | Hard | — |
|---|---|---:|---:|---:|---:|---:|
| M | ATP Tour | 19.253 | 63.820 | 15.261 | 75.206 | 1.812 |
| M | Challenger | 7.858 | 102.194 | 4.399 | 92.918 | 0 |
| M | Grand Slam | 0 | 10.656 | 12.189 | 16.839 | 0 |
| M | ITF | 20.382 | 269.880 | 3.678 | 227.475 | 0 |
| M | Other | 0 | 63 | 0 | 0 | 0 |
| M | Team | 1.601 | 6.074 | 639 | 5.675 | 1.148 |
| F | Challenger | 4.268 | 33.035 | 2.313 | 26.939 | 1.111 |
| F | Grand Slam | 0 | 10.673 | 12.352 | 18.157 | 0 |
| F | ITF | 16.601 | 222.891 | 6.841 | 226.068 | 0 |
| F | Other | 0 | 148 | 64 | 456 | 0 |
| F | Team | 371 | 5.809 | 186 | 5.269 | 603 |
| F | WTA Tour | 19.720 | 53.996 | 17.680 | 79.959 | 4.325 |

## Evidencia anti-fugas

Las consultas sobre ambos Parquet publicados confirmaron:

- `result_available_date-match_date` vale exactamente 21 días en las
  1.728.855 filas;
- no existe ninguna `ranking_date_a` ni `ranking_date_b` igual o posterior a
  `match_date`;
- las columnas de mercado y sus timestamps tienen cero valores no nulos, pues
  Sackmann no aporta una observación causal de cuotas;
- el manifiesto enlaza exactamente los dos runs Elo v5 activos y declara
  `historical_identity_exclusion=disabled`;
- ganador, perdedor, `y`, probabilidad de modelo y edge no forman parte de la
  allowlist del modelo;
- el hash de orientación no cambia al reordenar filas o añadir partidos futuros;
  el balance observado es 49,9356 % de `y=1` en `M` y 50,0427 % en `F`;
- la inferencia es antisimétrica: al intercambiar A/B, las probabilidades cruda
  y calibrada suman 1 con precisión de 15 decimales.

La invariancia de Elo/features ante futuro, el corte estricto de ranking, la
orientación y el mercado incompleto están cubiertos por tests específicos. El
resultado automatizado consolidado es `329/329 tests correctos`.

## Vector comentado real

Fila `426e27b4...30e3`: Carlos Alcaraz (`A`, 207989) contra Jannik Sinner
(`B`, 206173), ATP Tour, final sobre clay, `best_of=3`. La fecha fuente es
2026-04-05 y el label solo queda disponible el 2026-04-26.

| Grupo | Valores as-of antes del partido | Lectura |
|---|---|---|
| Elo general | A 2783,42; B 2808,87; diferencia -25,46 | Ventaja general ligera para B |
| Elo clay combinado | A 2710,65; B 2666,52; diferencia +44,13 | Ventaja de superficie para A |
| Últimos 10 | A 0,9000; B 0,8667; diferencia +0,0333 | Forma corta favorable a A |
| Últimos 3 meses | A 0,9412; B 0,8667; diferencia +0,0745 | Forma temporal favorable a A |
| H2H global | balance A +0,2941 en 17 partidos | Historial global favorable a A |
| H2H clay | balance A +0,6000 en 5 partidos | Historial de superficie favorable a A |
| Descanso | 32 / 32 días; diferencia 0 | Sin ventaja observada |
| Ranking | 1 / 2 a 2026-03-30; diferencia -1 | Snapshot estrictamente anterior |
| Puntos | 13.590 / 12.400; diferencia +1.190 | Ventaja para A |
| Edad | 22,9190 / 24,6357; diferencia -1,7167 | A es más joven |
| Distancia a 30 | 7,0810 / 5,3643; diferencia +1,7167 | B está más cerca de 30 |
| Mercado | nulo / nulo | No se inventa cuota histórica |
| Etiqueta | `y=0` | Ganó B; no aparece entre predictors |

Este ejemplo ilustra una fila causal, no una recomendación de apuesta. La fecha
fuente sigue siendo aproximada y está protegida por el embargo documentado.

El cierre de modelos y las limitaciones se consolidan en [`audit.md`](audit.md).
