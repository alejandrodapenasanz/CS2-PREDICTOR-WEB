# Informe de ejecución de la fase 8

## Ejecución real

El pipeline se ejecutó para la cartelera del **2026-07-30**. Reutilizó el HTML
íntegro capturado a `2026-07-30T09:48:28Z` y generó la inferencia a
`2026-07-30T14:01:32.476755Z`. La salida completa se publicó en:

```text
data/processed/predictions/
predictions_2026-07-30_20260730T140132476755Z.csv
```

No se modificó `start.ps1`; el lanzador independiente se validó desde la raíz
del repositorio con `--help` y política de ejecución de un solo proceso.

## Cobertura

| Género | Nivel | Partidos con predicción |
|---|---|---:|
| F | ITF | 86 |
| F | WTA | 10 |
| M | ATP | 2 |
| M | Challenger | 19 |
| M | ITF | 87 |
| **Total** |  | **204** |

La cartelera contenía 313 partidos: 228 `scheduled`, 76 `finished` y 9
`unknown`. De los programados, 24 no tenían ambos IDs resueltos. Por tanto:

- 204 filas recibieron probabilidad raw y calibrada;
- 201 tenían dos cuotas y edge de-vigado;
- 117 tenían superficie explícita;
- 109 filas conservaron probabilidades nulas por estado o mapping.

## Confianza y frescura

Las 204 predicciones quedaron `LOW`. No es una etiqueta derivada de sus
probabilidades: responde a fuentes desactualizadas.

| Fuente | Corte M | Corte F | Antigüedad a D |
|---|---|---|---:|
| Features/modelo/Elo | 2026-06-01 | 2026-06-02 | 59 / 58 días |
| Rankings | 2026-06-08 | 2026-06-08 | 52 días |

Además, los 87 Futures masculinos predichos no tenían superficie. Las flags
completas se conservan por fila. Las más frecuentes fueron:

| Flag | Filas |
|---|---:|
| `ranking_source_stale_52d` | 204 |
| `best_of_missing` | 204 |
| `round_missing` | 204 |
| `daily_raw_level_not_sackmann_code` | 204 |
| `history_stale_59d` | 108 |
| `history_stale_58d` | 96 |
| `surface_missing` | 87 |
| `market_missing` | 3 |

La etiqueta daily de nivel se conserva literalmente y también se mapea a la
categoría canónica. No se sustituye por un código Sackmann ficticio; por eso se
expone la flag correspondiente.

## Lectura de los edges

Los mayores desacuerdos absolutos son diagnósticos, no recomendaciones:

| Torneo | A | B | P modelo A | P mercado A | Edge A | Confianza |
|---|---|---|---:|---:|---:|---|
| Rogaska Slatina ITF | Drobysheva M. | Luciano L. | 27,65 % | 78,31 % | -50,66 pp | LOW |
| Futures 2026 | Krstic V. | Chestovaliev S. | 54,76 % | 91,65 % | -36,90 pp | LOW |
| Huamantla 4 ITF | Nava Elkin A. | Allegre C. | 51,07 % | 16,52 % | +34,55 pp | LOW |
| Huamantla 4 ITF | Angel M. | Watanabe Y. | 56,43 % | 23,59 % | +32,85 pp | LOW |
| Futures 2026 | Bonding O. | Baris O. | 64,42 % | 32,30 % | +32,11 pp | LOW |

El modelo activo es `sports_only`; las cuotas no entraron a LightGBM. Fase 7
no pudo evaluar superioridad frente al mercado por cobertura histórica 0 %.
Los desacuerdos grandes, especialmente ITF con datos viejos, deben motivar
revisión, no apuestas.

## Validación

La suite completa terminó:

```text
Ran 258 tests in 30.060s
OK
```

Las pruebas nuevas cubren reconstrucción dirigida `< D`, invariancia al añadir
futuro, bloqueo de modelos entrenados en/tras D, orientación A/B, filas no
mapeadas o terminadas, confianza, publicación CSV, superficie del mismo HTML,
lanzador y documentación de integración.

