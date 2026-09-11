# Puerta champion/challenger de TENNIS

## Contrato

`scripts/retrain_models.py`, tanto desde `run_tennis.ps1` como de forma manual,
registra primero un challenger inmutable. `training.py` no publica el puntero
activo. La única conmutación normal pasa por `src/modeling/promotion.py`.

La evaluación usa el último año completo del backtest expansivo. Para 2025,
cada modelo base se ajusta como máximo con temporadas hasta 2023, Platt se
ajusta con 2024 y el test es 2025. Champion y challenger deben contener
exactamente el mismo conjunto de partidos. No se usa shuffle ni se puntúa el
estimador final sobre su propio histórico.

Los parámetros están en `config/model_promotion.json`:

- log-loss mínimo: `0,001`;
- Brier mínimo para desempatar dentro de la banda de log-loss: `0,0005`;
- tolerancia de repetición: `1e-12`;
- soporte mínimo: 120 partidos;
- retención: últimos 2 + champion + `last_good`.

Un empate no cambia producción. Cada decisión queda en
`models/phase7/promotion/decisions/<challenger>.json`, con periodo, N,
configuración de calibración, hashes, métricas globales, accuracy y desgloses.

## Estado de migración (2026-08-31)

La puerta se inicializó sin reescribir ningún run:

- champion inicial: `d080b15c768228596b17f613a45b0757d790b193ce2380674fa1c4b73f71c1a5`;
- `last_good` inicial: `2320aad663fddd960fc08e78f5dc6723c278584eb419126ecd39ba098d59b594`.

La primera keep-list conservaba ambos runs y no tenía candidatos de borrado.
No se ejecutó ninguna poda; la aprobación inicial sigue pendiente hasta que el
operador vea una preview que incluya cualquier eliminación real.

## Verificación real (2026-08-31)

El pipeline automático registró el challenger final
`8b02e29cd947a951d64f5d2b6feba639961c939dd3ec57eeb26374cc5f8a0918`.
La puerta lo comparó con `d080b15c…` sobre exactamente 55.798 partidos entre
2025-01-06 y 2025-12-29:

| Artefacto | Accuracy | Log-loss | Brier |
|---|---:|---:|---:|
| Champion `d080b15c…` | 69,1889 % | 0,574738 | 0,196566 |
| Challenger `8b02e29…` | 69,1889 % | 0,574738 | 0,196566 |

Las probabilidades y las tres métricas se reprodujeron con delta máximo 0 bajo
tolerancia `1e-12`. Resultado: `REJECT / brier_tiebreak_not_improved`; no hubo
promoción y ambos punteros conservaron sus valores anteriores.

La preview posterior conserva los dos challengers más recientes, el champion
y `last_good`; no contiene eliminaciones. No se aprobó ni ejecutó ninguna poda.
