# Calibración por segmento — backtest temporal

Modelo/política evaluada: `nested_model_policy`. Predicciones OOS: 7978.

| Dimensión | Segmento | N | Prob. media | Tasa real | Gap | ECE | Estado |
|---|---|---:|---:|---:|---:|---:|---|
| elo_gap | 0-50 | 3439 | 0.5436 | 0.5362 | -0.0074 | 0.0122 | compatible_with_sampling_noise |
| elo_gap | 50-100 | 2329 | 0.5571 | 0.5530 | -0.0040 | 0.0166 | compatible_with_sampling_noise |
| elo_gap | 100-200 | 1891 | 0.5820 | 0.6060 | +0.0241 | 0.0344 | compatible_with_sampling_noise |
| elo_gap | 200-300 | 293 | 0.6411 | 0.6621 | +0.0210 | 0.0657 | compatible_with_sampling_noise |
| elo_gap | 300+ | 26 | 0.6307 | 0.6154 | -0.0154 | 0.1322 | insufficient_sample |
| stage | unknown | 7041 | - | - | - | - | unknown_segment |
| stage | group | 339 | 0.5623 | 0.5428 | -0.0195 | 0.0566 | deviation_detected |
| stage | swiss | 214 | 0.5808 | 0.6495 | +0.0687 | 0.0939 | deviation_detected |
| stage | quarter | 115 | 0.5698 | 0.5565 | -0.0132 | 0.0765 | compatible_with_sampling_noise |
| stage | final | 46 | 0.5314 | 0.4783 | -0.0531 | 0.0957 | insufficient_sample |
| stage | league | 2 | 0.6142 | 0.0000 | -0.6142 | 0.6142 | insufficient_sample |
| stage | semi | 103 | 0.5405 | 0.5825 | +0.0421 | 0.0912 | compatible_with_sampling_noise |
| stage | ro16 | 98 | 0.5696 | 0.5204 | -0.0492 | 0.0694 | insufficient_sample |
| stage | upper_bracket | 2 | 0.6985 | 1.0000 | +0.3015 | 0.3015 | insufficient_sample |
| stage | lower_bracket | 18 | 0.4822 | 0.5556 | +0.0734 | 0.2718 | insufficient_sample |
| environment | unknown | 6991 | - | - | - | - | unknown_segment |
| environment | lan | 259 | 0.5600 | 0.5521 | -0.0079 | 0.0823 | deviation_detected |
| environment | online | 728 | 0.5596 | 0.5646 | +0.0049 | 0.0318 | compatible_with_sampling_noise |

> `insufficient_sample` no se interpreta. `deviation_detected` es una señal descriptiva, no una orden de cambiar el modelo: el siguiente escalón debe ganar el hold-out temporal y la puerta champion/challenger.
