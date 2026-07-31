# Best Opportunity backtest

- Policy: `best_opportunity_v1`.
- Score: `decision_confidence * reliability_score`.
- Gates: confidence >= 60%; reliability >= 45%.
- Deduplication: latest real prematch snapshot; rank within UTC match day.
- Sample: 240 matches, 2026-07-08 to 2026-07-25.

| Strategy | Picks | Correct | Accuracy | Wilson 95% CI | Days |
|---|---:|---:|---:|---:|---:|
| All latest prematch predictions | 240 | 148 | 61.67% | 55.4%-67.6% | 18 |
| All eligible Best Opportunity cards | 105 | 69 | 65.71% | 56.2%-74.1% | 17 |
| Best Opportunity #1 per day | 17 | 11 | 64.71% | 41.3%-82.7% | 17 |
| Top 3 Best Opportunity per day | 50 | 37 | 74.00% | 60.4%-84.1% | 17 |

Accuracy measures winner selection only. It is not an ROI backtest and does not imply that every eligible card has positive expected value.
