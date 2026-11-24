# Final combined optimization screen

All candidates use the same 2021–2025 walk-forward folds and the same score model. Parameters were fixed before this run. The seasons have already been examined in earlier stages, so results are screening-only.

| Candidate | Overall score MAE | 2025 score MAE | 2025 margin MAE | 2025 total MAE |
|---|---:|---:|---:|---:|
| Production baseline | 9.1816 | 9.0541 | 12.6112 | 12.8998 |
| Recency ratings | 9.1642 | 9.0379 | 12.5428 | 12.8902 |
| Recency + pace | 9.1303 | 8.9792 | 12.5700 | 12.7886 |
| Recency + pace + role/site | 9.1224 | 8.9557 | 12.5551 | 12.7706 |
| Core + Stage 5 score prior | 9.1315 | 8.9463 | 12.5379 | 12.7666 |
| Core + Stage 4B efficiency | 9.1217 | 8.9163 | 12.4416 | 12.7762 |

## Opening-week and neutral-site MAE

| Candidate | Weeks 0–3 | Neutral-site all seasons | Neutral-site 2025 |
|---|---:|---:|---:|
| Production baseline | 9.4647 | 9.6521 | 9.5405 |
| Recency ratings | 9.4280 | 9.6122 | 9.5674 |
| Recency + pace | 9.3661 | 9.4596 | 9.3458 |
| Recency + pace + role/site | 9.3590 | 9.4624 | 9.3437 |
| Core + Stage 5 score prior | 9.4038 | 9.5055 | 9.3391 |
| Core + Stage 4B efficiency | 9.3661 | 9.5046 | 9.3020 |

## Seasonal score MAE

| Season | Production baseline | Recency ratings | Recency + pace | Recency + pace + role/site | Core + Stage 5 score prior | Core + Stage 4B efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 2021 | 9.3149 | 9.3362 | 9.2620 | 9.2617 | 9.2718 | 9.2329 |
| 2022 | 9.0956 | 9.0642 | 9.0710 | 9.0561 | 9.0926 | 9.0969 |
| 2023 | 9.2396 | 9.1921 | 9.1756 | 9.1805 | 9.2003 | 9.2057 |
| 2024 | 9.2080 | 9.1958 | 9.1689 | 9.1637 | 9.1534 | 9.1631 |
| 2025 | 9.0541 | 9.0379 | 8.9792 | 8.9557 | 8.9463 | 8.9163 |

## Paired final-season intervals

| Candidate | Overall 95% CI for score-MAE delta vs baseline |
|---|---:|
| Recency ratings | [-0.0715, 0.0358] |
| Recency + pace | [-0.1334, -0.0028] |
| Recency + pace + role/site | [-0.1523, -0.0382] |
| Core + Stage 5 score prior | [-0.1680, -0.0441] |
| Core + Stage 4B efficiency | [-0.1956, -0.0760] |

## Recommendation

Treat the cumulative stack as a screening candidate only. Prefer the simplest recency-plus-pace combination unless stabilized role/site features add repeatable value across seasons and neutral games. Keep Stage 5 and Stage 4B as ablations; do not select a winner from the previously examined 2025 block.

## Reproduction

- `python -m src.cfb_model.train.final_combined_experiment`
- `python -m pytest`
