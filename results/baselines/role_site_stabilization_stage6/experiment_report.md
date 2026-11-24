# Stage 6: role/site-feature stabilization

## Method

Role-specific same-season estimates are blended toward overall team estimates with 4 pseudo-games. Role observation counts and effective weights are included. Neutral-site targets use overall estimates directly; the existing neutral-site indicator is retained.

## Results

| Model | Overall score MAE | Overall margin MAE | Overall total MAE | 2025 score MAE |
|---|---:|---:|---:|---:|
| Production baseline | 9.1816 | 12.9349 | 12.9378 | 9.0541 |
| Stabilized role/site challenger | 9.1813 | 12.9435 | 12.9727 | 9.0447 |

### Required subsets

| Subset | Baseline MAE | Challenger MAE |
|---|---:|---:|
| Neutral site, all seasons | 9.6521 | 9.6236 |
| Neutral site, 2025 | 9.5405 | 9.5564 |
| Weeks 0-3, all seasons | 9.4647 | 9.4597 |

### Seasonal MAE

| Season | Baseline | Challenger |
|---:|---:|---:|
| 2021 | 9.3149 | 9.3220 |
| 2022 | 9.0956 | 9.0785 |
| 2023 | 9.2396 | 9.2503 |
| 2024 | 9.2080 | 9.2152 |
| 2025 | 9.0541 | 9.0447 |

## Decision and limitations

This is an isolated evaluation only. The 2025 block has been examined in earlier stages and is not an untouched final test. No production or published candidate changed.

Paired final-season bootstrap: `role_site_stabilization_stage6/paired_week_bootstrap_ci.json`.

## Reproduction

- `python -m src.cfb_model.train.role_site_stabilization_experiment`
- `python -m pytest`
- No new API fetch or cache is required; the runner uses the existing warehouse.
