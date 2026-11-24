# Stage 5: prior-season scoring prior

## Method

The challenger adds prior-season team scoring and scoring-allowed averages, shrunk toward current-season results with 4 prior pseudo-games. Features use only completed FBS-vs-FBS results before the target UTC date. The prior's weight is `4 / (prior current-season games + 4)` when prior-season data exist.

Returning-production endpoint values were not used because historical publication-time snapshots could not be verified. Existing roster priors and Elo features remain unchanged.

## Results

| Model | Overall score MAE | Overall margin MAE | Overall total MAE | 2025 score MAE |
|---|---:|---:|---:|---:|
| Production baseline | 9.1816 | 12.9349 | 12.9378 | 9.0541 |
| Prior-season score prior (4 pseudo-games) | 9.1662 | 12.9361 | 12.8832 | 9.0107 |

### Early-season Weeks 0–3

Baseline MAE: 9.4647 (949 games); challenger MAE: 9.4696 (949 games).

### Score MAE by week range

| Week range | Baseline | Challenger |
|---|---:|---:|
| Weeks 0-3 | 9.4647 | 9.4696 |
| Weeks 4-6 | 9.0286 | 9.0170 |
| Weeks 7+ | 9.1156 | 9.0899 |

### Seasonal MAE

| Season | Baseline | Challenger |
|---:|---:|---:|
| 2021 | 9.3149 | 9.3317 |
| 2022 | 9.0956 | 9.0806 |
| 2023 | 9.2396 | 9.2208 |
| 2024 | 9.2080 | 9.1927 |
| 2025 | 9.0541 | 9.0107 |

## Decision and limitations

This is an isolated challenger only; no production or published candidate changed. The 2025 block is not considered untouched because it was examined during earlier sequential experiments. The paired week-cluster interval and early-season subset should be considered before any promotion decision.

Paired final-season bootstrap: `preseason_score_prior_stage5/paired_week_bootstrap_ci.json`.

## Reproduction

- `python -m src.cfb_model.train.preseason_score_prior_experiment`
- `python -m pytest`
- No new API fetch or cache is required; the runner uses the existing warehouse.
