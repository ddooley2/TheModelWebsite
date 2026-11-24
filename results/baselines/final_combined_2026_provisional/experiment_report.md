# Provisional 2026 combined-model comparison

All candidates were evaluated on the same 695 completed FBS-vs-FBS games in the pinned 2026 snapshot (latest game 2026-11-22). Each candidate was trained only on prior seasons and no parameters were tuned. Feature builders exclude games on each target's UTC date, but CFBD statistics and roster priors are pinned current snapshots rather than archived as-of-game-date snapshots, so provider revisions are a limitation. The season is incomplete, and a Week 8 production evaluation predates this comparison; treat it as provisional, not a fully untouched test.

| Candidate | Games | 2026 score MAE | 2026 margin MAE | 2026 total MAE |
|---|---:|---:|---:|---:|
| Production baseline | 695 | 9.0426 | 12.6849 | 12.5633 |
| Recency ratings | 695 | 9.0119 | 12.5995 | 12.5982 |
| Recency + pace | 695 | 9.0200 | 12.6047 | 12.5824 |
| Recency + pace + role/site | 695 | 9.0043 | 12.6600 | 12.5257 |
| Core + Stage 5 score prior | 695 | 9.0037 | 12.6660 | 12.6132 |
| Core + Stage 4B efficiency | 695 | 9.0181 | 12.6590 | 12.5703 |

## Opening-week and neutral-site MAE

| Candidate | Weeks 0–3 | Neutral-site MAE |
|---|---:|---:|
| Production baseline | 8.8267 | 8.9687 |
| Recency ratings | 8.8296 | 9.0583 |
| Recency + pace | 8.7787 | 8.6541 |
| Recency + pace + role/site | 8.8053 | 8.7896 |
| Core + Stage 5 score prior | 8.8543 | 9.0122 |
| Core + Stage 4B efficiency | 8.9386 | 8.4799 |

## Seasonal score MAE

| Season | Production baseline | Recency ratings | Recency + pace | Recency + pace + role/site | Core + Stage 5 score prior | Core + Stage 4B efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 9.0426 | 9.0119 | 9.0200 | 9.0043 | 9.0037 | 9.0181 |

## Paired 2026 intervals

| Candidate | Paired week-cluster 95% CI for score-MAE delta vs baseline |
|---|---:|
| Recency ratings | [-0.1024, 0.0385] |
| Recency + pace | [-0.1312, 0.0768] |
| Recency + pace + role/site | [-0.1285, 0.0455] |
| Core + Stage 5 score prior | [-0.1483, 0.0608] |
| Core + Stage 4B efficiency | [-0.1195, 0.0650] |

## Recommendation

Treat this as a provisional out-of-time signal only. Do not tune or promote using this incomplete season; keep production unchanged and repeat the locked comparison after the 2026 season and data reconciliation are complete.

## Reproduction

- `python -m src.cfb_model.train.final_combined_2026_experiment`
- `python -m pytest`
