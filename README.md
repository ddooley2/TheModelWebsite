# Score-first college football model

This repository predicts home and away final scores for FBS-vs-FBS games. Margin and total are derived from those scores. The production candidate uses **only pre-kickoff, market-free features**: point-in-time form, Elo, opponent-adjusted ratings, shrunk running averages of CFBD advanced stats (PPA, success rate, explosiveness), preseason Elo, talent/recruiting priors, rest, week, and neutral-site status.

## Data integrity

- The feature list deliberately excludes betting lines and market columns.
- Each game receives form and rating features before its own result—or any shared-date result—is applied.
- CFBD lines appear only in the static site’s comparison layer. They never enter model fitting, candidate selection, or prediction.
- When `db/warehouse.duckdb` is absent, commands rebuild it from the versioned `data/raw/*.parquet` files without changing those source files.

## Backtest and challenger selection

```bash
python -m src.cfb_model.train.score_model --backtest
```

The headline metric is per-team score MAE. The evaluation is a season-forward backtest: each validation season uses training data from prior seasons only.

`src/cfb_model.train.experiments` evaluates seven market-free candidates on identical folds:

1. `direct_score_l1` predicts home and away scores directly.
2. `environment_total_margin_l1` predicts total points and home margin, then recombines them into scores.
3. Five role-based candidates add point-in-time team form split by home/away role; these include points-only, form, advanced-stat, both-roles, and relevant-role-plus-overall variants.

The selector marks a challenger eligible only if it lowers unweighted mean seasonal score MAE **and** wins at least three validation seasons. The publish workflow currently pins `role_relevant_and_overall_score_l1` for new weekly predictions; it still generates the comparison report for review.

## Build a prediction file

```bash
python -m src.cfb_model.train.score_model \
  --season 2026 --predict-week 8 \
  --candidate role_relevant_and_overall_score_l1
```

This reproduces the production candidate. The command writes `results/score_predictions_<season>_week<week>.csv`.

## GitHub Pages dashboard

The dashboard is a static GitHub Pages site. It includes the weekly game board, projected scores/spread/total, kickoff/venue, records/ranks where returned by CFBD, CFBD opening/current lines, pick/edge, methodology, data coverage, feature sparsity, distributions, and seasonal performance.

### One-time repository settings

1. In **Settings → Pages**, select **GitHub Actions** as the deployment source.
2. In **Settings → Secrets and variables → Actions**, create the secret `CFBD_API_KEY`.
3. Run **Refresh and publish CFB dashboard** manually once. The workflow runs weekly predictions and snapshots each provider's published opening lines on Monday at 10:00 UTC. Tuesday through Sunday at 10:00 UTC, it refreshes lines for the currently published week without recalculating predictions or changing opening lines. The upcoming season and FBS week are resolved from CFBD's schedule; repository variables are not needed. Manual runs can override season/week or choose a daily line refresh.

For new weekly predictions, the reference spread and total are captured from live CFBD lines immediately before model prediction. That snapshot is retained while later daily runs update only current lines. CFBD's `/lines` endpoint requires bearer authentication.

The model depends on `duckdb`, `lightgbm`, `numpy`, `pandas`, `pyarrow`, `requests`, `scikit-learn`, `scipy`, `matplotlib`, and `python-dotenv`.
