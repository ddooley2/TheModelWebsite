# Score-first college football model

This repository predicts home and away final scores for FBS-vs-FBS games. Margin and total are derived from those scores. The production candidate uses **only pre-kickoff, market-free features**: point-in-time form, Elo, opponent-adjusted ratings, preseason Elo, talent/recruiting priors, rest, week, and neutral-site status.

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

`src/cfb_model.train.experiments` evaluates two market-free candidates on identical folds:

1. `direct_score_l1` predicts home and away scores directly.
2. `environment_total_margin_l1` predicts total points and home margin, then recombines them into scores.

A challenger is promoted only if it lowers unweighted mean seasonal score MAE **and** wins at least three validation seasons. Otherwise the direct score model remains the winner. The generated report retains row-level out-of-fold predictions, so the decision is auditable.

## Build a prediction file

```bash
python -m src.cfb_model.train.score_model \
  --season 2026 --predict-week 8 \
  --candidate direct_score_l1
```

Choose the candidate only after reading the experiment report. The command writes `results/score_predictions_<season>_week<week>.csv`.

## GitHub Pages dashboard

The dashboard is a static GitHub Pages site. It includes the weekly game board, projected scores/spread/total, kickoff/network/venue, records/ranks where returned by CFBD, CFBD opening/current lines, pick/edge, methodology, data coverage, feature sparsity, distributions, and seasonal performance.

### One-time repository settings

1. In **Settings → Pages**, select **GitHub Actions** as the deployment source.
2. In **Settings → Secrets and variables → Actions**, create the secret `CFBD_API_KEY`.
3. Create repository variables `CFB_SEASON` and `CFB_WEEK` for scheduled refreshes.
4. Run **Refresh and publish CFB dashboard** manually once. The hourly schedule thereafter refreshes score projections and display-only line snapshots.

The workflow records the first observed line for each CFBD game/provider as the opening line and overwrites only the current snapshot later. This means a new game will show the opening line beginning with the first successful workflow run; CFBD’s documented `/lines` endpoint returns historical lines/results and requires bearer authentication.

The model depends on `duckdb`, `lightgbm`, `numpy`, `pandas`, `pyarrow`, `requests`, `scikit-learn`, `scipy`, `matplotlib`, and `python-dotenv`.
