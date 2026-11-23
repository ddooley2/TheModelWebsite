# Score-first college football model

This repository predicts home and away final scores for FBS-vs-FBS games. Margin and total are derived from those scores. The production candidate uses **only pre-kickoff, market-free features**: point-in-time form, Elo, opponent-adjusted ratings, shrunk running averages of CFBD advanced stats (PPA, success rate, explosiveness), preseason Elo, talent/recruiting priors, rest, week, and neutral-site status.

## Data integrity

- The feature list deliberately excludes betting lines and market columns.
- Each game receives form and rating features before its own result—or any shared-date result—is applied.
- Training and prediction targets, plus competitive score statistics, remain FBS-versus-FBS. Rest and games-played features use the separate completed-game schedule history for each FBS team, including FCS opponents.
- Current-season ESPN team IDs are crosswalked one-to-one to CFBD's FBS IDs for that season. Unmatched FBS names/IDs are logged with a mapping candidate and fail the refresh instead of dropping target games.
- CFBD lines appear only in the static site’s comparison layer. They never enter model fitting, candidate selection, or prediction.
- When `db/warehouse.duckdb` is absent, commands rebuild it from the versioned `data/raw/*.parquet` files without changing those source files.

## CFBD advanced-game-stat cache

The advanced-game endpoint is fetched only by the explicit ingestion command, not by model feature construction. Raw response bytes, a normalized Parquet table, and a manifest are stored under `data/raw/cfbd/advanced_game/year=<season>/`. The manifest records the request parameters, `excludeGarbageTime=false` policy (matching the existing historical source), response and normalized-data hashes, row count, code hash, and fetch time. Normalized rows use `(game_id, team)` as the primary key; `season_type`, `week`, `opponent`, optional team IDs, and flattened numeric fields such as `offense_passingPlays_ppa` are retained.

Refresh a season explicitly, or load its existing cache without network access:

```bash
python -m src.cfb_model.data.advanced_game_stats --season 2026 --refresh
python -m src.cfb_model.data.advanced_game_stats --season 2026
```

Weekly prediction jobs refresh the season cache before predicting. Historical backtests continue to use the versioned warehouse data; adding or refreshing a cache does not change model features until a separate modeling experiment opts into it.

| Column | Meaning |
|---|---|
| `game_id` | CFBD game ID; nullable only when the provider omits it. |
| `season`, `season_type`, `week` | CFBD season identity and schedule period. |
| `team`, `opponent` | CFBD team names for this team's game-level row. |
| `team_id`, `opponent_id` | Provider identifiers when present; otherwise null. |
| `offense_<path>`, `defense_<path>` | Numeric leaves copied from the nested response, preserving their source path and scale (for example `offense_plays`, `defense_ppa`, `offense_passingPlays_successRate`). Null source metrics remain null. |

## Backtest and challenger selection

```bash
python -m src.cfb_model.train.score_model --backtest
```

The headline metric is per-team score MAE. The evaluation is a season-forward backtest: each validation season uses training data from prior seasons only.

To save the canonical market-free baseline with per-game predictions, a machine-readable run manifest, and seasonal metrics:

```bash
python -m src.cfb_model.train.score_model \
  --backtest \
  --run-dir results/baselines/direct_score_l1-<run-id>
```

The run directory must not already exist. Its manifest records the candidate, source commit, feature-schema hash, cutoff policy, held-out game IDs, game count, market-independence status, and overall/season metrics. The per-game CSV includes both score predictions and actuals, score errors, and the recorded game-date cutoff. Betting-derived feature names are rejected before model fitting.

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
