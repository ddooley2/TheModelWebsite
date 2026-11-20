# Repository guide

## Commands

```bash
python -m pytest
python -m pytest tests/test_score_model.py::test_features_use_only_prior_game_results
python -m src.cfb_model.train.score_model --backtest
python -m src.cfb_model.train.score_model --season 2026 --predict-week 8 --candidate direct_score_l1
python -m src.cfb_model.site.build_site --season 2026 --week 8 --skip-diagnostics
```

## Architecture

`data/raw/*.parquet` are versioned source data. `data/warehouse.py` builds `db/warehouse.duckdb` when missing; training loads completed games and builds point-in-time form, opponent-adjusted ratings, preseason priors, and advanced-stat features in `features/`. `train/score_model.py` trains and predicts home/away scores; `train/experiments.py` compares candidates using season-forward folds, and `reporting/diagnostics.py` produces the evaluation artifacts used for challenger selection.

`site/build_site.py` combines saved predictions with schedule and CFBD data to write `public/data/dashboard.json` for the static site in `public/`. `.github/workflows/publish-pages.yml` runs weekly predictions or daily line refreshes and publishes the site.

## Repository-specific invariants

- Keep every model feature knowable before kickoff; compute features before applying that game's result, and batch same-date games so they cannot use one another's outcomes.
- Keep betting lines out of model features, fitting, and candidate selection. Lines are for dashboard comparison only.
- Preserve `data/raw/` as source of truth. Warehouse creation is automatic when the DuckDB file is absent.
- `data/site/line_history.json` preserves the first opening-line observation; daily refreshes update current lines without recalculating predictions.

## Local CFBD access

- Local commands that call CFBD must load `.env` with `python-dotenv` before reading `CFBD_API_KEY`. Never print or expose the key.
