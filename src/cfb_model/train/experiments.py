"""Market-free score-model challengers and strictly forward experiment reporting."""

from __future__ import annotations

from collections.abc import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor

from src.cfb_model.train.score_model import (
    BACKTEST_SEASONS,
    FIRST_TRAIN_SEASON,
    MODEL_COLUMNS,
    TARGET_COLUMNS,
    _make_model,
    score_metrics,
)


def _make_target_model() -> MultiOutputRegressor:
    """Use the baseline tree family for a different, score-coherent target basis."""
    estimator = lgb.LGBMRegressor(
        objective="l1",
        n_estimators=800,
        learning_rate=0.015,
        max_depth=4,
        num_leaves=7,
        min_child_samples=80,
        reg_lambda=20.0,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    return MultiOutputRegressor(estimator)


def _fit_direct(train_x: pd.DataFrame, train_y: np.ndarray, valid_x: pd.DataFrame) -> np.ndarray:
    model = _make_model()
    model.fit(train_x, train_y)
    return model.predict(valid_x)


def _fit_environment(
    train_x: pd.DataFrame, train_y: np.ndarray, valid_x: pd.DataFrame
) -> np.ndarray:
    """Predict total and home margin, then recombine into two internally consistent scores."""
    targets = np.column_stack((
        train_y[:, 0] + train_y[:, 1],
        train_y[:, 0] - train_y[:, 1],
    ))
    model = _make_target_model()
    model.fit(train_x, targets)
    total, margin = model.predict(valid_x).T
    return np.column_stack(((total + margin) / 2, (total - margin) / 2))


CANDIDATES: dict[str, Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray]] = {
    "direct_score_l1": _fit_direct,
    "environment_total_margin_l1": _fit_environment,
}


def run_candidate_backtest(
    games: pd.DataFrame,
    features: pd.DataFrame,
    val_seasons: list[int] = BACKTEST_SEASONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return season metrics and row-level, untouched out-of-fold predictions.

    Every candidate uses identical folds and feature columns. Betting-market
    columns are rejected even if they are present in ``features``.
    """
    forbidden = [column for column in features if "market" in column.lower() or "line" in column.lower()]
    if forbidden:
        raise ValueError(f"Market-derived columns are forbidden: {forbidden}")

    summaries: list[dict] = []
    predictions: list[pd.DataFrame] = []
    for season in val_seasons:
        train_mask = (games["season"] >= FIRST_TRAIN_SEASON) & (games["season"] < season)
        valid_mask = games["season"] == season
        if not train_mask.any() or not valid_mask.any():
            continue

        train_x = features.loc[train_mask, MODEL_COLUMNS]
        valid_x = features.loc[valid_mask, MODEL_COLUMNS]
        train_y = games.loc[train_mask, TARGET_COLUMNS].to_numpy(dtype=float)
        valid_y = games.loc[valid_mask, TARGET_COLUMNS].to_numpy(dtype=float)

        for name, fit_predict in CANDIDATES.items():
            predicted = fit_predict(train_x, train_y, valid_x)
            summaries.append({"candidate": name, "season": season, "games": len(valid_y), **score_metrics(valid_y, predicted)})
            fold = games.loc[valid_mask, ["game_id", "season", "week", "game_date", "home_team", "away_team"]].copy()
            fold["candidate"] = name
            fold["actual_home_score"] = valid_y[:, 0]
            fold["actual_away_score"] = valid_y[:, 1]
            fold["predicted_home_score"] = predicted[:, 0]
            fold["predicted_away_score"] = predicted[:, 1]
            fold["score_absolute_error"] = np.abs(valid_y - predicted).mean(axis=1)
            predictions.append(fold)

    if not summaries:
        raise ValueError("No requested validation seasons had both training and validation games.")
    return pd.DataFrame(summaries), pd.concat(predictions, ignore_index=True)


def choose_winner(metrics: pd.DataFrame, baseline_name: str = "direct_score_l1") -> dict:
    """Choose only a reliable winner; otherwise retain the baseline.

    A challenger must reduce unweighted mean seasonal score MAE and improve at
    least three of five requested seasons. This rule is decided without market
    data and avoids accepting a single-season fluctuation as an improvement.
    """
    seasonal = metrics.pivot(index="season", columns="candidate", values="score_mae")
    if baseline_name not in seasonal:
        raise ValueError(f"Baseline candidate {baseline_name!r} is missing.")
    baseline = seasonal[baseline_name]
    evaluations = []
    for candidate in seasonal.columns:
        candidate_values = seasonal[candidate]
        delta = candidate_values - baseline
        evaluations.append({
            "candidate": candidate,
            "mean_score_mae": float(candidate_values.mean()),
            "mean_delta_vs_baseline": float(delta.mean()),
            "seasons_improved": int((delta < 0).sum()),
            "eligible": bool(
                candidate == baseline_name
                or (candidate_values.mean() < baseline.mean() and (delta < 0).sum() >= 3)
            ),
        })
    eligible = [row for row in evaluations if row["eligible"]]
    winner = min(eligible, key=lambda row: row["mean_score_mae"])
    return {"winner": winner["candidate"], "candidates": evaluations}


def predict_candidate(
    candidate: str, train_x: pd.DataFrame, train_y: np.ndarray, predict_x: pd.DataFrame
) -> np.ndarray:
    """Fit one registered market-free candidate and return score predictions."""
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown candidate {candidate!r}; choose from {sorted(CANDIDATES)}")
    return CANDIDATES[candidate](train_x, train_y, predict_x)
