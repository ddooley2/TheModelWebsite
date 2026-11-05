import pandas as pd

import src.cfb_model.reporting.diagnostics as diagnostics
from src.cfb_model.train.experiments import candidate_feature_columns, choose_winner
from src.cfb_model.train.score_model import MODEL_COLUMNS


def test_challenger_requires_lower_mean_and_three_season_improvements():
    metrics = pd.DataFrame([
        {"candidate": "direct_score_l1", "season": season, "score_mae": 9.2}
        for season in range(2021, 2026)
    ] + [
        {"candidate": "environment_total_margin_l1", "season": 2021, "score_mae": 9.1},
        {"candidate": "environment_total_margin_l1", "season": 2022, "score_mae": 9.1},
        {"candidate": "environment_total_margin_l1", "season": 2023, "score_mae": 9.1},
        {"candidate": "environment_total_margin_l1", "season": 2024, "score_mae": 9.3},
        {"candidate": "environment_total_margin_l1", "season": 2025, "score_mae": 9.3},
    ])

    result = choose_winner(metrics)

    assert result["winner"] == "environment_total_margin_l1"


def test_challenger_is_rejected_when_it_wins_too_few_seasons():
    metrics = pd.DataFrame([
        {"candidate": "direct_score_l1", "season": season, "score_mae": 9.2}
        for season in range(2021, 2026)
    ] + [
        {"candidate": "environment_total_margin_l1", "season": 2021, "score_mae": 9.1},
        {"candidate": "environment_total_margin_l1", "season": 2022, "score_mae": 9.3},
        {"candidate": "environment_total_margin_l1", "season": 2023, "score_mae": 9.3},
        {"candidate": "environment_total_margin_l1", "season": 2024, "score_mae": 9.1},
        {"candidate": "environment_total_margin_l1", "season": 2025, "score_mae": 9.3},
    ])

    result = choose_winner(metrics)

    assert result["winner"] == "direct_score_l1"


def test_role_candidates_use_only_their_own_feature_formulation():
    points_columns = candidate_feature_columns("role_points_score_l1")
    role_columns = candidate_feature_columns("role_both_roles_score_l1")
    advanced_columns = candidate_feature_columns("role_advanced_score_l1")

    assert points_columns[:len(MODEL_COLUMNS)] == MODEL_COLUMNS
    assert role_columns[:len(MODEL_COLUMNS)] == MODEL_COLUMNS
    assert advanced_columns[:len(MODEL_COLUMNS)] == MODEL_COLUMNS
    assert "home_team_prior_home_points_for" in points_columns
    assert "home_team_prior_away_points_for" not in points_columns
    assert "home_team_prior_away_points_for" in role_columns
    assert "home_team_prior_home_offense_ppa" in advanced_columns


def test_diagnostics_pass_advanced_stats_into_candidate_backtest(tmp_path, monkeypatch):
    games = pd.DataFrame({"season": [2025]})
    features = pd.DataFrame({"feature": [1.0]})
    advanced = pd.DataFrame({"offense_ppa": [0.2]})
    coverage = pd.DataFrame({"season": [2025], "source": ["games"], "rows": [1]})
    captured = {}
    metrics = pd.DataFrame([
        {"candidate": "direct_score_l1", "season": 2025, "score_mae": 10.0}
    ])

    monkeypatch.setattr(diagnostics, "load_historical_games", lambda db: games)
    monkeypatch.setattr(diagnostics, "load_roster_priors", lambda db: pd.DataFrame())
    monkeypatch.setattr(diagnostics, "load_advanced_stats", lambda db: advanced)
    monkeypatch.setattr(diagnostics, "build_model_features", lambda *args: features)
    monkeypatch.setattr(diagnostics, "_source_coverage", lambda db: coverage)
    monkeypatch.setattr(diagnostics, "_plot_missingness", lambda *args: pd.DataFrame())
    monkeypatch.setattr(diagnostics, "_plot_distributions", lambda *args: pd.DataFrame())
    monkeypatch.setattr(diagnostics, "_plot_source_coverage", lambda *args: None)
    monkeypatch.setattr(diagnostics, "_plot_performance", lambda *args: None)

    def backtest(games_arg, features_arg, advanced=None):
        captured["advanced"] = advanced
        return metrics, pd.DataFrame()

    monkeypatch.setattr(diagnostics, "run_candidate_backtest", backtest)

    diagnostics.generate_diagnostics(tmp_path / "warehouse.duckdb", tmp_path / "report")

    assert captured["advanced"] is advanced
