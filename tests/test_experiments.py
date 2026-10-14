import pandas as pd

from src.cfb_model.train.experiments import choose_winner


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
