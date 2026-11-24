import numpy as np
import pandas as pd

from src.cfb_model.train.recency_experiment import (
    _paired_final_season,
    _week_cluster_bootstrap,
    select_recency_parameters,
)


def test_parameter_selection_uses_only_tuning_seasons():
    metrics = pd.DataFrame(
        [
            {"half_life_days": half_life, "season_prior_strength": 5.0,
             "ridge_alpha": 20.0, "season": season, "score_mae": score_mae}
            for half_life, scores in (
                (28, {2023: 9.0, 2024: 9.0, 2025: 1.0}),
                (56, {2023: 8.0, 2024: 8.0, 2025: 20.0}),
            )
            for season, score_mae in scores.items()
        ] + [
            {"half_life_days": 28, "season_prior_strength": 0.0,
             "ridge_alpha": 10.0, "season": season, "score_mae": 0.1}
            for season in (2023, 2024, 2025)
        ]
    )

    selected = select_recency_parameters(metrics, [2023, 2024])

    assert selected["half_life_days"] == 56


def test_final_season_paired_errors_use_per_team_score_absolute_error():
    baseline = pd.DataFrame(
        [
            {
                "game_id": 1, "season": 2025, "week": 2, "home_team": "A",
                "away_team": "B", "game_date": "2025-09-01",
                "home_score_absolute_error": 1.0, "score_absolute_error": 0.5,
            },
            {
                "game_id": 2, "season": 2025, "week": 5, "home_team": "C",
                "away_team": "D", "game_date": "2025-09-20",
                "home_score_absolute_error": 2.0, "score_absolute_error": 1.0,
            },
        ]
    )
    challenger = baseline.copy()
    challenger["home_score_absolute_error"] = [5.0, 0.0]
    challenger["score_absolute_error"] = [0.2, 1.4]

    paired = _paired_final_season(baseline, challenger, 2025)

    np.testing.assert_allclose(paired["score_absolute_error_delta"], [-0.3, 0.4])
    assert paired["week_block"].tolist() == ["Weeks 0-3", "Weeks 4-8"]
    intervals = _week_cluster_bootstrap(paired, replicates=100, seed=1)
    assert [row["block"] for row in intervals] == [
        "Overall", "Weeks 0-3", "Weeks 4-8", "Weeks 9+"
    ]
