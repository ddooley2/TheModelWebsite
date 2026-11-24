import pandas as pd

from src.cfb_model.train.preseason_score_prior_experiment import (
    _schedule_after_fcs_flags,
    _week_block,
)


def test_fbs_after_fcs_subset_uses_only_strictly_prior_schedule_games():
    targets = pd.DataFrame(
        [
            {"game_date": "2024-09-01", "home_team": "Alpha", "away_team": "Beta"},
            {"game_date": "2024-09-07", "home_team": "Alpha", "away_team": "Beta"},
            {"game_date": "2024-09-14", "home_team": "Alpha", "away_team": "Beta"},
        ],
        index=[5, 6, 7],
    )
    schedule = pd.DataFrame(
        [
            {
                "game_date": "2024-08-31",
                "home_team": "Alpha",
                "away_team": "Gamma",
                "home_is_fbs": True,
                "away_is_fbs": True,
            },
            {
                "game_date": "2024-09-01",
                "home_team": "Alpha",
                "away_team": "FCS Team",
                "home_is_fbs": True,
                "away_is_fbs": False,
            },
        ]
    )

    flags = _schedule_after_fcs_flags(targets, schedule)

    assert flags.tolist() == [False, True, True]
    assert flags.index.tolist() == [5, 6, 7]


def test_week_ranges_isolate_opening_weeks():
    assert [_week_block(week) for week in (0, 3, 4, 6, 7, 12)] == [
        "Weeks 0-3",
        "Weeks 0-3",
        "Weeks 4-6",
        "Weeks 4-6",
        "Weeks 7+",
        "Weeks 7+",
    ]
