from src.cfb_model.train.final_combined_2026_experiment import (
    _completed_fbs_games,
)


def test_completed_fbs_schedule_keeps_fcs_games_but_excludes_them_as_targets():
    games = _completed_fbs_games([
        {
            "id": 101,
            "week": 1,
            "startDate": "2026-09-01T00:00:00Z",
            "homeTeam": "FBS A",
            "awayTeam": "FBS B",
            "homePoints": 24,
            "awayPoints": 17,
            "homeClassification": "fbs",
            "awayClassification": "fbs",
        },
        {
            "id": 102,
            "week": 1,
            "startDate": "2026-09-02T00:00:00Z",
            "homeTeam": "FBS A",
            "awayTeam": "FCS C",
            "homePoints": 35,
            "awayPoints": 10,
            "homeClassification": "fbs",
            "awayClassification": "fcs",
        },
        {
            "id": 103,
            "week": 1,
            "startDate": "2026-09-03T00:00:00Z",
            "homeTeam": "FBS A",
            "awayTeam": "FBS D",
            "homePoints": None,
            "awayPoints": None,
            "homeClassification": "fbs",
            "awayClassification": "fbs",
        },
        {
            "id": 104,
            "week": 1,
            "startDate": "2026-09-04T00:00:00Z",
            "homeTeam": "FCS E",
            "awayTeam": "FCS F",
            "homePoints": 21,
            "awayPoints": 14,
            "homeClassification": "fcs",
            "awayClassification": "fcs",
        },
    ])

    assert games["game_id"].tolist() == [101, 102]
    assert (games["home_is_fbs"] & games["away_is_fbs"]).sum() == 1
