import numpy as np
import pandas as pd
import pytest

from src.cfb_model.features.cfbd_game_features import (
    BASE_GAME_STAT_COLUMNS,
    EFFICIENCY_FEATURE_COLUMNS,
    PACE_FEATURE_COLUMNS,
    SPLIT_STAT_COLUMNS,
    build_cfbd_efficiency_features,
    build_cfbd_pace_features,
    cfbd_feature_data_dictionary,
    combine_cfbd_game_stat_sources,
)


def _game_stats(rows):
    base = []
    split = []
    for game_id, season, week, team, opponent, plays, ppa, pass_plays, pass_ppa in rows:
        base.append({
            "game_id": game_id,
            "season": season,
            "week": week,
            "team": team,
            "opponent": opponent,
            "offense_plays": plays,
            "offense_drives": 10,
            "offense_ppa": ppa,
            "offense_totalPPA": ppa * plays,
            "offense_successRate": 0.5,
            "offense_explosiveness": 1.0,
            "defense_plays": plays - 5,
            "defense_drives": 9,
            "defense_ppa": -0.1,
            "defense_totalPPA": -0.1 * (plays - 5),
            "defense_successRate": 0.4,
            "defense_explosiveness": 0.8,
        })
        split_row = {
            "game_id": game_id,
            "season": season,
            "week": week,
            "team": team,
            "opponent": opponent,
        }
        for field in SPLIT_STAT_COLUMNS:
            split_row[field] = 0.25
        split_row["offense_passingPlays_ppa"] = pass_ppa
        split_row["offense_passingPlays_totalPPA"] = pass_ppa * pass_plays
        split_row["offense_passingPlays_successRate"] = 0.4
        split_row["offense_passingPlays_explosiveness"] = 1.2
        split_row["offense_rushingPlays_ppa"] = 0.5
        split_row["offense_rushingPlays_totalPPA"] = 0.5 * (plays - pass_plays)
        split_row["offense_rushingPlays_successRate"] = 0.6
        split_row["offense_rushingPlays_explosiveness"] = 0.9
        split.append(split_row)
    return pd.DataFrame(base), pd.DataFrame(split)


def _fixture():
    games = pd.DataFrame(
        [
            {"game_id": 1, "season": 2023, "week": 1, "game_date": "2023-08-31",
             "home_team": "Alpha", "away_team": "Beta", "neutral_site": False},
            {"game_id": 2, "season": 2024, "week": 1, "game_date": "2024-08-31",
             "home_team": "Alpha", "away_team": "Beta", "neutral_site": False},
            {"game_id": 3, "season": 2024, "week": 2, "game_date": "2024-09-07",
             "home_team": "Alpha", "away_team": "Gamma", "neutral_site": True},
            {"game_id": 4, "season": 2024, "week": 2, "game_date": "2024-09-07",
             "home_team": "Alpha", "away_team": "Delta", "neutral_site": False},
            {"game_id": 5, "season": 2024, "week": 3, "game_date": "2024-09-14",
             "home_team": "Alpha", "away_team": "Epsilon", "neutral_site": False},
            {"game_id": 6, "season": 2024, "week": 4, "game_date": "2024-09-21",
             "home_team": "Alpha", "away_team": "Zeta", "neutral_site": False},
        ],
        index=[8, 3, 10, 4, 12, 5],
    )
    stat_rows = []
    for game_id, season, week, home, away, home_plays, home_ppa, pass_plays, pass_ppa in (
        (1, 2023, 1, "Alpha", "Beta", 80, 0.2, 40, 0.2),
        (2, 2024, 1, "Alpha", "Beta", 100, 0.5, 100, 0.4),
        (3, 2024, 2, "Alpha", "Gamma", 20, 0.1, 20, 0.2),
        (4, 2024, 2, "Alpha", "Delta", 150, 0.9, 100, 0.9),
        (5, 2024, 3, "Alpha", "Epsilon", 10, 0.0, 10, 0.0),
        (6, 2024, 4, "Alpha", "Zeta", 50, 0.2, 20, 0.3),
    ):
        for team, opponent, plays, ppa, passes, pass_ppa in (
            (home, away, home_plays, home_ppa, pass_plays, pass_ppa),
            (away, home, 60, 0.3, 30, 0.3),
        ):
            stat_rows.append(
                (game_id, season, week, team, opponent, plays, ppa, passes, pass_ppa)
            )
    base, split = _game_stats(stat_rows)
    return games, combine_cfbd_game_stat_sources(base, split)


def test_pace_and_efficiency_use_denominators_and_preserve_rows():
    games, stats = _fixture()
    pace = build_cfbd_pace_features(games, stats)
    efficiency = build_cfbd_efficiency_features(games, stats)

    assert pace.index.tolist() == games.index.tolist()
    assert efficiency.index.tolist() == games.index.tolist()
    assert pace.loc[10, "home_pace_offense_plays_per_game_s2d"] == pytest.approx(85.0)
    assert efficiency.loc[10, "home_eff_off_ppa_s2d"] == pytest.approx(98 / 340)
    assert efficiency.loc[10, "home_eff_off_pass_ppa_s2d"] == pytest.approx(64 / 220)
    assert np.isnan(pace.loc[10, "home_off_plays_minus_away_def_plays_s2d"])


def test_same_date_and_future_observations_do_not_change_target_features():
    games, stats = _fixture()
    baseline = build_cfbd_efficiency_features(games, stats).loc[10]
    changed = stats.copy()
    changed.loc[changed["game_id"].isin([3, 4, 5]), [
        "offense_totalPPA", "offense_ppa", "offense_successRate",
        "offense_explosiveness",
    ]] = [900.0, 9.0, 1.0, 9.0]
    updated = build_cfbd_efficiency_features(games, changed).loc[10]

    pd.testing.assert_series_equal(baseline, updated)


def test_season_to_date_uses_prior_season_shrinkage_and_recent_is_time_weighted():
    games, stats = _fixture()
    pace = build_cfbd_pace_features(games, stats)

    assert pace.loc[3, "home_pace_offense_plays_per_game_s2d"] == pytest.approx(80.0)
    assert pace.loc[10, "home_pace_offense_plays_per_game_s2d"] == pytest.approx(85.0)
    assert pace.loc[10, "home_pace_n_games_s2d"] == 1
    assert pace.loc[10, "home_pace_n_games_recent_ess"] > 1


def test_zero_split_ppa_has_missing_denominator_instead_of_bad_weighting():
    games, stats = _fixture()
    baseline = build_cfbd_efficiency_features(games, stats).loc[5]
    changed = stats.copy()
    changed.loc[
        (changed["game_id"] == 5) & (changed["team"] == "Alpha"),
        "offense_passingPlays_ppa",
    ] = 0.1
    updated = build_cfbd_efficiency_features(games, changed).loc[5]

    assert np.isfinite(baseline["home_eff_off_pass_ppa_s2d"])
    pd.testing.assert_series_equal(baseline, updated)


def test_stat_source_join_and_canonical_identity_are_validated():
    games, stats = _fixture()
    base, split = _game_stats([(1, 2023, 1, "Alpha", "Beta", 80, 0.2, 40, 0.2)])
    split = pd.concat([split, split], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        combine_cfbd_game_stat_sources(base, split)

    changed = stats.copy()
    changed.loc[changed["game_id"] == 1, "opponent"] = "Wrong opponent"
    with pytest.raises(ValueError, match="identity"):
        build_cfbd_pace_features(games, changed)


def test_feature_dictionary_covers_both_feature_groups():
    dictionary = cfbd_feature_data_dictionary()

    assert set(PACE_FEATURE_COLUMNS + EFFICIENCY_FEATURE_COLUMNS) <= set(dictionary["feature"])
    assert dictionary["availability_cutoff"].eq(
        "before the target UTC calendar-date batch"
    ).all()
