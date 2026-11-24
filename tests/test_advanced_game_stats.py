import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from src.cfb_model.data import advanced_game_stats as ingestion


FIXTURE = Path(__file__).parent / "fixtures" / "cfbd_advanced_game.json"


def _canonical_games(game_id=401000001):
    return pd.DataFrame([
        {
            "game_id": game_id,
            "season": 2025,
            "week": 1,
            "game_date": "2025-08-30T16:00:00Z",
            "home_team": "Alpha",
            "away_team": "Beta",
            "home_score": 28.0,
            "away_score": 14.0,
        }
    ])


def test_normalize_advanced_game_response_flattens_nested_metrics():
    payload = json.loads(FIXTURE.read_text())

    stats = ingestion.normalize_advanced_game_stats(payload, 2025)

    assert stats[["game_id", "season", "season_type", "week", "team", "opponent"]].iloc[0].to_dict() == {
        "game_id": 401000001,
        "season": 2025,
        "season_type": "regular",
        "week": 1,
        "team": "Alpha",
        "opponent": "Beta",
    }
    assert stats.loc[0, "offense_passingPlays_ppa"] == 0.31
    assert stats.loc[0, "offense_plays"] == 60
    assert stats.loc[1, "defense_ppa"] == -0.21


def test_normalize_rejects_nonreciprocal_or_duplicate_team_records():
    payload = json.loads(FIXTURE.read_text())
    payload[1]["opponent"] = "Gamma"
    with pytest.raises(ValueError, match="not reciprocal"):
        ingestion.normalize_advanced_game_stats(payload, 2025)

    payload = json.loads(FIXTURE.read_text())
    payload[1]["team"] = "Alpha"
    payload[1]["opponent"] = "Beta"
    with pytest.raises(ValueError, match="duplicate"):
        ingestion.normalize_advanced_game_stats(payload, 2025)


def test_reconcile_matches_by_id_then_unique_season_week_matchup():
    stats = ingestion.normalize_advanced_game_stats(json.loads(FIXTURE.read_text()), 2025)

    by_id, id_report = ingestion.reconcile_advanced_game_stats(
        stats, _canonical_games()
    )
    assert set(by_id["match_method"]) == {"game_id"}
    assert id_report["match_rate"] == 1.0

    by_matchup, matchup_report = ingestion.reconcile_advanced_game_stats(
        stats, _canonical_games(game_id=987654321)
    )
    assert set(by_matchup["match_method"]) == {"season_week_team_opponent"}
    assert matchup_report["records_matched"] == 2
    assert by_matchup["game_date"].notna().all()


def test_reconcile_trusts_game_id_when_non_fbs_name_differs():
    stats = ingestion.normalize_advanced_game_stats(json.loads(FIXTURE.read_text()), 2025)
    games = _canonical_games().assign(
        away_team="Beta alternate",
        home_is_fbs=True,
        away_is_fbs=False,
    )

    matched, report = ingestion.reconcile_advanced_game_stats(stats, games)

    assert len(matched) == 2
    assert set(matched["match_method"]) == {"game_id"}
    assert report["match_rate"] == 1.0


def test_reconcile_rejects_ambiguous_fallback_and_partial_game():
    stats = ingestion.normalize_advanced_game_stats(json.loads(FIXTURE.read_text()), 2025)
    duplicate_games = pd.concat(
        [_canonical_games(game_id=111), _canonical_games(game_id=222)],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="multiple canonical game matches"):
        ingestion.reconcile_advanced_game_stats(stats, duplicate_games)

    one_side = stats.iloc[:1].copy()
    with pytest.raises(ValueError, match="1 matched stat rows"):
        ingestion.reconcile_advanced_game_stats(
            one_side, _canonical_games(game_id=987654321)
        )


def test_cache_round_trip_preserves_raw_response_and_avoids_network(tmp_path, monkeypatch):
    payload_bytes = FIXTURE.read_bytes()

    class Response:
        content = payload_bytes

        def raise_for_status(self):
            pass

    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr(ingestion.requests, "get", get)
    stats, manifest = ingestion.refresh_advanced_game_stats(
        2025,
        cache_root=tmp_path,
        api_key="test-key",
        canonical_games=_canonical_games(),
    )
    paths = ingestion._cache_paths(2025, tmp_path)
    assert paths["response"].read_bytes() == payload_bytes
    assert manifest["response_sha256"] == hashlib.sha256(payload_bytes).hexdigest()
    assert manifest["request_parameters"] == {
        "year": 2025,
        "seasonType": "both",
        "excludeGarbageTime": False,
    }
    assert manifest["reconciliation"]["match_rate"] == 1.0
    assert calls[0][1]["params"]["excludeGarbageTime"] is False

    def unexpected_request(*args, **kwargs):
        raise AssertionError("Cache read made a network request.")

    monkeypatch.setattr(ingestion.requests, "get", unexpected_request)
    cached = ingestion.load_cached_advanced_game_stats(2025, tmp_path)
    pd.testing.assert_frame_equal(stats, cached)


def test_cached_data_hash_mismatch_fails_loudly(tmp_path, monkeypatch):
    payload_bytes = FIXTURE.read_bytes()

    class Response:
        content = payload_bytes

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        ingestion.requests, "get", lambda *args, **kwargs: Response()
    )
    ingestion.refresh_advanced_game_stats(
        2025, cache_root=tmp_path, api_key="test-key"
    )
    paths = ingestion._cache_paths(2025, tmp_path)
    paths["response"].write_bytes(payload_bytes + b" ")

    with pytest.raises(ValueError, match="raw response hash mismatch"):
        ingestion.load_cached_advanced_game_stats(2025, tmp_path)


@pytest.mark.skipif(
    os.environ.get("CFBD_LIVE_INTEGRATION") != "1"
    or not os.environ.get("CFBD_API_KEY"),
    reason="Set CFBD_LIVE_INTEGRATION=1 and CFBD_API_KEY to run the live integration check.",
)
def test_live_advanced_game_ingestion_is_opt_in(tmp_path):
    season = int(os.environ.get("CFBD_LIVE_TEST_SEASON", "2025"))

    stats, manifest = ingestion.refresh_advanced_game_stats(
        season,
        cache_root=tmp_path,
        api_key=os.environ["CFBD_API_KEY"],
    )

    assert stats["season"].eq(season).all()
    assert manifest["exclude_garbage_time"] is False
