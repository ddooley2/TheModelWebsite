import pandas as pd
import pytest

from src.cfb_model.reporting.diagnostics import plot_mae_by_season_comparison


def test_plot_mae_by_season_comparison_saves_paired_chart(tmp_path):
    metrics = pd.DataFrame([
        {"candidate": "baseline", "season": 2024, "score_mae": 9.2},
        {"candidate": "baseline", "season": 2025, "score_mae": 9.1},
        {"candidate": "stage2", "season": 2024, "score_mae": 9.2},
        {"candidate": "stage2", "season": 2025, "score_mae": 9.1},
    ])
    path = tmp_path / "comparison.png"

    plot_mae_by_season_comparison(metrics, path)

    assert path.is_file()
    assert path.stat().st_size > 0


def test_plot_mae_by_season_comparison_rejects_mismatched_folds(tmp_path):
    metrics = pd.DataFrame([
        {"candidate": "baseline", "season": 2024, "score_mae": 9.2},
        {"candidate": "stage2", "season": 2025, "score_mae": 9.1},
    ])

    with pytest.raises(ValueError, match="every season"):
        plot_mae_by_season_comparison(metrics, tmp_path / "comparison.png")
