"""Build the local DuckDB warehouse from the versioned raw Parquet extracts."""

from __future__ import annotations

from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_DB_PATH = PROJECT_ROOT / "db" / "warehouse.duckdb"


def ensure_warehouse(db_path: Path = DEFAULT_DB_PATH, raw_dir: Path = RAW_DIR) -> Path:
    """Create a read-only-consumable warehouse when it is absent.

    The raw Parquet files are the source of truth. Existing warehouses are left
    untouched so prediction/backtest commands remain non-destructive.
    """
    if db_path.exists():
        return db_path
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    parquet_files = sorted(raw_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {raw_dir}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as connection:
        for parquet_path in parquet_files:
            table = parquet_path.stem
            connection.execute(
                f'CREATE OR REPLACE TABLE "{table}" AS '
                f"SELECT * FROM read_parquet(?)",
                [str(parquet_path)],
            )
    return db_path
