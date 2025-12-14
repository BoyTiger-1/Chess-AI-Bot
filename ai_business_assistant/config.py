from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Central configuration for ETL jobs.

    This project intentionally keeps configuration file-light; callers can
    construct Settings directly or via env var wrappers in their own runtime.
    """

    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = Path("data")
    warehouse_path: Path = Path("data") / "warehouse.duckdb"

    raw_dir: Path = Path("data") / "raw"
    staging_dir: Path = Path("data") / "staging"
    archive_dir: Path = Path("data") / "archive"

    checkpoint_dir: Path = Path("data") / "checkpoints"

    def abs_path(self, p: Path) -> Path:
        if p.is_absolute():
            return p
        return self.project_root / p

    @property
    def abs_data_dir(self) -> Path:
        return self.abs_path(self.data_dir)

    @property
    def abs_warehouse_path(self) -> Path:
        return self.abs_path(self.warehouse_path)

    def ensure_dirs(self) -> None:
        for p in [
            self.abs_data_dir,
            self.abs_path(self.raw_dir),
            self.abs_path(self.staging_dir),
            self.abs_path(self.archive_dir),
            self.abs_path(self.checkpoint_dir),
        ]:
            p.mkdir(parents=True, exist_ok=True)
