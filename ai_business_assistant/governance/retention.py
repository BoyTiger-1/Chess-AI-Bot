from __future__ import annotations

import gzip
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass(frozen=True)
class RetentionPolicy:
    """Simple file-based retention policy for raw/staging partitions."""

    keep_days: int = 365
    archive: bool = True


def apply_retention_policy(
    *,
    raw_dir: Path,
    archive_dir: Path,
    policy: RetentionPolicy,
    now: datetime | None = None,
) -> list[Path]:
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=policy.keep_days)

    archived: list[Path] = []
    if not raw_dir.exists():
        return archived

    archive_dir.mkdir(parents=True, exist_ok=True)

    for p in raw_dir.rglob("*"):
        if not p.is_file():
            continue
        mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        if mtime >= cutoff:
            continue

        if policy.archive:
            gz_path = archive_dir / (p.name + ".gz")
            with p.open("rb") as src, gzip.open(gz_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            archived.append(gz_path)

        p.unlink(missing_ok=True)

    return archived
