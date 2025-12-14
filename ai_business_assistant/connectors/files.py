from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class CsvConnector:
    path: Path
    read_kwargs: dict[str, Any] | None = None

    def read(self) -> pd.DataFrame:
        return pd.read_csv(self.path, **(self.read_kwargs or {}))

    def iter_chunks(self, *, chunksize: int = 250_000) -> Iterable[pd.DataFrame]:
        yield from pd.read_csv(self.path, chunksize=chunksize, **(self.read_kwargs or {}))


@dataclass(frozen=True)
class JsonLinesConnector:
    path: Path

    def iter_rows(self) -> Iterable[dict[str, Any]]:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def read_df(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.iter_rows()))
