from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ai_business_assistant.monitoring.alerts import AlertSink, StdoutAlertSink


@dataclass
class PollingIngestor:
    """Polling-based ingestion with checkpointing.

    Designed for sources without streaming APIs. The ingest_fn should return a DataFrame
    containing new records since the last checkpoint.
    """

    checkpoint_path: Path
    ingest_fn: Callable[[dict[str, Any]], pd.DataFrame]
    sink_fn: Callable[[pd.DataFrame], None]
    alert_sink: AlertSink = StdoutAlertSink()

    def load_checkpoint(self) -> dict[str, Any]:
        if not self.checkpoint_path.exists():
            return {"last_run": None}
        return json.loads(self.checkpoint_path.read_text(encoding="utf-8"))

    def save_checkpoint(self, state: dict[str, Any]) -> None:
        tmp = self.checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.checkpoint_path)

    def run_once(self) -> int:
        state = self.load_checkpoint()
        try:
            df = self.ingest_fn(state)
            if df.empty:
                state["last_run"] = datetime.now(timezone.utc).isoformat()
                self.save_checkpoint(state)
                return 0
            self.sink_fn(df)
            state["last_run"] = datetime.now(timezone.utc).isoformat()
            state["last_row_count"] = int(df.shape[0])
            self.save_checkpoint(state)
            return int(df.shape[0])
        except Exception as e:  # noqa: BLE001
            self.alert_sink.send(
                title="Polling ingestion failed",
                message=str(e),
                context={"checkpoint_path": str(self.checkpoint_path)},
            )
            raise

    def run_forever(self, *, interval_s: float = 5.0) -> None:
        while True:
            self.run_once()
            time.sleep(interval_s)
