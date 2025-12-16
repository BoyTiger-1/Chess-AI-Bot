from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

import requests

logger = logging.getLogger(__name__)


class AlertSink(Protocol):
    def send(self, *, title: str, message: str, context: dict[str, Any] | None = None) -> None: ...


@dataclass(frozen=True)
class StdoutAlertSink:
    def send(self, *, title: str, message: str, context: dict[str, Any] | None = None) -> None:
        payload = {"title": title, "message": message, "context": context or {}}
        logger.error("ALERT: %s", json.dumps(payload, sort_keys=True))


@dataclass(frozen=True)
class WebhookAlertSink:
    url: str
    timeout_s: float = 10.0

    def send(self, *, title: str, message: str, context: dict[str, Any] | None = None) -> None:
        payload = {"title": title, "message": message, "context": context or {}}
        try:
            resp = requests.post(self.url, json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
        except Exception:  # noqa: BLE001
            logger.exception("Failed sending webhook alert")
