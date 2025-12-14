from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import requests
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass(frozen=True)
class HttpApiConnector:
    """Small requests-based API connector with retries and pagination helpers."""

    base_url: str
    default_headers: dict[str, str] | None = None
    timeout_s: float = 20.0

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=0.5, min=0.5, max=8))
    def get_json(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")
        resp = requests.get(url, headers=self.default_headers, params=params, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def iter_paginated(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        next_page: Callable[[Any], dict[str, Any] | None] | None = None,
        sleep_s: float = 0.0,
    ) -> Iterable[Any]:
        current_params: dict[str, Any] = dict(params or {})
        while True:
            payload = self.get_json(path, params=current_params)
            yield payload
            if next_page is None:
                break
            nxt = next_page(payload)
            if not nxt:
                break
            current_params.update(nxt)
            if sleep_s:
                time.sleep(sleep_s)


@dataclass(frozen=True)
class MockApiConnector:
    """Deterministic in-memory connector for tests and offline runs."""

    routes: dict[tuple[str, str], Any]

    def get_json(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        key = (path, "" if not params else str(sorted(params.items())))
        if key not in self.routes:
            raise KeyError(f"No mock route for {key}")
        return self.routes[key]
