"""Shared tool primitives for Phase 2.

Provides:
- typed tool configuration
- a tiny in-memory TTL cache
- retry wrapper with bounded attempts
- a uniform `execute` API for all tools
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

RequestModelT = TypeVar("RequestModelT", bound=BaseModel)
ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


class ToolMode(StrEnum):
    """Runtime mode for all tools."""

    MOCK = "mock"
    REAL = "real"


class ToolConfig(BaseModel):
    """Cross-cutting behavior for tools."""

    model_config = ConfigDict(frozen=True)

    mode: ToolMode = ToolMode.MOCK
    timeout_seconds: float = Field(default=3.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    retry_backoff_seconds: float = Field(default=0.01, ge=0)
    cache_ttl_seconds: float = Field(default=120.0, gt=0)


class ToolError(RuntimeError):
    """Raised when a tool cannot complete after retries."""


class ToolTimeoutError(ToolError):
    """Raised when execution time exceeds configured timeout."""


class ToolRateLimitError(ToolError):
    """Raised when upstream reports a rate limit response."""


class ToolUpstreamError(ToolError):
    """Raised when upstream service returns a non-retryable server failure."""


class ToolPayloadError(ToolError):
    """Raised when a tool response cannot be validated."""


class ToolConfigurationError(ToolError):
    """Raised when tool runtime mode is unsupported or misconfigured."""


@dataclass(slots=True)
class ToolStats:
    """Lightweight counters useful for tests and diagnostics."""

    attempts: int = 0
    cache_hits: int = 0
    mock_calls: int = 0
    real_calls: int = 0


class _TTLCache(Generic[ResponseModelT]):
    """Very small in-memory TTL cache for deterministic repeated calls."""

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl_seconds = ttl_seconds
        self._items: dict[str, tuple[float, ResponseModelT]] = {}

    def get(self, key: str) -> ResponseModelT | None:
        entry = self._items.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at < time.monotonic():
            del self._items[key]
            return None
        return value

    def set(self, key: str, value: ResponseModelT) -> None:
        expires_at = time.monotonic() + self._ttl_seconds
        self._items[key] = (expires_at, value)


class BaseTool(ABC, Generic[RequestModelT, ResponseModelT]):
    """Base class for all typed tools with uniform reliability behavior."""

    name: str
    request_model: type[RequestModelT]
    response_model: type[ResponseModelT]

    def __init__(self, config: ToolConfig | None = None) -> None:
        self.config = config or ToolConfig()
        self.stats = ToolStats()
        self._cache: _TTLCache[ResponseModelT] = _TTLCache(self.config.cache_ttl_seconds)

    def execute(self, request: RequestModelT | dict[str, object]) -> ResponseModelT:
        """Validate input, apply cache/retry policy, and return typed output."""
        validated_request = self.request_model.model_validate(request)
        cache_key = self._cache_key(validated_request)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached

        max_attempts = self.config.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            self.stats.attempts += 1
            try:
                raw = self._invoke_with_timeout(validated_request)
                validated_response = self.response_model.model_validate(raw)
                self._cache.set(cache_key, validated_response)
                return validated_response
            except Exception as exc:  # pragma: no cover - loop behavior tested directly
                normalized = self._normalize_error(exc)
                last_error = normalized
                if isinstance(normalized, ToolConfigurationError):
                    raise normalized from exc
                if attempt + 1 >= max_attempts:
                    break
                if self.config.retry_backoff_seconds > 0:
                    time.sleep(self.config.retry_backoff_seconds)

        if isinstance(
            last_error,
            ToolConfigurationError
            | ToolTimeoutError
            | ToolRateLimitError
            | ToolUpstreamError
            | ToolPayloadError,
        ):
            raise last_error
        raise ToolError(f"{self.name} failed after {max_attempts} attempts") from last_error

    def _cache_key(self, request: RequestModelT) -> str:
        payload = request.model_dump(mode="json")
        return f"{self.name}:{json.dumps(payload, sort_keys=True, separators=(',', ':'))}"

    def _call_backend(self, request: RequestModelT) -> ResponseModelT | dict[str, object]:
        if self.config.mode == ToolMode.MOCK:
            self.stats.mock_calls += 1
            return self._run_mock(request)
        self.stats.real_calls += 1
        return self._run_real(request)

    def _invoke_with_timeout(self, request: RequestModelT) -> ResponseModelT | dict[str, object]:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._call_backend, request)
        try:
            return future.result(timeout=self.config.timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"{self.name} exceeded timeout {self.config.timeout_seconds:.2f}s"
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _normalize_error(self, exc: Exception) -> ToolError:
        if isinstance(exc, ToolError):
            return exc
        if isinstance(exc, TimeoutError):
            return ToolTimeoutError(str(exc))
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return ToolRateLimitError(f"{self.name} rate limited by upstream")
        if isinstance(status_code, int) and status_code >= 500:
            return ToolUpstreamError(f"{self.name} upstream failure (status={status_code})")
        if isinstance(exc, ValueError | TypeError):
            return ToolPayloadError(f"{self.name} produced invalid payload: {exc}")
        if isinstance(exc, ConnectionError):
            return ToolUpstreamError(f"{self.name} network failure: {exc}")
        return ToolError(f"{self.name} execution failed: {exc}")

    @abstractmethod
    def _run_mock(self, request: RequestModelT) -> ResponseModelT | dict[str, object]:
        """Mock implementation used in tests/dev."""

    def _run_real(self, request: RequestModelT) -> ResponseModelT | dict[str, object]:
        """Default real-mode behavior for phase 2.

        Real integrations are deferred to later phases. We fail explicitly so
        real mode never silently falls back to mocked data.
        """
        raise ToolConfigurationError(f"{self.name} does not support real mode yet")
