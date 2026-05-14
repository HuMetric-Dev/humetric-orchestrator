from __future__ import annotations

from dataclasses import dataclass

from humetric_core import HumetricError


@dataclass(frozen=True, slots=True)
class BackendCallFailed(HumetricError):
    backend: str
    detail: str


@dataclass(frozen=True, slots=True)
class BackendUnavailable(HumetricError):
    backend: str
    reason: str


@dataclass(frozen=True, slots=True)
class BackendMisconfigured(HumetricError):
    backend: str
    detail: str


@dataclass(frozen=True, slots=True)
class ParseRejected(HumetricError):
    detail: str


@dataclass(frozen=True, slots=True)
class HistoryReadFailed(HumetricError):
    path: str
    reason: str


@dataclass(frozen=True, slots=True)
class HistoryWriteFailed(HumetricError):
    path: str
    reason: str


@dataclass(frozen=True, slots=True)
class FeedWriteFailed(HumetricError):
    detail: str


type OrchestratorError = (
    BackendCallFailed
    | BackendUnavailable
    | BackendMisconfigured
    | ParseRejected
    | HistoryReadFailed
    | HistoryWriteFailed
    | FeedWriteFailed
)
