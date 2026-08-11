"""Serializable domain models for sessions, context plans, and token usage."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


SCHEMA_VERSION = 1
DEFAULT_SYSTEM_PROMPT = "你是一个有帮助、诚实且简洁的 AI 助手。"


def utc_now() -> str:
    """Return a stable, timezone-aware ISO 8601 timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _require_dict(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_str(value: object, name: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{name} must be a string")
    return value


def _optional_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or null")
    return value


@dataclass(slots=True)
class BudgetConfig:
    """Token-budget policy for a session."""

    context_window: int = 32_768
    max_output_tokens: int = 4_096
    safety_margin_tokens: int = 512
    probe_ratio: float = 0.80
    warning_ratio: float = 0.90
    trim_target_ratio: float = 0.80

    def __post_init__(self) -> None:
        for name in ("context_window", "max_output_tokens", "safety_margin_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.context_window <= self.max_output_tokens + self.safety_margin_tokens:
            raise ValueError("context_window must exceed output reserve plus safety margin")
        if not 0 < self.trim_target_ratio <= self.probe_ratio <= self.warning_ratio < 1:
            raise ValueError(
                "ratios must satisfy 0 < trim_target <= probe <= warning < 1"
            )

    @property
    def input_budget(self) -> int:
        return self.context_window - self.max_output_tokens - self.safety_margin_tokens

    @property
    def probe_threshold(self) -> int:
        return int(self.input_budget * self.probe_ratio)

    @property
    def warning_threshold(self) -> int:
        return int(self.input_budget * self.warning_ratio)

    @property
    def trim_target(self) -> int:
        return int(self.input_budget * self.trim_target_ratio)

    def to_dict(self) -> dict[str, int | float]:
        return {
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
            "probe_ratio": self.probe_ratio,
            "warning_ratio": self.warning_ratio,
            "trim_target_ratio": self.trim_target_ratio,
        }

    @classmethod
    def from_dict(cls, value: object) -> BudgetConfig:
        data = _require_dict(value, "budget")
        try:
            return cls(
                context_window=int(data["context_window"]),
                max_output_tokens=int(data["max_output_tokens"]),
                safety_margin_tokens=int(data["safety_margin_tokens"]),
                probe_ratio=float(data["probe_ratio"]),
                warning_ratio=float(data["warning_ratio"]),
                trim_target_ratio=float(data["trim_target_ratio"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid budget configuration: {exc}") from exc


@dataclass(slots=True)
class TokenUsage:
    """Token counts reported by Ollama for one request."""

    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        self.input_tokens = _optional_int(self.input_tokens, "input_tokens")
        self.output_tokens = _optional_int(self.output_tokens, "output_tokens")

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens

    def add(self, other: TokenUsage) -> None:
        """Accumulate known counts, preserving unknown values."""

        if other.input_tokens is not None:
            self.input_tokens = (self.input_tokens or 0) + other.input_tokens
        if other.output_tokens is not None:
            self.output_tokens = (self.output_tokens or 0) + other.output_tokens

    def to_dict(self) -> dict[str, int | None]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, value: object) -> TokenUsage:
        data = _require_dict(value, "token usage")
        return cls(
            input_tokens=_optional_int(data.get("input_tokens"), "input_tokens"),
            output_tokens=_optional_int(data.get("output_tokens"), "output_tokens"),
        )


TurnStatus = Literal[
    "pending",
    "complete",
    "truncated",
    "blocked",
    "interrupted",
    "failed",
    "no_answer",
]


@dataclass(slots=True)
class ContextTrace:
    """Audit record describing the exact context decision for a turn."""

    included_turn_ids: list[str] = field(default_factory=list)
    omitted_turn_ids: list[str] = field(default_factory=list)
    estimated_upper_tokens: int | None = None
    exact_input_tokens: int | None = None
    input_budget: int = 0
    decision: str = "none"
    active_context_start_before: int = 0
    active_context_start_after: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "included_turn_ids": list(self.included_turn_ids),
            "omitted_turn_ids": list(self.omitted_turn_ids),
            "estimated_upper_tokens": self.estimated_upper_tokens,
            "exact_input_tokens": self.exact_input_tokens,
            "input_budget": self.input_budget,
            "decision": self.decision,
            "active_context_start_before": self.active_context_start_before,
            "active_context_start_after": self.active_context_start_after,
        }

    @classmethod
    def from_dict(cls, value: object) -> ContextTrace:
        data = _require_dict(value, "context trace")
        included = data.get("included_turn_ids", [])
        omitted = data.get("omitted_turn_ids", [])
        if not isinstance(included, list) or not all(isinstance(item, str) for item in included):
            raise ValueError("included_turn_ids must be a string array")
        if not isinstance(omitted, list) or not all(isinstance(item, str) for item in omitted):
            raise ValueError("omitted_turn_ids must be a string array")
        return cls(
            included_turn_ids=list(included),
            omitted_turn_ids=list(omitted),
            estimated_upper_tokens=_optional_int(
                data.get("estimated_upper_tokens"), "estimated_upper_tokens"
            ),
            exact_input_tokens=_optional_int(
                data.get("exact_input_tokens"), "exact_input_tokens"
            ),
            input_budget=int(data.get("input_budget", 0)),
            decision=_require_str(data.get("decision", "none"), "decision"),
            active_context_start_before=int(data.get("active_context_start_before", 0)),
            active_context_start_after=int(data.get("active_context_start_after", 0)),
        )


@dataclass(slots=True)
class Turn:
    """A user request, its assistant result, and the associated trace."""

    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    status: TurnStatus = "pending"
    user_content: str = ""
    assistant_content: str = ""
    thinking: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    probe_usage: TokenUsage = field(default_factory=lambda: TokenUsage(0, 0))
    context_trace: ContextTrace = field(default_factory=ContextTrace)
    done_reason: str | None = None
    error: str | None = None

    @property
    def context_eligible(self) -> bool:
        return self.status in {"complete", "truncated"} and bool(self.assistant_content)

    def touch(self) -> None:
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "user_content": self.user_content,
            "assistant_content": self.assistant_content,
            "thinking": self.thinking,
            "usage": self.usage.to_dict(),
            "probe_usage": self.probe_usage.to_dict(),
            "context_trace": self.context_trace.to_dict(),
            "done_reason": self.done_reason,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: object) -> Turn:
        data = _require_dict(value, "turn")
        status = _require_str(data.get("status"), "turn.status", allow_empty=False)
        valid_statuses = {
            "pending",
            "complete",
            "truncated",
            "blocked",
            "interrupted",
            "failed",
            "no_answer",
        }
        if status not in valid_statuses:
            raise ValueError(f"unsupported turn status: {status}")
        done_reason = data.get("done_reason")
        error = data.get("error")
        if done_reason is not None and not isinstance(done_reason, str):
            raise ValueError("turn.done_reason must be a string or null")
        if error is not None and not isinstance(error, str):
            raise ValueError("turn.error must be a string or null")
        return cls(
            id=_require_str(data.get("id"), "turn.id", allow_empty=False),
            created_at=_require_str(data.get("created_at"), "turn.created_at"),
            updated_at=_require_str(data.get("updated_at"), "turn.updated_at"),
            status=status,  # type: ignore[arg-type]
            user_content=_require_str(data.get("user_content", ""), "turn.user_content"),
            assistant_content=_require_str(
                data.get("assistant_content", ""), "turn.assistant_content"
            ),
            thinking=_require_str(data.get("thinking", ""), "turn.thinking"),
            usage=TokenUsage.from_dict(data.get("usage", {})),
            probe_usage=TokenUsage.from_dict(data.get("probe_usage", {})),
            context_trace=ContextTrace.from_dict(data.get("context_trace", {})),
            done_reason=done_reason,
            error=error,
        )


SessionStatus = Literal["active", "paused"]


@dataclass(slots=True)
class Session:
    """A complete, file-backed conversation with an independent context cursor."""

    id: str
    model: str = "qwen3.5:9b"
    ollama_host: str = "http://127.0.0.1:11434"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    think: bool = True
    title: str = "新会话"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    status: SessionStatus = "active"
    active_context_start_index: int = 0
    turns: list[Turn] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("session id cannot be empty")
        if not 0 <= self.active_context_start_index <= len(self.turns):
            raise ValueError("active_context_start_index is out of range")

    def touch(self) -> None:
        self.updated_at = utc_now()

    def eligible_turns(
        self, *, before_index: int | None = None, honor_active_start: bool = True
    ) -> list[tuple[int, Turn]]:
        start = self.active_context_start_index if honor_active_start else 0
        stop = len(self.turns) if before_index is None else before_index
        return [
            (index, turn)
            for index, turn in enumerate(self.turns[start:stop], start=start)
            if turn.context_eligible
        ]

    def cumulative_usage(self) -> TokenUsage:
        total = TokenUsage(0, 0)
        for turn in self.turns:
            if turn.usage.input_tokens is not None and turn.usage.output_tokens is not None:
                total.add(turn.usage)
        return total

    def cumulative_probe_usage(self) -> TokenUsage:
        total = TokenUsage(0, 0)
        for turn in self.turns:
            total.add(turn.probe_usage)
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "model": self.model,
            "ollama_host": self.ollama_host,
            "system_prompt": self.system_prompt,
            "think": self.think,
            "budget": self.budget.to_dict(),
            "active_context_start_index": self.active_context_start_index,
            "turns": [turn.to_dict() for turn in self.turns],
            "cumulative_usage": self.cumulative_usage().to_dict(),
            "cumulative_probe_usage": self.cumulative_probe_usage().to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Session:
        data = _require_dict(value, "session")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported session schema version: {data.get('schema_version')!r}"
            )
        raw_turns = data.get("turns")
        if not isinstance(raw_turns, list):
            raise ValueError("session.turns must be an array")
        status = _require_str(data.get("status"), "session.status", allow_empty=False)
        if status not in {"active", "paused"}:
            raise ValueError(f"unsupported session status: {status}")
        think = data.get("think")
        if not isinstance(think, bool):
            raise ValueError("session.think must be a boolean")
        turns = [Turn.from_dict(item) for item in raw_turns]
        session = cls(
            id=_require_str(data.get("id"), "session.id", allow_empty=False),
            title=_require_str(data.get("title", "新会话"), "session.title"),
            created_at=_require_str(data.get("created_at"), "session.created_at"),
            updated_at=_require_str(data.get("updated_at"), "session.updated_at"),
            status=status,  # type: ignore[arg-type]
            model=_require_str(data.get("model"), "session.model", allow_empty=False),
            ollama_host=_require_str(
                data.get("ollama_host"), "session.ollama_host", allow_empty=False
            ),
            system_prompt=_require_str(
                data.get("system_prompt", ""), "session.system_prompt"
            ),
            think=think,
            budget=BudgetConfig.from_dict(data.get("budget")),
            active_context_start_index=int(data.get("active_context_start_index", 0)),
            turns=turns,
        )
        return session


@dataclass(slots=True)
class ContextPlan:
    """Messages selected for one model request and their budget metadata."""

    messages: list[dict[str, str]]
    included_turn_ids: list[str]
    omitted_turn_ids: list[str]
    selected_history_indices: list[int]
    estimated_upper_tokens: int | None = None
    exact_input_tokens: int | None = None
    input_budget: int = 0

    @property
    def usage_ratio(self) -> float | None:
        value = self.exact_input_tokens
        if value is None:
            value = self.estimated_upper_tokens
        if value is None or self.input_budget <= 0:
            return None
        return value / self.input_budget


EventKind = Literal["thinking", "content", "usage", "completed"]


@dataclass(slots=True)
class ChatEvent:
    """A streaming event emitted by the Ollama client and ChatEngine."""

    kind: EventKind
    text: str = ""
    live_output_tokens: int | None = None
    usage: TokenUsage | None = None
    done_reason: str | None = None
