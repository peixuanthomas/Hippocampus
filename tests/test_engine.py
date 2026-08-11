from __future__ import annotations

from collections.abc import Callable

import pytest

from hippocampus.engine import ChatEngine, LimitAction
from hippocampus.models import BudgetConfig, ChatEvent, Session, TokenUsage, Turn
from hippocampus.ollama_client import OllamaConnectionError
from hippocampus.store import SessionStore


def eligible_turn(index: int) -> Turn:
    return Turn(
        status="complete",
        user_content=f"user-{index}",
        assistant_content=f"assistant-{index}",
        thinking=f"thinking-{index}",
        usage=TokenUsage(10, 5),
    )


class FakeClient:
    def __init__(
        self,
        count: int | Callable[[list[dict[str, str]]], int] = 100,
        *,
        render_supported: bool = True,
        stream_events: list[ChatEvent] | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        self.count = count
        self.render_supported = render_supported
        self.stream_events = stream_events or [
            ChatEvent(kind="content", text="answer", live_output_tokens=2),
            ChatEvent(kind="usage", live_output_tokens=2),
            ChatEvent(
                kind="completed",
                live_output_tokens=2,
                usage=TokenUsage(100, 2),
                done_reason="stop",
            ),
        ]
        self.stream_error = stream_error
        self.probe_calls: list[list[dict[str, str]]] = []
        self.stream_calls: list[dict[str, object]] = []

    def token_count(self, messages: list[dict[str, str]]) -> int:
        return self.count(messages) if callable(self.count) else self.count

    def render_prompt(self, **kwargs) -> str | None:
        if not self.render_supported:
            return None
        return "x" * self.token_count(kwargs["messages"])

    def probe(self, **kwargs) -> TokenUsage:
        messages = kwargs["messages"]
        self.probe_calls.append(messages)
        return TokenUsage(self.token_count(messages), 1)

    def stream_chat(self, **kwargs):
        self.stream_calls.append(kwargs)
        for event in self.stream_events:
            yield event
        if self.stream_error is not None:
            raise self.stream_error


def small_budget() -> BudgetConfig:
    return BudgetConfig(
        context_window=1_000,
        max_output_tokens=100,
        safety_margin_tokens=0,
        probe_ratio=0.80,
        warning_ratio=0.90,
        trim_target_ratio=0.80,
    )


def make_session(store: SessionStore, *, turns: list[Turn] | None = None) -> Session:
    session = store.create(budget=small_budget(), think=True)
    session.turns.extend(turns or [])
    store.save(session)
    return session


def test_below_probe_threshold_does_not_probe(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    client = FakeClient(count=100)
    session = make_session(store)

    prepared = ChatEngine(store, client).prepare_turn(session, "hello")

    assert prepared.ready
    assert prepared.plan.exact_input_tokens is None
    assert client.probe_calls == []


def test_render_fallback_forces_exact_probe(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    client = FakeClient(count=100, render_supported=False)
    session = make_session(store)

    prepared = ChatEngine(store, client).prepare_turn(session, "hello")

    assert prepared.ready
    assert prepared.plan.exact_input_tokens == 100
    assert len(client.probe_calls) == 1
    assert session.turns[-1].probe_usage.total_tokens == 101


def test_probe_threshold_is_inclusive_and_warning_threshold_is_exact(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    probe_client = FakeClient(count=720)
    probe_session = make_session(store)

    prepared = ChatEngine(store, probe_client).prepare_turn(probe_session, "at 80%")

    assert prepared.ready
    assert prepared.plan.exact_input_tokens == 720
    assert len(probe_client.probe_calls) == 1

    warning_client = FakeClient(count=810)
    warning_session = make_session(store)

    warned = ChatEngine(store, warning_client).prepare_turn(
        warning_session, "at 90%"
    )

    assert warned.needs_limit_decision
    assert warned.plan.exact_input_tokens == 810
    assert warning_client.stream_calls == []


def test_warning_can_end_without_generation(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    client = FakeClient(count=850)
    session = make_session(store)
    engine = ChatEngine(store, client)

    prepared = engine.prepare_turn(session, "near limit")
    ended = engine.resolve_limit(session, prepared, LimitAction.END_SESSION)

    assert prepared.needs_limit_decision is False  # object was transitioned
    assert ended.status == "ended"
    assert session.status == "paused"
    assert session.turns[-1].status == "blocked"
    assert client.stream_calls == []


def test_continue_keeps_maximum_suffix_at_trim_target(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")

    def count(messages: list[dict[str, str]]) -> int:
        history_turns = sum(1 for message in messages if message["role"] == "assistant")
        return 300 + 250 * history_turns

    client = FakeClient(count=count)
    history = [eligible_turn(0), eligible_turn(1), eligible_turn(2)]
    session = make_session(store, turns=history)
    engine = ChatEngine(store, client)

    prepared = engine.prepare_turn(session, "current")
    resumed = engine.resolve_limit(
        session, prepared, LimitAction.CONTINUE_WITH_TRIM
    )

    assert prepared.needs_limit_decision
    assert resumed.ready
    assert resumed.plan.exact_input_tokens == 550
    assert resumed.plan.included_turn_ids == [history[-1].id]
    assert session.active_context_start_index == 2
    assert session.turns[-1].context_trace.decision == "trimmed_and_continued"


def test_mandatory_input_over_budget_is_blocked(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    client = FakeClient(count=950)
    session = make_session(store)

    prepared = ChatEngine(store, client).prepare_turn(session, "too large")

    assert prepared.status == "blocked"
    assert session.status == "paused"
    assert session.turns[-1].status == "blocked"
    assert client.stream_calls == []


def test_stream_persists_exact_usage_and_excludes_thinking_from_next_prompt(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    events = [
        ChatEvent(kind="thinking", text="reason", live_output_tokens=1),
        ChatEvent(kind="usage", live_output_tokens=1),
        ChatEvent(kind="content", text="answer", live_output_tokens=3),
        ChatEvent(kind="usage", live_output_tokens=3),
        ChatEvent(
            kind="completed",
            live_output_tokens=3,
            usage=TokenUsage(100, 3),
            done_reason="stop",
        ),
    ]
    client = FakeClient(count=100, stream_events=events)
    session = make_session(store)
    engine = ChatEngine(store, client)
    prepared = engine.prepare_turn(session, "question")

    received = list(engine.stream_turn(session, prepared))

    assert received[-1].kind == "completed"
    turn = session.turns[-1]
    assert turn.status == "complete"
    assert turn.thinking == "reason"
    assert turn.assistant_content == "answer"
    assert turn.usage.total_tokens == 103
    next_plan = engine.assembler.assemble(session, "next")
    assert "reason" not in str(next_plan.messages)
    assert "answer" in str(next_plan.messages)


def test_final_usage_corrects_live_logprob_count(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    events = [
        ChatEvent(kind="content", text="answer", live_output_tokens=2),
        ChatEvent(kind="usage", live_output_tokens=2),
        ChatEvent(
            kind="completed",
            live_output_tokens=2,
            usage=TokenUsage(100, 3),
            done_reason="stop",
        ),
    ]
    client = FakeClient(count=100, stream_events=events)
    session = make_session(store)
    engine = ChatEngine(store, client)

    received = list(engine.stream_turn(session, engine.prepare_turn(session, "question")))

    assert received[-1].live_output_tokens == 2
    assert received[-1].usage.output_tokens == 3
    assert session.turns[-1].usage.total_tokens == 103
    assert session.cumulative_usage().total_tokens == 103


def test_output_limit_with_final_body_remains_context_eligible(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    client = FakeClient(
        count=100,
        stream_events=[
            ChatEvent(kind="content", text="partial answer", live_output_tokens=4),
            ChatEvent(
                kind="completed",
                live_output_tokens=4,
                usage=TokenUsage(100, 4),
                done_reason="length",
            ),
        ],
    )
    session = make_session(store)
    engine = ChatEngine(store, client)

    list(engine.stream_turn(session, engine.prepare_turn(session, "question")))

    turn = session.turns[-1]
    assert turn.status == "truncated"
    assert turn.context_eligible
    assert "达到输出 token 上限" in (turn.error or "")


def test_interrupted_stream_saves_only_confirmed_partial_counts(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    client = FakeClient(
        count=100,
        stream_events=[ChatEvent(kind="content", text="partial", live_output_tokens=2)],
        stream_error=OllamaConnectionError("lost"),
    )
    session = make_session(store)
    engine = ChatEngine(store, client)
    prepared = engine.prepare_turn(session, "question")

    with pytest.raises(OllamaConnectionError):
        list(engine.stream_turn(session, prepared))

    turn = session.turns[-1]
    assert turn.status == "interrupted"
    assert turn.context_eligible is False
    assert turn.usage.input_tokens is None
    assert turn.usage.output_tokens == 2


def test_prepared_turn_cannot_be_used_with_another_session(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    client = FakeClient(count=100)
    engine = ChatEngine(store, client)
    first = make_session(store)
    second = make_session(store)
    prepared = engine.prepare_turn(first, "question")

    with pytest.raises(ValueError, match="different session"):
        list(engine.stream_turn(second, prepared))
