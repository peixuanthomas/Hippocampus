from __future__ import annotations

import json

import pytest

from hippocampus.models import ContextTrace, TokenUsage, Turn
from hippocampus.store import (
    AmbiguousSessionError,
    CorruptSessionError,
    SessionStore,
)


def test_session_round_trip_and_cumulative_usage(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create(think=True)
    session.turns.append(
        Turn(
            status="complete",
            user_content="hello",
            assistant_content="world",
            thinking="private trace",
            usage=TokenUsage(12, 4),
            probe_usage=TokenUsage(12, 1),
            context_trace=ContextTrace(
                included_turn_ids=[],
                omitted_turn_ids=[],
                estimated_upper_tokens=30,
                exact_input_tokens=12,
                input_budget=session.budget.input_budget,
                decision="ready",
            ),
        )
    )
    session.active_context_start_index = 1
    path = store.save(session)

    restored = store.load(session.id[:12])

    assert restored.id == session.id
    assert restored.think is True
    assert restored.active_context_start_index == 1
    assert restored.turns[0].thinking == "private trace"
    assert restored.turns[0].usage.total_tokens == 16
    assert restored.cumulative_usage().total_tokens == 16
    assert restored.cumulative_probe_usage().total_tokens == 13
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert not list(path.parent.glob("*.tmp"))


def test_corrupt_session_is_rejected_without_overwrite(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    store.root.mkdir(parents=True)
    path = store.root / "broken.json"
    original = "{not-json"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(CorruptSessionError):
        store.load("broken")

    assert path.read_text(encoding="utf-8") == original


def test_unsupported_schema_is_rejected(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    store.root.mkdir(parents=True)
    path = store.root / "future.json"
    path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")

    with pytest.raises(CorruptSessionError, match="schema"):
        store.load("future")


def test_ambiguous_prefix_is_rejected(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    first = store.create()
    second = store.create()
    first.id = "shared-one"
    second.id = "shared-two"
    store.save(first)
    store.save(second)

    with pytest.raises(AmbiguousSessionError):
        store.load("shared-")
