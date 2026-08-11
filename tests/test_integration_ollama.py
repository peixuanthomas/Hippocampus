from __future__ import annotations

import os

import pytest

from hippocampus import BudgetConfig, ChatEngine, LimitAction, OllamaClient, SessionStore
from hippocampus.models import Turn


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("HIPPOCAMPUS_RUN_OLLAMA_INTEGRATION") != "1",
        reason="set HIPPOCAMPUS_RUN_OLLAMA_INTEGRATION=1 to use local Ollama",
    ),
]


def test_live_multiturn_resume_thinking_and_authoritative_usage(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    client = OllamaClient(timeout=120)
    client.check_model("qwen3.5:9b", 2_048)
    session = store.create(
        think=False,
        budget=BudgetConfig(
            context_window=2_048,
            max_output_tokens=96,
            safety_margin_tokens=32,
        ),
    )
    engine = ChatEngine(store, client)

    first = engine.prepare_turn(session, "只回复：第一轮")
    first_events = list(engine.stream_turn(session, first))
    assert session.turns[-1].status == "complete"
    assert first_events[-1].usage == session.turns[-1].usage

    restored = store.load(session.id[:12])
    second = engine.prepare_turn(restored, "只回复：第二轮")
    assert second.plan.included_turn_ids == [restored.turns[0].id]
    second_events = list(engine.stream_turn(restored, second))
    assert restored.turns[-1].status == "complete"
    assert second_events[-1].usage == restored.turns[-1].usage

    restored.think = True
    store.save(restored)
    thinking = engine.prepare_turn(restored, "只回复：开")
    thinking_events = list(engine.stream_turn(restored, thinking))
    final = thinking_events[-1]

    assert any(event.kind == "thinking" for event in thinking_events)
    assert restored.turns[-1].thinking
    assert final.usage == restored.turns[-1].usage
    assert final.live_output_tokens <= final.usage.output_tokens
    assert restored.cumulative_usage().total_tokens == sum(
        turn.usage.total_tokens for turn in restored.turns
    )
    assert restored.turns[-1].thinking not in str(
        engine.assembler.assemble(restored, "下一轮").messages
    )

    reloaded = store.load(restored.id)
    assert reloaded.think is True
    assert reloaded.active_context_start_index == restored.active_context_start_index
    assert reloaded.cumulative_usage() == restored.cumulative_usage()


def test_live_small_budget_reaches_limit_decision_and_trims(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    client = OllamaClient(timeout=120)
    client.check_model("qwen3.5:9b", 512)
    session = store.create(
        think=False,
        budget=BudgetConfig(
            context_window=512,
            max_output_tokens=64,
            safety_margin_tokens=32,
        ),
    )
    history = Turn(
        status="complete",
        user_content="历史" * 170,
        assistant_content="回答" * 170,
    )
    session.turns.append(history)
    store.save(session)
    engine = ChatEngine(store, client)

    prepared = engine.prepare_turn(session, "继续")

    assert prepared.needs_limit_decision
    assert prepared.plan.exact_input_tokens >= session.budget.warning_threshold
    assert session.turns[-1].status == "pending"

    resumed = engine.resolve_limit(session, prepared, LimitAction.CONTINUE_WITH_TRIM)

    assert resumed.ready
    assert resumed.plan.included_turn_ids == []
    assert resumed.plan.exact_input_tokens <= session.budget.trim_target
    assert session.active_context_start_index == resumed.turn_index
    assert store.load(session.id).active_context_start_index == resumed.turn_index
