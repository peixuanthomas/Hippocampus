from __future__ import annotations

from hippocampus.context import ContextAssembler
from hippocampus.models import BudgetConfig, Session, Turn


def completed_turn(user: str, assistant: str, *, thinking: str = "") -> Turn:
    return Turn(
        status="complete",
        user_content=user,
        assistant_content=assistant,
        thinking=thinking,
    )


def test_default_budget_values() -> None:
    budget = BudgetConfig()

    assert budget.input_budget == 28_160
    assert budget.probe_threshold == 22_528
    assert budget.warning_threshold == 25_344
    assert budget.trim_target == 22_528


def test_context_order_and_thinking_is_not_reinjected() -> None:
    old = completed_turn("old-user", "old-assistant", thinking="old-secret")
    recent = completed_turn("recent-user", "recent-assistant", thinking="new-secret")
    session = Session(id="one", system_prompt="system", turns=[old, recent])
    session.active_context_start_index = 1

    plan = ContextAssembler().assemble(session, "current")

    assert plan.messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "recent-user"},
        {"role": "assistant", "content": "recent-assistant"},
        {"role": "user", "content": "current"},
    ]
    assert "old-secret" not in str(plan.messages)
    assert "new-secret" not in str(plan.messages)
    assert plan.included_turn_ids == [recent.id]
    assert plan.omitted_turn_ids == [old.id]


def test_incomplete_and_no_answer_turns_are_not_context_eligible() -> None:
    complete = completed_turn("u1", "a1")
    interrupted = Turn(
        status="interrupted", user_content="u2", assistant_content="partial"
    )
    no_answer = Turn(status="no_answer", user_content="u3", thinking="only thought")
    truncated = Turn(
        status="truncated", user_content="u4", assistant_content="usable partial"
    )
    session = Session(
        id="eligibility", turns=[complete, interrupted, no_answer, truncated]
    )

    plan = ContextAssembler().assemble(session, "current")

    assert plan.included_turn_ids == [complete.id, truncated.id]
    contents = [message["content"] for message in plan.messages]
    assert "partial" not in contents
    assert "only thought" not in contents
    assert "usable partial" in contents


def test_sessions_never_share_context() -> None:
    session_a = Session(id="a", turns=[completed_turn("A fact", "A answer")])
    session_b = Session(id="b", turns=[completed_turn("B fact", "B answer")])
    assembler = ContextAssembler()

    plan_a = assembler.assemble(session_a, "ask A")
    plan_b = assembler.assemble(session_b, "ask B")

    assert "B fact" not in str(plan_a.messages)
    assert "A fact" not in str(plan_b.messages)
