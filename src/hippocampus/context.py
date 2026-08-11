"""Pure recent-turn context assembly and conservative token estimation."""

from __future__ import annotations

from collections.abc import Sequence

from hippocampus.models import ContextPlan, Session, Turn


class ContextAssembler:
    """Build system + recent complete turns + current user input in strict order."""

    _FALLBACK_BASE_OVERHEAD = 256
    _FALLBACK_MESSAGE_OVERHEAD = 64

    def assemble(
        self,
        session: Session,
        current_user: str,
        *,
        history: Sequence[tuple[int, Turn]] | None = None,
        current_turn_index: int | None = None,
    ) -> ContextPlan:
        if history is None:
            history = session.eligible_turns(before_index=current_turn_index)
        selected = list(history)

        messages: list[dict[str, str]] = []
        if session.system_prompt:
            messages.append({"role": "system", "content": session.system_prompt})
        for _, turn in selected:
            messages.append({"role": "user", "content": turn.user_content})
            # Deliberately do not reinject stored thinking traces.
            messages.append({"role": "assistant", "content": turn.assistant_content})
        messages.append({"role": "user", "content": current_user})

        all_eligible = session.eligible_turns(
            before_index=current_turn_index, honor_active_start=False
        )
        included = {turn.id for _, turn in selected}
        omitted = [turn.id for _, turn in all_eligible if turn.id not in included]

        return ContextPlan(
            messages=messages,
            included_turn_ids=[turn.id for _, turn in selected],
            omitted_turn_ids=omitted,
            selected_history_indices=[index for index, _ in selected],
            estimated_upper_tokens=self.estimate_upper_bound(messages),
            input_budget=session.budget.input_budget,
        )

    @classmethod
    def estimate_upper_bound(cls, messages: Sequence[dict[str, str]]) -> int:
        """Return a conservative UTF-8 byte bound including chat wrapper slack."""

        return cls._FALLBACK_BASE_OVERHEAD + sum(
            len(message.get("content", "").encode("utf-8"))
            + len(message.get("role", "").encode("utf-8"))
            + cls._FALLBACK_MESSAGE_OVERHEAD
            for message in messages
        )

    @staticmethod
    def apply_rendered_upper_bound(plan: ContextPlan, rendered_prompt: str) -> None:
        """Replace the fallback bound with the rendered prompt's UTF-8 byte bound."""

        plan.estimated_upper_tokens = len(rendered_prompt.encode("utf-8"))
