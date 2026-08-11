"""Conversation orchestration, exact budget decisions, and durable turn state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from hippocampus.context import ContextAssembler
from hippocampus.models import ContextPlan, ContextTrace, Session, TokenUsage, Turn
from hippocampus.ollama_client import (
    OllamaClient,
    OllamaContextLengthError,
    OllamaProtocolError,
)
from hippocampus.store import SessionStore


class LimitAction(str, Enum):
    """User response to a near-limit context decision."""

    CONTINUE_WITH_TRIM = "continue_with_trim"
    END_SESSION = "end_session"


PreparationStatus = str


@dataclass(slots=True)
class PreparedTurn:
    """A pending turn that is ready, awaiting a decision, or blocked."""

    session_id: str
    turn_id: str
    turn_index: int
    plan: ContextPlan
    status: PreparationStatus
    message: str = ""

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def needs_limit_decision(self) -> bool:
        return self.status == "limit_warning"


class ChatEngine:
    """Coordinate session persistence, context selection, probes, and generation."""

    def __init__(
        self,
        store: SessionStore,
        client: OllamaClient,
        assembler: ContextAssembler | None = None,
    ) -> None:
        self.store = store
        self.client = client
        self.assembler = assembler or ContextAssembler()

    def prepare_turn(self, session: Session, user_content: str) -> PreparedTurn:
        """Persist a pending input and determine whether it may be generated."""

        if not user_content.strip():
            raise ValueError("用户输入不能为空")
        self._recover_stale_pending(session)
        session.status = "active"
        if not session.turns:
            compact = " ".join(user_content.split())
            session.title = compact[:40] + ("…" if len(compact) > 40 else "")

        start_before = session.active_context_start_index
        turn = Turn(user_content=user_content)
        session.turns.append(turn)
        turn_index = len(session.turns) - 1
        self.store.save(session)

        try:
            history = session.eligible_turns(before_index=turn_index)
            plan, render_supported = self._build_plan(
                session, turn_index, history, user_content
            )
            if (not render_supported) or (
                plan.estimated_upper_tokens is not None
                and plan.estimated_upper_tokens >= session.budget.probe_threshold
            ):
                self._probe_plan(session, turn, plan)

            metric = self._plan_metric(plan)
            if metric >= session.budget.warning_threshold:
                mandatory, _ = self._build_plan(
                    session, turn_index, [], user_content
                )
                # An exact mandatory probe distinguishes an oversized current
                # input from removable history.
                self._probe_plan(session, turn, mandatory)
                if self._plan_metric(mandatory) > session.budget.input_budget:
                    return self._block_mandatory(
                        session, turn, turn_index, mandatory, start_before
                    )

                self._apply_trace(
                    turn,
                    plan,
                    decision="limit_warning",
                    start_before=start_before,
                    start_after=start_before,
                )
                turn.touch()
                self.store.save(session)
                return PreparedTurn(
                    session_id=session.id,
                    turn_id=turn.id,
                    turn_index=turn_index,
                    plan=plan,
                    status="limit_warning",
                    message=(
                        "上下文已达到临界阈值；请选择丢弃最旧完整轮次后继续，"
                        "或暂停当前会话。"
                    ),
                )

            self._apply_trace(
                turn,
                plan,
                decision="ready",
                start_before=start_before,
                start_after=start_before,
            )
            turn.touch()
            self.store.save(session)
            return PreparedTurn(
                session_id=session.id,
                turn_id=turn.id,
                turn_index=turn_index,
                plan=plan,
                status="ready",
            )
        except Exception as exc:
            if turn.status == "pending":
                turn.status = "failed"
                turn.error = str(exc)
                turn.touch()
                self.store.save(session)
            raise

    def resolve_limit(
        self, session: Session, prepared: PreparedTurn, action: LimitAction
    ) -> PreparedTurn:
        """Resolve a 90%-threshold warning without silently changing history."""

        if not prepared.needs_limit_decision:
            raise ValueError("prepared turn does not require a limit decision")
        turn = self._pending_turn(session, prepared)
        start_before = session.active_context_start_index

        if action is LimitAction.END_SESSION:
            turn.status = "blocked"
            turn.error = "用户选择在上下文临界点暂停会话；消息未发送给模型"
            turn.context_trace.decision = "paused_by_user"
            turn.touch()
            session.status = "paused"
            self.store.save(session)
            prepared.status = "ended"
            prepared.message = turn.error
            return prepared

        if action is not LimitAction.CONTINUE_WITH_TRIM:
            raise ValueError(f"unsupported limit action: {action}")

        history = session.eligible_turns(before_index=prepared.turn_index)
        cache: dict[int, ContextPlan] = {len(history): prepared.plan}

        mandatory = self.assembler.assemble(
            session,
            turn.user_content,
            history=[],
            current_turn_index=prepared.turn_index,
        )
        self._probe_plan(session, turn, mandatory)
        cache[0] = mandatory
        mandatory_tokens = self._plan_metric(mandatory)
        if mandatory_tokens > session.budget.input_budget:
            return self._block_mandatory(
                session, turn, prepared.turn_index, mandatory, start_before
            )

        target = session.budget.trim_target
        if mandatory_tokens > target:
            turn.status = "blocked"
            turn.error = (
                "系统提示与当前输入超过 80% 安全裁剪目标，请缩短系统提示或当前输入"
            )
            self._apply_trace(
                turn,
                mandatory,
                decision="mandatory_above_trim_target",
                start_before=start_before,
                start_after=start_before,
            )
            turn.touch()
            session.status = "paused"
            self.store.save(session)
            return PreparedTurn(
                session_id=session.id,
                turn_id=turn.id,
                turn_index=prepared.turn_index,
                plan=mandatory,
                status="blocked",
                message=turn.error,
            )
        else:
            low, high = 0, len(history)
            best_count = 0
            while low <= high:
                middle = (low + high) // 2
                candidate = cache.get(middle)
                if candidate is None:
                    selected = history[-middle:] if middle else []
                    candidate = self.assembler.assemble(
                        session,
                        turn.user_content,
                        history=selected,
                        current_turn_index=prepared.turn_index,
                    )
                    self._probe_plan(session, turn, candidate)
                    cache[middle] = candidate
                if self._plan_metric(candidate) <= target:
                    best_count = middle
                    low = middle + 1
                else:
                    high = middle - 1

        selected_plan = cache.get(best_count)
        if selected_plan is None:
            selected = history[-best_count:] if best_count else []
            selected_plan = self.assembler.assemble(
                session,
                turn.user_content,
                history=selected,
                current_turn_index=prepared.turn_index,
            )
            self._probe_plan(session, turn, selected_plan)

        if best_count:
            new_start = selected_plan.selected_history_indices[0]
        else:
            # The current pending turn becomes the first eligible turn after it
            # receives a final assistant body.
            new_start = prepared.turn_index
        session.active_context_start_index = new_start
        session.status = "active"
        self._apply_trace(
            turn,
            selected_plan,
            decision="trimmed_and_continued",
            start_before=start_before,
            start_after=new_start,
        )
        turn.touch()
        self.store.save(session)
        return PreparedTurn(
            session_id=session.id,
            turn_id=turn.id,
            turn_index=prepared.turn_index,
            plan=selected_plan,
            status="ready",
            message=f"已保留最近 {best_count} 个完整轮次并继续。",
        )

    def stream_turn(self, session: Session, prepared: PreparedTurn):
        """Generate a prepared turn and atomically persist its final state."""

        if not prepared.ready:
            raise ValueError("turn is not ready for generation")
        turn = self._pending_turn(session, prepared)
        thinking_parts: list[str] = []
        content_parts: list[str] = []
        live_output_tokens = 0
        final_usage: TokenUsage | None = None

        try:
            for event in self.client.stream_chat(
                model=session.model,
                messages=prepared.plan.messages,
                think=session.think,
                num_ctx=session.budget.context_window,
                num_predict=session.budget.max_output_tokens,
            ):
                if event.live_output_tokens is not None:
                    live_output_tokens = event.live_output_tokens
                if event.kind == "thinking":
                    thinking_parts.append(event.text)
                elif event.kind == "content":
                    content_parts.append(event.text)
                elif event.kind == "completed":
                    if event.usage is None:
                        raise OllamaProtocolError("完成事件缺少精确 token usage")
                    final_usage = event.usage
                    if (
                        prepared.plan.exact_input_tokens is not None
                        and final_usage.input_tokens != prepared.plan.exact_input_tokens
                    ):
                        raise OllamaProtocolError(
                            "精确探测与正式请求的输入 token 不一致；拒绝将该轮加入上下文"
                        )
                    turn.thinking = "".join(thinking_parts)
                    turn.assistant_content = "".join(content_parts)
                    turn.usage = final_usage
                    turn.done_reason = event.done_reason
                    prepared.plan.exact_input_tokens = final_usage.input_tokens
                    turn.context_trace.exact_input_tokens = final_usage.input_tokens
                    if not turn.assistant_content:
                        turn.status = "no_answer"
                        turn.error = "模型未返回可作为后续上下文的正文"
                    elif event.done_reason == "length":
                        turn.status = "truncated"
                        turn.error = "回答达到输出 token 上限，正文可能不完整"
                    else:
                        turn.status = "complete"
                    turn.touch()
                    session.status = "active"
                    self.store.save(session)
                    yield event
                    return
                yield event
        except KeyboardInterrupt:
            self.interrupt_turn(
                session,
                prepared,
                thinking="".join(thinking_parts),
                content="".join(content_parts),
                live_output_tokens=live_output_tokens,
            )
            raise
        except Exception as exc:
            turn.thinking = "".join(thinking_parts)
            turn.assistant_content = "".join(content_parts)
            if final_usage is not None:
                turn.usage = final_usage
            else:
                turn.usage = TokenUsage(
                    None,
                    live_output_tokens if live_output_tokens else None,
                )
            if isinstance(exc, OllamaContextLengthError):
                turn.status = "blocked"
                session.status = "paused"
            elif thinking_parts or content_parts or live_output_tokens:
                turn.status = "interrupted"
            else:
                turn.status = "failed"
            turn.error = str(exc)
            turn.touch()
            self.store.save(session)
            raise

        # A conforming client emits a completed event or raises. This guard is
        # retained for injected test clients and future transports.
        turn.status = "interrupted"
        turn.thinking = "".join(thinking_parts)
        turn.assistant_content = "".join(content_parts)
        turn.usage = TokenUsage(
            None,
            live_output_tokens if live_output_tokens else None,
        )
        turn.error = "模型流在完成事件之前结束"
        turn.touch()
        self.store.save(session)
        raise OllamaProtocolError(turn.error)

    def interrupt_turn(
        self,
        session: Session,
        prepared: PreparedTurn,
        *,
        thinking: str = "",
        content: str = "",
        live_output_tokens: int = 0,
    ) -> None:
        """Persist a user-interrupted pending turn without inventing final usage."""

        try:
            turn = self._pending_turn(session, prepared)
        except ValueError:
            return
        turn.thinking = thinking
        turn.assistant_content = content
        turn.usage = TokenUsage(None, live_output_tokens if live_output_tokens else None)
        turn.status = "interrupted"
        turn.error = "用户中断生成，未收到最终权威 token 计数"
        turn.touch()
        session.status = "paused"
        self.store.save(session)

    def _build_plan(
        self,
        session: Session,
        turn_index: int,
        history: list[tuple[int, Turn]],
        user_content: str,
    ) -> tuple[ContextPlan, bool]:
        plan = self.assembler.assemble(
            session,
            user_content,
            history=history,
            current_turn_index=turn_index,
        )
        try:
            rendered = self.client.render_prompt(
                model=session.model,
                messages=plan.messages,
                think=session.think,
                num_ctx=session.budget.context_window,
            )
        except OllamaContextLengthError as exc:
            plan.exact_input_tokens = (
                exc.prompt_tokens
                if exc.prompt_tokens is not None
                else session.budget.context_window + 1
            )
            return plan, True
        if rendered is not None:
            self.assembler.apply_rendered_upper_bound(plan, rendered)
            return plan, True
        return plan, False

    def _probe_plan(self, session: Session, turn: Turn, plan: ContextPlan) -> None:
        if plan.exact_input_tokens is not None:
            return
        try:
            usage = self.client.probe(
                model=session.model,
                messages=plan.messages,
                think=session.think,
                num_ctx=session.budget.context_window,
            )
        except OllamaContextLengthError as exc:
            if exc.prompt_tokens is None:
                # It is known to exceed the actual context, so this value is a
                # safe ordering sentinel rather than a fabricated usage count.
                plan.exact_input_tokens = session.budget.context_window + 1
            else:
                plan.exact_input_tokens = exc.prompt_tokens
            return
        plan.exact_input_tokens = usage.input_tokens
        turn.probe_usage.add(usage)

    @staticmethod
    def _plan_metric(plan: ContextPlan) -> int:
        if plan.exact_input_tokens is not None:
            return plan.exact_input_tokens
        if plan.estimated_upper_tokens is not None:
            return plan.estimated_upper_tokens
        raise ValueError("context plan has neither exact nor estimated token count")

    @staticmethod
    def _apply_trace(
        turn: Turn,
        plan: ContextPlan,
        *,
        decision: str,
        start_before: int,
        start_after: int,
    ) -> None:
        turn.context_trace = ContextTrace(
            included_turn_ids=list(plan.included_turn_ids),
            omitted_turn_ids=list(plan.omitted_turn_ids),
            estimated_upper_tokens=plan.estimated_upper_tokens,
            exact_input_tokens=plan.exact_input_tokens,
            input_budget=plan.input_budget,
            decision=decision,
            active_context_start_before=start_before,
            active_context_start_after=start_after,
        )

    def _block_mandatory(
        self,
        session: Session,
        turn: Turn,
        turn_index: int,
        plan: ContextPlan,
        start_before: int,
    ) -> PreparedTurn:
        turn.status = "blocked"
        turn.error = (
            "系统提示与当前输入本身已超过输入预算；请缩短输入或提高上下文配置"
        )
        self._apply_trace(
            turn,
            plan,
            decision="mandatory_input_exceeded",
            start_before=start_before,
            start_after=start_before,
        )
        turn.touch()
        session.status = "paused"
        self.store.save(session)
        return PreparedTurn(
            session_id=session.id,
            turn_id=turn.id,
            turn_index=turn_index,
            plan=plan,
            status="blocked",
            message=turn.error,
        )

    @staticmethod
    def _pending_turn(session: Session, prepared: PreparedTurn) -> Turn:
        if session.id != prepared.session_id:
            raise ValueError("prepared turn belongs to a different session")
        try:
            turn = session.turns[prepared.turn_index]
        except IndexError as exc:
            raise ValueError("prepared turn index is no longer valid") from exc
        if turn.id != prepared.turn_id or turn.status != "pending":
            raise ValueError("prepared turn no longer references a pending turn")
        return turn

    def _recover_stale_pending(self, session: Session) -> None:
        changed = False
        for turn in session.turns:
            if turn.status == "pending":
                turn.status = "interrupted"
                turn.error = "上次进程在该轮完成前退出"
                turn.touch()
                changed = True
        if changed:
            self.store.save(session)
