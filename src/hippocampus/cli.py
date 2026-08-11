"""Interactive terminal interface for one local Ollama chat session."""

from __future__ import annotations

import argparse
import builtins
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from hippocampus.context import ContextAssembler
from hippocampus.engine import ChatEngine, LimitAction, PreparedTurn
from hippocampus.models import BudgetConfig, ChatEvent, DEFAULT_SYSTEM_PROMPT, Session, TokenUsage
from hippocampus.ollama_client import OllamaClient
from hippocampus.store import SessionStore, SessionStoreError


def _unknown(value: int | None) -> str:
    return "未知" if value is None else str(value)


def _usage_line(label: str, usage: TokenUsage) -> str:
    return (
        f"{label} input={_unknown(usage.input_tokens)}, "
        f"output={_unknown(usage.output_tokens)}, total={_unknown(usage.total_tokens)}"
    )


def _budget_lines(session: Session, prepared: PreparedTurn | None = None) -> list[str]:
    budget = session.budget
    lines = [
        "预算："
        f"context={budget.context_window}, input={budget.input_budget}, "
        f"output reserve={budget.max_output_tokens}, safety={budget.safety_margin_tokens}",
        f"阈值：80%={budget.probe_threshold}, 90%={budget.warning_threshold}；"
        f"活动起点={session.active_context_start_index}",
    ]
    if prepared is not None:
        trace = prepared.plan
        exact_or_estimate = trace.exact_input_tokens
        source = "精确" if exact_or_estimate is not None else "估计上界"
        if exact_or_estimate is None:
            exact_or_estimate = trace.estimated_upper_tokens
        ratio = "未知" if exact_or_estimate is None else f"{exact_or_estimate / budget.input_budget:.1%}"
        lines.append(
            f"当前 trace：{source} input={_unknown(exact_or_estimate)} ({ratio})；"
            f"included={len(trace.included_turn_ids)}, omitted={len(trace.omitted_turn_ids)}"
        )
    elif session.turns:
        trace = session.turns[-1].context_trace
        lines.append(
            "最近 trace："
            f"估计={_unknown(trace.estimated_upper_tokens)}, 精确 input={_unknown(trace.exact_input_tokens)}；"
            f"included={len(trace.included_turn_ids)}, omitted={len(trace.omitted_turn_ids)}"
        )
    lines.extend(
        [
            _usage_line("回答累计：", session.cumulative_usage()),
            _usage_line("probe 累计：", session.cumulative_probe_usage()),
        ]
    )
    return lines


def _render_live(thinking: str, content: str, live_tokens: int, session: Session) -> Panel:
    budget = session.budget
    body = Text()
    if thinking:
        body.append("Thinking（仅本轮展示，不会回注上下文）\n", style="yellow")
        body.append(thinking, style="yellow")
        body.append("\n\n")
    body.append("Assistant\n", style="bold green")
    body.append(content or "…", style="white")
    body.append("\n\n")
    body.append(
        f"实时输出（未最终确认）: {live_tokens}；最终权威计数将在完成事件校正\n"
        f"输入预算: {budget.input_budget}；included/omitted 见本轮 trace；"
        f"probe 累计: {_unknown(session.cumulative_probe_usage().total_tokens)}",
        style="dim",
    )
    return Panel(body, title="Hippocampus", border_style="cyan")


def _stream_prepared(
    console: Console, engine: ChatEngine, session: Session, prepared: PreparedTurn
) -> None:
    thinking = ""
    content = ""
    live_tokens = 0

    def consume(event: ChatEvent) -> None:
        nonlocal thinking, content, live_tokens
        if event.live_output_tokens is not None:
            live_tokens = event.live_output_tokens
        if event.kind == "thinking":
            thinking += event.text
        elif event.kind == "content":
            content += event.text

    if console.is_terminal:
        with Live(_render_live(thinking, content, live_tokens, session), console=console, refresh_per_second=12) as live:
            for event in engine.stream_turn(session, prepared):
                consume(event)
                live.update(_render_live(thinking, content, live_tokens, session))
    else:
        for event in engine.stream_turn(session, prepared):
            consume(event)
            if event.kind == "thinking" and event.text:
                console.print(f"思考：{event.text}")
            elif event.kind == "content" and event.text:
                console.print(event.text, end="")
        if content:
            console.print()

    final = session.turns[-1].usage
    console.print(_usage_line("本轮最终（权威）：", final))
    console.print(_usage_line("回答累计：", session.cumulative_usage()))
    console.print(_usage_line("probe 累计：", session.cumulative_probe_usage()))


def _show_session(console: Console, session: Session) -> None:
    console.print(f"会话 {session.id}｜{session.status}｜{session.title}")
    console.print(f"模型：{session.model}｜Ollama：{session.ollama_host}｜thinking：{'on' if session.think else 'off'}")
    console.print(f"系统提示：{session.system_prompt}")
    for line in _budget_lines(session):
        console.print(line)
    for index, turn in enumerate(session.turns, start=1):
        user = " ".join(turn.user_content.split())[:80]
        answer = " ".join(turn.assistant_content.split())[:80]
        console.print(f"第 {index} 轮 [{turn.status}] 用户：{user}")
        console.print(f"  正文：{answer or '（无）'}")
        console.print("  " + _usage_line("精确：", turn.usage))
        console.print("  " + _usage_line("probe：", turn.probe_usage))


def _list_sessions(console: Console, store: SessionStore) -> int:
    sessions = store.list_sessions()
    if not sessions:
        console.print("没有保存的会话。")
        return 0
    for session in sessions:
        console.print(
            f"{session.id}｜{session.status}｜{session.updated_at}｜{session.model}｜"
            f"{len(session.turns)} 轮｜{session.title}"
        )
    return 0


def _chat(
    console: Console,
    store: SessionStore,
    session: Session,
    client: Any,
    model_info: Any,
    input_fn: Callable[[str], str],
) -> int:
    console.print(
        f"Ollama {model_info.version}｜模型最大上下文 {model_info.context_length}｜会话 {session.id}"
    )
    for line in _budget_lines(session):
        console.print(line)
    console.print("输入 /help 查看命令。")
    engine = ChatEngine(store, client, ContextAssembler())
    while True:
        try:
            value = input_fn("你> ")
        except EOFError:
            store.save(session)
            console.print("收到 EOF，已保存并退出。")
            return 0
        except KeyboardInterrupt:
            console.print("已中断；未伪造 token 计数。")
            return 130
        command = value.strip()
        if command == "/exit":
            path = store.save(session)
            console.print(f"已保存并退出：{path}")
            return 0
        if command == "/save":
            console.print(f"已原子保存：{store.save(session)}")
            continue
        if command == "/help":
            console.print("/budget、/think on|off、/save、/help、/exit")
            continue
        if command == "/budget":
            for line in _budget_lines(session):
                console.print(line)
            continue
        if command.startswith("/think"):
            parts = command.split()
            if len(parts) == 1:
                console.print(f"thinking：{'on' if session.think else 'off'}")
            elif len(parts) == 2 and parts[1] in {"on", "off"}:
                session.think = parts[1] == "on"
                store.save(session)
                console.print(f"thinking 已设为：{parts[1]}")
            else:
                console.print("用法：/think on|off")
            continue
        if command.startswith("/"):
            console.print("未知命令；输入 /help 查看命令。")
            continue
        try:
            prepared = engine.prepare_turn(session, value)
            if prepared.status == "blocked":
                console.print(f"无法生成：{prepared.message}")
                continue
            if prepared.needs_limit_decision:
                console.print(prepared.message)
                while True:
                    answer = input_fn("上下文临界。继续/结束：").strip()
                    if answer == "继续":
                        prepared = engine.resolve_limit(session, prepared, LimitAction.CONTINUE_WITH_TRIM)
                        console.print(prepared.message)
                        break
                    if answer == "结束":
                        engine.resolve_limit(session, prepared, LimitAction.END_SESSION)
                        console.print("已结束本会话；消息未发送给模型。")
                        return 0
                    console.print("请输入“继续”或“结束”。")
            _stream_prepared(console, engine, session, prepared)
        except (ValueError, SessionStoreError, Exception) as exc:
            console.print(f"错误：{exc}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hippocampus", description="本地 Ollama 单会话聊天")
    parser.add_argument("--sessions-dir", default="sessions")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    sub = parser.add_subparsers(dest="command", required=True)
    new = sub.add_parser("new", help="创建并进入新会话")
    new.add_argument("--model", default="qwen3.5:9b")
    new.add_argument("--context-window", type=int, default=32768)
    new.add_argument("--max-output-tokens", type=int, default=4096)
    new.add_argument("--safety-margin-tokens", type=int, default=512)
    think = new.add_mutually_exclusive_group()
    think.add_argument("--think", dest="think", action="store_true", default=True)
    think.add_argument("--no-think", dest="think", action="store_false")
    prompt = new.add_mutually_exclusive_group()
    prompt.add_argument("--system-prompt")
    prompt.add_argument("--system-prompt-file")
    sub.add_parser("list", help="列出会话")
    show = sub.add_parser("show", help="只读查看会话")
    show.add_argument("identifier")
    resume = sub.add_parser("resume", help="恢复会话")
    resume.add_argument("identifier")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    input_fn: Callable[[str], str] | None = None,
    console: Console | None = None,
    client_factory: Callable[..., OllamaClient] = OllamaClient,
) -> int:
    """Run the command-line application and return a process exit status."""

    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    console = console or Console()
    store = SessionStore(args.sessions_dir)
    try:
        if args.command == "list":
            return _list_sessions(console, store)
        if args.command == "show":
            _show_session(console, store.load(args.identifier))
            return 0
        if args.command == "new":
            prompt = args.system_prompt
            if args.system_prompt_file:
                prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
            budget = BudgetConfig(
                context_window=args.context_window,
                max_output_tokens=args.max_output_tokens,
                safety_margin_tokens=args.safety_margin_tokens,
            )
            session = store.create(
                model=args.model, ollama_host=args.host,
                system_prompt=DEFAULT_SYSTEM_PROMPT if prompt is None else prompt,
                budget=budget, think=args.think,
            )
            client = client_factory(host=session.ollama_host)
            info = client.check_model(session.model, session.budget.context_window)
        else:
            session = store.load(args.identifier)
            store.reopen(session)
            client = client_factory(host=session.ollama_host)
            info = client.check_model(session.model, session.budget.context_window)
        return _chat(console, store, session, client, info, input_fn or builtins.input)
    except (OSError, ValueError, SessionStoreError, Exception) as exc:
        console.print(f"错误：{exc}")
        return 1
