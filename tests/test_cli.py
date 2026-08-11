from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

from rich.console import Console

import hippocampus.cli as cli_module
from hippocampus.cli import _plan_trace_line, main
from hippocampus.engine import PreparedTurn
from hippocampus.models import BudgetConfig, ChatEvent, ContextPlan, Session, TokenUsage, Turn
from hippocampus.store import SessionStore


class FakeClient:
    made: list["FakeClient"] = []

    def __init__(self, host: str) -> None:
        self.host = host
        self.checked: list[tuple[str, int]] = []
        self.stream_calls = 0
        FakeClient.made.append(self)

    def check_model(self, model: str, requested_context: int):
        self.checked.append((model, requested_context))
        return SimpleNamespace(version="test", context_length=65536)

    def render_prompt(self, **kwargs):
        return "x" * 100

    def probe(self, **kwargs):
        return TokenUsage(100, 1)

    def stream_chat(self, **kwargs):
        self.stream_calls += 1
        yield ChatEvent(kind="thinking", text="reason", live_output_tokens=1)
        yield ChatEvent(kind="content", text="answer", live_output_tokens=2)
        yield ChatEvent(kind="completed", usage=TokenUsage(100, 3), live_output_tokens=2)


def run(tmp_path, argv, answers):
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    iterator = iter(answers)
    code = main(argv, input_fn=lambda _: next(iterator), console=console, client_factory=FakeClient)
    return code, output.getvalue()


def test_new_health_stream_and_authoritative_final_usage(tmp_path) -> None:
    FakeClient.made.clear()
    code, text = run(tmp_path, ["--sessions-dir", str(tmp_path), "new"], ["hello", "/exit"])
    assert code == 0 and len(FakeClient.made) == 1
    assert FakeClient.made[0].checked
    assert "当前 trace：估计上界 input=100 / 28160 (0.4%)；included=0, omitted=0" in text
    assert "本轮最终（权威）： input=100, output=3, total=103" in text
    assert "实时输出校正：live 2 → final 3" in text


def test_resume_prefix_reopens_and_think_persists(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session = store.create(think=True)
    code, text = run(tmp_path, ["--sessions-dir", str(tmp_path), "resume", session.id[:12]], ["/think off", "/exit"])
    assert code == 0 and "thinking 已设为：off" in text
    assert store.load(session.id).think is False


def test_non_terminal_trace_uses_current_prepared_plan() -> None:
    session = Session(
        id="session",
        budget=BudgetConfig(
            context_window=1_000,
            max_output_tokens=100,
            safety_margin_tokens=0,
        ),
    )
    prepared = PreparedTurn(
        session_id=session.id,
        turn_id="turn",
        turn_index=0,
        status="ready",
        plan=ContextPlan(
            messages=[],
            included_turn_ids=["new-a", "new-b"],
            omitted_turn_ids=["old"],
            selected_history_indices=[4, 5],
            estimated_upper_tokens=10,
            exact_input_tokens=750,
            input_budget=900,
        ),
    )
    assert _plan_trace_line(session, prepared) == (
        "当前 trace：精确 input=750 / 900 (83.3%)；included=2, omitted=1"
    )


def test_list_show_are_read_only_and_sessions_do_not_cross(tmp_path) -> None:
    store = SessionStore(tmp_path)
    first = store.create(model="first")
    second = store.create(model="second")
    FakeClient.made.clear()
    code, text = run(tmp_path, ["--sessions-dir", str(tmp_path), "list"], [])
    assert code == 0 and first.id in text and second.id in text and not FakeClient.made
    code, text = run(tmp_path, ["--sessions-dir", str(tmp_path), "show", first.id], [])
    assert code == 0 and "first" in text and "second" not in text and not FakeClient.made


def test_show_renders_session_data_and_status_as_literal_text(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session = store.create(
        model="[blue]model[/blue]",
        ollama_host="http://[red]host[/red]",
        system_prompt="[green]system[/green]",
    )
    session.title = "[yellow]title[/yellow]"
    session.turns.append(
        Turn(
            status="complete",
            user_content="[red]user[/red]",
            assistant_content="[blue]assistant[/blue]",
            usage=TokenUsage(10, 2),
        )
    )
    store.save(session)

    code, text = run(tmp_path, ["--sessions-dir", str(tmp_path), "show", session.id], [])

    assert code == 0
    for literal in (
        "[yellow]title[/yellow]",
        "[blue]model[/blue]",
        "[red]host[/red]",
        "[green]system[/green]",
        "[complete]",
        "[red]user[/red]",
        "[blue]assistant[/blue]",
    ):
        assert literal in text


def test_non_tty_stream_renders_markup_like_chunks_literally(tmp_path) -> None:
    class MarkupClient(FakeClient):
        def stream_chat(self, **kwargs):
            self.stream_calls += 1
            yield ChatEvent(kind="thinking", text="[yellow]thought[/yellow]", live_output_tokens=1)
            yield ChatEvent(kind="content", text="[red]literal[/red]", live_output_tokens=2)
            yield ChatEvent(kind="completed", usage=TokenUsage(100, 3), live_output_tokens=2)

    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    answers = iter(["hello", "/exit"])
    code = main(
        ["--sessions-dir", str(tmp_path), "new"],
        input_fn=lambda _: next(answers),
        console=console,
        client_factory=MarkupClient,
    )

    text = output.getvalue()
    assert code == 0
    assert "[yellow]thought[/yellow]" in text
    assert "[red]literal[/red]" in text
    assert "本轮最终（权威）： input=100, output=3, total=103" in text


def test_budget_reports_answer_and_probe(tmp_path) -> None:
    code, text = run(tmp_path, ["--sessions-dir", str(tmp_path), "new"], ["hello", "/budget", "/exit"])
    assert code == 0 and "回答累计： input=100, output=3, total=103" in text
    assert "probe 累计： input=0, output=0, total=0" in text
    assert "最近本轮 probe： input=0, output=0, total=0" in text


def test_limit_end_does_not_stream(tmp_path) -> None:
    class NearLimit(FakeClient):
        def render_prompt(self, **kwargs):
            return "x" * 26000

        def probe(self, **kwargs):
            return TokenUsage(26000, 1)

    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    answers = iter(["hello", "结束"])
    code = main(
        ["--sessions-dir", str(tmp_path), "new"],
        input_fn=lambda _: next(answers),
        console=console,
        client_factory=NearLimit,
    )
    assert code == 0 and NearLimit.made[-1].stream_calls == 0


def test_limit_choice_eof_or_keyboard_interrupt_pauses_without_stream(tmp_path) -> None:
    class NearLimit(FakeClient):
        def render_prompt(self, **kwargs):
            return "x" * 26000

        def probe(self, **kwargs):
            return TokenUsage(26000, 1)

    for interruption, expected_code in ((EOFError(), 0), (KeyboardInterrupt(), 130)):
        root = tmp_path / str(expected_code)
        output = StringIO()
        console = Console(file=output, force_terminal=False, color_system=None)
        answers = iter(["hello", interruption])

        def input_fn(_):
            value = next(answers)
            if isinstance(value, BaseException):
                raise value
            return value

        code = main(
            ["--sessions-dir", str(root), "new"],
            input_fn=input_fn,
            console=console,
            client_factory=NearLimit,
        )
        session = SessionStore(root).list_sessions()[0]
        assert code == expected_code
        assert session.status == "paused"
        assert session.turns[-1].status == "blocked"
        assert NearLimit.made[-1].stream_calls == 0


def test_keyboard_interrupt_during_stream_is_persisted_and_reported(tmp_path) -> None:
    class InterruptingClient(FakeClient):
        def stream_chat(self, **kwargs):
            self.stream_calls += 1
            yield ChatEvent(kind="content", text="partial", live_output_tokens=2)
            raise KeyboardInterrupt

    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    answers = iter(["hello"])
    code = main(
        ["--sessions-dir", str(tmp_path), "new"],
        input_fn=lambda _: next(answers),
        console=console,
        client_factory=InterruptingClient,
    )
    session = SessionStore(tmp_path).list_sessions()[0]
    turn = session.turns[-1]
    assert code == 130 and "Traceback" not in output.getvalue()
    assert "生成已被用户中断" in output.getvalue()
    assert session.status == "paused" and turn.status == "interrupted"
    assert turn.assistant_content == "partial" and turn.usage == TokenUsage(None, 2)


def test_tty_live_displays_trace_probe_and_final_correction(tmp_path) -> None:
    class ProbingClient(FakeClient):
        def render_prompt(self, **kwargs):
            return None

    output = StringIO()
    console = Console(file=output, force_terminal=True, color_system=None, width=200)
    answers = iter(["hello", "/exit"])
    code = main(
        ["--sessions-dir", str(tmp_path), "new"],
        input_fn=lambda _: next(answers),
        console=console,
        client_factory=ProbingClient,
    )
    text = output.getvalue()
    assert code == 0
    assert "当前 input（精确）: 100 / 28160 (0.4%)" in text
    assert "included=0, omitted=0" in text
    assert "本轮 probe： input=100, output=1, total=101" in text
    assert "probe 累计： input=100, output=1, total=101" in text
    assert "live 2 → final 3" in text
    assert "本轮最终（权威）： input=100, output=3, total=103" in text


def test_tty_live_update_interrupt_is_persisted(tmp_path, monkeypatch) -> None:
    class RaisingLive:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def update(self, value) -> None:
            raise KeyboardInterrupt

    monkeypatch.setattr(cli_module, "Live", RaisingLive)
    output = StringIO()
    console = Console(file=output, force_terminal=True, color_system=None)
    answers = iter(["hello"])
    code = main(
        ["--sessions-dir", str(tmp_path), "new"],
        input_fn=lambda _: next(answers),
        console=console,
        client_factory=FakeClient,
    )
    session = SessionStore(tmp_path).list_sessions()[0]
    assert code == 130
    assert session.status == "paused"
    assert session.turns[-1].status == "interrupted"
    assert session.turns[-1].assistant_content == ""
    assert session.turns[-1].usage == TokenUsage(None, 1)


def test_tty_interrupt_after_completed_keeps_authoritative_usage(tmp_path, monkeypatch) -> None:
    class RaisingAfterCompletedLive:
        updates = 0

        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def update(self, value) -> None:
            type(self).updates += 1
            if type(self).updates == 3:
                raise KeyboardInterrupt

    monkeypatch.setattr(cli_module, "Live", RaisingAfterCompletedLive)
    output = StringIO()
    console = Console(file=output, force_terminal=True, color_system=None)
    answers = iter(["hello"])
    code = main(
        ["--sessions-dir", str(tmp_path), "new"],
        input_fn=lambda _: next(answers),
        console=console,
        client_factory=FakeClient,
    )
    session = SessionStore(tmp_path).list_sessions()[0]
    turn = session.turns[-1]
    text = output.getvalue()
    assert code == 130 and turn.status == "complete"
    assert turn.usage == TokenUsage(100, 3)
    assert "权威 token 计数已保存" in text
    assert "本轮最终（权威）： input=100, output=3, total=103" in text
    assert "本轮非最终" not in text and "未收到最终" not in text


def test_non_tty_prints_current_and_cumulative_probe(tmp_path) -> None:
    class ProbingClient(FakeClient):
        def render_prompt(self, **kwargs):
            return None

    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    answers = iter(["hello", "/exit"])
    code = main(
        ["--sessions-dir", str(tmp_path), "new"],
        input_fn=lambda _: next(answers),
        console=console,
        client_factory=ProbingClient,
    )
    text = output.getvalue()
    assert code == 0
    assert "本轮 probe： input=100, output=1, total=101" in text
    assert "probe 累计： input=100, output=1, total=101" in text


def test_corrupt_session_is_chinese_error(tmp_path) -> None:
    (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")
    code, text = run(tmp_path, ["--sessions-dir", str(tmp_path), "show", "bad"], [])
    assert code == 1 and "错误：会话文件损坏" in text
