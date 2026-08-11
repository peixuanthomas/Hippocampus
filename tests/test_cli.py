from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from hippocampus.cli import main
from hippocampus.models import ChatEvent, TokenUsage
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
    assert "本轮最终（权威）： input=100, output=3, total=103" in text
    assert "实时输出" not in text


def test_resume_prefix_reopens_and_think_persists(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session = store.create(think=True)
    code, text = run(tmp_path, ["--sessions-dir", str(tmp_path), "resume", session.id[:12]], ["/think off", "/exit"])
    assert code == 0 and "thinking 已设为：off" in text
    assert store.load(session.id).think is False


def test_list_show_are_read_only_and_sessions_do_not_cross(tmp_path) -> None:
    store = SessionStore(tmp_path)
    first = store.create(model="first")
    second = store.create(model="second")
    FakeClient.made.clear()
    code, text = run(tmp_path, ["--sessions-dir", str(tmp_path), "list"], [])
    assert code == 0 and first.id in text and second.id in text and not FakeClient.made
    code, text = run(tmp_path, ["--sessions-dir", str(tmp_path), "show", first.id], [])
    assert code == 0 and "first" in text and "second" not in text and not FakeClient.made


def test_budget_reports_answer_and_probe(tmp_path) -> None:
    code, text = run(tmp_path, ["--sessions-dir", str(tmp_path), "new"], ["hello", "/budget", "/exit"])
    assert code == 0 and "回答累计： input=100, output=3, total=103" in text
    assert "probe 累计： input=0, output=0, total=0" in text


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


def test_corrupt_session_is_chinese_error(tmp_path) -> None:
    (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")
    code, text = run(tmp_path, ["--sessions-dir", str(tmp_path), "show", "bad"], [])
    assert code == 1 and "错误：会话文件损坏" in text
