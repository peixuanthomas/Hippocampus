from __future__ import annotations

import io
import json
from urllib.error import HTTPError

import pytest

import hippocampus.ollama_client as client_module
from hippocampus.ollama_client import (
    OllamaClient,
    OllamaContextLengthError,
    OllamaStreamError,
)


class FakeResponse:
    def __init__(self, lines: list[dict[str, object]]) -> None:
        self.lines = [(json.dumps(item) + "\n").encode() for item in lines]

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def __iter__(self):
        return iter(self.lines)


def test_stream_parses_channels_and_exact_usage(monkeypatch) -> None:
    captured = {}
    response = FakeResponse(
        [
            {
                "message": {"role": "assistant", "thinking": "think"},
                "done": False,
                "logprobs": [{"token": "a"}],
            },
            {
                "message": {"role": "assistant", "content": ""},
                "done": False,
                "logprobs": [{"token": "b"}, {"token": "c"}],
            },
            {
                "message": {"role": "assistant", "content": "answer"},
                "done": False,
                "logprobs": [{"token": "d"}],
            },
            {
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 12,
                "eval_count": 4,
            },
        ]
    )

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return response

    monkeypatch.setattr(client_module, "urlopen", fake_urlopen)
    client = OllamaClient(timeout=7)

    events = list(
        client.stream_chat(
            model="qwen3.5:9b",
            messages=[{"role": "user", "content": "hello"}],
            think=True,
            num_ctx=32_768,
            num_predict=4_096,
        )
    )

    assert [event.kind for event in events] == [
        "thinking",
        "usage",
        "usage",
        "content",
        "usage",
        "completed",
    ]
    assert [event.live_output_tokens for event in events if event.kind == "usage"] == [
        1,
        3,
        4,
    ]
    assert events[-1].usage.total_tokens == 16
    assert captured["payload"]["truncate"] is False
    assert captured["payload"]["shift"] is False
    assert captured["payload"]["logprobs"] is True
    assert captured["payload"]["think"] is True
    assert captured["payload"]["options"] == {
        "num_ctx": 32_768,
        "num_predict": 4_096,
    }


def test_stream_without_final_event_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        client_module,
        "urlopen",
        lambda request, timeout: FakeResponse(
            [
                {
                    "message": {"role": "assistant", "content": "partial"},
                    "done": False,
                    "logprobs": [{"token": "a"}],
                }
            ]
        ),
    )

    with pytest.raises(OllamaStreamError) as error:
        list(
            OllamaClient().stream_chat(
                model="model",
                messages=[{"role": "user", "content": "hello"}],
                think=False,
                num_ctx=2_048,
                num_predict=10,
            )
        )

    assert error.value.live_output_tokens == 1


def test_nested_context_error_exposes_counts(monkeypatch) -> None:
    nested = {
        "error": json.dumps(
            {
                "error": {
                    "message": "request exceeds the available context size",
                    "n_prompt_tokens": 6_012,
                    "n_ctx": 2_048,
                }
            }
        )
    }
    http_error = HTTPError(
        "http://localhost/api/chat",
        400,
        "bad request",
        hdrs=None,
        fp=io.BytesIO(json.dumps(nested).encode()),
    )
    monkeypatch.setattr(
        client_module, "urlopen", lambda request, timeout: (_ for _ in ()).throw(http_error)
    )

    with pytest.raises(OllamaContextLengthError) as error:
        OllamaClient().probe(
            model="model",
            messages=[{"role": "user", "content": "long"}],
            think=False,
            num_ctx=2_048,
        )

    assert error.value.prompt_tokens == 6_012
    assert error.value.context_tokens == 2_048
