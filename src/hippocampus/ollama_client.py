"""Small native Ollama HTTP client with strict context and streaming metrics."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any, NoReturn
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from hippocampus.models import ChatEvent, TokenUsage


class OllamaError(Exception):
    """Base class for Ollama communication failures."""


class OllamaConnectionError(OllamaError):
    """Raised when the local Ollama service cannot be reached."""


class OllamaModelNotFoundError(OllamaError):
    """Raised when the configured model is not installed."""


class OllamaProtocolError(OllamaError):
    """Raised when Ollama returns an incomplete or malformed response."""


class OllamaStreamError(OllamaError):
    """Raised when a stream ends before its authoritative final event."""

    def __init__(self, message: str, *, live_output_tokens: int = 0) -> None:
        super().__init__(message)
        self.live_output_tokens = live_output_tokens


class OllamaContextLengthError(OllamaError):
    """Raised when a rendered prompt exceeds the server context allocation."""

    def __init__(
        self,
        message: str,
        *,
        prompt_tokens: int | None = None,
        context_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.context_tokens = context_tokens


@dataclass(frozen=True, slots=True)
class ModelInfo:
    version: str
    name: str
    context_length: int


class OllamaClient:
    """Native client for the subset of Ollama required by the chat engine."""

    def __init__(self, host: str = "http://127.0.0.1:11434", *, timeout: float = 600) -> None:
        normalized = host.rstrip("/")
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"invalid Ollama host: {host!r}")
        self.host = normalized
        self.timeout = timeout

    def check_model(self, model: str, requested_context: int) -> ModelInfo:
        """Check service health, local model availability, and context capacity."""

        version_payload = self._request_json("GET", "/api/version")
        version = str(version_payload.get("version", "unknown"))
        tags = self._request_json("GET", "/api/tags")
        raw_models = tags.get("models", [])
        available = {
            str(item.get("name") or item.get("model"))
            for item in raw_models
            if isinstance(item, dict)
        }
        if model not in available:
            raise OllamaModelNotFoundError(
                f"本地未安装模型 {model!r}；请先运行: ollama pull {model}"
            )

        details = self._request_json("POST", "/api/show", {"model": model})
        context_length = self._extract_context_length(details)
        if context_length <= 0:
            raise OllamaProtocolError(f"Ollama 未返回模型 {model!r} 的上下文长度")
        if requested_context > context_length:
            raise OllamaError(
                f"配置的上下文 {requested_context} 超过模型上限 {context_length}"
            )
        return ModelInfo(version=version, name=model, context_length=context_length)

    def render_prompt(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        think: bool,
        num_ctx: int,
    ) -> str | None:
        """Ask Ollama to render its native chat template without inference.

        ``_debug_render_only`` is feature-detected. If unavailable, callers must
        use an exact probe instead of trusting the local fallback estimator.
        """

        payload = self._chat_payload(
            model=model,
            messages=messages,
            think=think,
            num_ctx=num_ctx,
            num_predict=1,
            stream=False,
        )
        payload["_debug_render_only"] = True
        response = self._request_json("POST", "/api/chat", payload)
        debug = response.get("_debug_info")
        if not isinstance(debug, dict):
            return None
        rendered = debug.get("rendered_template")
        return rendered if isinstance(rendered, str) else None

    def probe(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        think: bool,
        num_ctx: int,
    ) -> TokenUsage:
        """Run a deterministic one-token call to obtain exact prompt usage."""

        payload = self._chat_payload(
            model=model,
            messages=messages,
            think=think,
            num_ctx=num_ctx,
            num_predict=1,
            stream=False,
        )
        options = payload["options"]
        assert isinstance(options, dict)
        options.update({"temperature": 0, "seed": 0})
        response = self._request_json("POST", "/api/chat", payload)
        prompt_count = response.get("prompt_eval_count")
        output_count = response.get("eval_count")
        if not isinstance(prompt_count, int) or not isinstance(output_count, int):
            raise OllamaProtocolError("精确探测响应缺少 prompt_eval_count 或 eval_count")
        return TokenUsage(prompt_count, output_count)

    def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        think: bool,
        num_ctx: int,
        num_predict: int,
    ):
        """Yield thinking, content, live usage, and one authoritative final event."""

        payload = self._chat_payload(
            model=model,
            messages=messages,
            think=think,
            num_ctx=num_ctx,
            num_predict=num_predict,
            stream=True,
        )
        payload.update({"logprobs": True, "top_logprobs": 0})
        request = self._build_request("POST", "/api/chat", payload)
        live_output_tokens = 0
        saw_done = False
        try:
            response = urlopen(request, timeout=self.timeout)
            with response:
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    try:
                        chunk = json.loads(raw_line)
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise OllamaStreamError(
                            f"Ollama 流返回了无效 JSON: {exc}",
                            live_output_tokens=live_output_tokens,
                        ) from exc
                    if not isinstance(chunk, dict):
                        raise OllamaStreamError(
                            "Ollama 流事件不是 JSON 对象",
                            live_output_tokens=live_output_tokens,
                        )
                    if "error" in chunk:
                        self._raise_api_error(chunk, status=None)

                    logprobs = chunk.get("logprobs")
                    increment = len(logprobs) if isinstance(logprobs, list) else 0
                    live_output_tokens += increment
                    message = chunk.get("message")
                    if isinstance(message, dict):
                        thinking = message.get("thinking")
                        content = message.get("content")
                        if isinstance(thinking, str) and thinking:
                            yield ChatEvent(
                                kind="thinking",
                                text=thinking,
                                live_output_tokens=live_output_tokens,
                            )
                        if isinstance(content, str) and content:
                            yield ChatEvent(
                                kind="content",
                                text=content,
                                live_output_tokens=live_output_tokens,
                            )
                    if increment:
                        yield ChatEvent(kind="usage", live_output_tokens=live_output_tokens)

                    if chunk.get("done") is True:
                        prompt_count = chunk.get("prompt_eval_count")
                        output_count = chunk.get("eval_count")
                        if not isinstance(prompt_count, int) or not isinstance(output_count, int):
                            raise OllamaStreamError(
                                "Ollama 最终事件缺少精确 token 计数",
                                live_output_tokens=live_output_tokens,
                            )
                        saw_done = True
                        yield ChatEvent(
                            kind="completed",
                            live_output_tokens=live_output_tokens,
                            usage=TokenUsage(prompt_count, output_count),
                            done_reason=(
                                chunk.get("done_reason")
                                if isinstance(chunk.get("done_reason"), str)
                                else None
                            ),
                        )
                        break
        except HTTPError as exc:
            self._raise_http_error(exc)
        except (URLError, TimeoutError, socket.timeout, ConnectionError) as exc:
            raise OllamaConnectionError(f"无法连接 Ollama ({self.host}): {exc}") from exc

        if not saw_done:
            raise OllamaStreamError(
                "Ollama 流在最终计数事件之前结束",
                live_output_tokens=live_output_tokens,
            )

    @staticmethod
    def _extract_context_length(payload: dict[str, Any]) -> int:
        details = payload.get("details")
        if isinstance(details, dict):
            direct = details.get("context_length")
            if isinstance(direct, int):
                return direct
        model_info = payload.get("model_info")
        if isinstance(model_info, dict):
            values = [
                value
                for key, value in model_info.items()
                if key.endswith(".context_length") and isinstance(value, int)
            ]
            if values:
                return max(values)
        return 0

    @staticmethod
    def _chat_payload(
        *,
        model: str,
        messages: list[dict[str, str]],
        think: bool,
        num_ctx: int,
        num_predict: int,
        stream: bool,
    ) -> dict[str, Any]:
        return {
            "model": model,
            "messages": messages,
            "stream": stream,
            "think": think,
            "truncate": False,
            "shift": False,
            "keep_alive": "5m",
            "options": {"num_ctx": num_ctx, "num_predict": num_predict},
        }

    def _request_json(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        request = self._build_request(method, path, payload)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            self._raise_http_error(exc)
        except (URLError, TimeoutError, socket.timeout, ConnectionError) as exc:
            raise OllamaConnectionError(f"无法连接 Ollama ({self.host}): {exc}") from exc
        try:
            decoded = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise OllamaProtocolError(f"Ollama 返回了无效 JSON: {exc}") from exc
        if not isinstance(decoded, dict):
            raise OllamaProtocolError("Ollama 响应不是 JSON 对象")
        if "error" in decoded:
            self._raise_api_error(decoded, status=None)
        return decoded

    def _build_request(
        self, method: str, path: str, payload: dict[str, Any] | None
    ) -> Request:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        return Request(f"{self.host}{path}", data=body, headers=headers, method=method)

    def _raise_http_error(self, exc: HTTPError) -> NoReturn:
        try:
            raw = exc.read()
            payload = json.loads(raw) if raw else {"error": str(exc)}
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {"error": str(exc)}
        if not isinstance(payload, dict):
            payload = {"error": str(payload)}
        self._raise_api_error(payload, status=exc.code)

    @staticmethod
    def _raise_api_error(payload: dict[str, Any], status: int | None) -> NoReturn:
        raw_error = payload.get("error", "unknown Ollama error")
        message = str(raw_error)
        details: dict[str, Any] | None = None
        if isinstance(raw_error, dict):
            details = raw_error
        elif isinstance(raw_error, str):
            try:
                nested = json.loads(raw_error)
            except json.JSONDecodeError:
                nested = None
            if isinstance(nested, dict):
                possible = nested.get("error", nested)
                if isinstance(possible, dict):
                    details = possible
                    message = str(possible.get("message", raw_error))

        prompt_tokens = details.get("n_prompt_tokens") if details else None
        context_tokens = details.get("n_ctx") if details else None
        lowered = message.lower()
        if (
            isinstance(prompt_tokens, int)
            or "exceeds the available context" in lowered
            or "context length" in lowered
            or "prompt is too long" in lowered
        ):
            raise OllamaContextLengthError(
                message,
                prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
                context_tokens=context_tokens if isinstance(context_tokens, int) else None,
            )
        if status == 404 or "not found" in lowered:
            raise OllamaModelNotFoundError(message)
        prefix = f"Ollama HTTP {status}: " if status is not None else "Ollama: "
        raise OllamaError(prefix + message)
