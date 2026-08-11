"""Atomic, one-JSON-file-per-session persistence."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from hippocampus.models import BudgetConfig, DEFAULT_SYSTEM_PROMPT, Session


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class SessionStoreError(Exception):
    """Base class for session persistence failures."""


class SessionNotFoundError(SessionStoreError):
    """Raised when a session identifier cannot be resolved."""


class AmbiguousSessionError(SessionStoreError):
    """Raised when a session prefix matches more than one file."""


class CorruptSessionError(SessionStoreError):
    """Raised when a session file cannot be decoded or validated."""


class SessionStore:
    """Persist independent sessions as human-readable, atomically replaced JSON."""

    def __init__(self, root: str | Path = "sessions") -> None:
        self.root = Path(root).expanduser().resolve()

    def create(
        self,
        *,
        model: str = "qwen3.5:9b",
        ollama_host: str = "http://127.0.0.1:11434",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        budget: BudgetConfig | None = None,
        think: bool = True,
    ) -> Session:
        now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        session = Session(
            id=f"{now}-{uuid4().hex[:8]}",
            model=model,
            ollama_host=ollama_host.rstrip("/"),
            system_prompt=system_prompt,
            budget=budget or BudgetConfig(),
            think=think,
        )
        self.save(session)
        return session

    def save(self, session: Session) -> Path:
        if not _SAFE_ID.fullmatch(session.id):
            raise SessionStoreError(f"unsafe session id: {session.id!r}")
        session.touch()
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{session.id}.json"
        payload = json.dumps(
            session.to_dict(), ensure_ascii=False, indent=2, sort_keys=False
        )
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.root, prefix=f".{session.id}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return target

    def load(self, identifier: str) -> Session:
        path = self.resolve(identifier)
        try:
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            return Session.from_dict(raw)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError) as exc:
            raise CorruptSessionError(f"会话文件损坏或格式不受支持: {path}: {exc}") from exc
        except OSError as exc:
            raise SessionStoreError(f"无法读取会话文件 {path}: {exc}") from exc

    def resolve(self, identifier: str) -> Path:
        if not _SAFE_ID.fullmatch(identifier):
            raise SessionNotFoundError(f"无效会话标识: {identifier!r}")
        exact = self.root / f"{identifier}.json"
        if exact.is_file():
            return exact
        if not self.root.exists():
            raise SessionNotFoundError(f"找不到会话: {identifier}")
        matches = sorted(self.root.glob(f"{identifier}*.json"))
        if not matches:
            raise SessionNotFoundError(f"找不到会话: {identifier}")
        if len(matches) > 1:
            names = ", ".join(path.stem for path in matches[:5])
            raise AmbiguousSessionError(f"会话前缀不唯一: {identifier}（匹配 {names}）")
        return matches[0]

    def list_sessions(self) -> list[Session]:
        if not self.root.exists():
            return []
        sessions = [self.load(path.stem) for path in sorted(self.root.glob("*.json"))]
        return sorted(sessions, key=lambda session: session.updated_at, reverse=True)

    def reopen(self, session: Session) -> None:
        session.status = "active"
        self.save(session)
