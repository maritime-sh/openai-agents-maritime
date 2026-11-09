"""A stand-in for the Maritime SDK that really runs commands, on this machine.

Commands run with subprocess and files live on the local disk, so the SDK's
own helpers (path checks, tar snapshots, apply_patch) execute for real. Only
the network calls to Maritime are replaced.
"""

from __future__ import annotations

import itertools
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from maritime import MaritimeSandboxError

_ids = itertools.count(1)


@dataclass
class FakeCommand:
    exit_code: int | None
    stdout: str
    stderr: str


@dataclass
class FakeData:
    id: str
    state: str


class FakeFiles:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self._sandbox = sandbox

    def read(self, path: str) -> bytes:
        self._sandbox.calls.append(("files.read", path))
        self._sandbox.check_alive()
        p = Path(path)
        if not p.is_file():
            raise MaritimeSandboxError(404, code="file_not_found", message=f"no file at {path}")
        return p.read_bytes()

    def write(self, path: str, data: str | bytes) -> None:
        self._sandbox.calls.append(("files.write", path))
        self._sandbox.check_alive()
        if self._sandbox.fail_writes:
            raise MaritimeSandboxError(503, code="unavailable", message="try again")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data.encode() if isinstance(data, str) else data)

    def remove(self, path: str, *, recursive: bool = False) -> None:
        self._sandbox.calls.append(("files.remove", path))
        p = Path(path)
        if p.is_dir():
            shutil.rmtree(p) if recursive else p.rmdir()
        elif p.exists():
            p.unlink()


@dataclass
class FakeSandbox:
    create_kwargs: dict[str, Any]
    id: str = field(default_factory=lambda: f"sbx-fake{next(_ids)}")
    state: str = "running"
    calls: list[tuple[Any, ...]] = field(default_factory=list)
    fail_runs_with: BaseException | None = None
    fail_writes: bool = False
    exit_code_override: int | None | str = "unset"

    def __post_init__(self) -> None:
        self.files = FakeFiles(self)

    def check_alive(self) -> None:
        if self.state == "ended":
            raise MaritimeSandboxError(410, code="sandbox_ended", message="the sandbox has ended")
        if self.state == "paused":
            self.state = "running"  # Maritime resumes a paused sandbox on its next call.

    def run(
        self,
        cmd: str | list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_seconds: int | None = None,
        **_: Any,
    ) -> FakeCommand:
        self.calls.append(("run", cmd, cwd, env, timeout_seconds))
        self.check_alive()
        if self.fail_runs_with is not None:
            raise self.fail_runs_with
        merged = {**os.environ, **(self.create_kwargs.get("env") or {}), **(env or {})}
        try:
            done = subprocess.run(
                cmd,
                shell=isinstance(cmd, str),
                cwd=cwd,
                env=merged,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as e:
            raise MaritimeSandboxError(408, code="command_timeout", message="the command timed out") from e
        exit_code = done.returncode if self.exit_code_override == "unset" else self.exit_code_override
        return FakeCommand(exit_code=exit_code, stdout=done.stdout, stderr=done.stderr)  # type: ignore[arg-type]

    def refresh(self) -> FakeData:
        self.calls.append(("refresh",))
        return FakeData(id=self.id, state=self.state)

    def pause(self) -> FakeData:
        self.calls.append(("pause",))
        self.state = "paused"
        return FakeData(id=self.id, state=self.state)

    def kill(self) -> FakeData:
        self.calls.append(("kill",))
        self.state = "ended"
        return FakeData(id=self.id, state=self.state)

    def url(self, port: int, *, token: bool = True) -> str:
        self.calls.append(("url", port, token))
        return f"https://{port}-{self.id}.sbx.maritime.sh/?token=tok_secret"


class FakeSandboxes:
    def __init__(self) -> None:
        self.created: list[FakeSandbox] = []
        self.by_id: dict[str, FakeSandbox] = {}
        self.create_error: BaseException | None = None

    def create(self, **kwargs: Any) -> FakeSandbox:
        if self.create_error is not None:
            raise self.create_error
        sandbox = FakeSandbox(create_kwargs=kwargs)
        self.created.append(sandbox)
        self.by_id[sandbox.id] = sandbox
        return sandbox

    def get(self, id_or_name: str) -> FakeSandbox:
        if id_or_name not in self.by_id:
            raise MaritimeSandboxError(404, code="resource_not_found", message="no such sandbox")
        return self.by_id[id_or_name]


class FakeMaritime:
    def __init__(self) -> None:
        self.sandboxes = FakeSandboxes()
