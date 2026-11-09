"""Maritime sandbox backend for the OpenAI Agents SDK.

A Maritime sandbox is a Linux microVM with its own filesystem, network and
processes. This module lets a ``SandboxAgent`` run its shell, file and patch
tools inside one:

    from agents import RunConfig, Runner
    from agents.sandbox import SandboxAgent, SandboxRunConfig
    from openai_agents_maritime import MaritimeSandboxClient, MaritimeSandboxClientOptions

    run_config = RunConfig(
        sandbox=SandboxRunConfig(
            client=MaritimeSandboxClient(),  # reads MARITIME_API_KEY
            options=MaritimeSandboxClientOptions(),
        )
    )

The Maritime Python SDK is synchronous, so every call runs in a worker thread
and never blocks the event loop.
"""

from __future__ import annotations

import asyncio
import io
import math
import shlex
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeVar
from urllib.parse import urlsplit

from agents.sandbox.entries import Mount
from agents.sandbox.errors import (
    ExecTimeoutError,
    ExecTransportError,
    ExposedPortUnavailableError,
    SandboxError,
    WorkspaceArchiveReadError,
    WorkspaceArchiveWriteError,
    WorkspaceReadNotFoundError,
    WorkspaceStartError,
    WorkspaceWriteTypeError,
)
from agents.sandbox.manifest import Manifest
from agents.sandbox.session import SandboxSession, SandboxSessionState
from agents.sandbox.session.base_sandbox_session import BaseSandboxSession
from agents.sandbox.session.dependencies import Dependencies
from agents.sandbox.session.manager import Instrumentation
from agents.sandbox.session.runtime_helpers import (
    RESOLVE_WORKSPACE_PATH_HELPER,
    RuntimeHelperScript,
)
from agents.sandbox.session.sandbox_client import BaseSandboxClient, BaseSandboxClientOptions
from agents.sandbox.session.tar_workspace import shell_tar_exclude_args
from agents.sandbox.snapshot import SnapshotBase, SnapshotSpec, resolve_snapshot
from agents.sandbox.types import ExecResult, ExposedPortEndpoint, User
from agents.sandbox.util.tar_utils import UnsafeTarMemberError, validate_tar_bytes
from agents.sandbox.workspace_paths import (
    coerce_posix_path,
    posix_path_as_path,
    sandbox_path_str,
)
from maritime import Maritime, MaritimeAPIError, MaritimeSandboxError, Sandbox
from pydantic import ConfigDict, Field

__all__ = [
    "DEFAULT_MARITIME_WORKSPACE_ROOT",
    "MaritimeSandboxClient",
    "MaritimeSandboxClientOptions",
    "MaritimeSandboxSession",
    "MaritimeSandboxSessionState",
]

DEFAULT_MARITIME_WORKSPACE_ROOT = "/home/user/workspace"
"""Where the agent's workspace lives inside the sandbox unless the manifest says otherwise."""

BACKEND_ID = "maritime"

# Maritime states in which the sandbox still holds its disk. A paused sandbox
# resumes on its next command or file call, so it counts as alive.
_ALIVE_STATES = frozenset({"starting", "running", "pausing", "paused", "resuming"})
_NOT_FOUND_CODES = frozenset({"file_not_found", "resource_not_found", "not_found"})
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

T = TypeVar("T")

SandboxSize = Literal["small", "medium", "large", "xlarge"]
IdleAction = Literal["pause", "kill"]


def _is_retryable(error: BaseException) -> bool | None:
    status = getattr(error, "status", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS
    return None


def _provider_context(error: BaseException) -> dict[str, object]:
    """Error details that are safe to surface: the code and request id, never credentials."""
    context: dict[str, object] = {"backend": BACKEND_ID}
    code = getattr(error, "code", None)
    if isinstance(code, str) and code:
        context["provider_error_code"] = code
    request_id = getattr(error, "request_id", None)
    if isinstance(request_id, str) and request_id:
        context["provider_request_id"] = request_id
    return context


class MaritimeSandboxClientOptions(BaseSandboxClientOptions):
    """How to create a Maritime sandbox.

    Unknown fields are rejected, so a typo never silently drops a setting.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["maritime"] = "maritime"
    template: str | None = None
    """A saved snapshot, a stock or custom template, or a public image. Defaults to ``base``."""
    size: SandboxSize | None = None
    """A size preset. ``vcpu``, ``memory_mib`` and ``disk_gib`` override its values."""
    vcpu: int | None = Field(default=None, ge=1, le=16)
    memory_mib: int | None = Field(default=None, ge=1024, le=65536)
    disk_gib: int | None = Field(default=None, ge=5, le=500)
    env: dict[str, str] = Field(default_factory=dict)
    """Environment variables set on the sandbox for every command."""
    metadata: dict[str, str] = Field(default_factory=dict)
    """Labels you can filter on when listing sandboxes."""
    name: str | None = None
    """A unique name. Creating with a name that already runs returns that sandbox."""
    end_user: str | None = None
    """Your own id for the person this sandbox works for."""
    idle_timeout_seconds: int | None = Field(default=600, ge=60)
    """After this long with no activity Maritime applies ``on_idle``."""
    on_idle: IdleAction = "pause"
    time_limit_seconds: int | None = Field(default=None, ge=60, le=86400)
    """Hard limit: the sandbox ends this long after it starts, whatever its activity."""
    pause_on_exit: bool = False
    """Pause instead of ending the sandbox when the run's session shuts down."""
    exposed_ports: tuple[int, ...] = ()
    """Ports the agent may expose. Each gets a private HTTPS URL with a token."""
    exec_timeout_unbounded_s: int = Field(default=3600, ge=1, le=86400)
    """The timeout for commands the SDK runs without one."""


class MaritimeSandboxSessionState(SandboxSessionState):
    """What the runner saves to reconnect to a Maritime sandbox later.

    ``env`` is saved so a sandbox that ended can be recreated with the same
    settings. Do not put secrets you would not store with your run state there;
    use Maritime account secrets instead.
    """

    type: Literal["maritime"] = "maritime"
    sandbox_id: str
    template: str | None = None
    size: SandboxSize | None = None
    vcpu: int | None = None
    memory_mib: int | None = None
    disk_gib: int | None = None
    env: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    name: str | None = None
    end_user: str | None = None
    idle_timeout_seconds: int | None = 600
    on_idle: IdleAction = "pause"
    time_limit_seconds: int | None = None
    pause_on_exit: bool = False
    exec_timeout_unbounded_s: int = 3600


class MaritimeSandboxSession(BaseSandboxSession):
    """Runs the SDK's commands and file operations inside one Maritime sandbox."""

    state: MaritimeSandboxSessionState

    def __init__(self, *, state: MaritimeSandboxSessionState, sandbox: Sandbox) -> None:
        self.state = state
        self._sandbox = sandbox
        self._root_prepared = False

    @classmethod
    def from_state(cls, state: MaritimeSandboxSessionState, *, sandbox: Sandbox) -> MaritimeSandboxSession:
        return cls(state=state, sandbox=sandbox)

    @property
    def sandbox(self) -> Sandbox:
        """The Maritime SDK ``Sandbox`` client, for anything this adapter does not wrap."""
        return self._sandbox

    @property
    def sandbox_id(self) -> str:
        return self.state.sandbox_id

    async def _call(self, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
        return await asyncio.to_thread(fn, *args, **kwargs)

    # Workspace setup -------------------------------------------------------

    def _runtime_helpers(self) -> tuple[RuntimeHelperScript, ...]:
        return (RESOLVE_WORKSPACE_PATH_HELPER,)

    async def _validate_path_access(self, path: Path | str, *, for_write: bool = False) -> Path:
        return await self._validate_remote_path_access(path, for_write=for_write)

    async def _prepare_backend_workspace(self) -> None:
        root = sandbox_path_str(self.state.manifest.root)
        quoted = shlex.quote(root)
        # Commands run as the unprivileged user ``user``. A root outside its home,
        # such as the SDK's default /workspace, is created with sudo and handed over.
        script = (
            f"mkdir -p -- {quoted} 2>/dev/null || "
            f'{{ sudo -n mkdir -p -- {quoted} && sudo -n chown "$(id -u):$(id -g)" -- {quoted}; }}'
        )
        try:
            command = await self._call(self._sandbox.run, ["sh", "-c", script], timeout_seconds=60)
        except Exception as e:
            raise WorkspaceStartError(
                path=Path(root),
                context={**_provider_context(e), "reason": "workspace_root_setup_failed"},
                cause=e,
            ) from e
        if command.exit_code != 0:
            raise WorkspaceStartError(
                path=Path(root),
                context={
                    "backend": BACKEND_ID,
                    "reason": "workspace_root_nonzero_exit",
                    "exit_code": command.exit_code,
                    "output": command.stderr or command.stdout,
                },
                message=f"failed to start session: could not create {root} in the Maritime sandbox",
            )
        self._root_prepared = True

    async def _resolved_envs(self) -> dict[str, str]:
        return dict(await self.state.manifest.environment.resolve())

    def _exec_timeout_seconds(self, timeout: float | None) -> int:
        if timeout is None:
            return self.state.exec_timeout_unbounded_s
        return max(1, math.ceil(timeout))

    # Commands --------------------------------------------------------------

    async def _exec_internal(self, *command: str | Path, timeout: float | None = None) -> ExecResult:
        argv = [str(part) for part in command]
        envs = await self._resolved_envs()
        cwd = (
            sandbox_path_str(self.state.manifest.root)
            if self._root_prepared or self.state.workspace_root_ready
            else None
        )
        try:
            result = await self._call(
                self._sandbox.run,
                argv,
                cwd=cwd,
                env=envs or None,
                timeout_seconds=self._exec_timeout_seconds(timeout),
            )
        except MaritimeSandboxError as e:
            if e.code == "command_timeout":
                raise ExecTimeoutError(command=command, timeout_s=timeout, cause=e) from e
            raise ExecTransportError(
                command=command,
                context=_provider_context(e),
                cause=e,
                retryable=_is_retryable(e),
            ) from e
        except Exception as e:
            raise ExecTransportError(
                command=command,
                context=_provider_context(e),
                cause=e,
                retryable=_is_retryable(e),
            ) from e
        # A process ended by a signal has no exit code; report it the way a shell does.
        exit_code = result.exit_code if result.exit_code is not None else 128 + 9
        return ExecResult(
            stdout=(result.stdout or "").encode("utf-8", errors="replace"),
            stderr=(result.stderr or "").encode("utf-8", errors="replace"),
            exit_code=exit_code,
        )

    # Files -----------------------------------------------------------------

    async def read(self, path: Path | str, *, user: str | User | None = None) -> io.IOBase:
        error_path = posix_path_as_path(coerce_posix_path(path))
        if user is not None:
            workspace_path = await self._check_read_with_exec(path, user=user)
        else:
            workspace_path = await self._validate_path_access(path)
        try:
            data = await self._call(self._sandbox.files.read, sandbox_path_str(workspace_path))
        except MaritimeSandboxError as e:
            if e.code in _NOT_FOUND_CODES:
                raise WorkspaceReadNotFoundError(path=error_path, cause=e) from e
            raise WorkspaceArchiveReadError(
                path=error_path, context=_provider_context(e), cause=e, retryable=_is_retryable(e)
            ) from e
        except Exception as e:
            raise WorkspaceArchiveReadError(
                path=error_path, context=_provider_context(e), cause=e, retryable=_is_retryable(e)
            ) from e
        return io.BytesIO(data)

    async def write(self, path: Path | str, data: io.IOBase, *, user: str | User | None = None) -> None:
        error_path = posix_path_as_path(coerce_posix_path(path))
        if user is not None:
            await self._check_write_with_exec(path, user=user)
        payload = data.read()
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if not isinstance(payload, bytes | bytearray):
            raise WorkspaceWriteTypeError(path=error_path, actual_type=type(payload).__name__)
        workspace_path = await self._validate_path_access(path, for_write=True)
        try:
            await self._call(self._sandbox.files.write, sandbox_path_str(workspace_path), bytes(payload))
        except Exception as e:
            raise WorkspaceArchiveWriteError(
                path=workspace_path, context=_provider_context(e), cause=e
            ) from e

    # Lifecycle -------------------------------------------------------------

    async def running(self) -> bool:
        if not self.state.workspace_root_ready:
            return False
        try:
            data = await self._call(self._sandbox.refresh)
        except Exception:
            return False
        return data.state in _ALIVE_STATES

    async def _shutdown_backend(self) -> None:
        try:
            if self.state.pause_on_exit:
                await self._call(self._sandbox.pause)
            else:
                await self._call(self._sandbox.kill)
        except Exception:
            # Shutdown must tolerate partial startup and repeated calls. A sandbox
            # that cannot be reached here still ends on its idle or time limit.
            pass

    async def _resolve_exposed_port(self, port: int) -> ExposedPortEndpoint:
        try:
            url = await self._call(self._sandbox.url, port, token=True)
            split = urlsplit(url)
            if split.hostname is None:
                raise ValueError("the URL has no host")
        except Exception as e:
            raise ExposedPortUnavailableError(
                port=port,
                exposed_ports=self.state.exposed_ports,
                reason="backend_unavailable",
                context=_provider_context(e),
                cause=e,
            ) from e
        tls = split.scheme == "https"
        return ExposedPortEndpoint(
            host=split.hostname,
            port=split.port or (443 if tls else 80),
            tls=tls,
            query=split.query,
        )

    # Workspace snapshots ---------------------------------------------------

    async def persist_workspace(self) -> io.IOBase:
        root = self._workspace_root_path()
        tar_path = f"/tmp/maritime-persist-{self.state.session_id.hex}.tar"
        excludes = " ".join(shell_tar_exclude_args(self._persist_workspace_skip_relpaths()))
        tar_cmd = f"tar {excludes} -C {shlex.quote(root.as_posix())} -cf {shlex.quote(tar_path)} ."
        try:
            result = await self.exec(tar_cmd, timeout=600)
            if not result.ok():
                raise WorkspaceArchiveReadError(
                    path=root,
                    context={
                        "backend": BACKEND_ID,
                        "reason": "tar_failed",
                        "output": result.stderr.decode("utf-8", errors="replace"),
                    },
                    retryable=False,
                )
            raw = await self._call(self._sandbox.files.read, tar_path)
        except SandboxError:
            raise
        except Exception as e:
            raise WorkspaceArchiveReadError(
                path=root, context=_provider_context(e), cause=e, retryable=_is_retryable(e)
            ) from e
        finally:
            try:
                await self._call(self._sandbox.files.remove, tar_path)
            except Exception:
                pass
        return io.BytesIO(raw)

    async def hydrate_workspace(self, data: io.IOBase) -> None:
        root = self._workspace_root_path()
        tar_path = f"/tmp/maritime-hydrate-{self.state.session_id.hex}.tar"
        payload = data.read()
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if not isinstance(payload, bytes | bytearray):
            raise WorkspaceWriteTypeError(path=Path(tar_path), actual_type=type(payload).__name__)
        try:
            validate_tar_bytes(bytes(payload), allow_external_symlink_targets=False)
        except UnsafeTarMemberError as e:
            raise WorkspaceArchiveWriteError(
                path=root,
                context={"reason": "unsafe_or_invalid_tar", "member": e.member, "detail": str(e)},
                cause=e,
            ) from e
        try:
            await self._call(self._sandbox.files.write, tar_path, bytes(payload))
            result = await self.exec(
                f"mkdir -p -- {shlex.quote(root.as_posix())} && "
                f"tar -C {shlex.quote(root.as_posix())} -xf {shlex.quote(tar_path)}",
                timeout=600,
            )
            if not result.ok():
                raise WorkspaceArchiveWriteError(
                    path=root,
                    context={
                        "backend": BACKEND_ID,
                        "reason": "tar_extract_failed",
                        "output": result.stderr.decode("utf-8", errors="replace"),
                    },
                )
        except SandboxError:
            raise
        except Exception as e:
            raise WorkspaceArchiveWriteError(path=root, context=_provider_context(e), cause=e) from e
        finally:
            try:
                await self._call(self._sandbox.files.remove, tar_path)
            except Exception:
                pass


class MaritimeSandboxClient(BaseSandboxClient[MaritimeSandboxClientOptions]):
    """Creates, resumes and deletes Maritime sandboxes for the agent runner.

    Pass an API key, or set ``MARITIME_API_KEY``. You can also pass a
    ``maritime.Maritime`` client you already have.
    """

    backend_id = BACKEND_ID

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Maritime | None = None,
        instrumentation: Instrumentation | None = None,
        dependencies: Dependencies | None = None,
    ) -> None:
        if client is not None and (api_key is not None or base_url is not None):
            raise ValueError("pass either client or api_key/base_url, not both")
        self._maritime = client if client is not None else Maritime(api_key=api_key, base_url=base_url)
        self._instrumentation = instrumentation if instrumentation is not None else Instrumentation()
        self._dependencies = dependencies

    @property
    def maritime(self) -> Maritime:
        return self._maritime

    def _validate_manifest_for_create(self, manifest: Manifest) -> Manifest:
        manifest = super()._validate_manifest_for_create(manifest)
        mounts = [str(path) for path, entry in manifest.iter_entries() if isinstance(entry, Mount)]
        if mounts:
            raise ValueError(
                "Maritime sandboxes do not support manifest mounts yet: "
                + ", ".join(mounts)
                + ". Copy the files in with File or Dir entries instead."
            )
        return manifest

    def _create_sandbox(self, state: MaritimeSandboxSessionState) -> Sandbox:
        return self._maritime.sandboxes.create(
            name=state.name,
            from_=state.template,
            size=state.size,
            vcpu=state.vcpu,
            memory_mib=state.memory_mib,
            disk_gib=state.disk_gib,
            end_user=state.end_user,
            env=state.env or None,
            metadata=state.metadata or None,
            idle_timeout_seconds=state.idle_timeout_seconds,
            on_idle=state.on_idle,
            time_limit_seconds=state.time_limit_seconds,
        )

    async def create(
        self,
        *,
        snapshot: SnapshotSpec | SnapshotBase | None = None,
        manifest: Manifest | None = None,
        options: MaritimeSandboxClientOptions,
    ) -> SandboxSession:
        if not isinstance(options, MaritimeSandboxClientOptions):
            raise TypeError(
                f"MaritimeSandboxClient requires MaritimeSandboxClientOptions, got {type(options).__name__}"
            )
        if manifest is None:
            manifest = Manifest(root=DEFAULT_MARITIME_WORKSPACE_ROOT)
        manifest = self._validate_manifest_for_create(manifest)

        session_id = uuid.uuid4()
        state = MaritimeSandboxSessionState(
            session_id=session_id,
            manifest=manifest,
            snapshot=resolve_snapshot(snapshot, str(session_id)),
            sandbox_id="",
            template=options.template,
            size=options.size,
            vcpu=options.vcpu,
            memory_mib=options.memory_mib,
            disk_gib=options.disk_gib,
            env=dict(options.env),
            metadata=dict(options.metadata),
            name=options.name,
            end_user=options.end_user,
            idle_timeout_seconds=options.idle_timeout_seconds,
            on_idle=options.on_idle,
            time_limit_seconds=options.time_limit_seconds,
            pause_on_exit=options.pause_on_exit,
            exposed_ports=options.exposed_ports,
            exec_timeout_unbounded_s=options.exec_timeout_unbounded_s,
        )
        sandbox = await asyncio.to_thread(self._create_sandbox, state)
        state.sandbox_id = sandbox.id
        inner = MaritimeSandboxSession.from_state(state, sandbox=sandbox)
        return self._wrap_session(inner, instrumentation=self._instrumentation)

    async def delete(self, session: SandboxSession) -> SandboxSession:
        inner = session._inner
        if not isinstance(inner, MaritimeSandboxSession):
            raise TypeError("MaritimeSandboxClient.delete expects a MaritimeSandboxSession")
        try:
            await inner.shutdown()
        except Exception:
            pass
        return session

    async def resume(self, state: SandboxSessionState) -> SandboxSession:
        if not isinstance(state, MaritimeSandboxSessionState):
            raise TypeError("MaritimeSandboxClient.resume expects a MaritimeSandboxSessionState")
        state.assert_path_grants_rebound()

        existing: Sandbox | None = None
        if state.sandbox_id:
            try:
                found: Sandbox = await asyncio.to_thread(self._maritime.sandboxes.get, state.sandbox_id)
            except (MaritimeSandboxError, MaritimeAPIError):
                found = None  # type: ignore[assignment]
            if found is not None and found.state in _ALIVE_STATES:
                existing = found
        reconnected = existing is not None
        if existing is not None:
            sandbox = existing
        else:
            sandbox = await asyncio.to_thread(self._create_sandbox, state)
            state.sandbox_id = sandbox.id
            state.workspace_root_ready = False

        inner = MaritimeSandboxSession.from_state(state, sandbox=sandbox)
        inner._set_start_state_preserved(reconnected, system=reconnected)
        return self._wrap_session(inner, instrumentation=self._instrumentation)

    def deserialize_session_state(self, payload: dict[str, object]) -> SandboxSessionState:
        return self._deserialize_session_state_payload(payload, MaritimeSandboxSessionState)
