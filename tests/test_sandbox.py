from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from agents import RunConfig, Runner
from agents.sandbox import Manifest, SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Filesystem, Shell
from agents.sandbox.entries import File, InContainerMountStrategy, MountpointMountPattern, S3Mount
from agents.sandbox.errors import (
    ExecTimeoutError,
    ExecTransportError,
    ExposedPortUnavailableError,
    InvalidManifestPathError,
    WorkspaceArchiveWriteError,
    WorkspaceReadNotFoundError,
    WorkspaceStartError,
)
from agents.sandbox.manifest import Environment
from agents.sandbox.session import BaseSandboxClientOptions
from agents.sandbox.session.sandbox_session_state import SandboxSessionState
from agents.testing import ScriptedModel, assistant_message, function_call
from maritime import MaritimeSandboxError
from pydantic import ValidationError

from openai_agents_maritime import (
    MaritimeSandboxClient,
    MaritimeSandboxClientOptions,
    MaritimeSandboxSession,
    MaritimeSandboxSessionState,
)

from .fakes import FakeMaritime, FakeSandbox


@pytest.fixture
def fake() -> FakeMaritime:
    return FakeMaritime()


@pytest.fixture
def client(fake: FakeMaritime) -> MaritimeSandboxClient:
    return MaritimeSandboxClient(client=fake)  # type: ignore[arg-type]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "workspace"


async def _started(client: MaritimeSandboxClient, root: Path, **options: object):
    session = await client.create(
        manifest=Manifest(root=str(root)),
        options=MaritimeSandboxClientOptions(**options),  # type: ignore[arg-type]
    )
    await session.start()
    return session


def _inner(session) -> MaritimeSandboxSession:
    inner = session._inner
    assert isinstance(inner, MaritimeSandboxSession)
    return inner


# Options and state -----------------------------------------------------------


def test_options_register_on_import_in_a_fresh_interpreter() -> None:
    code = (
        "import openai_agents_maritime\n"
        "from agents.sandbox.session import BaseSandboxClientOptions\n"
        "o = BaseSandboxClientOptions.parse({'type': 'maritime', 'size': 'large', 'env': {'A': 'b'}})\n"
        "print(type(o).__name__, o.size, o.env)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "MaritimeSandboxClientOptions large {'A': 'b'}"


def test_options_reject_unknown_fields_and_bad_values() -> None:
    with pytest.raises(ValidationError):
        MaritimeSandboxClientOptions(sizee="large")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        MaritimeSandboxClientOptions(size="huge")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        MaritimeSandboxClientOptions(vcpu=0)
    with pytest.raises(ValidationError):
        MaritimeSandboxClientOptions(on_idle="sleep")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        BaseSandboxClientOptions.parse({"type": "maritime", "network_isolated": True})


def test_options_round_trip_through_json() -> None:
    options = MaritimeSandboxClientOptions(template="my-snapshot", vcpu=4, exposed_ports=(8080,))
    again = BaseSandboxClientOptions.parse(json.loads(options.model_dump_json()))
    assert again == options


async def test_state_round_trips_through_the_client(client: MaritimeSandboxClient, root: Path) -> None:
    session = await _started(client, root, size="medium", env={"MODE": "test"}, pause_on_exit=True)
    payload = client.serialize_session_state(session.state)
    restored = client.deserialize_session_state(json.loads(json.dumps(payload)))
    assert isinstance(restored, MaritimeSandboxSessionState)
    assert restored.sandbox_id == session.state.sandbox_id
    assert (restored.size, restored.env, restored.pause_on_exit) == ("medium", {"MODE": "test"}, True)
    assert isinstance(SandboxSessionState.parse(payload), MaritimeSandboxSessionState)


def test_deserialize_rejects_malformed_state(client: MaritimeSandboxClient) -> None:
    with pytest.raises(ValueError):
        client.deserialize_session_state({"type": "maritime", "manifest": {}})


def test_client_refuses_client_and_key_together(fake: FakeMaritime) -> None:
    with pytest.raises(ValueError):
        MaritimeSandboxClient(client=fake, api_key="mk_x")  # type: ignore[arg-type]


# Create ----------------------------------------------------------------------


async def test_create_passes_every_option_to_maritime(
    client: MaritimeSandboxClient, fake: FakeMaritime
) -> None:
    await client.create(
        options=MaritimeSandboxClientOptions(
            template="py-snap",
            size="large",
            vcpu=4,
            memory_mib=8192,
            disk_gib=20,
            env={"A": "1"},
            metadata={"team": "x"},
            name="job-7",
            end_user="cust_1",
            idle_timeout_seconds=120,
            on_idle="kill",
            time_limit_seconds=900,
        )
    )
    (sandbox,) = fake.sandboxes.created
    assert sandbox.create_kwargs == {
        "name": "job-7",
        "from_": "py-snap",
        "size": "large",
        "vcpu": 4,
        "memory_mib": 8192,
        "disk_gib": 20,
        "end_user": "cust_1",
        "env": {"A": "1"},
        "metadata": {"team": "x"},
        "idle_timeout_seconds": 120,
        "on_idle": "kill",
        "time_limit_seconds": 900,
    }


async def test_create_defaults_the_workspace_root(client: MaritimeSandboxClient) -> None:
    session = await client.create(options=MaritimeSandboxClientOptions())
    assert str(session.state.manifest.root) == "/home/user/workspace"


async def test_create_rejects_other_options(client: MaritimeSandboxClient, fake: FakeMaritime) -> None:
    class OtherOptions(BaseSandboxClientOptions):
        type: str = "not_maritime_test"

    with pytest.raises(TypeError):
        await client.create(options=OtherOptions())  # type: ignore[arg-type]
    assert fake.sandboxes.created == []


async def test_create_rejects_mounts_before_allocating(
    client: MaritimeSandboxClient, fake: FakeMaritime
) -> None:
    mount = S3Mount(bucket="b", mount_strategy=InContainerMountStrategy(pattern=MountpointMountPattern()))
    manifest = Manifest(entries={"data": mount})
    with pytest.raises(ValueError, match="mounts"):
        await client.create(manifest=manifest, options=MaritimeSandboxClientOptions())
    assert fake.sandboxes.created == []


async def test_create_surfaces_maritime_errors(client: MaritimeSandboxClient, fake: FakeMaritime) -> None:
    fake.sandboxes.create_error = MaritimeSandboxError(402, code="plan_required", message="upgrade")
    with pytest.raises(MaritimeSandboxError, match="plan_required"):
        await client.create(options=MaritimeSandboxClientOptions())


# Commands --------------------------------------------------------------------


async def test_exec_runs_in_the_workspace_with_manifest_env(
    client: MaritimeSandboxClient, root: Path
) -> None:
    manifest = Manifest(root=str(root), environment=Environment(value={"GREETING": "ahoy"}))
    session = await client.create(manifest=manifest, options=MaritimeSandboxClientOptions())
    await session.start()
    result = await session.exec("pwd; echo $GREETING; exit 3")
    assert result.exit_code == 3
    assert result.stdout.decode().split() == [str(root), "ahoy"]


async def test_exec_omits_cwd_until_the_root_exists(
    client: MaritimeSandboxClient, fake: FakeMaritime, root: Path
) -> None:
    session = await client.create(manifest=Manifest(root=str(root)), options=MaritimeSandboxClientOptions())
    sandbox = fake.sandboxes.created[0]
    await _inner(session)._exec_internal("true")
    assert sandbox.calls[-1][2] is None
    await session.start()
    await session.exec("true")
    assert sandbox.calls[-1][2] == str(root)


async def test_start_creates_a_missing_root(client: MaritimeSandboxClient, root: Path) -> None:
    assert not root.exists()
    await _started(client, root)
    assert root.is_dir()


async def test_start_fails_clearly_when_the_root_cannot_be_created(
    client: MaritimeSandboxClient, fake: FakeMaritime, tmp_path: Path
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    session = await client.create(
        manifest=Manifest(root=str(blocker / "workspace")), options=MaritimeSandboxClientOptions()
    )
    with pytest.raises(WorkspaceStartError):
        await session.start()


async def test_exec_maps_maritime_timeout(client: MaritimeSandboxClient, root: Path) -> None:
    session = await _started(client, root)
    with pytest.raises(ExecTimeoutError):
        await session.exec("sleep 5", timeout=1)


async def test_exec_rounds_timeouts_up_to_whole_seconds(
    client: MaritimeSandboxClient, fake: FakeMaritime, root: Path
) -> None:
    session = await _started(client, root, exec_timeout_unbounded_s=77)
    await session.exec("true", timeout=0.2)
    assert fake.sandboxes.created[0].calls[-1][4] == 1
    await session.exec("true", timeout=2.5)
    assert fake.sandboxes.created[0].calls[-1][4] == 3
    await session.exec("true")
    assert fake.sandboxes.created[0].calls[-1][4] == 77


async def test_exec_maps_transport_failures_without_leaking_secrets(
    client: MaritimeSandboxClient, fake: FakeMaritime, root: Path
) -> None:
    session = await _started(client, root)
    fake.sandboxes.created[0].fail_runs_with = MaritimeSandboxError(
        503, code="unavailable", message="host busy", request_id="req_1"
    )
    with pytest.raises(ExecTransportError) as caught:
        await session.exec("true")
    assert caught.value.context["provider_error_code"] == "unavailable"
    assert caught.value.context["provider_request_id"] == "req_1"
    assert caught.value.retryable is True


async def test_exec_reports_a_signal_kill_as_137(
    client: MaritimeSandboxClient, fake: FakeMaritime, root: Path
) -> None:
    session = await _started(client, root)
    fake.sandboxes.created[0].exit_code_override = None
    result = await session.exec("true")
    assert result.exit_code == 137


# Files -----------------------------------------------------------------------


async def test_write_then_read(client: MaritimeSandboxClient, root: Path) -> None:
    session = await _started(client, root)
    await session.write(root / "dir" / "a.bin", io.BytesIO(b"\x00\x01data"))
    assert (await session.read(root / "dir" / "a.bin")).read() == b"\x00\x01data"
    await session.write(root / "t.txt", io.StringIO("text"))
    assert (root / "t.txt").read_text() == "text"


async def test_read_missing_file(client: MaritimeSandboxClient, root: Path) -> None:
    session = await _started(client, root)
    with pytest.raises(WorkspaceReadNotFoundError):
        await session.read(root / "nope.txt")


async def test_read_and_write_refuse_paths_outside_the_workspace(
    client: MaritimeSandboxClient, root: Path, tmp_path: Path
) -> None:
    session = await _started(client, root)
    outside = tmp_path / "secret.txt"
    outside.write_text("keep out")
    with pytest.raises(InvalidManifestPathError):
        await session.read(outside)
    with pytest.raises(InvalidManifestPathError):
        await session.write(root / ".." / "escape.txt", io.BytesIO(b"x"))
    assert not (tmp_path / "escape.txt").exists()


async def test_write_failure_is_a_workspace_error(
    client: MaritimeSandboxClient, fake: FakeMaritime, root: Path
) -> None:
    session = await _started(client, root)
    fake.sandboxes.created[0].fail_writes = True
    with pytest.raises(WorkspaceArchiveWriteError):
        await session.write(root / "a.txt", io.BytesIO(b"x"))


# Snapshots -------------------------------------------------------------------


async def test_persist_and_hydrate_round_trip(
    client: MaritimeSandboxClient, root: Path, tmp_path: Path
) -> None:
    session = await _started(client, root)
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("print('hi')\n")
    archive = (await _inner(session).persist_workspace()).read()
    names = tarfile.open(fileobj=io.BytesIO(archive)).getnames()
    assert "./src/main.py" in names
    assert not list(Path("/tmp").glob(f"maritime-persist-{session.state.session_id.hex}.tar"))

    other_root = tmp_path / "other"
    other = await _started(client, other_root)
    await _inner(other).hydrate_workspace(io.BytesIO(archive))
    assert (other_root / "src" / "main.py").read_text() == "print('hi')\n"


async def test_hydrate_refuses_unsafe_archives(client: MaritimeSandboxClient, root: Path) -> None:
    session = await _started(client, root)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(WorkspaceArchiveWriteError):
        await _inner(session).hydrate_workspace(io.BytesIO(buffer.getvalue()))
    assert not (root.parent / "escape.txt").exists()


# Lifecycle -------------------------------------------------------------------


async def test_running_reflects_the_sandbox_state(
    client: MaritimeSandboxClient, fake: FakeMaritime, root: Path
) -> None:
    session = await _started(client, root)
    sandbox = fake.sandboxes.created[0]
    assert await session.running() is True
    sandbox.state = "paused"
    assert await session.running() is True  # paused keeps its disk and resumes on the next call
    sandbox.state = "ended"
    assert await session.running() is False


async def test_shutdown_kills_by_default(
    client: MaritimeSandboxClient, fake: FakeMaritime, root: Path
) -> None:
    session = await _started(client, root)
    await client.delete(session)
    assert fake.sandboxes.created[0].state == "ended"
    await client.delete(session)  # repeated cleanup is safe


async def test_shutdown_pauses_when_asked(
    client: MaritimeSandboxClient, fake: FakeMaritime, root: Path
) -> None:
    session = await _started(client, root, pause_on_exit=True)
    await client.delete(session)
    assert fake.sandboxes.created[0].state == "paused"
    assert ("kill",) not in fake.sandboxes.created[0].calls


async def test_resume_reconnects_to_a_live_sandbox(
    client: MaritimeSandboxClient, fake: FakeMaritime, root: Path
) -> None:
    session = await _started(client, root, pause_on_exit=True)
    (root / "keep.txt").write_text("still here")
    await client.delete(session)
    state = client.deserialize_session_state(client.serialize_session_state(session.state))

    resumed = await client.resume(state)
    await resumed.start()
    assert len(fake.sandboxes.created) == 1
    assert resumed.state.sandbox_id == session.state.sandbox_id
    assert (await resumed.read(root / "keep.txt")).read() == b"still here"


async def test_resume_recreates_an_ended_sandbox_with_the_same_settings(
    client: MaritimeSandboxClient, fake: FakeMaritime, root: Path
) -> None:
    session = await _started(client, root, size="large", env={"K": "v"})
    await client.delete(session)
    state = client.deserialize_session_state(client.serialize_session_state(session.state))

    resumed = await client.resume(state)
    assert len(fake.sandboxes.created) == 2
    replacement = fake.sandboxes.created[1]
    assert resumed.state.sandbox_id == replacement.id != session.state.sandbox_id
    assert resumed.state.workspace_root_ready is False
    assert replacement.create_kwargs["size"] == "large"
    assert replacement.create_kwargs["env"] == {"K": "v"}


async def test_resume_recreates_a_sandbox_maritime_no_longer_knows(
    client: MaritimeSandboxClient, fake: FakeMaritime, root: Path
) -> None:
    session = await _started(client, root)
    del fake.sandboxes.by_id[session.state.sandbox_id]
    resumed = await client.resume(session.state)
    assert resumed.state.sandbox_id == fake.sandboxes.created[1].id


async def test_resume_and_delete_reject_foreign_types(client: MaritimeSandboxClient) -> None:
    with pytest.raises(TypeError):
        await client.resume(object())  # type: ignore[arg-type]

    class Other:
        _inner = object()

    with pytest.raises(TypeError):
        await client.delete(Other())  # type: ignore[arg-type]


# Ports -----------------------------------------------------------------------


async def test_exposed_port_keeps_the_token_in_the_query(
    client: MaritimeSandboxClient, fake: FakeMaritime, root: Path
) -> None:
    session = await _started(client, root, exposed_ports=(8080,))
    endpoint = await session.resolve_exposed_port(8080)
    sandbox = fake.sandboxes.created[0]
    assert (endpoint.host, endpoint.port, endpoint.tls) == (f"8080-{sandbox.id}.sbx.maritime.sh", 443, True)
    assert endpoint.query == "token=tok_secret"
    assert endpoint.url_for("http").startswith("https://8080-")
    assert ("url", 8080, True) in sandbox.calls


async def test_unconfigured_port_is_refused(client: MaritimeSandboxClient, root: Path) -> None:
    session = await _started(client, root, exposed_ports=(8080,))
    with pytest.raises(ExposedPortUnavailableError):
        await session.resolve_exposed_port(9000)


# The public Runner path -------------------------------------------------------


async def test_runner_uses_the_sandbox_and_cleans_it_up(
    client: MaritimeSandboxClient, fake: FakeMaritime, root: Path
) -> None:
    model = ScriptedModel(
        [
            [
                function_call(
                    "exec_command", {"cmd": "cat notes.txt && echo from-shell > out.txt"}, call_id="c1"
                )
            ],
            [assistant_message("done")],
        ]
    )
    agent = SandboxAgent(
        name="t",
        model=model,
        capabilities=[Shell(), Filesystem()],
        default_manifest=Manifest(root=str(root), entries={"notes.txt": File(content=b"manifest file\n")}),
    )
    result = await Runner.run(
        agent,
        "go",
        run_config=RunConfig(
            tracing_disabled=True,
            sandbox=SandboxRunConfig(client=client, options=MaritimeSandboxClientOptions()),
        ),
    )
    model.assert_complete()
    outputs = [str(i.output) for i in result.new_items if i.type == "tool_call_output_item"]
    assert any("manifest file" in o for o in outputs)
    assert (root / "out.txt").read_text() == "from-shell\n"
    (sandbox,) = fake.sandboxes.created
    assert sandbox.state == "ended"


async def test_runner_cleans_up_after_a_tool_error(
    client: MaritimeSandboxClient, fake: FakeMaritime, root: Path
) -> None:
    model = ScriptedModel(
        [
            [function_call("exec_command", {"cmd": "exit 9"}, call_id="c1")],
            [assistant_message("it failed")],
        ]
    )
    agent = SandboxAgent(
        name="t", model=model, capabilities=[Shell()], default_manifest=Manifest(root=str(root))
    )
    await Runner.run(
        agent,
        "go",
        run_config=RunConfig(
            tracing_disabled=True,
            sandbox=SandboxRunConfig(client=client, options=MaritimeSandboxClientOptions()),
        ),
    )
    assert fake.sandboxes.created[0].state == "ended"


async def test_runner_leaves_an_application_owned_session_running(
    client: MaritimeSandboxClient, fake: FakeMaritime, root: Path
) -> None:
    session = await client.create(manifest=Manifest(root=str(root)), options=MaritimeSandboxClientOptions())
    async with session:
        model = ScriptedModel(
            [[function_call("exec_command", {"cmd": "echo owned"}, call_id="c1")], [assistant_message("ok")]]
        )
        agent = SandboxAgent(name="t", model=model, capabilities=[Shell()])
        await Runner.run(
            agent,
            "go",
            run_config=RunConfig(tracing_disabled=True, sandbox=SandboxRunConfig(session=session)),
        )
        assert fake.sandboxes.created[0].state == "running"
    await client.delete(session)
    assert fake.sandboxes.created[0].state == "ended"


def test_fake_matches_the_sdk_surface_we_use() -> None:
    """Guard against the fake drifting from the real Maritime SDK."""
    from maritime import Sandbox, SandboxFiles
    from maritime.sandboxes.resources import Sandboxes

    for name in ("run", "refresh", "pause", "kill", "url"):
        assert hasattr(Sandbox, name) and hasattr(FakeSandbox, name)
    for name in ("read", "write", "remove"):
        assert hasattr(SandboxFiles, name)
    for name in ("create", "get"):
        assert hasattr(Sandboxes, name)
