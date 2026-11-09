"""Tests against a real Maritime account. Run with: pytest -m live (needs MARITIME_API_KEY)."""

from __future__ import annotations

import io
import os
import time
import urllib.request
from pathlib import Path

import pytest
from agents import RunConfig, Runner
from agents.sandbox import Manifest, SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Filesystem, Shell
from agents.sandbox.entries import File
from agents.sandbox.errors import ExecTimeoutError, WorkspaceReadNotFoundError
from agents.testing import ScriptedModel, assistant_message, function_call

from openai_agents_maritime import MaritimeSandboxClient, MaritimeSandboxClientOptions

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("MARITIME_API_KEY"), reason="needs MARITIME_API_KEY"),
]

OPTIONS = dict(idle_timeout_seconds=300, on_idle="kill", time_limit_seconds=1800, metadata={"test": "live"})


async def test_commands_files_and_snapshots_on_a_real_sandbox() -> None:
    client = MaritimeSandboxClient()
    session = await client.create(options=MaritimeSandboxClientOptions(**OPTIONS))
    try:
        await session.start()
        root = Path(session.state.manifest.root)

        result = await session.exec("pwd; whoami; echo out; echo err >&2; exit 4")
        assert result.exit_code == 4
        assert result.stdout.decode().split() == [str(root), "user", "out"]
        assert result.stderr.decode().strip() == "err"

        await session.write(root / "dir" / "data.bin", io.BytesIO(bytes(range(256))))
        assert (await session.read(root / "dir" / "data.bin")).read() == bytes(range(256))
        with pytest.raises(WorkspaceReadNotFoundError):
            await session.read(root / "missing.txt")

        as_nobody = await session.exec("id -un", user="nobody")
        assert as_nobody.stdout.decode().strip() == "nobody"

        with pytest.raises(ExecTimeoutError):
            await session.exec("sleep 10", timeout=2)

        archive = (await session._inner.persist_workspace()).read()
        await session.exec("rm -rf dir")
        await session._inner.hydrate_workspace(io.BytesIO(archive))
        assert (await session.read(root / "dir" / "data.bin")).read() == bytes(range(256))
    finally:
        await client.delete(session)
    assert client.maritime.sandboxes.get(session.state.sandbox_id).state == "ended"


async def test_default_sdk_workspace_outside_home_is_created() -> None:
    client = MaritimeSandboxClient()
    session = await client.create(manifest=Manifest(), options=MaritimeSandboxClientOptions(**OPTIONS))
    try:
        await session.start()
        result = await session.exec("pwd; touch made-here && ls -ld .")
        assert result.ok(), result.stderr
        assert result.stdout.decode().startswith("/workspace\n")
        assert " user user " in result.stdout.decode()
    finally:
        await client.delete(session)


async def test_exposed_port_serves_over_https_with_its_token() -> None:
    client = MaritimeSandboxClient()
    session = await client.create(options=MaritimeSandboxClientOptions(exposed_ports=(8000,), **OPTIONS))
    try:
        await session.start()
        await session.exec("echo served-by-maritime > index.html")
        await session.exec("nohup python3 -m http.server 8000 >/tmp/http.log 2>&1 &")
        endpoint = await session.resolve_exposed_port(8000)
        assert endpoint.tls and endpoint.query
        url = endpoint.url_for("http")
        body = ""
        for _ in range(20):
            try:
                with urllib.request.urlopen(url, timeout=10) as response:
                    body = response.read().decode()
                    break
            except Exception:
                time.sleep(2)
        assert "served-by-maritime" in body
    finally:
        await client.delete(session)


async def test_pause_on_exit_then_resume_keeps_the_workspace() -> None:
    client = MaritimeSandboxClient()
    session = await client.create(options=MaritimeSandboxClientOptions(pause_on_exit=True, **OPTIONS))
    try:
        await session.start()
        await session.exec("echo kept > note.txt")
        await client.delete(session)
        assert client.maritime.sandboxes.get(session.state.sandbox_id).state == "paused"

        state = client.deserialize_session_state(client.serialize_session_state(session.state))
        resumed = await client.resume(state)
        await resumed.start()
        assert resumed.state.sandbox_id == session.state.sandbox_id
        result = await resumed.exec("cat note.txt")
        assert result.stdout.decode().strip() == "kept"
    finally:
        client.maritime.sandboxes.get(session.state.sandbox_id).kill()


async def test_runner_end_to_end_with_a_scripted_model() -> None:
    client = MaritimeSandboxClient()
    model = ScriptedModel(
        [
            [
                function_call(
                    "exec_command", {"cmd": "cat notes.txt && python3 -c 'print(6*7)'"}, call_id="c1"
                )
            ],
            [assistant_message("done")],
        ]
    )
    agent = SandboxAgent(
        name="live",
        model=model,
        capabilities=[Shell(), Filesystem()],
        default_manifest=Manifest(entries={"notes.txt": File(content=b"from the manifest\n")}),
    )
    result = await Runner.run(
        agent,
        "go",
        run_config=RunConfig(
            tracing_disabled=True,
            sandbox=SandboxRunConfig(client=client, options=MaritimeSandboxClientOptions(**OPTIONS)),
        ),
    )
    model.assert_complete()
    outputs = " ".join(str(i.output) for i in result.new_items if i.type == "tool_call_output_item")
    assert "from the manifest" in outputs and "42" in outputs
    alive = client.maritime.sandboxes.list(state=["running", "paused"], metadata={"test": "live"})
    assert alive.data == [], "the runner left a sandbox alive"
