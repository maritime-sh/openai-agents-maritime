<p align="center">
  <a href="https://maritime.sh">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/maritime-mark-white.png">
      <img src="assets/maritime-mark-black.png" width="64" alt="Maritime">
    </picture>
  </a>
</p>

<h1 align="center">openai-agents-maritime</h1>

<p align="center"><a href="https://pypi.org/project/openai-agents-maritime/"><img src="https://img.shields.io/pypi/v/openai-agents-maritime?style=flat-square&color=2563eb" alt="PyPI"></a> <a href="https://pypi.org/project/openai-agents-maritime/"><img src="https://img.shields.io/pypi/pyversions/openai-agents-maritime?style=flat-square" alt="Python versions"></a> <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-0a0a0a?style=flat-square" alt="MIT license"></a> <a href="https://github.com/maritime-sh/openai-agents-maritime/actions/workflows/test.yml"><img src="https://img.shields.io/github/actions/workflow/status/maritime-sh/openai-agents-maritime/test.yml?branch=main&style=flat-square&label=tests" alt="Tests"></a> <a href="https://maritime.sh/docs"><img src="https://img.shields.io/badge/docs-maritime.sh-0a0a0a?style=flat-square" alt="Docs"></a> <a href="https://discord.gg/wvPAvXNZqm"><img src="https://img.shields.io/badge/Discord-join-5865F2?style=flat-square&logo=discord&logoColor=white" alt="Discord"></a></p>

<p align="center">
  <b>Give every <a href="https://github.com/openai/openai-agents-python">OpenAI Agents SDK</a> sandbox agent its own cloud computer on <a href="https://maritime.sh">Maritime</a>.</b><br>
  A real Linux machine per run: the agent's shell, files and patches run there, never on your laptop or server.
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#options">Options</a> ·
  <a href="#keep-a-sandbox-between-runs">Resume</a> ·
  <a href="#what-is-supported">What is supported</a> ·
  <a href="https://github.com/maritime-sh/agents">More agents on Maritime</a>
</p>

## Quickstart

```bash
pip install openai-agents-maritime
export MARITIME_API_KEY=mk_...     # maritime.sh/settings/api-keys
export OPENAI_API_KEY=sk-...
```

```python
import asyncio

from agents import RunConfig, Runner
from agents.sandbox import SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Filesystem, Shell

from openai_agents_maritime import MaritimeSandboxClient, MaritimeSandboxClientOptions

agent = SandboxAgent(
    name="Coder",
    model="gpt-5.4",
    instructions="You work inside a Linux sandbox. Use the shell and files to do the task.",
    capabilities=[Shell(), Filesystem()],
)


async def main() -> None:
    result = await Runner.run(
        agent,
        "Write fizzbuzz.py for 1 to 15, run it, and tell me the last three lines.",
        run_config=RunConfig(
            sandbox=SandboxRunConfig(
                client=MaritimeSandboxClient(),
                options=MaritimeSandboxClientOptions(),
            )
        ),
    )
    print(result.final_output)


asyncio.run(main())
```

When the run starts, Maritime boots a fresh sandbox in a few seconds. The agent writes and runs code there, and the sandbox ends when the run ends.

Two runnable examples are in [`examples/`](examples):

- [`scripted_demo.py`](examples/scripted_demo.py) runs a real sandbox with the SDK's `ScriptedModel`, so it needs only `MARITIME_API_KEY`.
- [`llm_agent.py`](examples/llm_agent.py) is the coding agent above.

No OpenAI account? [Maritime LLM](https://maritime.sh/docs/llm) is OpenAI-compatible: set `OPENAI_BASE_URL=https://api.maritime.sh/api/llm/v1` and use your Maritime LLM token as `OPENAI_API_KEY`.

## What the agent gets

- **Its own Linux machine.** Ubuntu 24.04 with Python 3.12, Node, git and curl, 2 vCPUs, 4 GiB of memory and 12 GiB of disk by default, up to 16 vCPUs and 64 GiB.
- **Real isolation.** Each sandbox is a microVM with its own kernel, filesystem, network and processes.
- **Root when it needs it.** Commands run as `user`, with passwordless `sudo` for installs.
- **Your files on arrival.** Manifest `File` and `Dir` entries are written into the workspace before the first turn.
- **Private web previews.** A port in `exposed_ports` gets an HTTPS URL that carries an access token, so a dev server the agent starts is reachable by you and nobody else.
- **Limits that clean up after you.** An idle timeout pauses or ends a forgotten sandbox, and `time_limit_seconds` ends it no matter what.

## Options

`MaritimeSandboxClientOptions` rejects unknown fields, so a typo never silently drops a setting.

| Option | Default | What it does |
| --- | --- | --- |
| `template` | `base` | A saved snapshot, a template, or a public image to start from. |
| `size` | `small` | A preset: `small`, `medium`, `large` or `xlarge`. |
| `vcpu`, `memory_mib`, `disk_gib` | from `size` | Override the preset: 1 to 16 vCPUs, 1,024 to 65,536 MiB, 5 to 500 GiB. |
| `env` | `{}` | Environment variables for every command. |
| `metadata` | `{}` | Labels you can filter on when you list sandboxes. |
| `name` | none | A unique name. Creating a name that already runs returns that sandbox. |
| `end_user` | none | Your id for the person this sandbox works for. |
| `idle_timeout_seconds` | `600` | How long without activity before `on_idle` applies. |
| `on_idle` | `pause` | `pause` keeps the disk and resumes on the next call. `kill` ends the sandbox. |
| `time_limit_seconds` | none | End the sandbox this long after it starts, from 60 to 86,400. |
| `pause_on_exit` | `False` | Pause instead of ending the sandbox when the run's session closes. |
| `exposed_ports` | `()` | Ports the agent may expose over HTTPS. |
| `exec_timeout_unbounded_s` | `3600` | The timeout for commands the SDK runs without one. |

The workspace is `/home/user/workspace` unless your manifest sets `root`. A root outside the home directory, such as the SDK's default `/workspace`, is created with `sudo` and handed to `user`.

## Keep a sandbox between runs

With `pause_on_exit=True` the sandbox pauses when the run ends instead of ending. Save the run state, and the next run reconnects to the same machine with everything the agent installed still there. If the sandbox has ended since, a new one is created with the same settings and the SDK restores the workspace from its snapshot.

```python
options = MaritimeSandboxClientOptions(pause_on_exit=True, idle_timeout_seconds=1800)
```

You can also own the session yourself and use it across several runs:

```python
client = MaritimeSandboxClient()
session = await client.create(options=MaritimeSandboxClientOptions())
async with session:
    await Runner.run(agent, "Clone the repo and install it.", run_config=RunConfig(sandbox=SandboxRunConfig(session=session)))
    await Runner.run(agent, "Now run the tests.", run_config=RunConfig(sandbox=SandboxRunConfig(session=session)))
await client.delete(session)
```

For anything this adapter does not wrap, `session._inner.sandbox` is the [Maritime Python SDK](https://pypi.org/project/maritime/) `Sandbox`.

## What is supported

| Feature | Status |
| --- | --- |
| `Shell`, `Filesystem` and `apply_patch` tools | Supported |
| Manifest `File` and `Dir` entries, environment | Supported |
| Workspace snapshots and resume | Supported |
| Pause on exit, reconnect on resume | Supported |
| Exposed ports | Supported, private HTTPS URL with a token |
| Running commands as another user | Supported, through `sudo -u` |
| Interactive PTY sessions (`write_stdin`) | Not yet: commands run to completion |
| Manifest mounts (S3, GCS, R2, Azure) | Not yet: rejected before a sandbox is created |

`env` is saved in the run state so an ended sandbox can be recreated with the same settings. Keep secrets you would not store with your run state out of it.

## Compatibility

| Package | Versions |
| --- | --- |
| Python | 3.10 to 3.13 |
| `openai-agents` | 0.24.x |
| `maritime` | 0.11.x |

The SDK's sandbox interfaces are new and change between minor versions, so this package pins the range it is tested on.

## Develop

```bash
uv sync --group dev
uv run pytest                    # unit tests, no account needed
MARITIME_API_KEY=mk_... uv run pytest -m live   # real sandboxes on your account
```

The unit tests swap only the network calls: commands really run and files really move, so the SDK's path checks, tar snapshots and patching are exercised for real. The live tests create real sandboxes, check commands, files, timeouts, snapshots, an HTTPS port, pause and resume, and confirm every sandbox ends.

## License

[MIT](LICENSE), from [Maritime](https://maritime.sh).
