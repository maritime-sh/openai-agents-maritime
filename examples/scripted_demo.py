"""Run a SandboxAgent on a Maritime sandbox with a scripted model: no model key needed.

MARITIME_API_KEY=... python examples/scripted_demo.py
"""

import asyncio

from agents import RunConfig, Runner
from agents.sandbox import Manifest, SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Filesystem, Shell
from agents.sandbox.entries import File
from agents.testing import ScriptedModel, assistant_message, function_call

from openai_agents_maritime import MaritimeSandboxClient, MaritimeSandboxClientOptions


async def main() -> None:
    model = ScriptedModel(
        [
            [
                function_call(
                    "exec_command",
                    {"cmd": "cat notes.txt && uname -sm && python3 -c 'print(6 * 7)'"},
                    call_id="call-1",
                )
            ],
            [assistant_message("Done.")],
        ]
    )
    agent = SandboxAgent(
        name="Maritime demo",
        model=model,
        capabilities=[Shell(), Filesystem()],
        default_manifest=Manifest(entries={"notes.txt": File(content=b"hello from the manifest\n")}),
    )
    result = await Runner.run(
        agent,
        "Run the command.",
        run_config=RunConfig(
            tracing_disabled=True,
            sandbox=SandboxRunConfig(
                client=MaritimeSandboxClient(),
                options=MaritimeSandboxClientOptions(idle_timeout_seconds=300, on_idle="kill"),
            ),
        ),
    )
    model.assert_complete()
    outputs = [str(item.output) for item in result.new_items if item.type == "tool_call_output_item"]
    print("\n".join(outputs))
    if not any("hello from the manifest" in o and "42" in o for o in outputs):
        raise SystemExit("the command did not run in the sandbox")
    print("openai-agents-maritime works")


if __name__ == "__main__":
    asyncio.run(main())
