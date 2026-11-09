"""A coding agent that works inside its own Maritime sandbox.

export MARITIME_API_KEY=...       # https://maritime.sh/dashboard/settings
export OPENAI_API_KEY=...         # an OpenAI key, or a Maritime LLM token
export OPENAI_BASE_URL=https://api.maritime.sh/api/llm/v1   # only for Maritime LLM
python examples/llm_agent.py
"""

import asyncio
import os

from agents import RunConfig, Runner
from agents.sandbox import SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Filesystem, Shell

from openai_agents_maritime import MaritimeSandboxClient, MaritimeSandboxClientOptions

TASK = (
    "Write fizzbuzz.py that prints FizzBuzz for 1 to 15, run it with python3, "
    "then tell me the last three lines it printed and the Linux kernel version of this machine."
)


async def main() -> None:
    agent = SandboxAgent(
        name="Maritime coder",
        model=os.environ.get("MODEL", "gpt-5.4"),
        instructions=(
            "You work inside a Linux sandbox. Use the shell and files to do the task, then answer briefly."
        ),
        capabilities=[Shell(), Filesystem()],
    )
    client = MaritimeSandboxClient()
    result = await Runner.run(
        agent,
        TASK,
        run_config=RunConfig(
            sandbox=SandboxRunConfig(
                client=client,
                options=MaritimeSandboxClientOptions(idle_timeout_seconds=300, on_idle="kill"),
            ),
        ),
    )
    for item in result.new_items:
        if item.type == "tool_call_item":
            raw = item.raw_item
            name = getattr(raw, "name", None) or (raw.get("name") if isinstance(raw, dict) else None)
            args = getattr(raw, "arguments", None) or (raw.get("arguments") if isinstance(raw, dict) else "")
            print(f"> {name} {args}")
    print("\n" + str(result.final_output))


if __name__ == "__main__":
    asyncio.run(main())
