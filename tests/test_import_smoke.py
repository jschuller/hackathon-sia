"""Importing the package works with no environment at all and has no visible side effects.

Each case runs in a fresh interpreter with an empty environment (no API keys,
no MCP settings) and a temp working directory, so no .env can be read.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _run(code: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    prelude = f"import sys; sys.path.insert(0, {str(REPO)!r})\n"
    return subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)],
        cwd=cwd,
        env={},  # no keys, no MCP settings
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_importing_tools_does_not_build_the_agent(tmp_path: Path) -> None:
    result = _run(
        """
        import sys
        import self_improving_agent.tools
        assert "self_improving_agent.agent" not in sys.modules
        assert "self_improving_agent.mcp_servers" not in sys.modules
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_root_agent_imports_with_no_env_and_prints_nothing(tmp_path: Path) -> None:
    result = _run(
        """
        from google.adk.agents import BaseAgent, LlmAgent
        from google.adk.tools.mcp_tool import McpToolset
        from self_improving_agent import root_agent, agent

        assert isinstance(root_agent, BaseAgent)
        assert root_agent.name == "self_healing_incident_agent"
        for llm in (agent.triage_agent, agent.resolution_agent, agent.narrator_agent):
            assert not [t for t in llm.tools if isinstance(t, McpToolset)]
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""


def test_adk_web_dot_finds_the_agent(tmp_path: Path) -> None:
    # What `adk web .` does from the repo root: treat it as a folder of agents.
    result = _run(
        f"""
        import os
        os.environ["ADK_DISABLE_LOAD_DOTENV"] = "1"  # never read a real .env
        from google.adk.cli.utils.agent_loader import AgentLoader
        loader = AgentLoader({str(REPO)!r})
        assert loader.load_agent("self_improving_agent").name == "self_healing_incident_agent"
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
