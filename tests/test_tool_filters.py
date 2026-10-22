"""Agents that read untrusted content get read-only MCP tools.

Builds the toolsets with dummy settings; McpToolset connects lazily, so no
MCP server starts.
"""

import importlib
from collections.abc import Iterator
from types import ModuleType

import pytest
from google.adk.tools.mcp_tool import McpToolset

from self_improving_agent import agent as sia
from self_improving_agent import mcp_servers

SERVICENOW_WRITE_TOOLS = {
    "create_record",
    "update_record",
    "delete_record",
    "create_ci",
    "update_ci",
    "create_update_set",
    "set_current_update_set",
}


def toolsets(llm_agent) -> list[McpToolset]:
    return [t for t in llm_agent.tools if isinstance(t, McpToolset)]


@pytest.fixture
def wired(monkeypatch) -> Iterator[ModuleType]:
    """agent.py re-imported as if every MCP key were set in .env."""
    monkeypatch.setattr(mcp_servers, "_find_bin", lambda name: f"/opt/bin/{name}")
    for key in ("SERVICENOW_INSTANCE_URL", "ELEVENLABS_API_KEY", "PERPLEXITY_API_KEY"):
        monkeypatch.setenv(key, "dummy")
    yield importlib.reload(sia)
    for key in ("SERVICENOW_INSTANCE_URL", "ELEVENLABS_API_KEY", "PERPLEXITY_API_KEY"):
        monkeypatch.delenv(key)
    importlib.reload(sia)


def test_servicenow_allowlist_has_no_write_tools() -> None:
    assert not SERVICENOW_WRITE_TOOLS & set(mcp_servers.SERVICENOW_READ_TOOLS)


def test_triage_and_resolution_get_only_read_tools(wired: ModuleType) -> None:
    for llm_agent in (wired.triage_agent, wired.resolution_agent):
        sets = toolsets(llm_agent)
        assert len(sets) == 2  # ServiceNow + Perplexity, no ElevenLabs
        allowed: set[str] = set()
        for toolset in sets:
            assert isinstance(toolset.tool_filter, list), "an unfiltered toolset exposes every tool"
            allowed |= set(toolset.tool_filter)
        assert allowed == set(mcp_servers.SERVICENOW_READ_TOOLS) | {"perplexity_search_web"}
        assert not allowed & SERVICENOW_WRITE_TOOLS


def test_servicenow_server_runs_read_only(wired: ModuleType) -> None:
    servicenow = [
        t for t in toolsets(wired.triage_agent) if "list_records" in (t.tool_filter or [])
    ]
    assert len(servicenow) == 1
    params = servicenow[0]._connection_params.server_params
    assert params.env["SERVICENOW_READ_ONLY"] == "true"
    assert mcp_servers.SERVICENOW_MCP_SOURCE in params.args


def test_only_the_narrator_can_speak(wired: ModuleType) -> None:
    assert [t.tool_filter for t in toolsets(wired.narrator_agent)] == [["text_to_speech"]]
    for llm_agent in (wired.critic_agent, wired.refiner_agent):
        assert toolsets(llm_agent) == []


def test_stdio_servers_are_pinned(wired: ModuleType) -> None:
    sets = toolsets(wired.triage_agent) + toolsets(wired.narrator_agent)
    args = [" ".join(t._connection_params.server_params.args) for t in sets]
    assert sorted(args) == sorted([
        f"--from {mcp_servers.SERVICENOW_MCP_SOURCE} mcp-server-servicenow",
        "-y @jschuller/perplexity-mcp@3.0.0",
        "elevenlabs-mcp==0.12.2",
    ])


def test_no_datadog_mcp_factory() -> None:
    # It sent the Datadog application key to a remote MCP that no agent used.
    assert not hasattr(mcp_servers, "datadog_mcp")
