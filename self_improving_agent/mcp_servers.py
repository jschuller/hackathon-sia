"""
MCP Server connection factories for ADK McpToolset.
Each function returns an McpToolset ready to plug into an agent's tools list.

All STDIO servers use uvx/npx — portable across Mac and Linux (ParrotOS).
"""

import os
import shutil

from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import (
    StdioConnectionParams,
    SseConnectionParams,
)
from mcp import StdioServerParameters


def _find_bin(name: str) -> str:
    """Find a binary on PATH, raising if missing."""
    path = shutil.which(name)
    if not path:
        raise FileNotFoundError(f"{name} not found on PATH")
    return path


# ── STDIO servers (local processes via uvx/npx) ─────────────

def elevenlabs_mcp() -> McpToolset:
    """ElevenLabs MCP — TTS, STT, voice cloning."""
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=_find_bin("uvx"),
                args=["elevenlabs-mcp"],
                env={
                    "ELEVENLABS_API_KEY": os.environ.get("ELEVENLABS_API_KEY", ""),
                    "ELEVENLABS_MCP_OUTPUT_MODE": "both",
                },
            ),
            timeout=30,
        ),
    )


def servicenow_mcp() -> McpToolset:
    """ServiceNow MCP — jschuller/mcp-server-servicenow (18 tools)."""
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=_find_bin("mcp-servicenow"),
                args=[],
                env={
                    "SERVICENOW_INSTANCE_URL": os.environ.get(
                        "SERVICENOW_INSTANCE_URL", ""
                    ),
                    "SERVICENOW_USERNAME": os.environ.get("SERVICENOW_USERNAME", ""),
                    "SERVICENOW_PASSWORD": os.environ.get("SERVICENOW_PASSWORD", ""),
                    "SERVICENOW_AUTH_TYPE": os.environ.get("SERVICENOW_AUTH_TYPE", "basic"),
                },
            ),
            timeout=30,
        ),
    )


def postman_mcp_local() -> McpToolset:
    """Postman MCP (local STDIO — full 100+ tools)."""
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=_find_bin("npx"),
                args=["-y", "@postman/postman-mcp-server", "--full"],
                env={
                    "POSTMAN_API_KEY": os.environ.get("POSTMAN_API_KEY", ""),
                },
            ),
            timeout=30,
        ),
    )


def perplexity_mcp() -> McpToolset:
    """Perplexity MCP — web search."""
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=_find_bin("uvx"),
                args=["perplexity-mcp"],
                env={
                    "PERPLEXITY_API_KEY": os.environ.get("PERPLEXITY_API_KEY", ""),
                },
            ),
            timeout=30,
        ),
    )


# ── SSE / Streamable HTTP servers (remote — zero install) ────

def postman_mcp_remote(mode: str = "minimal") -> McpToolset:
    """Postman MCP (remote — zero local setup).
    mode: 'minimal', 'mcp' (full), or 'code'
    """
    url_map = {
        "minimal": "https://mcp.postman.com/minimal",
        "mcp": "https://mcp.postman.com/mcp",
        "full": "https://mcp.postman.com/mcp",
        "code": "https://mcp.postman.com/code",
    }
    return McpToolset(
        connection_params=SseConnectionParams(
            url=url_map.get(mode, url_map["minimal"]),
            headers={
                "Authorization": f"Bearer {os.environ.get('POSTMAN_API_KEY', '')}",
            },
        ),
    )


def braintrust_mcp() -> McpToolset:
    """Braintrust MCP — query eval experiments and scores."""
    return McpToolset(
        connection_params=SseConnectionParams(
            url="https://api.braintrust.dev/mcp",
            headers={
                "Authorization": f"Bearer {os.environ.get('BRAINTRUST_API_KEY', '')}",
            },
        ),
    )


def datadog_mcp() -> McpToolset:
    """Datadog MCP — logs, metrics, traces (96 tools)."""
    return McpToolset(
        connection_params=SseConnectionParams(
            url="https://mcp.datadoghq.com/sse",
            headers={
                "DD-API-KEY": os.environ.get("DD_API_KEY", ""),
                "DD-APPLICATION-KEY": os.environ.get("DD_APP_KEY", ""),
            },
        ),
    )
