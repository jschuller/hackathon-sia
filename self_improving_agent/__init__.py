"""Self-Improving IT Incident Agent — ADK Package.

ADK finds the agent through ``root_agent`` here. It is resolved lazily, so
``import self_improving_agent.tools`` (the eval, the dashboard) does not build
the agents or their MCP toolsets.
"""

from typing import Any

__all__ = ["root_agent"]


def __getattr__(name: str) -> Any:
    if name == "root_agent":
        from .agent import root_agent

        return root_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
