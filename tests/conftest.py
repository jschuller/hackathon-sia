"""Keep tests offline: never read .env and never start an MCP server."""

import os

import dotenv
import pytest

# agent.py calls load_dotenv() on import; the repo's .env holds real keys.
dotenv.load_dotenv = lambda *args, **kwargs: False

# Without these, agent.py attaches no MCP toolsets at import.
for _key in ("SERVICENOW_INSTANCE_URL", "ELEVENLABS_API_KEY", "PERPLEXITY_API_KEY"):
    os.environ.pop(_key, None)


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch):
    """Point the experience memory at a temp file."""
    from self_improving_agent import tools

    monkeypatch.setattr(tools, "_MEMORY_FILE", tmp_path / "experience_memory.json")
    monkeypatch.setattr(tools, "_EXPERIENCE_MEMORY", [])
