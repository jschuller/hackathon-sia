#!/usr/bin/env bash
# Run the agent with Datadog LLM Observability instrumentation
set -euo pipefail

# Load environment
set -a; source .env 2>/dev/null; set +a

echo "Starting Self-Improving Agent with Datadog instrumentation..."
echo "  Model: ${ADK_MODEL:-gemini-2.5-flash}"
echo "  DD ML App: ${DD_LLMOBS_ML_APP:-self-improving-agent}"

# ddtrace-run auto-instruments all Gemini calls, ADK decisions, MCP tool calls
uv run ddtrace-run adk web self_improving_agent --port "${PORT:-8000}"
