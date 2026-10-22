#!/usr/bin/env bash
# Run the agent with Datadog LLM Observability instrumentation
set -euo pipefail

# Run from the repo root, where .env and the agent package live
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "run.sh: no .env in $(pwd)." >&2
  echo "  Create it from the template and add your API keys:" >&2
  echo "    cp .env.template .env && nano .env" >&2
  exit 1
fi

# Load environment
set -a; source .env; set +a

echo "Starting Self-Improving Agent with Datadog instrumentation..."
echo "  Model: ${ADK_MODEL:-gemini-2.5-flash}"
echo "  DD ML App: ${DD_LLMOBS_ML_APP:-self-improving-agent}"

# ddtrace-run auto-instruments all Gemini calls, ADK decisions, MCP tool calls
uv run ddtrace-run adk web . --port "${PORT:-8000}"
