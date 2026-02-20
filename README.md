# 🔄 Self-Improving IT Incident Agent

> **Self-Improving Agents Hack NYC** — Feb 21, 2026

An autonomous IT incident resolution agent that **gets measurably better** with
every interaction. Built with Google ADK, Gemini 2.5 Flash, and MCP.

## Architecture

```
Incident → [Triage Agent] → [Resolution Agent] → [Critic ↔ Refiner Loop] → Resolution
                                                         ↑
                                                  Experience Memory
                                                  (learns from past)
```

**Reflexion pattern**: The critic evaluates the resolution on 5 dimensions.
If score < 0.85, the refiner improves it. Loop runs up to 5 iterations.
Successful resolutions are stored and retrieved for future incidents.

## Quick Start

```bash
# 1. Clone + bootstrap (ParrotOS / Debian / Ubuntu)
git clone git@github.com:jschuller/hackathon-sia.git
cd hackathon-sia
chmod +x bootstrap.sh && ./bootstrap.sh

# 2. Add your API keys
nano .env

# 3. Run the agent
uv run adk web self_improving_agent

# 4. Run with Datadog observability
bash scripts/run.sh

# 5. Run evals
uv run python -m eval.braintrust_eval
```

## macOS Quick Start

```bash
uv sync                                    # install deps
cp .env.template .env && nano .env         # add keys
uv run adk web self_improving_agent        # start
```

## Sponsor Tools

| Sponsor | Role | Integration |
|---------|------|-------------|
| ServiceNow | Incident/CMDB data | MCP (STDIO via uvx) |
| ElevenLabs | Voice narration | MCP (STDIO via uvx) |
| Datadog | LLM Observability | ddtrace auto-instrument |
| Braintrust | Eval scoring | Python SDK + MCP |
| Postman | API management | MCP (remote SSE + local npx) |

## License

MIT — Hackathon project
