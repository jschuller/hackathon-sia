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
| ServiceNow | Incident/CMDB data | MCP (STDIO via uvx, pinned) |
| ElevenLabs | Voice narration | MCP (STDIO via uvx, pinned) |
| Datadog | LLM Observability | ddtrace auto-instrument |
| Braintrust | Eval scoring | Python SDK (`eval/braintrust_eval.py`) |
| Perplexity | Web search for triage and resolution | MCP (STDIO via npx, pinned) |

## Known limitations

- **Read-only tools.** Incident text, CMDB records and web results are
  untrusted input, so triage and resolution get only ServiceNow read tools
  (`SERVICENOW_READ_ONLY=true` plus a `tool_filter`) and Perplexity search.
  ElevenLabs is attached to the narrator alone, limited to `text_to_speech`.
  The agent proposes resolutions; it cannot create incidents, add work notes
  or change CIs.
- **The eval does not measure the agent.** `eval/braintrust_eval.py` calls
  Gemini directly with a hand-written prompt; it never runs `root_agent`, the
  critic loop or the MCP tools. Its five incidents are the same ones
  `scripts/seed_servicenow.py` seeds, and v2 only adds stored experiences to
  the prompt, so the v1/v2 before/after numbers are not evidence that the
  agent improves.

## License

MIT — Hackathon project
