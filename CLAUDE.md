# CLAUDE.md — Context for Claude Code

## Project
Self-Improving IT Incident Agent for Self-Improving Agents Hack NYC (Feb 21, 2026).

## Architecture
- **Framework**: Google ADK (Agent Development Kit) v1.25+
- **LLM**: Gemini 2.5 Flash via `google-adk`
- **Pattern**: Reflexion loop — Triage → Resolve → Critic → Refine → Learn
- **MCP Servers**: ServiceNow, ElevenLabs, Postman, Perplexity, Datadog, Braintrust

## Package Manager
Uses **uv** — NOT pip, NOT conda. All commands via `uv run`.
- Install deps: `uv sync`
- Run agent: `uv run adk web self_improving_agent`
- Run with DD: `uv run ddtrace-run adk web self_improving_agent`
- Run evals: `uv run python -m eval.braintrust_eval`

## Key Files
- `self_improving_agent/agent.py` — root_agent (SequentialAgent → LoopAgent)
- `self_improving_agent/mcp_servers.py` — MCP connection factories (uvx/npx)
- `self_improving_agent/tools.py` — experience memory (store/retrieve/stats)
- `eval/braintrust_eval.py` — before/after scoring with Braintrust
- `.env` — API keys (gitignored, copy from .env.template)

## Sponsor Tools (must use 3+ for judging)
1. **ServiceNow** — incident/CMDB data via MCP
2. **ElevenLabs** — voice narration of resolutions via MCP
3. **Datadog** — LLM Observability traces via ddtrace
4. **Braintrust** — eval scoring before/after improvement
5. **Postman** — API management + MCP server generation

## Judging Criteria
1. Self-improvement: Agent must get measurably better over time
2. Autonomy: Agent decides what to improve without human guidance
3. Tool depth: Each sponsor tool must be genuinely essential
4. Demo impact: 3-minute live demo showing improvement delta
