"""
Self-Improving IT Incident Agent
=================================
Google ADK + MCP + Reflexion-style self-improvement loop.

Run:  uv run adk web self_improving_agent
With DD: uv run ddtrace-run adk web self_improving_agent
"""

import os
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.sequential_agent import SequentialAgent

from .tools import (
    store_experience,
    retrieve_experiences,
    get_improvement_stats,
    clear_experience_memory,
)

# ── Configuration ────────────────────────────────────────────
MODEL = os.environ.get("ADK_MODEL", "gemini-2.5-flash")


# ── MCP Toolsets (loaded conditionally based on .env) ────────
def _load_mcp_tools() -> list:
    """Return MCP toolsets for keys that are present in .env."""
    tools = []
    try:
        from .mcp_servers import servicenow_mcp
        if os.environ.get("SERVICENOW_INSTANCE_URL"):
            tools.append(servicenow_mcp())
            print("  ✓ ServiceNow MCP loaded")
    except Exception as e:
        print(f"  ⚠ ServiceNow MCP skipped: {e}")

    try:
        from .mcp_servers import elevenlabs_mcp
        if os.environ.get("ELEVENLABS_API_KEY"):
            tools.append(elevenlabs_mcp())
            print("  ✓ ElevenLabs MCP loaded")
    except Exception as e:
        print(f"  ⚠ ElevenLabs MCP skipped: {e}")

    try:
        from .mcp_servers import perplexity_mcp
        if os.environ.get("PERPLEXITY_API_KEY"):
            tools.append(perplexity_mcp())
            print("  ✓ Perplexity MCP loaded")
    except Exception as e:
        print(f"  ⚠ Perplexity MCP skipped: {e}")

    return tools


# ── Exit tool for the LoopAgent ──────────────────────────────
def exit_loop(tool_context) -> dict:
    """Signal that the resolution meets quality standards.
    Call this ONLY when the critic scores the resolution >= 0.85.
    """
    tool_context.actions.escalate = True
    return {"status": "quality_achieved", "action": "exiting_improvement_loop"}


# ── Load MCP tools once ──────────────────────────────────────
print("Loading MCP toolsets...")
_mcp_tools = _load_mcp_tools()
print(f"  Total MCP toolsets: {len(_mcp_tools)}")


# ══════════════════════════════════════════════════════════════
# AGENT 1: TRIAGE — classifies incidents
# ══════════════════════════════════════════════════════════════
triage_agent = LlmAgent(
    model=MODEL,
    name="triage_agent",
    description="Classifies incoming IT incidents by severity, category, and impact.",
    instruction="""You are a senior IT incident triage specialist with 28 years
of enterprise experience including ServiceNow CTA certification.

When you receive an incident report:
1. Call retrieve_experiences with the most likely category to check past patterns
2. Extract key symptoms: what is failing, severity indicators, affected systems
3. If ServiceNow MCP tools are available, query CMDB for affected CIs and
   search for similar past incidents
4. Classify the incident:
   - Priority: P1 (critical) / P2 (high) / P3 (medium) / P4 (low)
   - Category: cpu / memory / disk / network / ssl / application / database / security
   - Estimated blast radius: single server / service / region / global
5. Note any similar past experiences found and their resolutions

Output a structured triage report. Be decisive — this is production.""",
    tools=[retrieve_experiences] + _mcp_tools,
    output_key="triage_report",
)


# ══════════════════════════════════════════════════════════════
# AGENT 2: RESOLUTION — proposes fix steps
# ══════════════════════════════════════════════════════════════
resolution_agent = LlmAgent(
    model=MODEL,
    name="resolution_agent",
    description="Proposes concrete resolution steps for triaged IT incidents.",
    instruction="""You are an expert IT resolution engineer. You receive a triage
report and must propose concrete, actionable resolution steps.

Based on the triage report (available as session state key 'triage_report'):
1. Review the incident classification and affected systems
2. Call retrieve_experiences for the incident category — learn from past wins
3. If ServiceNow MCP tools are available, search knowledge base for runbooks
4. Propose a numbered, step-by-step resolution plan with specific commands
5. Include rollback steps in case the fix doesn't work
6. Estimate your confidence (0.0 to 1.0) in the proposed resolution

Your resolution should reference real systems, paths, and commands.
If past experiences exist, adapt and improve those proven patterns.
If confidence < 0.7, explicitly flag for human review.""",
    tools=[retrieve_experiences] + _mcp_tools,
    output_key="resolution_proposal",
)


# ══════════════════════════════════════════════════════════════
# AGENT 3: CRITIC — evaluates resolution quality
# ══════════════════════════════════════════════════════════════
critic_agent = LlmAgent(
    model=MODEL,
    name="critic_agent",
    description="Evaluates the quality of proposed incident resolutions.",
    instruction="""You are a senior IT operations reviewer. You evaluate resolution
proposals for quality and completeness. Be rigorous but fair.

Review the resolution proposal (session state key 'resolution_proposal') against
the triage report (session state key 'triage_report').

Score on these 5 dimensions (each 0.0 to 1.0):
1. **Completeness**: Does it address ALL symptoms from the triage report?
2. **Specificity**: Are steps concrete (real commands, paths, thresholds)?
3. **Safety**: Are there rollback steps and impact mitigation?
4. **Efficiency**: Is this the most direct path to resolution?
5. **Learning**: Does it incorporate patterns from past experiences?

Calculate overall score = average of all 5 dimensions.

If overall score >= 0.85:
- Say "QUALITY_ACHIEVED" clearly in your response
- Call store_experience with the category, a summary of the resolution, and the score
- Call get_improvement_stats to show the learning trajectory

If overall score < 0.85:
- Provide specific, actionable feedback for EACH low-scoring dimension
- Explain exactly what would raise the score
- Do NOT say "QUALITY_ACHIEVED"

Output your evaluation with all dimension scores and the overall score.""",
    tools=[store_experience, get_improvement_stats],
    output_key="evaluation_result",
)


# ══════════════════════════════════════════════════════════════
# AGENT 4: REFINER — improves based on critique
# ══════════════════════════════════════════════════════════════
refiner_agent = LlmAgent(
    model=MODEL,
    name="refiner_agent",
    description="Refines resolution proposals based on critic feedback.",
    instruction="""You are a resolution refinement specialist. You take critic
feedback and improve the resolution.

Read the evaluation result (session state key 'evaluation_result'):

If the evaluation says "QUALITY_ACHIEVED":
- Call the exit_loop tool immediately to end the improvement cycle
- The resolution is ready to deliver

If the evaluation identifies weaknesses:
- Address EACH specific piece of feedback from the critic
- Add missing details, commands, rollback steps, or safety measures
- Check retrieve_experiences for additional proven patterns
- Output the improved resolution — it will overwrite 'resolution_proposal'

Each iteration MUST show measurable improvement over the previous version.
Never repeat the same resolution without substantive changes.""",
    tools=[exit_loop, retrieve_experiences],
    output_key="resolution_proposal",
)


# ══════════════════════════════════════════════════════════════
# SELF-IMPROVEMENT LOOP (Reflexion pattern via LoopAgent)
# ══════════════════════════════════════════════════════════════
improvement_loop = LoopAgent(
    name="self_improvement_loop",
    description=(
        "Iteratively improves resolution quality through critique and "
        "refinement. Runs critic → refiner in a loop until quality >= 0.85 "
        "or max iterations reached."
    ),
    sub_agents=[critic_agent, refiner_agent],
    max_iterations=5,
)


# ══════════════════════════════════════════════════════════════
# ROOT AGENT — Full Pipeline (this is what ADK loads)
# ══════════════════════════════════════════════════════════════
root_agent = SequentialAgent(
    name="self_healing_incident_agent",
    description=(
        "Self-improving IT incident resolution agent. "
        "Triages → Resolves → Self-critiques → Refines → Learns. "
        "Gets measurably better with every interaction."
    ),
    sub_agents=[
        triage_agent,
        resolution_agent,
        improvement_loop,
    ],
)
