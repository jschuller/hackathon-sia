"""
Braintrust Evaluation — Measure Self-Improvement
=================================================
Runs incident scenarios through the agent and scores with Braintrust.
Run before and after experience accumulation to show improvement.

Usage:
  python -m eval.braintrust_eval
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from braintrust import Eval
from autoevals import Factuality

# ── Incident test scenarios with expected resolution quality ──
INCIDENT_DATASET = [
    {
        "input": (
            "High CPU utilization at 98% on web-prod-03. "
            "Deployment v2.4.1 rolled out 2 hours ago. "
            "Response times degraded from 200ms to 3s."
        ),
        "expected": (
            "Check recent deployment v2.4.1 for resource-intensive changes. "
            "Run top/htop to identify the offending process. "
            "Check auto-scaling status. If deployment-related, rollback with "
            "kubectl rollout undo. Restart service with graceful drain. "
            "Monitor for 15 minutes post-fix."
        ),
    },
    {
        "input": (
            "Memory leak detected in api-gateway-01. "
            "Heap at 94% and growing steadily. "
            "Connection pool exhaustion warnings in logs."
        ),
        "expected": (
            "Capture heap dump with jmap. Compare with baseline. "
            "Check connection pool configuration and recent code changes. "
            "Apply memory limit. Schedule restart during low-traffic window. "
            "Investigate connection pool leaks."
        ),
    },
    {
        "input": (
            "Disk space critical at 95% on db-replica-02. "
            "Log files consuming 40GB. Database temp files growing."
        ),
        "expected": (
            "Identify largest directories with du -sh. "
            "Rotate and compress old logs. Clear application caches. "
            "Check for orphaned temp files. "
            "Update logrotate to keep max 7 days. "
            "Set monitoring alert at 80% threshold."
        ),
    },
    {
        "input": (
            "Network latency spike to 500ms between app-tier and db-tier. "
            "Packet loss at 2%. Started after network maintenance window."
        ),
        "expected": (
            "Run traceroute and mtr between tiers. "
            "Check switch/router interface errors. "
            "Compare routing tables pre/post maintenance. "
            "Check for MTU mismatches. Engage network team if "
            "infrastructure change detected."
        ),
    },
    {
        "input": (
            "SSL certificate expiring in 48 hours for *.prod.company.com. "
            "Auto-renewal via Let's Encrypt failed. "
            "ACME challenge returning 404."
        ),
        "expected": (
            "Check certbot logs for ACME failure reason. "
            "Verify DNS records and web server .well-known path. "
            "Manually trigger renewal with certbot --dry-run. "
            "If blocked, use DNS challenge as fallback. "
            "Update monitoring to alert at 30 days."
        ),
    },
]


# ── Simulated agent runner (lightweight for eval) ────────────
async def run_agent_simple(incident: str) -> str:
    """Run a lightweight version of the agent for eval.
    Uses Gemini directly to simulate the ADK agent's output.
    """
    import google.generativeai as genai

    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""You are an IT incident resolution agent.
Given this incident, provide a concrete, step-by-step resolution plan
with specific commands and rollback procedures.

Incident: {incident}

Resolution:"""

    response = model.generate_content(prompt)
    return response.text


async def run_agent_with_experience(incident: str) -> str:
    """Run agent with experience memory context (simulates v2)."""
    import google.generativeai as genai
    from self_improving_agent.tools import retrieve_experiences

    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Guess category from incident text
    categories = ["cpu", "memory", "disk", "network", "ssl"]
    category = "general"
    for cat in categories:
        if cat in incident.lower():
            category = cat
            break

    past = retrieve_experiences(category)
    experience_ctx = ""
    if past["experiences"]:
        experience_ctx = "\n\nPast successful resolutions:\n"
        for exp in past["experiences"]:
            experience_ctx += f"- [Score {exp['score']}]: {exp['resolution'][:200]}\n"

    prompt = f"""You are a self-improving IT incident resolution agent.
You learn from past successful resolutions.
{experience_ctx}

Given this incident, provide a concrete, step-by-step resolution plan
with specific commands and rollback procedures.

Incident: {incident}

Resolution:"""

    response = model.generate_content(prompt)
    return response.text


# ── Braintrust eval runners ─────────────────────────────────
def run_eval_v1():
    """Baseline eval — no experience memory."""
    import asyncio

    Eval(
        "Self-Improving-Agent",
        experiment_name="v1-before-improvement",
        data=lambda: [
            {"input": d["input"], "expected": d["expected"]}
            for d in INCIDENT_DATASET
        ],
        task=lambda input: asyncio.run(run_agent_simple(input)),
        scores=[Factuality],
    )


def run_eval_v2():
    """Post-improvement eval — with experience memory."""
    import asyncio

    Eval(
        "Self-Improving-Agent",
        experiment_name="v2-after-improvement",
        data=lambda: [
            {"input": d["input"], "expected": d["expected"]}
            for d in INCIDENT_DATASET
        ],
        task=lambda input: asyncio.run(run_agent_with_experience(input)),
        scores=[Factuality],
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("v1", "both"):
        print("Running v1 eval (baseline)...")
        run_eval_v1()
    if mode in ("v2", "both"):
        print("Running v2 eval (with experience)...")
        run_eval_v2()
    if mode not in ("v1", "v2", "both"):
        print("Usage: python -m eval.braintrust_eval [v1|v2|both]")
