"""Run root_agent end to end with scripted models (no network, no keys)."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import Field

from self_improving_agent import agent as sia


def text(value: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=value)]))


def call(name: str, **args: Any) -> LlmResponse:
    part = types.Part(function_call=types.FunctionCall(name=name, args=args))
    return LlmResponse(content=types.Content(role="model", parts=[part]))


class ScriptedLlm(BaseLlm):
    """Returns the scripted responses in order; fails if called once too often."""

    script: list[LlmResponse] = Field(default_factory=list)
    calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        if not self.script:
            raise AssertionError(f"{self.model} was called more often than scripted")
        self.calls += 1
        yield self.script.pop(0)


def approve() -> list[LlmResponse]:
    return [
        call("store_experience", category="cpu", resolution="restart", score=0.9),
        call("exit_loop"),
        text("Overall score: 0.90.\nVERDICT: APPROVED"),
    ]


async def _run_many(incidents: list[str]) -> list[dict[str, Any]]:
    """Runs incidents one after another in one session; returns state after each."""
    service = InMemorySessionService()
    session = await service.create_session(app_name="sia-test", user_id="u")
    runner = Runner(agent=sia.root_agent, app_name="sia-test", session_service=service)
    states = []
    for incident in incidents:
        message = types.Content(role="user", parts=[types.Part(text=incident)])
        async for _ in runner.run_async(user_id="u", session_id=session.id, new_message=message):
            pass
        done = await service.get_session(app_name="sia-test", user_id="u", session_id=session.id)
        states.append(dict(done.state))
    return states


async def _run(incident: str) -> dict[str, Any]:
    return (await _run_many([incident]))[0]


@pytest.fixture
def models(monkeypatch) -> dict[str, ScriptedLlm]:
    fakes: dict[str, ScriptedLlm] = {}
    for name in ("triage_agent", "resolution_agent", "critic_agent", "refiner_agent"):
        fakes[name] = ScriptedLlm(model=name)
        monkeypatch.setattr(getattr(sia, name), "model", fakes[name])
    fakes["triage_agent"].script = [text("P2 cpu incident on web-prod-03")]
    fakes["resolution_agent"].script = [text("APPROVED PLAN: roll back v2.4.1")]
    return fakes


def test_approved_resolution_is_not_overwritten(models: dict[str, ScriptedLlm]) -> None:
    models["critic_agent"].script = approve()

    state = asyncio.run(_run("CPU at 98% on web-prod-03"))

    assert state["resolution_proposal"] == "APPROVED PLAN: roll back v2.4.1"
    assert models["refiner_agent"].calls == 0
    assert "quality_achieved" not in state


def test_refined_resolution_is_kept_once_approved(models: dict[str, ScriptedLlm]) -> None:
    models["critic_agent"].script = [text("Overall 0.60. Add rollback steps.")] + approve()
    models["refiner_agent"].script = [text("REFINED PLAN: roll back v2.4.1, then verify")]

    state = asyncio.run(_run("CPU at 98% on web-prod-03"))

    assert state["resolution_proposal"] == "REFINED PLAN: roll back v2.4.1, then verify"
    assert models["refiner_agent"].calls == 1
    assert models["critic_agent"].calls == 4


def test_text_only_approval_still_ends_the_loop(models: dict[str, ScriptedLlm]) -> None:
    # The critic approves in text but never calls exit_loop.
    models["critic_agent"].script = [text("Overall score: 0.90.\nVERDICT: APPROVED")]
    models["refiner_agent"].script = [text("UNREVIEWED DRAFT")] * 5

    state = asyncio.run(_run("CPU at 98% on web-prod-03"))

    assert state["resolution_proposal"] == "APPROVED PLAN: roll back v2.4.1"
    assert models["refiner_agent"].calls == 0  # gate stops the loop before it


def test_text_only_approval_after_a_refinement(models: dict[str, ScriptedLlm]) -> None:
    models["critic_agent"].script = [
        text("**Overall score**: 0.60. Add rollback steps.\nVERDICT: REVISE"),
        text("Overall: 0.88\nVERDICT: APPROVED"),
    ]
    models["refiner_agent"].script = [text("REFINED PLAN: roll back v2.4.1, then verify")]

    state = asyncio.run(_run("CPU at 98% on web-prod-03"))

    assert state["resolution_proposal"] == "REFINED PLAN: roll back v2.4.1, then verify"
    assert models["refiner_agent"].calls == 1


@pytest.mark.parametrize(
    ("evaluation", "approved"),
    [
        ("Overall score: 0.90.\nVERDICT: APPROVED", True),
        ("Scores...\n**VERDICT: APPROVED**", True),
        ("Overall score: 0.70.\nVERDICT: REVISE", False),
        ("VERDICT: APPROVED (first draft)\n...after rereading...\nVERDICT: REVISE", False),
        # Text without a verdict line is never an approval, whatever it says.
        ("**Overall score** = (0.90 + 0.80 + 0.50 + 0.60 + 0.70) / 5 = 0.70. Add rollback steps.", False),
        ("Overall, 1 critical gap remains: no rollback.", False),
        ("Overall score: 0.95. QUALITY_ACHIEVED", False),
        ("I would not give a VERDICT: APPROVED yet.", False),
        ("", False),
    ],
)
def test_critic_approved(evaluation: str, approved: bool) -> None:
    assert sia.critic_approved(evaluation) is approved


def test_stale_approval_from_an_earlier_incident_is_ignored(models: dict[str, ScriptedLlm]) -> None:
    # Incident 1 is approved. In incident 2 the critic first returns an empty reply,
    # which leaves output_key untouched; the gate must not reuse incident 1's verdict.
    models["triage_agent"].script = [text("P2 cpu incident"), text("P1 disk incident")]
    models["resolution_agent"].script = [text("PLAN A"), text("PLAN B")]
    models["critic_agent"].script = approve() + [LlmResponse(content=None)] + [
        text("Overall score: 0.50.\nVERDICT: REVISE")
    ] * 4
    models["refiner_agent"].script = [text(f"PLAN B v{i}") for i in range(2, 7)]

    states = asyncio.run(_run_many(["CPU at 98%", "Disk full on db-01"]))

    assert states[0]["resolution_proposal"] == "PLAN A"
    assert models["refiner_agent"].calls == 5  # every incident-2 round was refined
    assert states[1]["resolution_proposal"] == "PLAN B v6"


def test_refiner_cannot_exit_the_loop() -> None:
    refiner_tools = {getattr(t, "__name__", None) for t in sia.refiner_agent.tools}
    critic_tools = {getattr(t, "__name__", None) for t in sia.critic_agent.tools}

    assert "exit_loop" not in refiner_tools
    assert "exit_loop" in critic_tools
