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
        text("Overall 0.90. QUALITY_ACHIEVED"),
    ]


async def _run(incident: str) -> dict[str, Any]:
    service = InMemorySessionService()
    session = await service.create_session(app_name="sia-test", user_id="u")
    runner = Runner(agent=sia.root_agent, app_name="sia-test", session_service=service)
    message = types.Content(role="user", parts=[types.Part(text=incident)])
    async for _ in runner.run_async(user_id="u", session_id=session.id, new_message=message):
        pass
    done = await service.get_session(app_name="sia-test", user_id="u", session_id=session.id)
    return dict(done.state)


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
    models["critic_agent"].script = [text("Overall score: 0.90. QUALITY_ACHIEVED")]
    models["refiner_agent"].script = [text("UNREVIEWED DRAFT")] * 5

    state = asyncio.run(_run("CPU at 98% on web-prod-03"))

    assert state["resolution_proposal"] == "APPROVED PLAN: roll back v2.4.1"
    assert models["refiner_agent"].calls == 0  # gate stops the loop before it


def test_text_only_approval_after_a_refinement(models: dict[str, ScriptedLlm]) -> None:
    models["critic_agent"].script = [
        text("**Overall score**: 0.60. Add rollback steps."),
        text("Overall: 0.88 QUALITY_ACHIEVED"),
    ]
    models["refiner_agent"].script = [text("REFINED PLAN: roll back v2.4.1, then verify")]

    state = asyncio.run(_run("CPU at 98% on web-prod-03"))

    assert state["resolution_proposal"] == "REFINED PLAN: roll back v2.4.1, then verify"
    assert models["refiner_agent"].calls == 1


@pytest.mark.parametrize(
    ("evaluation", "approved"),
    [
        ("Overall score: 0.90. QUALITY_ACHIEVED", True),
        ("**Overall score** = 0.85", True),
        ("Overall: 0.84. Needs rollback steps.", False),
        ("Overall score 0.60 -- not QUALITY_ACHIEVED yet", False),
        ("Completeness 0.9 ... Overall score: 0.70 ... revised overall score: 0.91", True),
        ("QUALITY_ACHIEVED", True),
        ("Needs more detail.", False),
    ],
)
def test_critic_approved(evaluation: str, approved: bool) -> None:
    assert sia.critic_approved(evaluation) is approved


def test_refiner_cannot_exit_the_loop() -> None:
    refiner_tools = {getattr(t, "__name__", None) for t in sia.refiner_agent.tools}
    critic_tools = {getattr(t, "__name__", None) for t in sia.critic_agent.tools}

    assert "exit_loop" not in refiner_tools
    assert "exit_loop" in critic_tools
