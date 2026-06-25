import pytest
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import root_agent


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)


@pytest.fixture
def agent():
    return root_agent


def test_low_decibel_pre_filtering(agent) -> None:
    """Verifies that low-decibel signals (<50dB) bypass MCP and LLM entirely and log directly."""
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(user_id="test_user", app_name="test")
    runner = Runner(agent=agent, session_service=session_service, app_name="test")

    # A low decibel bird chirping sound (42dB)
    message = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text='{"sensor_id": "sensor-01", "location": "Sector 3B", "decibel_level": 42.0, "acoustic_signature": "Chirping Bird", "timestamp": "2026-06-22T08:00:00Z"}'
            )
        ],
    )

    events = list(
        runner.run(user_id="test_user", session_id=session.id, new_message=message)
    )

    # Verify the final event is logged with no threat assessment needed
    assert len(events) > 0
    final_output = events[-1].output
    assert final_output is not None
    assert final_output["status"] == "LOGGED"
    assert final_output["sensor_data"]["decibel_level"] == 42.0


def test_mcp_patrol_pre_filtering(agent) -> None:
    """Verifies that Elk Prairie events match ranger schedules via mock MCP and bypass LLM threat evaluation."""
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(user_id="test_user", app_name="test")
    runner = Runner(agent=agent, session_service=session_service, app_name="test")

    # A high decibel footsteps sound in Elk Prairie where patrol is scheduled
    message = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text='{"sensor_id": "sensor-02", "location": "Elk Prairie", "decibel_level": 68.0, "acoustic_signature": "Footsteps", "timestamp": "2026-06-22T08:00:00Z"}'
            )
        ],
    )

    events = list(
        runner.run(user_id="test_user", session_id=session.id, new_message=message)
    )

    assert len(events) > 0
    final_output = events[-1].output
    assert final_output is not None
    assert final_output["status"] == "LOGGED"
    assert final_output["sensor_data"]["location"] == "Elk Prairie"


def test_high_threat_hitl_escalation_and_resume(agent) -> None:
    """Verifies that a high threat anomaly triggers an interrupt, pauses the workflow, and resumes upon Ranger decision."""
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(user_id="test_user", app_name="test")
    runner = Runner(agent=agent, session_service=session_service, app_name="test")

    # Unscheduled sector high-threat event (Chainsaw in Sector 9, 95.5dB)
    message = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text='{"sensor_id": "sensor-03", "location": "Sector 9", "decibel_level": 95.5, "acoustic_signature": "Chainsaw Revving", "timestamp": "2026-06-22T08:00:00Z"}'
            )
        ],
    )

    events = list(
        runner.run(user_id="test_user", session_id=session.id, new_message=message)
    )

    # Search for the adk_request_input interrupt
    interrupted = False
    interrupt_id = None
    for e in events:
        if e.content and e.content.parts:
            for part in e.content.parts:
                if (
                    part.function_call
                    and part.function_call.name == "adk_request_input"
                ):
                    interrupted = True
                    interrupt_id = part.function_call.id
                    break

    assert interrupted, (
        "Workflow should have paused with an interrupt for a high threat level event"
    )
    assert interrupt_id is not None

    # Resume: Simulating Ranger approving drone deployment
    resume_message = types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=interrupt_id, response={"response": "approve"}
                )
            )
        ],
    )

    resume_events = list(
        runner.run(
            user_id="test_user", session_id=session.id, new_message=resume_message
        )
    )

    assert len(resume_events) > 0
    final_output = resume_events[-1].output
    assert final_output is not None
    assert final_output["status"] == "DRONE_DISPATCHED"
    assert final_output["sensor_data"]["acoustic_signature"] == "Chainsaw Revving"
