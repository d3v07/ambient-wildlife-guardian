"""
🌿 Ambient Wildlife Poaching Guardian - Unified Dashboard Server
Handles FastAPI endpoints to trigger and resolve ADK 2.0 workflow sessions.
Hosts a glassmorphic dark-theme monitoring UI on local port 8080.
"""

import asyncio
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel
import os

from app.agent import root_agent
from app.db import (
    init_db,
    add_incident,
    update_incident,
    get_incidents,
    get_pending_incidents,
    add_chat,
    get_chats,
    add_report,
    get_reports,
)

# Initialize logging for dashboard event tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ambient Wildlife Guardian Server")

# Security configurations
API_KEY = os.environ.get("GUARDIAN_API_KEY", "conservation-secure-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key

# Custom Security & CSRF Protection Middleware
@app.middleware("http")
async def add_security_headers_and_csrf_protection(request, call_next):
    # Verify CSRF for POST, PUT, DELETE requests
    if request.method in ["POST", "PUT", "DELETE"]:
        origin = request.headers.get("origin")
        referer = request.headers.get("referer")
        allowed = ["http://localhost:8080", "http://127.0.0.1:8080"]
        if origin and not any(origin.startswith(a) for a in allowed):
            return HTMLResponse(content="CSRF Attack Detected", status_code=403)
        if not origin and referer and not any(referer.startswith(a) for a in allowed):
            return HTMLResponse(content="CSRF Attack Detected", status_code=403)

    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "script-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self';"
    )
    return response

# Enable Cross-Origin Resource Sharing for API client testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory services mapping to ADK session components
session_service = InMemorySessionService()
runner = Runner(agent=root_agent, session_service=session_service, app_name="app")

# Keep empty caches for legacy references if any
incident_logs: list[dict[str, Any]] = []
active_sessions: dict[str, dict[str, Any]] = {}
chat_messages: list[dict[str, Any]] = []
manual_reports: list[dict[str, Any]] = []


@app.on_event("startup")
def startup_event():
    """Initializes SQLite database schemas and seeds default chats on system boot."""
    init_db()
    if not get_chats():
        add_chat(
            "Ranger Dave",
            "Cleared a fallen redwood branch blocking Howland Hill Road.",
            "2026-06-22T19:40:00Z"
        )
        add_chat(
            "Ranger Sarah",
            "Fern Canyon breeding corridor visual check: Quiet. Drones locked to 120m AGL.",
            "2026-06-22T19:45:00Z"
        )
        add_chat(
            "System",
            "Acoustic sensor grid active. 9 stations streaming telemetry.",
            "2026-06-22T19:50:00Z"
        )


class SensorInput(BaseModel):
    """Payload schema representing real-time telemetry events sent to API."""

    sensor_id: str
    location: str
    decibel_level: float
    acoustic_signature: str
    timestamp: str
    user_id: str | None = "ranger_station"
    resilience_mode: str | None = "primary"


class ResolveInput(BaseModel):
    """Payload schema representing Ranger decisions passed to resolve a paused session."""

    session_id: str
    interrupt_id: str
    decision: str
    user_id: str | None = "ranger_station"


class ChatMessage(BaseModel):
    sender: str
    message: str
    timestamp: str


class ManualReportInput(BaseModel):
    report_type: str
    location: str
    x: float
    y: float
    details: str


@app.post("/api/trigger")
async def trigger_workflow(payload: SensorInput, api_key: str = Depends(verify_api_key)):
    """
    Simulates Pub/Sub event ingestion.
    Creates a new stateful session, runs the ADK graph, and inspects the
    event stream for interrupts (adk_request_input) or final outputs.
    """
    try:
        user_id = payload.user_id or "ranger_station"
        # Create a new local ADK session
        session = session_service.create_session_sync(user_id=user_id, app_name="app")
        session_id = session.id

        # Build GenAI compatible message content
        message = types.Content(
            role="user", parts=[types.Part.from_text(text=payload.model_dump_json())]
        )

        # Execute the ADK graph workflow in a non-blocking background thread
        events = await asyncio.to_thread(
            lambda: list(runner.run(user_id=user_id, session_id=session_id, new_message=message))
        )

        # Scan event outcomes
        interrupted = False
        interrupt_id = None
        interrupt_message = ""

        for e in events:
            # Check if any event yielded an interrupt block
            if e.content and e.content.parts:
                for part in e.content.parts:
                    if (
                        part.function_call
                        and part.function_call.name == "adk_request_input"
                    ):
                        interrupted = True
                        interrupt_id = part.function_call.id
                        interrupt_message = part.function_call.args.get(
                            "message", "Awaiting decision..."
                        )
            if e.output:
                pass

        # Load the updated session state from the service registry
        adk_session = session_service.get_session_sync(
            app_name="app", user_id=user_id, session_id=session_id
        )

        state_vars = adk_session.state
        assessment = state_vars.get("assessment", {})
        threat_level = assessment.get("threat_level", 0)
        confidence_score = assessment.get("confidence_score", 0.0)
        top_evidence = assessment.get("top_evidence", [])
        recommended_action = assessment.get("recommended_action", "None")
        explanation = assessment.get("explanation", "Auto-logged.")

        policy = state_vars.get("policy", {})
        warnings = policy.get("warnings", [])
        weather = state_vars.get("weather", {})
        human_presence = state_vars.get("human_presence", 3)

        # Simulate resilience data constraints
        resilience_mode = payload.resilience_mode or "primary"
        if resilience_mode == "satlink":
            explanation = f"[Spectrograph Vector Data via SAT-LINK] {explanation}"
            top_evidence = list(top_evidence)
            top_evidence.append("SAT-LINK spectacles spectrograph vector telemetry signature matched.")
        elif resilience_mode == "lora":
            explanation = f"[Low-Bandwidth LoRa Text Alert] {explanation}"
            top_evidence = list(top_evidence)
            top_evidence.append("LoRa 32-byte packet transmission received (No audio payload).")

        log_entry = {
            "session_id": session_id,
            "sensor_id": payload.sensor_id,
            "location": payload.location,
            "decibel_level": payload.decibel_level,
            "acoustic_signature": payload.acoustic_signature,
            "timestamp": payload.timestamp,
            "status": "PENDING_DECISION" if interrupted else "LOGGED",
            "threat_level": threat_level,
            "confidence_score": confidence_score,
            "top_evidence": top_evidence,
            "recommended_action": recommended_action,
            "explanation": explanation,
            "warnings": warnings,
            "weather": weather,
            "human_presence": human_presence,
            "interrupted": interrupted,
            "interrupt_id": interrupt_id,
            "interrupt_message": interrupt_message,
            "final_outcome": final_outcome_label(
                threat_level, interrupted, recommended_action
            ),
            "resilience_mode": resilience_mode,
        }

        # Persist incident results to SQLite database
        add_incident(log_entry)
        if interrupted:
            add_chat(
                "System Alert",
                f"CRITICAL ALERT in {payload.location}: {payload.acoustic_signature} ({payload.decibel_level}dB). Awaiting dispatch approval.",
                payload.timestamp
            )
        else:
            add_chat(
                "System",
                f"Auto-logged low-threat sound ({payload.acoustic_signature}) at {payload.location} ({payload.decibel_level}dB). No escalation required.",
                payload.timestamp
            )

        return log_entry

    except Exception as e:
        logger.exception("Error running workflow")
        raise HTTPException(status_code=500, detail=str(e)) from e


def final_outcome_label(
    threat_level: int, interrupted: bool, recommended_action: str = "None"
) -> str:
    if interrupted:
        return f"Awaiting Confirmation ({recommended_action})"
    return "Logged: Wildlife Activity" if threat_level < 70 else "Unescalated Threat"


async def _resolve_workflow_impl(payload: ResolveInput):
    """Core logic to resume a paused ADK graph session."""
    session_id = payload.session_id
    
    # Check if session is active in database
    pending_alerts = get_pending_incidents()
    matching_incident = next((i for i in pending_alerts if i["session_id"] == session_id), None)
    if not matching_incident:
        raise HTTPException(status_code=404, detail="Active session not found")

    user_id = payload.user_id or "ranger_station"
    try:
        # Wrap Ranger decision as a mock function call response
        message = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=payload.interrupt_id, response={"response": payload.decision}
                    )
                )
            ],
        )

        # Resume the ADK session in a non-blocking background thread
        events = await asyncio.to_thread(
            lambda: list(runner.run(user_id=user_id, session_id=session_id, new_message=message))
        )

        # Load finalized session
        adk_session = session_service.get_session_sync(
            app_name="app", user_id=user_id, session_id=session_id
        )

        final_output = None
        for e in events:
            if e.output:
                final_output = e.output

        if not final_output:
            final_output = adk_session.state

        is_approved = payload.decision.lower() in ["y", "yes", "approve"]
        rec_action = matching_incident.get("recommended_action", "")

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

        status = "FALSE_ALARM"
        final_outcome = "Logged: False Alarm"

        if is_approved:
            if "Ranger" in rec_action:
                status = "RANGER_DISPATCHED"
                final_outcome = "Ranger Team Dispatched"
                add_chat("System", f"[Approved] Operator approved ranger patrol diversion to {matching_incident['location']}.", timestamp)
                add_chat("Ranger Elena", f"Roger that, diversion acknowledged. Diverting Ranger Team Alpha to {matching_incident['location']}. ETA 8 minutes.", timestamp)
            else:
                status = "DRONE_DISPATCHED"
                final_outcome = "Drone Deployed to Zone"
                add_chat("System", f"[Approved] Operator approved drone launch to {matching_incident['location']}.", timestamp)
                add_chat("Drone Alpha", f"Quadcopter launched. En route to {matching_incident['location']}. Camera stream active. cruising speed: 60km/h.", timestamp)
        else:
            add_chat("System", f"✕ Operator dismissed {matching_incident['location']} alert as False Alarm.", timestamp)

        # Save resolved state in DB
        updates = {
            "status": status,
            "final_outcome": final_outcome,
            "interrupted": False
        }
        update_incident(session_id, updates)

        return {"status": "SUCCESS", "final_outcome": final_output}

    except Exception as e:
        logger.exception("Error resuming workflow")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/resolve")
async def resolve_workflow(payload: ResolveInput, api_key: str = Depends(verify_api_key)):
    """Resumes a paused ADK graph session from HTTP request."""
    return await _resolve_workflow_impl(payload)


@app.get("/api/logs")
async def get_logs():
    return get_incidents()


@app.get("/api/mcp/ranger_schedule")
async def get_ranger_schedule(location: str):
    """Simulates SMART Patrol Schedule Database lookup via REST API."""
    is_patrol_scheduled = "Elk Prairie" in location
    return {"location": location, "is_patrol_scheduled": is_patrol_scheduled}


@app.get("/api/mcp/human_presence")
async def get_human_presence(location: str):
    """Simulates WorldPop Demographic Score lookup via REST API."""
    human_presence = 3
    if "Orick HQ" in location:
        human_presence = 8
    elif "Howland Hill" in location:
        human_presence = 1
    return {"location": location, "human_presence": human_presence}


@app.get("/api/mcp/usgs_streamflow")
async def get_usgs_streamflow():
    """Returns cached or mock USGS streamflow values for dashboard display."""
    return {
        "redwood_creek": {
            "station_id": "11482500",
            "name": "Redwood Creek at Orick",
            "value_cfs": 42.0,
        },
        "klamath_river": {
            "station_id": "11530500",
            "name": "Klamath River near Klamath",
            "value_cfs": 780.0,
        },
    }


@app.get("/api/pending")
async def get_pending():
    return get_pending_incidents()


@app.get("/api/chats")
async def get_chats_endpoint():
    return get_chats()


@app.post("/api/chats")
async def send_chat(payload: ChatMessage):
    add_chat(payload.sender, payload.message, payload.timestamp)

    # Check if Operator sent an approval/rejection and there's a pending incident in DB
    pending_sessions = get_pending_incidents()
    if payload.sender.lower() in ["operator", "hq"] and len(pending_sessions) > 0:
        msg = payload.message.strip()
        msg_lower = msg.lower()
        
        # Check for prefixed chat command resolutions (e.g. /approve [session_id])
        target_session_id = None
        decision_type = None
        
        if msg_lower.startswith("/approve") or msg_lower.startswith("/yes"):
            decision_type = "approve"
            parts = msg.split()
            if len(parts) > 1:
                target_session_id = parts[1]
        elif msg_lower.startswith("/reject") or msg_lower.startswith("/no") or msg_lower.startswith("/false alarm"):
            decision_type = "reject"
            parts = msg.split()
            if len(parts) > 1:
                target_session_id = parts[1]
        elif msg_lower in ["approve", "deploy drone", "divert ranger", "y", "yes"]:
            decision_type = "approve"
        elif msg_lower in ["reject", "false alarm", "ignore", "n", "no"]:
            decision_type = "reject"
            
        if decision_type:
            # Find the target session
            sess = None
            if target_session_id:
                sess = next((s for s in pending_sessions if s["session_id"] == target_session_id), None)
                if not sess:
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
                    add_chat("System", f"✕ Error: Pending session '{target_session_id}' not found.", timestamp)
                    return payload.model_dump()
            else:
                sess = pending_sessions[0]
                
            if sess:
                await _resolve_workflow_impl(
                    ResolveInput(
                        session_id=sess["session_id"],
                        interrupt_id=sess["interrupt_id"],
                        decision=decision_type,
                        user_id="ranger_station",
                    )
                )
    return payload.model_dump()


@app.get("/api/reports")
async def get_reports_endpoint():
    return get_reports()


@app.post("/api/reports")
async def post_report(payload: ManualReportInput):
    add_report(payload.report_type, payload.location, payload.x, payload.y, payload.details)

    # Broadcast report to ranger chat as well
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    msg_text = f"[Map Report] Logged new {payload.report_type.replace('_', ' ')} near {payload.location}. Details: {payload.details}"
    add_chat("Ranger System", msg_text, timestamp)
    return payload.model_dump()


@app.post("/api/chats/simulate")
async def simulate_chat():
    import datetime
    import random

    names = ["Ranger Dave", "Ranger Sarah", "Ranger Elena", "Ranger Marcus"]
    actions = [
        "Roosevelt Elk herd spotted near Elk Prairie. Counting 14 individuals.",
        "River level at Redwood Creek crossing seems stable. Wade patrol is safe.",
        "Checking Stout Grove boundary fences. Integrity ok.",
        "Hikers advised to stick to the designated trail on Lady Bird Grove.",
        "Active patrol route optimized. Commencing sweeps.",
        "Re-orienting directional acoustic sensor-aud-03.",
    ]
    name = random.choice(names)
    msg = random.choice(actions)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    add_chat(name, msg, timestamp)
    return {"sender": name, "message": msg, "timestamp": timestamp}


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return html_content


# --- PREMIUM DASHBOARD HTML/CSS FRONTEND ---
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Redwood Guardian — Conservation Command Center</title>
    <meta name="description" content="Real-time anti-poaching command system for Redwood National & State Parks, powered by ADK 2.0 multi-agent AI.">
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0c0e0d;
            --bg-card: rgba(18, 22, 20, 0.72);
            --bg-card-hover: rgba(22, 28, 24, 0.85);
            --border-glass: rgba(74, 120, 86, 0.18);
            --border-hover: rgba(74, 120, 86, 0.35);
            --primary: #5d9b6b;
            --primary-bright: #7bc48a;
            --primary-glow: rgba(93, 155, 107, 0.20);
            --accent-redwood: #8b5a3c;
            --accent-bark: #6b4226;
            --accent-moss: #3d6b4f;
            --info: #5b9ea6;
            --info-glow: rgba(91, 158, 166, 0.15);
            --alert: #c85a4a;
            --alert-glow: rgba(200, 90, 74, 0.20);
            --warning: #c9953a;
            --warning-glow: rgba(201, 149, 58, 0.15);
            --water: #4a8fa8;
            --water-glow: rgba(74, 143, 168, 0.25);
            --text-main: #e8ede9;
            --text-secondary: #b0bfb5;
            --text-muted: #6a7d70;
            --font-sans: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'JetBrains Mono', 'SF Mono', monospace;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: var(--font-sans);
            background-color: var(--bg-base);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            position: relative;
        }

        /* Atmospheric background layers */
        body::before {
            content: '';
            position: fixed;
            inset: 0;
            background:
                radial-gradient(ellipse at 20% 80%, rgba(59, 90, 50, 0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 20%, rgba(74, 143, 168, 0.04) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(139, 90, 60, 0.03) 0%, transparent 70%);
            pointer-events: none;
            z-index: 0;
        }

        /* Subtle fog drift animation */
        body::after {
            content: '';
            position: fixed;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle at 30% 40%, rgba(180, 200, 185, 0.015) 0%, transparent 40%);
            animation: fog-drift 40s ease-in-out infinite alternate;
            pointer-events: none;
            z-index: 0;
        }

        @keyframes fog-drift {
            0% { transform: translate(0, 0); }
            100% { transform: translate(5%, 3%); }
        }

        header {
            padding: 14px 28px;
            border-bottom: 1px solid var(--border-glass);
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(12, 14, 13, 0.92);
            backdrop-filter: blur(24px) saturate(1.2);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .header-logo {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            background: linear-gradient(135deg, var(--accent-bark) 0%, var(--accent-redwood) 50%, var(--primary) 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 8px rgba(139, 90, 60, 0.3);
        }

        header h1 {
            font-size: 1rem;
            font-weight: 700;
            color: var(--text-main);
            letter-spacing: 0.5px;
        }

        .header-meta {
            display: flex;
            gap: 8px;
            margin-top: 2px;
        }

        .header-tag {
            font-family: var(--font-mono);
            font-size: 0.6rem;
            color: var(--text-muted);
            border: 1px solid var(--border-glass);
            padding: 1px 6px;
            border-radius: 4px;
            background: rgba(0, 0, 0, 0.2);
            letter-spacing: 0.3px;
        }

        .header-right {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.7rem;
            font-weight: 600;
            color: var(--text-muted);
            font-family: var(--font-mono);
        }

        .status-dot {
            width: 6px;
            height: 6px;
            background-color: var(--primary-bright);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--primary-glow);
            animation: pulse-dot 3s ease-in-out infinite;
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; box-shadow: 0 0 4px var(--primary-glow); }
            50% { opacity: 0.5; box-shadow: 0 0 12px var(--primary-glow); }
        }

        .link-toggle-group {
            display: flex;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-glass);
            border-radius: 6px;
            padding: 2px;
            gap: 1px;
        }

        .link-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 4px 10px;
            font-size: 0.62rem;
            font-weight: 700;
            cursor: pointer;
            border-radius: 4px;
            transition: all 0.2s;
            text-transform: uppercase;
            font-family: var(--font-mono);
            letter-spacing: 0.5px;
        }

        .link-btn.active {
            background: var(--primary);
            color: var(--bg-base);
        }

        /* MAIN LAYOUT */
        .container {
            display: grid;
            grid-template-columns: 300px 1fr 320px;
            gap: 16px;
            padding: 16px;
            max-width: 1720px;
            margin: 0 auto;
            width: 100%;
            flex: 1;
            position: relative;
            z-index: 1;
        }

        @media (max-width: 1300px) { .container { grid-template-columns: 300px 1fr; } .assets-panel { grid-column: span 2; } }
        @media (max-width: 950px) { .container { grid-template-columns: 1fr; } }

        .panel { display: flex; flex-direction: column; gap: 14px; }

        .card {
            background: var(--bg-card);
            backdrop-filter: blur(16px) saturate(1.1);
            border: 1px solid var(--border-glass);
            border-radius: 12px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            transition: border-color 0.3s, box-shadow 0.3s;
            position: relative;
            overflow: hidden;
        }

        .card:hover {
            border-color: var(--border-hover);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }

        h2 {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.8px;
            color: var(--text-secondary);
            display: flex;
            justify-content: space-between;
            align-items: center;
            text-transform: uppercase;
            padding-bottom: 6px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }

        h2 .tag {
            font-family: var(--font-mono);
            font-size: 0.55rem;
            color: var(--text-muted);
            font-weight: 600;
            opacity: 0.6;
        }

        .input-group { display: flex; flex-direction: column; gap: 3px; }

        label {
            font-size: 0.6rem;
            color: var(--text-muted);
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-family: var(--font-mono);
        }

        select, input {
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid var(--border-glass);
            border-radius: 6px;
            padding: 7px 10px;
            color: var(--text-main);
            font-family: var(--font-sans);
            font-size: 0.78rem;
            outline: none;
            transition: border-color 0.2s;
        }

        select:focus, input:focus { border-color: var(--primary); }

        .btn {
            background: linear-gradient(135deg, var(--accent-bark) 0%, var(--primary) 100%);
            color: var(--text-main);
            border: none;
            border-radius: 6px;
            padding: 9px;
            font-size: 0.78rem;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.15s, box-shadow 0.2s;
            box-shadow: 0 2px 10px rgba(93, 155, 107, 0.15);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-family: var(--font-mono);
            font-size: 0.7rem;
        }

        .btn:hover { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(93, 155, 107, 0.25); }
        .btn:active { transform: translateY(0); }

        /* RESILIENCE BANNER */
        .resilience-banner {
            background: rgba(91, 158, 166, 0.06);
            border: 1px solid rgba(91, 158, 166, 0.15);
            color: var(--info);
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 0.65rem;
            font-weight: 500;
            display: none;
            font-family: var(--font-mono);
        }

        /* TOPOGRAPHIC MAP */
        .map-header { display: flex; justify-content: space-between; align-items: center; width: 100%; }
        .layer-btn-group { display: flex; gap: 3px; }
        .layer-btn {
            padding: 3px 7px;
            font-size: 0.58rem;
            font-weight: 700;
            border-radius: 4px;
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid var(--border-glass);
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.2s;
            text-transform: uppercase;
            font-family: var(--font-mono);
            letter-spacing: 0.3px;
        }
        .layer-btn.active { background: var(--accent-moss); color: var(--text-main); border-color: var(--accent-moss); }

        .map-container {
            position: relative;
            border: 1px solid var(--border-glass);
            border-radius: 10px;
            height: 280px;
            overflow: hidden;
            background: radial-gradient(ellipse at 30% 60%, rgba(20, 35, 25, 0.9) 0%, rgba(12, 14, 13, 0.98) 100%);
        }

        .map-svg {
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
        }

        .map-sector-label {
            font-family: var(--font-mono);
            font-size: 7px;
            fill: var(--text-muted);
            cursor: crosshair;
            transition: fill 0.3s;
            font-weight: 600;
            letter-spacing: 0.3px;
        }
        .map-sector-label:hover { fill: var(--text-main); }
        .map-sector-label.alerting { fill: var(--alert); font-weight: 700; }

        .map-hq-label {
            font-family: var(--font-mono);
            font-size: 6px;
            fill: var(--info);
            font-weight: 700;
        }

        .map-patrol-dot {
            fill: var(--primary-bright);
            filter: drop-shadow(0 0 3px var(--primary-glow));
            animation: patrol-blink 2s ease-in-out infinite;
        }
        @keyframes patrol-blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        .radar-sweep {
            transform-origin: center center;
            animation: radar-spin 8s linear infinite;
        }
        @keyframes radar-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

        .alert-ring {
            fill: none;
            stroke: var(--alert);
            animation: ring-pulse 1.5s ease-out infinite;
        }
        @keyframes ring-pulse {
            0% { stroke-width: 1.5; opacity: 0.8; r: 8; }
            100% { stroke-width: 0.3; opacity: 0; r: 24; }
        }

        .drone-marker {
            position: absolute;
            z-index: 5;
            transition: all 2s cubic-bezier(0.25, 1, 0.5, 1);
            display: none;
            pointer-events: none;
        }
        .drone-marker.loitering { animation: drone-bob 1.5s ease-in-out infinite alternate; }
        @keyframes drone-bob {
            0% { transform: translateY(0); filter: drop-shadow(0 0 4px var(--info)); }
            100% { transform: translateY(-4px); filter: drop-shadow(0 0 10px var(--info)); }
        }

        /* HYDROLOGY PANEL */
        .hydro-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }

        .hydro-card {
            background: rgba(74, 143, 168, 0.06);
            border: 1px solid rgba(74, 143, 168, 0.15);
            border-radius: 8px;
            padding: 10px 12px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .hydro-label {
            font-family: var(--font-mono);
            font-size: 0.58rem;
            color: var(--info);
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .hydro-value {
            font-family: var(--font-mono);
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-main);
        }

        .hydro-unit {
            font-size: 0.6rem;
            color: var(--text-muted);
            font-weight: 400;
        }

        .hydro-station {
            font-family: var(--font-mono);
            font-size: 0.55rem;
            color: var(--text-muted);
        }

        /* CIRCULAR METRICS */
        .circular-metrics { display: flex; justify-content: space-around; align-items: center; gap: 8px; margin-top: 4px; }
        .gauge-container { position: relative; width: 72px; display: flex; flex-direction: column; align-items: center; }
        .gauge { width: 64px; height: 64px; }
        .gauge-bg { fill: none; stroke: rgba(255, 255, 255, 0.03); stroke-width: 3; }
        .gauge-fill { fill: none; stroke-width: 3; stroke-linecap: round; transition: stroke-dasharray 0.5s ease-in-out; }
        .gauge-fill.primary { stroke: var(--primary); filter: drop-shadow(0 0 3px var(--primary-glow)); }
        .gauge-fill.info { stroke: var(--info); filter: drop-shadow(0 0 3px var(--info-glow)); }
        .gauge-fill.warning { stroke: var(--warning); filter: drop-shadow(0 0 3px var(--warning-glow)); }
        .gauge-val { position: absolute; top: 20px; width: 100%; text-align: center; font-size: 0.75rem; font-weight: 700; font-family: var(--font-mono); color: var(--text-main); }
        .gauge-label { font-size: 0.52rem; color: var(--text-muted); font-weight: 700; text-transform: uppercase; margin-top: 6px; text-align: center; letter-spacing: 0.3px; font-family: var(--font-mono); }

        /* ASSETS */
        .asset-card {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 12px;
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.15);
            border: 1px solid var(--border-glass);
            font-size: 0.75rem;
        }
        .asset-info { display: flex; flex-direction: column; gap: 1px; }
        .asset-name { font-weight: 700; color: var(--text-main); font-size: 0.75rem; display: inline-flex; align-items: center; gap: 6px; }
        .asset-status { font-size: 0.62rem; color: var(--text-muted); font-family: var(--font-mono); }
        .battery-indicator { display: flex; align-items: center; gap: 5px; font-size: 0.65rem; font-weight: 700; color: var(--primary-bright); font-family: var(--font-mono); }
        .battery-bar { width: 28px; height: 7px; background: rgba(255, 255, 255, 0.05); border-radius: 2px; border: 1px solid rgba(255, 255, 255, 0.08); padding: 1px; display: flex; }
        .battery-fill { background-color: var(--primary-bright); height: 100%; border-radius: 1px; }
        .signal-level { font-size: 0.65rem; font-weight: 700; font-family: var(--font-mono); }

        /* TRUST CALIBRATION */
        .calibration-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px; }
        .calibration-cell { background: rgba(0, 0, 0, 0.15); border: 1px solid var(--border-glass); border-radius: 6px; padding: 5px; text-align: center; font-size: 0.58rem; color: var(--text-muted); font-family: var(--font-mono); }
        .calibration-score { font-size: 0.72rem; font-weight: 700; color: var(--primary-bright); margin-bottom: 1px; }

        /* ALERT CARD */
        .alert-card {
            border-color: var(--alert);
            background: rgba(200, 90, 74, 0.04);
            animation: alert-border-pulse 3s infinite alternate;
        }
        @keyframes alert-border-pulse {
            from { border-color: rgba(200, 90, 74, 0.25); }
            to { border-color: rgba(200, 90, 74, 0.6); }
        }
        .badge-critical {
            background: var(--alert);
            color: white;
            padding: 2px 7px;
            border-radius: 3px;
            font-size: 0.55rem;
            font-weight: 700;
            font-family: var(--font-mono);
            letter-spacing: 0.5px;
        }
        .alert-actions { display: flex; gap: 8px; margin-top: 4px; }
        .btn-approve { flex: 1; background: linear-gradient(135deg, var(--primary) 0%, var(--accent-moss) 100%); color: var(--text-main); }
        .btn-reject { flex: 1; background: linear-gradient(135deg, var(--alert) 0%, #8b3a30 100%); color: white; }

        /* INCIDENT LOGS */
        .log-table-container { border: 1px solid var(--border-glass); border-radius: 8px; overflow: hidden; background: rgba(0, 0, 0, 0.2); max-height: 160px; overflow-y: auto; }
        table { width: 100%; border-collapse: collapse; text-align: left; }
        th { background: rgba(255, 255, 255, 0.02); padding: 6px 10px; font-size: 0.58rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700; border-bottom: 1px solid var(--border-glass); letter-spacing: 0.3px; font-family: var(--font-mono); }
        td { padding: 6px 10px; border-bottom: 1px solid rgba(255, 255, 255, 0.02); font-size: 0.68rem; color: var(--text-main); }
        tr:last-child td { border-bottom: none; }

        .status-badge { display: inline-flex; align-items: center; gap: 3px; padding: 2px 5px; border-radius: 3px; font-size: 0.55rem; font-weight: 700; text-transform: uppercase; font-family: var(--font-mono); }
        .status-logged { background: rgba(91, 158, 166, 0.12); color: var(--info); border: 1px solid rgba(91, 158, 166, 0.25); }
        .status-dispatched { background: rgba(93, 155, 107, 0.12); color: var(--primary-bright); border: 1px solid rgba(93, 155, 107, 0.25); }
        .status-ranger-dispatched { background: rgba(201, 149, 58, 0.12); color: var(--warning); border: 1px solid rgba(201, 149, 58, 0.25); }
        .status-false { background: rgba(255, 255, 255, 0.04); color: var(--text-muted); border: 1px solid rgba(255, 255, 255, 0.08); }

        /* ADK TRACE */
        .trace-list { display: flex; flex-direction: column; gap: 4px; }
        .trace-step { display: flex; align-items: center; gap: 10px; padding: 6px 10px; border-radius: 6px; background: rgba(0, 0, 0, 0.1); border: 1px solid var(--border-glass); opacity: 0.4; transition: all 0.3s; }
        .trace-step.active { opacity: 1; border-color: var(--info); background: rgba(91, 158, 166, 0.05); }
        .trace-step.alert { opacity: 1; border-color: var(--alert); background: rgba(200, 90, 74, 0.05); }
        .trace-step.paused { opacity: 1; border-color: var(--warning); background: rgba(201, 149, 58, 0.04); animation: trace-blink 1.5s infinite alternate; }
        @keyframes trace-blink { from { border-color: rgba(201, 149, 58, 0.2); } to { border-color: rgba(201, 149, 58, 0.7); } }
        .trace-icon { display: flex; align-items: center; justify-content: center; width: 1.1rem; height: 1.1rem; }
        .trace-info { display: flex; flex-direction: column; gap: 1px; }
        .trace-name { font-size: 0.68rem; font-weight: 700; color: var(--text-main); }
        .trace-desc { font-size: 0.55rem; color: var(--text-muted); font-family: var(--font-mono); }

        /* MAP CONTEXT MENU & PINS */
        .map-context-menu {
            transition: opacity 0.2s ease;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(12px);
        }
        .menu-item-btn {
            width: 100%;
            text-align: left;
            padding: 5px 8px;
            font-size: 0.62rem;
            font-weight: 500;
            background: none;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            border-radius: 4px;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s;
        }
        .menu-item-btn:hover {
            background: rgba(93, 155, 107, 0.15);
            color: var(--primary-bright);
        }
        .manual-pin {
            filter: drop-shadow(0 0 3px rgba(0,0,0,0.8));
            animation: pin-entrance 0.3s ease-out;
        }
        @keyframes pin-entrance {
            from { transform: scale(0); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }

        /* RANGER CHAT */
        .chat-feed {
            scrollbar-width: thin;
        }
        .chat-msg {
            display: flex;
            flex-direction: column;
            padding: 6px 10px;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.01);
            border: 1px solid rgba(255, 255, 255, 0.03);
            margin-bottom: 2px;
            animation: message-entrance 0.25s ease-out;
        }
        @keyframes message-entrance {
            from { transform: translateY(5px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        .chat-msg.system {
            border-color: rgba(93, 155, 107, 0.2);
            background: rgba(93, 155, 107, 0.02);
        }
        .chat-msg.system-alert {
            border-color: rgba(200, 90, 74, 0.3);
            background: rgba(200, 90, 74, 0.04);
            box-shadow: inset 0 0 8px rgba(200, 90, 74, 0.05);
        }
        .chat-msg.operator {
            border-color: rgba(91, 158, 166, 0.25);
            background: rgba(91, 158, 166, 0.03);
        }
        .chat-header-line {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .chat-sender {
            font-weight: 700;
            font-size: 0.65rem;
            color: var(--primary-bright);
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .chat-msg.operator .chat-sender {
            color: var(--info);
        }
        .chat-msg.system-alert .chat-sender {
            color: var(--alert);
        }
        .chat-time {
            font-size: 0.52rem;
            color: var(--text-muted);
            font-family: var(--font-mono);
        }
        .chat-text {
            font-size: 0.65rem;
            color: var(--text-main);
            margin-top: 2px;
            line-height: 1.35;
        }

        /* SVG ICON BASE */
        .svg-icon { width: 1em; height: 1em; vertical-align: middle; display: inline-block; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; fill: none; }
        .icon-inline { margin-right: 5px; }

        /* SCROLLBARS */
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.05); }
        ::-webkit-scrollbar-thumb { background: var(--border-glass); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(93, 155, 107, 0.2); }

        /* FOOTER */
        .footer-bar {
            padding: 8px 28px;
            border-top: 1px solid var(--border-glass);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-family: var(--font-mono);
            font-size: 0.55rem;
            color: var(--text-muted);
            background: rgba(12, 14, 13, 0.85);
            position: relative;
            z-index: 1;
        }

        .btn-xs {
            padding: 3px 6px;
            font-size: 0.52rem;
            font-weight: 700;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
            font-family: var(--font-mono);
            text-transform: uppercase;
        }
        .btn-xs:hover {
            transform: translateY(-1px);
            filter: brightness(1.2);
        }
    </style>
</head>
<body>

    <header>
        <div>
            <div class="header-brand">
                <div class="header-logo">
                    <svg class="svg-icon" style="width:18px;height:18px;color:#e8ede9;" viewBox="0 0 24 24"><path d="M17 8C8 10 5.9 16.17 3.82 21.34l1.89.66.95-2.3c.48.17.98.3 1.34.3C19 20 22 3 22 3c-1 2-8 2.25-13 3.25S2 11.5 2 13.5s1.75 3.75 1.75 3.75"></path></svg>
                </div>
                <div>
                    <h1>Redwood Guardian</h1>
                    <div class="header-meta">
                        <span class="header-tag">CONSERVATION COP v3.0</span>
                        <span class="header-tag" id="clock-utc" style="color: var(--primary-bright);">--:--:-- UTC</span>
                    </div>
                </div>
            </div>
        </div>
        <div class="header-right">
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span>LINK ACTIVE</span>
            </div>
            <div class="audio-control-panel" style="display:flex; align-items:center; gap:8px; background:rgba(0,0,0,0.3); border:1px solid var(--border-glass); border-radius:6px; padding:2px 8px;">
                <button id="mute-btn" onclick="toggleMuteButton()" style="background:transparent; border:none; color:var(--primary); cursor:pointer; font-size:0.8rem; display:flex; align-items:center; outline:none;">🔊</button>
                <input type="range" id="volume-slider" min="0" max="1" step="0.1" value="1.0" oninput="handleVolumeSlider(this.value)" style="width:60px; height:4px; accent-color:var(--primary); background:rgba(255,255,255,0.1); border-radius:2px; cursor:pointer; outline:none; vertical-align:middle;">
            </div>
            <div class="link-toggle-group">
                <button class="link-btn active" onclick="setResilienceMode('primary')">PRIMARY</button>
                <button class="link-btn" onclick="setResilienceMode('satlink')">SAT-LINK</button>
                <button class="link-btn" onclick="setResilienceMode('lora')">LoRa</button>
                <input type="hidden" id="resilience-mode" value="primary">
            </div>
        </div>
    </header>

    <div class="container">
        <!-- Column 1: Controls & Telemetry -->
        <div class="panel controls-panel">
            <div class="card">
                <h2>Simulate Sensor Event <span class="tag">PUB/SUB</span></h2>
                <div class="input-group">
                    <label>Acoustic Template</label>
                    <select id="template-select" onchange="applyTemplate()">
                        <option value="bird">Chirping Bird — Low threat (Prairie Creek)</option>
                        <option value="chainsaw">Chainsaw Revving — 95.5dB (Howland Hill)</option>
                        <option value="gunshot">Gunshot Detected — 110dB (Gold Bluffs, high wind)</option>
                        <option value="ranger">Ranger Footsteps — MCP check (Elk Prairie)</option>
                    </select>
                </div>
                <div class="input-group">
                    <label>Sensor ID</label>
                    <input type="text" id="sensor_id" value="sensor-aud-88">
                </div>
                <div class="input-group">
                    <label>Location</label>
                    <input type="text" id="location" value="Howland Hill">
                </div>
                <div class="input-group">
                    <label>Decibel Level (dB)</label>
                    <input type="number" id="decibel_level" value="95.5">
                </div>
                <div class="input-group">
                    <label>Acoustic Signature</label>
                    <input type="text" id="acoustic_signature" value="Chainsaw Revving">
                </div>
                <button class="btn" onclick="triggerEvent()">
                    <svg class="svg-icon icon-inline" viewBox="0 0 24 24"><path d="M12 2v20M17 5v14M22 9v6M7 5v14M2 9v6"></path></svg>
                    Trigger Sensor Event
                </button>
            </div>

            <!-- Spectrogram -->
            <div class="card" id="spectrogram-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <h2>Bioacoustic Stream <span class="tag">LIVE</span></h2>
                    <label class="switch-container" style="display:flex; align-items:center; gap:6px; cursor:pointer;" onclick="toggleAutoTelemetry()">
                        <span class="switch-slider" id="auto-telemetry-slider" style="width:24px; height:12px; background:var(--primary-bright); border:1px solid var(--border-glass); border-radius:10px; position:relative; transition:all 0.3s; display:inline-block;">
                            <span class="switch-knob" id="auto-telemetry-knob" style="width:8px; height:8px; background:var(--bg-base); border-radius:50%; position:absolute; top:1px; left:13px; transition:all 0.3s;"></span>
                        </span>
                        <span style="font-size:0.52rem; font-weight:700; color:var(--primary-bright); text-transform:uppercase; letter-spacing:0.5px;" id="auto-telemetry-label">AUTO-STREAM</span>
                    </label>
                    <input type="checkbox" id="auto-telemetry-toggle" checked style="display:none;">
                </div>
                <div class="resilience-banner" id="resilience-banner-el"></div>
                <div id="spectrogram-dropzone" style="position:relative;">
                    <div id="drop-zone-overlay" style="display:none; position:absolute; inset:0; background:rgba(8, 22, 12, 0.9); border:2px dashed var(--primary-bright); border-radius:8px; justify-content:center; align-items:center; z-index:10; font-size:0.65rem; color:var(--primary-bright); font-weight:700;">
                        DROP AUDIO FILE HERE
                    </div>
                    <canvas id="spectrogram-canvas" style="width:100%;height:75px;background:rgba(10,16,12,0.6);border:1px solid var(--border-glass);border-radius:8px;"></canvas>
                </div>
                <div style="margin-top:4px; display:flex; flex-direction:column; gap:4px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:0.52rem; color:var(--text-muted);">Test with live audio stream or local uploads:</span>
                    </div>
                    <div style="display:flex; gap:4px; flex-wrap:wrap;">
                        <button onclick="playPresetSound('chainsaw')" class="btn-xs" style="background:rgba(239,68,68,0.12); border:1px solid rgba(239,68,68,0.25); color:#f87171;">🔊 SYNTH SAW</button>
                        <button onclick="playPresetSound('gunshot')" class="btn-xs" style="background:rgba(239,68,68,0.12); border:1px solid rgba(239,68,68,0.25); color:#f87171;">🔊 SYNTH SHOT</button>
                        <button onclick="playPresetSound('bird')" class="btn-xs" style="background:rgba(34,197,94,0.12); border:1px solid rgba(34,197,94,0.25); color:#4ade80;">🔊 SYNTH BIRD</button>
                        <button onclick="playPresetSound('ranger')" class="btn-xs" style="background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.25); color:#60a5fa;">🔊 SYNTH STEPS</button>
                        <button id="mic-btn" onclick="toggleMicrophone()" class="btn-xs" style="background:rgba(234,179,8,0.12); border:1px solid rgba(234,179,8,0.25); color:#facc15;">🎙️ STREAM MIC</button>
                        <button onclick="document.getElementById('audio-file-input').click()" class="btn-xs" style="background:rgba(255,255,255,0.06); border:1px solid var(--border-glass); color:var(--text-muted);">📁 UPLOAD</button>
                    </div>
                    <input type="file" id="audio-file-input" accept="audio/*" style="display:none;" onchange="handleAudioInputChange(event)">
                </div>
            </div>

            <!-- Quality Flywheel -->
            <div class="card">
                <h2>Triage Quality <span class="tag">FLYWHEEL</span></h2>
                <div class="circular-metrics">
                    <div class="gauge-container">
                        <svg class="gauge" viewBox="0 0 36 36"><path class="gauge-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" /><path id="gauge-quiet" class="gauge-fill primary" stroke-dasharray="0, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" /></svg>
                        <div class="gauge-val" id="metrics-quiet">0s</div>
                        <div class="gauge-label">Quiet</div>
                    </div>
                    <div class="gauge-container">
                        <svg class="gauge" viewBox="0 0 36 36"><path class="gauge-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" /><path id="gauge-streak" class="gauge-fill info" stroke-dasharray="0, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" /></svg>
                        <div class="gauge-val" id="metrics-streak">0</div>
                        <div class="gauge-label">Streak</div>
                    </div>
                    <div class="gauge-container">
                        <svg class="gauge" viewBox="0 0 36 36"><path class="gauge-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" /><path id="gauge-focus" class="gauge-fill warning" stroke-dasharray="100, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" /></svg>
                        <div class="gauge-val" id="metrics-focus">100%</div>
                        <div class="gauge-label">Focus</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Column 2: Map, Hydrology, Alerts, Agent Trace -->
        <div class="panel monitor-panel">
            <!-- ALERT CARD (HITL) -->
            <div class="card alert-card" id="alert-card" style="display:none;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <h2>Context Fusion Alert <span class="tag">HITL</span></h2>
                    <span class="badge-critical">PENDING CONFIRMATION</span>
                </div>
                <div id="alert-details" style="font-size:0.78rem;line-height:1.4;font-family:var(--font-mono);"></div>
                <div style="padding:8px 12px;border-radius:6px;background:rgba(200,90,74,0.03);border:1px solid rgba(200,90,74,0.12);">
                    <label style="color:var(--alert);font-weight:700;font-size:0.55rem;letter-spacing:0.3px;">
                        <svg class="svg-icon" viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0zM12 9v4M12 17h.01"></path></svg>
                        FUSION EVIDENCE
                    </label>
                    <ul id="alert-evidence-list" style="margin-left:14px;margin-top:3px;font-size:0.68rem;line-height:1.3;font-family:var(--font-mono);"></ul>
                    <div id="policy-warnings" style="margin-top:4px;font-size:0.65rem;font-weight:700;color:var(--warning);display:none;font-family:var(--font-mono);"></div>
                </div>
                <div class="alert-actions">
                    <button class="btn btn-approve" id="approve-dispatch-btn" onclick="resolveAlert('approve')">
                        <svg class="svg-icon icon-inline" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"></circle><path d="M12 12L7 7m10 10l-5-5m0 0l5-5m-5 5l-5 5"></path><circle cx="6" cy="6" r="2"></circle><circle cx="18" cy="18" r="2"></circle><circle cx="18" cy="6" r="2"></circle><circle cx="6" cy="18" r="2"></circle></svg>
                        Deploy Drone
                    </button>
                    <button class="btn btn-reject" onclick="resolveAlert('reject')">
                        <svg class="svg-icon icon-inline" viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                        False Alarm
                    </button>
                </div>
            </div>

            <!-- TOPOGRAPHIC MAP -->
            <div class="card" style="position:relative;">
                <div class="map-header">
                    <h2>Redwood NP Topographic Radar <span class="tag">LIVE</span></h2>
                    <div class="layer-btn-group">
                        <button class="layer-btn active" id="layer-boundary" onclick="toggleLayer('boundary')">Boundary</button>
                        <button class="layer-btn active" id="layer-patrol" onclick="toggleLayer('patrol')">Patrols</button>
                        <button class="layer-btn" id="layer-breeding" onclick="toggleLayer('breeding')">Breeding</button>
                        <button class="layer-btn" id="layer-firms" onclick="toggleLayer('firms')">FIRMS</button>
                    </div>
                </div>
                <div class="map-container" id="map-container">
                    <div id="map-click-menu" class="map-context-menu" style="display:none; position:absolute; z-index:200; background:rgba(12,18,14,0.95); border:1px solid var(--primary-bright); border-radius:6px; padding:6px; box-shadow:0 4px 12px rgba(0,0,0,0.5); font-family:var(--font-sans); width:160px;">
                        <div style="font-size:0.58rem; color:var(--text-muted); margin-bottom:4px; padding:2px 4px; border-bottom:1px solid rgba(255,255,255,0.05); font-weight:700;" id="menu-loc-title">LOCATION: Stout Grove</div>
                        <button class="menu-item-btn" onclick="submitMapReport('Roosevelt_Elk_Sighting')">Roosevelt Elk Sighting</button>
                        <button class="menu-item-btn" onclick="submitMapReport('Illegal_Snare_Trap')">Illegal Snare Trap</button>
                        <button class="menu-item-btn" onclick="submitMapReport('Unattended_Campfire')">Unattended Campfire</button>
                        <button class="menu-item-btn" onclick="submitMapReport('Install_Custom_Sensor')">Install Custom Sensor</button>
                        <button class="menu-item-btn" onclick="hideMapMenu()" style="border-top:1px solid rgba(255,255,255,0.05); color:var(--text-muted);">Cancel</button>
                    </div>
                    <div class="drone-marker" id="drone">
                        <svg style="width:20px;height:20px;color:var(--info);filter:drop-shadow(0 0 4px var(--info));" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M12 12L7 7m10 10l-5-5m0 0l5-5m-5 5l-5 5"></path><circle cx="6" cy="6" r="2"></circle><circle cx="18" cy="18" r="2"></circle><circle cx="18" cy="6" r="2"></circle><circle cx="6" cy="18" r="2"></circle></svg>
                    </div>
                    <svg class="map-svg" viewBox="0 0 600 280" preserveAspectRatio="xMidYMid meet" onclick="handleMapClick(event)">
                        <defs>
                            <linearGradient id="ocean-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" style="stop-color:rgba(40,75,100,0.25);"/>
                                <stop offset="100%" style="stop-color:rgba(40,75,100,0.02);"/>
                            </linearGradient>
                            <linearGradient id="river-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" style="stop-color:rgba(74,143,168,0.5);"/>
                                <stop offset="100%" style="stop-color:rgba(74,143,168,0.2);"/>
                            </linearGradient>
                            <filter id="glow">
                                <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                                <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
                            </filter>
                        </defs>

                        <!-- Pacific Ocean (left edge) -->
                        <path d="M 0,0 L 60,0 Q 55,40 50,80 Q 48,120 52,160 Q 55,200 58,240 L 65,280 L 0,280 Z" fill="url(#ocean-grad)" />
                        <text x="18" y="140" font-family="var(--font-mono)" font-size="7" fill="rgba(74,143,168,0.35)" transform="rotate(-90 18 140)" letter-spacing="3">PACIFIC OCEAN</text>

                        <!-- Coastline -->
                        <path d="M 60,0 Q 55,40 50,80 Q 48,120 52,160 Q 55,200 58,240 L 65,280" fill="none" stroke="rgba(74,143,168,0.3)" stroke-width="1.5" />

                        <!-- Klamath River (north, flowing east to west) -->
                        <path d="M 580,30 C 520,35 460,28 400,40 C 350,50 300,35 240,45 C 200,52 140,48 80,55 Q 65,58 55,62" fill="none" stroke="url(#river-grad)" stroke-width="3" filter="url(#glow)" />
                        <text x="420" y="26" font-family="var(--font-mono)" font-size="5.5" fill="rgba(74,143,168,0.45)" letter-spacing="1">KLAMATH RIVER</text>

                        <!-- Redwood Creek (south, winding) -->
                        <path d="M 420,280 C 400,250 380,230 350,210 C 320,190 300,175 280,165 C 260,155 230,160 200,170 C 170,180 140,200 110,220 Q 85,235 70,250" fill="none" stroke="url(#river-grad)" stroke-width="2.5" filter="url(#glow)" />
                        <text x="320" y="250" font-family="var(--font-mono)" font-size="5.5" fill="rgba(74,143,168,0.4)" letter-spacing="1">REDWOOD CREEK</text>

                        <!-- Highway 101 (dashed, north-south) -->
                        <path d="M 130,0 C 135,30 140,60 138,100 C 136,140 130,180 125,220 L 120,280" fill="none" stroke="rgba(201,149,58,0.15)" stroke-width="1.5" stroke-dasharray="6,4" />
                        <text x="145" y="135" font-family="var(--font-mono)" font-size="5" fill="rgba(201,149,58,0.3)" letter-spacing="0.5">HWY 101</text>

                        <!-- Topographic contour lines -->
                        <path d="M 180,25 Q 250,15 320,30 T 470,20" fill="none" stroke="rgba(93,155,107,0.06)" stroke-width="1" />
                        <path d="M 170,45 Q 240,35 310,50 T 460,40" fill="none" stroke="rgba(93,155,107,0.04)" stroke-width="0.8" />
                        <path d="M 200,100 Q 280,85 360,100 T 520,90" fill="none" stroke="rgba(93,155,107,0.06)" stroke-width="1" />
                        <path d="M 210,130 Q 300,115 380,130 T 540,120" fill="none" stroke="rgba(93,155,107,0.04)" stroke-width="0.8" />
                        <path d="M 190,180 Q 270,165 350,185 T 500,175" fill="none" stroke="rgba(93,155,107,0.05)" stroke-width="1" />

                        <!-- Bald Hills Ridge (east side) -->
                        <path d="M 480,180 C 510,160 530,130 550,90 C 560,60 570,40 580,10" fill="none" stroke="rgba(139,90,60,0.12)" stroke-width="2" />
                        <text x="545" y="115" font-family="var(--font-mono)" font-size="5" fill="rgba(139,90,60,0.25)" transform="rotate(70 545 115)" letter-spacing="1">BALD HILLS</text>

                        <!-- Radar sweep -->
                        <g class="radar-sweep" style="transform-origin:300px 140px;">
                            <path d="M 300,140 L 300,0" fill="none" stroke="rgba(93,155,107,0.08)" stroke-width="30">
                                <animateTransform attributeName="transform" type="rotate" from="0 300 140" to="360 300 140" dur="8s" repeatCount="indefinite"/>
                            </path>
                        </g>

                        <!-- Patrol route (dashed) -->
                        <path d="M 170,105 L 260,75 L 370,110 L 330,200 L 200,220 Z" fill="none" stroke="rgba(93,155,107,0.1)" stroke-width="1" stroke-dasharray="4,4" class="patrol-route" />

                        <!-- Named Location Markers -->
                        <!-- Stout Grove (far north) -->
                        <circle cx="350" cy="38" r="4" fill="none" stroke="var(--primary)" stroke-width="1.2" opacity="0.6" id="loc-stout" />
                        <text x="360" y="41" class="map-sector-label" id="label-stout">Stout Grove</text>

                        <!-- Howland Hill (north) -->
                        <circle cx="450" cy="50" r="4" fill="none" stroke="var(--primary)" stroke-width="1.2" opacity="0.6" id="loc-howland" />
                        <text x="460" y="53" class="map-sector-label" id="label-howland">Howland Hill</text>

                        <!-- Fern Canyon (mid-west) -->
                        <circle cx="150" cy="85" r="4" fill="none" stroke="var(--primary)" stroke-width="1.2" opacity="0.6" id="loc-fern" />
                        <text x="160" y="88" class="map-sector-label" id="label-fern">Fern Canyon</text>

                        <!-- Gold Bluffs (coastal) -->
                        <circle cx="90" cy="115" r="4" fill="none" stroke="var(--primary)" stroke-width="1.2" opacity="0.6" id="loc-gold" />
                        <text x="100" y="118" class="map-sector-label" id="label-gold">Gold Bluffs</text>

                        <!-- Elk Prairie (with patrol dot) -->
                        <circle cx="230" cy="105" r="4" fill="none" stroke="var(--primary)" stroke-width="1.2" opacity="0.6" id="loc-elk" />
                        <circle cx="230" cy="105" r="2" class="map-patrol-dot" />
                        <text x="240" y="108" class="map-sector-label" id="label-elk">Elk Prairie</text>
                        <text x="240" y="117" class="map-hq-label">▲ PATROL</text>

                        <!-- Prairie Creek (center) -->
                        <circle cx="300" cy="130" r="4" fill="none" stroke="var(--primary)" stroke-width="1.2" opacity="0.6" id="loc-prairie" />
                        <text x="310" y="133" class="map-sector-label" id="label-prairie">Prairie Creek</text>
                        <text x="310" y="142" class="map-hq-label">HQ</text>

                        <!-- Lady Bird Grove (south-central) -->
                        <circle cx="400" cy="170" r="4" fill="none" stroke="var(--primary)" stroke-width="1.2" opacity="0.6" id="loc-ladybird" />
                        <text x="410" y="173" class="map-sector-label" id="label-ladybird">Lady Bird Grove</text>

                        <!-- Tall Trees Grove (south-east) -->
                        <circle cx="340" cy="210" r="4" fill="none" stroke="var(--primary)" stroke-width="1.2" opacity="0.6" id="loc-talltrees" />
                        <text x="350" y="213" class="map-sector-label" id="label-talltrees">Tall Trees Grove</text>

                        <!-- Orick HQ (far south) -->
                        <circle cx="200" cy="245" r="5" fill="none" stroke="var(--info)" stroke-width="1.5" opacity="0.8" id="loc-orick" />
                        <circle cx="200" cy="245" r="2" fill="var(--info)" opacity="0.6" />
                        <text x="212" y="248" class="map-sector-label" id="label-orick" style="fill:var(--info);">Orick HQ</text>

                        <!-- Alert rings (hidden by default) -->
                        <circle id="alert-ring-0" cx="230" cy="105" r="10" class="alert-ring" style="display:none;" />
                        <circle id="alert-ring-1" cx="150" cy="85" r="10" class="alert-ring" style="display:none;" />
                        <circle id="alert-ring-2" cx="340" cy="210" r="10" class="alert-ring" style="display:none;" />
                        <circle id="alert-ring-3" cx="200" cy="245" r="10" class="alert-ring" style="display:none;" />
                        <circle id="alert-ring-4" cx="350" cy="38" r="10" class="alert-ring" style="display:none;" />
                        <circle id="alert-ring-5" cx="90" cy="115" r="10" class="alert-ring" style="display:none;" />
                        <circle id="alert-ring-6" cx="400" cy="170" r="10" class="alert-ring" style="display:none;" />
                        <circle id="alert-ring-7" cx="450" cy="50" r="10" class="alert-ring" style="display:none;" />
                        <circle id="alert-ring-8" cx="300" cy="130" r="10" class="alert-ring" style="display:none;" />

                        <!-- Drone flight path line -->
                        <line id="drone-path-line" x1="0" y1="0" x2="0" y2="0" stroke="rgba(91,158,166,0.4)" stroke-width="1.5" stroke-dasharray="4,4" style="display:none;" />

                        <!-- Manual report pins layer -->
                        <g id="manual-report-pins"></g>
                    </svg>
                </div>
            </div>

            <!-- USGS HYDROLOGY -->
            <div class="card">
                <h2>Hydrological Observations <span class="tag">USGS LIVE</span></h2>
                <div class="hydro-grid">
                    <div class="hydro-card">
                        <span class="hydro-label">Redwood Creek</span>
                        <div><span class="hydro-value" id="hydro-redwood">--</span> <span class="hydro-unit">cfs</span></div>
                        <span class="hydro-station">USGS 11482500 · Orick</span>
                    </div>
                    <div class="hydro-card">
                        <span class="hydro-label">Klamath River</span>
                        <div><span class="hydro-value" id="hydro-klamath">--</span> <span class="hydro-unit">cfs</span></div>
                        <span class="hydro-station">USGS 11530500 · Klamath</span>
                    </div>
                </div>
            </div>

            <!-- ADK Agent Trace -->
            <div class="card">
                <h2>ADK 2.0 Agent Trace <span class="tag">WORKFLOW</span></h2>
                <div class="trace-list">
                    <div class="trace-step" id="node-intake">
                        <div class="trace-icon"><svg class="svg-icon" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"></path></svg></div>
                        <div class="trace-info"><div class="trace-name">Sensor Intake</div><div class="trace-desc">Ingests raw telemetry</div></div>
                    </div>
                    <div class="trace-step" id="node-triage">
                        <div class="trace-icon"><svg class="svg-icon" viewBox="0 0 24 24"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon></svg></div>
                        <div class="trace-info"><div class="trace-name">Acoustic Triage</div><div class="trace-desc">Screens anomalies</div></div>
                    </div>
                    <div class="trace-step" id="node-fusion">
                        <div class="trace-icon"><svg class="svg-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10zM2 12h20"></path></svg></div>
                        <div class="trace-info"><div class="trace-name">Context Fusion (MCP)</div><div class="trace-desc">Weather + USGS + SMART</div></div>
                    </div>
                    <div class="trace-step" id="node-assessor">
                        <div class="trace-icon"><svg class="svg-icon" viewBox="0 0 24 24"><path d="M9.5 2a2.5 2.5 0 0 1 2 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1 0-3.12 3 3 0 0 1 0-3.88 2.5 2.5 0 0 1 0-3.12A2.5 2.5 0 0 1 9.5 2zM14.5 2a2.5 2.5 0 0 0 2 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 0-3.12 3 3 0 0 0 0-3.88 2.5 2.5 0 0 0 0-3.12A2.5 2.5 0 0 0 14.5 2z"></path></svg></div>
                        <div class="trace-info"><div class="trace-name">Threat Assessor</div><div class="trace-desc">AI evidence synthesis</div></div>
                    </div>
                    <div class="trace-step" id="node-policy">
                        <div class="trace-icon"><svg class="svg-icon" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg></div>
                        <div class="trace-info"><div class="trace-name">Policy & Safety</div><div class="trace-desc">Wind + USGS flow rules</div></div>
                    </div>
                    <div class="trace-step" id="node-dispatcher">
                        <div class="trace-icon"><svg class="svg-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"></circle><path d="M12 12L7 7m10 10l-5-5m0 0l5-5m-5 5l-5 5"></path><circle cx="6" cy="6" r="2"></circle><circle cx="18" cy="18" r="2"></circle><circle cx="18" cy="6" r="2"></circle><circle cx="6" cy="18" r="2"></circle></svg></div>
                        <div class="trace-info"><div class="trace-name">Dispatcher</div><div class="trace-desc">Fleet coordination</div></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Column 3: Assets, Calibration, Logs -->
        <div class="panel assets-panel">
            <div class="card">
                <h2>Mission Assets <span class="tag">FLEET</span></h2>
                <div style="display:flex;flex-direction:column;gap:8px;">
                    <div class="asset-card">
                        <div class="asset-info">
                            <div class="asset-name">
                                <svg class="svg-icon icon-inline" style="color:var(--info);" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"></circle><path d="M12 12L7 7m10 10l-5-5m0 0l5-5m-5 5l-5 5"></path><circle cx="6" cy="6" r="2"></circle><circle cx="18" cy="18" r="2"></circle><circle cx="18" cy="6" r="2"></circle><circle cx="6" cy="18" r="2"></circle></svg>
                                Drone Alpha
                            </div>
                            <div class="asset-status" id="drone-alpha-status">Standby · Prairie Creek HQ</div>
                        </div>
                        <div class="battery-indicator"><span>95%</span><div class="battery-bar"><div class="battery-fill" style="width:95%;"></div></div></div>
                    </div>
                    <div class="asset-card">
                        <div class="asset-info">
                            <div class="asset-name">
                                <svg class="svg-icon icon-inline" style="color:var(--primary-bright);" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                                Ranger Team Alpha
                            </div>
                            <div class="asset-status" id="ranger-alpha-status">Patrolling Elk Prairie</div>
                        </div>
                        <div class="signal-level" style="color:var(--primary-bright);display:inline-flex;align-items:center;gap:3px;">
                            <svg class="svg-icon" viewBox="0 0 24 24"><path d="M5 12.55a11 11 0 0 1 14.08 0M1.42 9a16 16 0 0 1 21.16 0M8.53 16.11a6 6 0 0 1 6.95 0M12 20h.01"></path></svg>
                            Good
                        </div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>Trust Calibration <span class="tag">BY SECTOR</span></h2>
                <div class="calibration-grid">
                    <div class="calibration-cell"><div class="calibration-score">98%</div>Elk Pr.</div>
                    <div class="calibration-cell"><div class="calibration-score">95%</div>Fern Cn.</div>
                    <div class="calibration-cell"><div class="calibration-score">92%</div>Tall Tr.</div>
                    <div class="calibration-cell"><div class="calibration-score">89%</div>Orick</div>
                    <div class="calibration-cell"><div class="calibration-score">100%</div>Stout</div>
                    <div class="calibration-cell"><div class="calibration-score">94%</div>Gold Bl.</div>
                    <div class="calibration-cell"><div class="calibration-score">96%</div>L. Bird</div>
                    <div class="calibration-cell"><div class="calibration-score">91%</div>Howland</div>
                    <div class="calibration-cell"><div class="calibration-score">88%</div>Prairie</div>
                </div>
            </div>

            <div class="card" style="flex:1; display:flex; flex-direction:column;">
                <div class="tab-header" style="display:flex; border-bottom:1px solid var(--border-glass); margin-bottom:10px; gap:8px;">
                    <button class="tab-btn active" id="tab-chat" onclick="switchPanelTab('chat')" style="flex:1; background:none; border:none; padding:8px; color:var(--text-main); font-weight:700; font-size:0.68rem; cursor:pointer; text-align:center; border-bottom:2px solid var(--primary-bright); transition:all 0.3s; font-family:var(--font-mono); letter-spacing:0.5px;">RANGER COMMS</button>
                    <button class="tab-btn" id="tab-logs" onclick="switchPanelTab('logs')" style="flex:1; background:none; border:none; padding:8px; color:var(--text-muted); font-weight:700; font-size:0.68rem; cursor:pointer; text-align:center; border-bottom:2px solid transparent; transition:all 0.3s; font-family:var(--font-mono); letter-spacing:0.5px;">INCIDENT DATABASE</button>
                </div>

                <!-- Tab 1: Comms Chat -->
                <div class="tab-content" id="content-chat" style="display:flex; flex-direction:column; flex:1; justify-content:space-between;">
                    <div class="chat-feed" id="chat-feed" style="flex:1; min-height:200px; max-height:260px; overflow-y:auto; display:flex; flex-direction:column; gap:6px; padding:8px; background:rgba(0,0,0,0.15); border-radius:6px; border:1px solid var(--border-glass);">
                        <!-- Messages populated by JS -->
                    </div>
                    <div class="chat-input-container" style="display:flex; gap:6px; margin-top:8px;">
                        <select id="chat-session-select" style="background:rgba(10,16,12,0.8); border:1px solid var(--border-glass); border-radius:6px; padding:6px; font-size:0.62rem; color:var(--text-main); font-family:var(--font-mono); outline:none; max-width:90px; cursor:pointer;">
                            <option value="">Auto</option>
                        </select>
                        <input type="text" id="chat-input-el" placeholder="Broadcast message (e.g. 'approve' or 'reject')..." onkeydown="handleChatKeyDown(event)" style="flex:1; background:rgba(10,16,12,0.8); border:1px solid var(--border-glass); border-radius:6px; padding:6px 10px; font-size:0.68rem; color:var(--text-main); font-family:var(--font-sans); outline:none;">
                        <button class="chat-send-btn" onclick="sendChatMessage()" style="background:var(--accent-moss); border:1px solid var(--border-glass); border-radius:6px; padding:6px 10px; cursor:pointer; color:var(--text-main); display:flex; align-items:center; justify-content:center; transition:all 0.2s;">
                            <svg class="svg-icon" style="width:12px;height:12px;" viewBox="0 0 24 24"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                        </button>
                    </div>
                </div>

                <!-- Tab 2: Logs Table -->
                <div class="tab-content" id="content-logs" style="display:none; flex-direction:column; flex:1;">
                    <div class="log-table-container" style="overflow-y:auto; max-height:280px;">
                        <table>
                            <thead><tr><th>Sensor / Location</th><th>Signature</th><th>Threat</th><th>Status</th></tr></thead>
                            <tbody id="logs-tbody"></tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="footer-bar">
        <span>Redwood National & State Parks · Humboldt/Del Norte Counties, CA</span>
        <span>ADK 2.0 · Open-Meteo · USGS NWIS · SMART/WorldPop MCP</span>
    </div>

    <script>
        // Chat Comms Icons (Defined at top to avoid Temporal Dead Zone reference errors)
        const chatIcons = {
            radio: `<svg class="svg-icon icon-inline" style="width:12px;height:12px;vertical-align:middle;margin-right:4px;" viewBox="0 0 24 24"><path d="M5 12.55a11 11 0 0 1 14.08 0M1.42 9a16 16 0 0 1 21.16 0M8.53 16.11a6 6 0 0 1 6.95 0M12 20h.01"></path></svg>`,
            alert: `<svg class="svg-icon icon-inline" style="width:12px;height:12px;vertical-align:middle;margin-right:4px;color:var(--alert);" viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0zM12 9v4M12 17h.01"></path></svg>`,
            system: `<svg class="svg-icon icon-inline" style="width:12px;height:12px;vertical-align:middle;margin-right:4px;color:var(--primary-bright);" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>`,
            operator: `<svg class="svg-icon icon-inline" style="width:12px;height:12px;vertical-align:middle;margin-right:4px;color:var(--info);" viewBox="0 0 24 24"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect><line x1="8" y1="21" x2="16" y2="21"></line><line x1="12" y1="17" x2="12" y2="21"></line></svg>`,
            drone: `<svg class="svg-icon icon-inline" style="width:12px;height:12px;vertical-align:middle;margin-right:4px;" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"></circle><path d="M12 12L7 7m10 10l-5-5m0 0l5-5m-5 5l-5 5"></path><circle cx="6" cy="6" r="2"></circle><circle cx="18" cy="18" r="2"></circle><circle cx="18" cy="6" r="2"></circle><circle cx="6" cy="18" r="2"></circle></svg>`,
            ranger: `<svg class="svg-icon icon-inline" style="width:12px;height:12px;vertical-align:middle;margin-right:4px;" viewBox="0 0 24 24"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>`
        };

        // Global state variables defined at the top to avoid Temporal Dead Zone (TDZ) reference errors
        let specCanvas = null;
        let specCtx = null;
        let waveformMode = "idle";
        let gunshotDecay = 0;
        let timeSeed = 0;
        let audioCtx = null;
        let globalVolume = 1.0;
        let globalMuted = false;
        let dropVolumeNode = null;
        let dropAudioCtx = null;
        let dropSource = null;
        let dropAnalyser = null;
        let dropBufferLength = 0;
        let dropDataArray = null;
        let dropIsPlaying = false;
        let activeSynthNodes = [];
        let micStream = null;
        let isMicStreaming = false;
        let currentActiveSession = null;

        const resilienceTexts = {
            primary: "Primary Link: High-fidelity MQTT mesh streaming telemetry active.",
            satlink: "Delayed Link: Store-carry-forward telemetry buffered (CBOR).",
            lora: "Low-Bandwidth Fallback: Packet constraint active (32-byte frames)."
        };

        const steps = ["intake", "triage", "fusion", "assessor", "policy", "dispatcher"];

        const templates = {
            bird: { sensor_id: "sensor-aud-10", location: "Prairie Creek", decibel_level: 42.0, acoustic_signature: "Chirping Bird" },
            chainsaw: { sensor_id: "sensor-aud-88", location: "Howland Hill", decibel_level: 95.5, acoustic_signature: "Chainsaw Revving" },
            gunshot: { sensor_id: "sensor-aud-01", location: "Gold Bluffs", decibel_level: 110.0, acoustic_signature: "Possible Gunshot" },
            ranger: { sensor_id: "sensor-aud-07", location: "Elk Prairie", decibel_level: 68.0, acoustic_signature: "Footsteps" }
        };

        // Map location names to SVG element IDs and alert ring IDs
        const locationMap = {
            "elk prairie":       { label: "label-elk",      ring: "alert-ring-0", droneTarget: [230, 105] },
            "fern canyon":       { label: "label-fern",     ring: "alert-ring-1", droneTarget: [150, 85] },
            "tall trees grove":  { label: "label-talltrees", ring: "alert-ring-2", droneTarget: [340, 210] },
            "orick hq":         { label: "label-orick",    ring: "alert-ring-3", droneTarget: [200, 245] },
            "stout grove":      { label: "label-stout",    ring: "alert-ring-4", droneTarget: [350, 38] },
            "gold bluffs":      { label: "label-gold",     ring: "alert-ring-5", droneTarget: [90, 115] },
            "lady bird grove":  { label: "label-ladybird", ring: "alert-ring-6", droneTarget: [400, 170] },
            "howland hill":     { label: "label-howland",   ring: "alert-ring-7", droneTarget: [450, 50] },
            "prairie creek":    { label: "label-prairie",   ring: "alert-ring-8", droneTarget: [300, 130] },
        };

        // Clock
        function updateClock() {
            const now = new Date();
            document.getElementById("clock-utc").textContent = now.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
        }
        setInterval(updateClock, 1000);
        updateClock();

        // Spectrogram
        specCanvas = document.getElementById("spectrogram-canvas");
        specCtx = specCanvas.getContext("2d");
        waveformMode = "idle";
        gunshotDecay = 0;
        timeSeed = 0;

        function resizeCanvas() {
            const rect = specCanvas.getBoundingClientRect();
            specCanvas.width = rect.width;
            specCanvas.height = rect.height;
        }
        window.addEventListener("resize", resizeCanvas);
        resizeCanvas();

        function drawSpectrogram() {
            timeSeed += 0.25;
            specCtx.drawImage(specCanvas, 1, 0, specCanvas.width - 1, specCanvas.height, 0, 0, specCanvas.width - 1, specCanvas.height);
            const x = specCanvas.width - 1;
            if (dropIsPlaying && dropAnalyser && dropDataArray) {
                dropAnalyser.getByteFrequencyData(dropDataArray);
            }
            for (let y = 0; y < specCanvas.height; y++) {
                let val = 0;
                if (dropIsPlaying && dropAnalyser && dropDataArray) {
                    const binIdx = Math.floor(((specCanvas.height - 1 - y) / specCanvas.height) * dropBufferLength);
                    val = dropDataArray[binIdx] || 0;
                } else if (waveformMode === "idle") { val = Math.random() * 20; }
                else if (waveformMode === "bird") {
                    const cf = Math.sin(timeSeed * 0.4 + y * 0.1) * 15 + 40;
                    val = Math.abs(y - cf) < 3 ? 180 + Math.random() * 55 : Math.random() * 15;
                }
                else if (waveformMode === "chainsaw") { val = Math.random() * 100 + 50; }
                else if (waveformMode === "gunshot") { val = (Math.random() * 200 + 30) * gunshotDecay; }
                else if (waveformMode === "ranger") {
                    const tick = Math.floor(timeSeed) % 8 === 0;
                    val = tick && Math.abs(y - 25) < 12 ? 120 : Math.random() * 15;
                }
                let r, g, b;
                if (dropIsPlaying && dropAnalyser && dropDataArray) {
                    r = val;
                    g = val * 0.6;
                    b = val * 0.15;
                } else if (waveformMode === "gunshot") { r = val; g = val / 5; b = val / 4; }
                else if (waveformMode === "chainsaw") { r = val * 0.9; g = val * 0.5; b = 0; }
                else if (waveformMode === "ranger") { r = 0; g = val * 0.4; b = val * 0.7; }
                else { r = val / 4; g = val * 0.7; b = val * 0.4; }
                specCtx.fillStyle = `rgb(${r},${g},${b})`;
                specCtx.fillRect(x, y, 1, 1);
            }
            if (gunshotDecay > 0) gunshotDecay = Math.max(0, gunshotDecay - 0.03);
            requestAnimationFrame(drawSpectrogram);
        }
        drawSpectrogram();

        // Gauges
        function updateCircularGauge(id, value, max) {
            const el = document.getElementById(id);
            if (el) el.style.strokeDasharray = `${Math.min(100, Math.max(0, (value / max) * 100))}, 100`;
        }

        // Audio
        audioCtx = null;
        globalVolume = 1.0;
        globalMuted = false;
        dropVolumeNode = null;

        function initAudio() { if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)(); }
        function playSonarPing() {
            if (globalMuted || globalVolume === 0) return;
            try { initAudio(); const o = audioCtx.createOscillator(), g = audioCtx.createGain(); o.connect(g); g.connect(audioCtx.destination); o.type = "sine"; o.frequency.setValueAtTime(1100, audioCtx.currentTime); g.gain.setValueAtTime(0.1 * globalVolume, audioCtx.currentTime); g.gain.exponentialRampToValueAtTime(0.001 * globalVolume, audioCtx.currentTime + 0.8); o.start(); o.stop(audioCtx.currentTime + 0.8); } catch(e) {}
        }
        function playSiren() {
            if (globalMuted || globalVolume === 0) return;
            try { initAudio(); const o = audioCtx.createOscillator(), g = audioCtx.createGain(); o.connect(g); g.connect(audioCtx.destination); o.type = "sawtooth"; o.frequency.setValueAtTime(400, audioCtx.currentTime); o.frequency.linearRampToValueAtTime(650, audioCtx.currentTime + 0.5); o.frequency.linearRampToValueAtTime(400, audioCtx.currentTime + 1.0); o.frequency.linearRampToValueAtTime(650, audioCtx.currentTime + 1.5); o.frequency.linearRampToValueAtTime(400, audioCtx.currentTime + 2.0); g.gain.setValueAtTime(0.05 * globalVolume, audioCtx.currentTime); g.gain.exponentialRampToValueAtTime(0.001 * globalVolume, audioCtx.currentTime + 2.0); o.start(); o.stop(audioCtx.currentTime + 2.0); } catch(e) {}
        }

        function updateVolume(val) {
            globalVolume = parseFloat(val);
            if (dropVolumeNode && dropAudioCtx) {
                dropVolumeNode.gain.setValueAtTime(globalMuted ? 0 : globalVolume, dropAudioCtx.currentTime);
            }
        }
        function toggleMute(muted) {
            globalMuted = muted;
            if (dropVolumeNode && dropAudioCtx) {
                dropVolumeNode.gain.setValueAtTime(globalMuted ? 0 : globalVolume, dropAudioCtx.currentTime);
            }
        }
        function toggleMuteButton() {
            globalMuted = !globalMuted;
            const btn = document.getElementById("mute-btn");
            if (globalMuted) {
                btn.innerText = "🔇";
                btn.style.color = "var(--text-muted)";
            } else {
                btn.innerText = "🔊";
                btn.style.color = "var(--primary)";
            }
            toggleMute(globalMuted);
        }
        function handleVolumeSlider(val) {
            updateVolume(val);
            const btn = document.getElementById("mute-btn");
            if (parseFloat(val) === 0) {
                btn.innerText = "🔇";
                btn.style.color = "var(--text-muted)";
            } else if (!globalMuted) {
                btn.innerText = "🔊";
                btn.style.color = "var(--primary)";
            }
        }

        function applyTemplate() {
            const val = document.getElementById("template-select").value;
            const d = templates[val];
            document.getElementById("sensor_id").value = d.sensor_id;
            document.getElementById("location").value = d.location;
            document.getElementById("decibel_level").value = d.decibel_level;
            document.getElementById("acoustic_signature").value = d.acoustic_signature;
            if (val === "chainsaw") waveformMode = "chainsaw";
            else if (val === "gunshot") { waveformMode = "gunshot"; gunshotDecay = 1.0; }
            else if (val === "ranger") waveformMode = "ranger";
            else waveformMode = "bird";
        }

        function toggleLayer(layer) {
            document.getElementById(`layer-${layer}`).classList.toggle("active");
        }

        // Resilience mode (Declared at top)
        function toggleResilienceMode() {
            const mode = document.getElementById("resilience-mode").value;
            const banner = document.getElementById("resilience-banner-el");
            banner.innerText = resilienceTexts[mode];
            banner.style.display = "block";
            playSonarPing();
        }
        function setResilienceMode(mode) {
            document.getElementById("resilience-mode").value = mode;
            document.querySelectorAll(".link-btn").forEach(b => b.classList.remove("active"));
            const btn = document.querySelector(`.link-btn[onclick*="${mode}"]`);
            if (btn) btn.classList.add("active");
            toggleResilienceMode();

            // Disable audio stream controls if in LoRa mode
            const micBtn = document.getElementById("mic-btn");
            const uploadBtn = document.querySelector("button[onclick*='audio-file-input']");
            if (mode === "lora") {
                if (micBtn) { micBtn.disabled = true; micBtn.style.opacity = 0.5; micBtn.title = "LoRa Mode: Audio streaming disabled"; }
                if (uploadBtn) { uploadBtn.disabled = true; uploadBtn.style.opacity = 0.5; uploadBtn.title = "LoRa Mode: Audio upload disabled"; }
                if (typeof isMicStreaming !== 'undefined' && isMicStreaming) { toggleMicrophone(); }
            } else {
                if (micBtn) { micBtn.disabled = false; micBtn.style.opacity = 1; micBtn.title = ""; }
                if (uploadBtn) { uploadBtn.disabled = false; uploadBtn.style.opacity = 1; uploadBtn.title = ""; }
            }
        }

        // Trace highlighting (steps declared at top)
        function highlightTrace(stepId, status="active") {
            steps.forEach(s => { const el = document.getElementById(`node-${s}`); if (el) el.className = "trace-step"; });
            if (stepId) { const el = document.getElementById(`node-${stepId}`); if (el) el.className = `trace-step ${status}`; }
        }

        // Tab switching
        function switchPanelTab(tabName) {
            document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(content => content.style.display = "none");

            const btnChat = document.getElementById("tab-chat");
            const btnLogs = document.getElementById("tab-logs");

            if (tabName === 'chat') {
                btnChat.classList.add("active");
                btnChat.style.color = "var(--text-main)";
                btnChat.style.borderBottomColor = "var(--primary-bright)";
                btnLogs.style.color = "var(--text-muted)";
                btnLogs.style.borderBottomColor = "transparent";
                document.getElementById("content-chat").style.display = "flex";
                loadChats();
            } else {
                btnLogs.classList.add("active");
                btnLogs.style.color = "var(--text-main)";
                btnLogs.style.borderBottomColor = "var(--primary-bright)";
                btnChat.style.color = "var(--text-muted)";
                btnChat.style.borderBottomColor = "transparent";
                document.getElementById("content-logs").style.display = "flex";
                loadLogs();
            }
        }

        async function loadChats() {
            try {
                const res = await fetch("/api/chats");
                const chats = await res.json();
                const feed = document.getElementById("chat-feed");

                const isScrolledToBottom = feed.scrollHeight - feed.clientHeight <= feed.scrollTop + 20;

                feed.innerHTML = "";
                chats.forEach(msg => {
                    const div = document.createElement("div");
                    let cls = "";
                    let iconHtml = chatIcons.radio;
                    if (msg.sender.includes("System Alert")) {
                        cls = "system-alert";
                        iconHtml = chatIcons.alert;
                    } else if (msg.sender.includes("System") || msg.sender.includes("Ranger System")) {
                        cls = "system";
                        iconHtml = chatIcons.system;
                    } else if (msg.sender.toLowerCase() === "operator" || msg.sender.toLowerCase() === "hq") {
                        cls = "operator";
                        iconHtml = chatIcons.operator;
                    } else if (msg.sender.includes("Drone")) {
                        iconHtml = chatIcons.drone;
                    } else if (msg.sender.includes("Ranger")) {
                        iconHtml = chatIcons.ranger;
                    }

                    div.className = `chat-msg ${cls}`;
                    div.innerHTML = `
                        <div class="chat-header-line">
                            <span class="chat-sender">${iconHtml}${msg.sender}</span>
                            <span class="chat-time">${msg.timestamp.substring(11, 19)}</span>
                        </div>
                        <div class="chat-text">${msg.message}</div>
                    `;
                    feed.appendChild(div);
                });

                if (isScrolledToBottom || feed.scrollTop === 0) {
                    feed.scrollTop = feed.scrollHeight;
                }
            } catch (e) {
                console.error("Error loading chats:", e);
            }
        }

        async function sendChatMessage() {
            const input = document.getElementById("chat-input-el");
            let val = input.value.trim();
            if (!val) return;

            const sessionSelect = document.getElementById("chat-session-select");
            const selectedSession = sessionSelect ? sessionSelect.value : "";
            const vLower = val.toLowerCase();
            if (selectedSession && (vLower === "approve" || vLower === "reject" || vLower === "yes" || vLower === "no" || vLower === "false alarm")) {
                val = `/${vLower} ${selectedSession}`;
            }

            const payload = {
                sender: "Operator",
                message: val,
                timestamp: new Date().toISOString()
            };

            input.value = "";

            try {
                await fetch("/api/chats", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                await loadChats();

                setTimeout(() => {
                    loadLogs();
                    loadPending();
                }, 600);
            } catch(e) {
                console.error(e);
            }
        }

        function handleChatKeyDown(event) {
            if (event.key === "Enter") {
                sendChatMessage();
            }
        }

        // Map Click Sighting Reports
        let lastClickCoords = { x: 0, y: 0 };
        let lastClickedLocName = "";

        function handleMapClick(event) {
            if (event.target.classList.contains("menu-item-btn")) return;

            const svg = event.currentTarget;
            const rect = svg.getBoundingClientRect();

            const clickX = ((event.clientX - rect.left) / rect.width) * 600;
            const clickY = ((event.clientY - rect.top) / rect.height) * 280;

            let nearestLoc = "Prairie Creek";
            let minDist = 999999;
            Object.keys(locationMap).forEach(key => {
                const l = locationMap[key];
                const dx = clickX - l.droneTarget[0];
                const dy = clickY - l.droneTarget[1];
                const dist = Math.sqrt(dx*dx + dy*dy);
                if (dist < minDist) {
                    minDist = dist;
                    nearestLoc = key.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
                }
            });

            lastClickCoords = { x: clickX, y: clickY };
            lastClickedLocName = nearestLoc;

            showMapMenu(nearestLoc, event.clientX - rect.left, event.clientY - rect.top);
        }

        function showMapMenu(locName, x, y) {
            const menu = document.getElementById("map-click-menu");
            document.getElementById("menu-loc-title").innerText = "LOC: " + locName;

            menu.style.left = `${x}px`;
            menu.style.top = `${y}px`;
            menu.style.display = "block";

            setTimeout(() => {
                window.addEventListener("click", hideMenuOnClickOutside);
            }, 100);
        }

        function hideMapMenu() {
            document.getElementById("map-click-menu").style.display = "none";
            window.removeEventListener("click", hideMenuOnClickOutside);
        }

        function hideMenuOnClickOutside(e) {
            const menu = document.getElementById("map-click-menu");
            if (!menu.contains(e.target)) {
                hideMapMenu();
            }
        }

        async function submitMapReport(reportType) {
            hideMapMenu();

            const payload = {
                report_type: reportType,
                location: lastClickedLocName,
                x: lastClickCoords.x,
                y: lastClickCoords.y,
                details: new Date().toISOString()
            };

            try {
                await fetch("/api/reports", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                await loadReports();
                await loadChats();
            } catch(e) {
                console.error(e);
            }
        }

        async function loadReports() {
            try {
                const res = await fetch("/api/reports");
                const reports = await res.json();
                const pinsGroup = document.getElementById("manual-report-pins");
                pinsGroup.innerHTML = "";

                reports.forEach(rep => {
                    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
                    g.setAttribute("transform", `translate(${rep.x}, ${rep.y})`);
                    g.setAttribute("class", "manual-pin");

                    let shapeHtml = "";
                    if (rep.report_type === "Roosevelt_Elk_Sighting") {
                        shapeHtml = `
                            <circle cx="0" cy="0" r="3.5" fill="var(--primary-bright)"/>
                            <circle cx="0" cy="0" r="6" fill="none" stroke="var(--primary-bright)" stroke-width="0.7" opacity="0.6"/>
                        `;
                    } else if (rep.report_type === "Illegal_Snare_Trap") {
                        shapeHtml = `
                            <circle cx="0" cy="0" r="4.5" fill="var(--warning)"/>
                            <line x1="-2.5" y1="-2.5" x2="2.5" y2="2.5" stroke="#000" stroke-width="1.2"/>
                            <line x1="2.5" y1="-2.5" x2="-2.5" y2="2.5" stroke="#000" stroke-width="1.2"/>
                        `;
                    } else if (rep.report_type === "Unattended_Campfire") {
                        shapeHtml = `
                            <circle cx="0" cy="0" r="4" fill="var(--alert)" filter="url(#glow)"/>
                            <polygon points="0,-4.5 -2.5,0.5 2.5,0.5" fill="#ffaa00"/>
                        `;
                    } else if (rep.report_type === "Install_Custom_Sensor") {
                        shapeHtml = `
                            <circle cx="0" cy="0" r="4.5" fill="rgba(91,158,166,0.3)" stroke="var(--info)" stroke-width="1.2"/>
                            <circle cx="0" cy="0" r="1.5" fill="var(--info)"/>
                        `;
                    }

                    g.innerHTML = shapeHtml;
                    pinsGroup.appendChild(g);
                });
            } catch(e) {
                console.error(e);
            }
        }

        // Map alerts
        function highlightMapSector(name, type) {
            document.querySelectorAll(".map-sector-label").forEach(el => el.classList.remove("alerting"));
            document.querySelectorAll(".alert-ring").forEach(el => el.style.display = "none");
            if (!name) return;
            const key = name.toLowerCase().trim();
            const loc = locationMap[key];
            if (loc && type === "alert") {
                const label = document.getElementById(loc.label);
                if (label) label.classList.add("alerting");
                const ring = document.getElementById(loc.ring);
                if (ring) ring.style.display = "block";
            }
        }

        // Drone animation
        function returnDroneToHQ() {
            const drone = document.getElementById("drone");
            drone.classList.remove("loitering");
            drone.style.transition = "left 2s ease-in-out, top 2s ease-in-out";
            drone.style.left = "46%";
            drone.style.top = "44%";
            document.getElementById("drone-alpha-status").innerText = "Standby · Prairie Creek HQ";
            const pl = document.getElementById("drone-path-line");
            if (pl) pl.style.display = "none";
            setTimeout(() => { drone.style.display = "none"; }, 2000);
        }

        function animateDrone(targetName, callback) {
            const key = targetName.toLowerCase().trim();
            const loc = locationMap[key];
            if (!loc) return;
            const drone = document.getElementById("drone");
            const container = document.getElementById("map-container");
            const rect = container.getBoundingClientRect();
            const [tx, ty] = loc.droneTarget;
            const percX = (tx / 600) * 100;
            const percY = (ty / 280) * 100;

            drone.style.transition = "none";
            drone.style.left = "46%";
            drone.style.top = "44%";
            drone.style.display = "block";
            drone.offsetHeight;

            const pl = document.getElementById("drone-path-line");
            if (pl) {
                pl.setAttribute("x1", "300"); pl.setAttribute("y1", "130");
                pl.setAttribute("x2", String(tx)); pl.setAttribute("y2", String(ty));
                pl.style.display = "block";
            }

            drone.style.transition = "left 2s ease-in-out, top 2s ease-in-out";
            drone.style.left = `${percX}%`;
            drone.style.top = `${percY}%`;
            document.getElementById("drone-alpha-status").innerText = `Flight to ${targetName}...`;
            setTimeout(() => {
                drone.classList.add("loitering");
                document.getElementById("drone-alpha-status").innerText = `Loitering · ${targetName}`;
                if (callback) callback();
            }, 2000);
        }

        // Flywheel metrics
        let quietTimeSeconds = 0, falseAlarmStreak = 0, totalProcessed = 0, actedOn = 0;
        setInterval(() => {
            quietTimeSeconds++;
            document.getElementById("metrics-quiet").innerText = `${quietTimeSeconds}s`;
            updateCircularGauge("gauge-quiet", quietTimeSeconds, 30);
        }, 1000);

        function updateFlywheelMetrics(status) {
            totalProcessed++;
            if (status === "LOGGED" || status === "FALSE_ALARM") { falseAlarmStreak++; }
            else { falseAlarmStreak = 0; actedOn++; }
            document.getElementById("metrics-streak").innerText = falseAlarmStreak;
            updateCircularGauge("gauge-streak", falseAlarmStreak, 10);
            const focus = totalProcessed > 0 ? Math.round((actedOn / totalProcessed) * 100) : 100;
            document.getElementById("metrics-focus").innerText = `${focus}%`;
            updateCircularGauge("gauge-focus", focus, 100);
            quietTimeSeconds = 0;
            updateCircularGauge("gauge-quiet", 0, 30);
        }

        // Fetch USGS hydrology
        async function loadHydrology() {
            try {
                const res = await fetch("/api/mcp/usgs_streamflow");
                const data = await res.json();
                document.getElementById("hydro-redwood").textContent = data.redwood_creek?.value_cfs ?? "--";
                document.getElementById("hydro-klamath").textContent = data.klamath_river?.value_cfs ?? "--";
            } catch(e) {
                document.getElementById("hydro-redwood").textContent = "42";
                document.getElementById("hydro-klamath").textContent = "780";
            }
        }
        loadHydrology();
        setInterval(loadHydrology, 30000);

        // Real-time drag and drop audio decoding and analyser setup (Variables declared at top)
        dropAudioCtx = null;
        dropSource = null;
        dropAnalyser = null;
        dropBufferLength = 0;
        dropDataArray = null;
        dropIsPlaying = false;
        activeSynthNodes = [];
        micStream = null;
        isMicStreaming = false;

        function stopActiveSynth() {
            stopMicrophone();
            if (dropSource) { try { dropSource.stop(); } catch(err) {} dropSource = null; }
            activeSynthNodes.forEach(node => {
                if (node.stop && typeof node.stop === "function") {
                    try { node.stop(); } catch(err) {}
                }
                try { node.disconnect(); } catch(err) {}
            });
            activeSynthNodes = [];
            dropIsPlaying = false;
        }

        // Sound Synthesizers using Web Audio API
        function synthesizeGunshot(ctx, destination) {
            const bufferSize = ctx.sampleRate * 1.5;
            const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
            const data = buffer.getChannelData(0);
            for (let i = 0; i < bufferSize; i++) {
                data[i] = Math.random() * 2 - 1;
            }
            const noise = ctx.createBufferSource();
            noise.buffer = buffer;

            const filter = ctx.createBiquadFilter();
            filter.type = 'lowpass';
            filter.frequency.setValueAtTime(1000, ctx.currentTime);
            filter.frequency.exponentialRampToValueAtTime(100, ctx.currentTime + 0.5);

            const gain = ctx.createGain();
            gain.gain.setValueAtTime(0.8, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.6);

            noise.connect(filter);
            filter.connect(gain);
            gain.connect(destination);

            noise.start(0);
            return [noise, filter, gain];
        }

        function synthesizeChainsaw(ctx, destination) {
            const osc1 = ctx.createOscillator();
            osc1.type = 'sawtooth';
            osc1.frequency.setValueAtTime(90, ctx.currentTime);

            const modulator = ctx.createOscillator();
            modulator.type = 'sine';
            modulator.frequency.setValueAtTime(12, ctx.currentTime);

            const modGain = ctx.createGain();
            modGain.gain.setValueAtTime(15, ctx.currentTime);

            osc1.frequency.linearRampToValueAtTime(140, ctx.currentTime + 1.5);
            osc1.frequency.linearRampToValueAtTime(90, ctx.currentTime + 2.5);
            osc1.frequency.linearRampToValueAtTime(130, ctx.currentTime + 3.8);
            osc1.frequency.linearRampToValueAtTime(80, ctx.currentTime + 4.0);

            const filter = ctx.createBiquadFilter();
            filter.type = 'bandpass';
            filter.frequency.setValueAtTime(300, ctx.currentTime);
            filter.Q.setValueAtTime(1.0, ctx.currentTime);

            const gain = ctx.createGain();
            gain.gain.setValueAtTime(0.3, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 4.0);

            modulator.connect(modGain);
            modGain.connect(osc1.frequency);
            osc1.connect(filter);
            filter.connect(gain);
            gain.connect(destination);

            osc1.start(0);
            modulator.start(0);
            osc1.stop(ctx.currentTime + 4.0);
            modulator.stop(ctx.currentTime + 4.0);

            return [osc1, modulator, modGain, filter, gain];
        }

        function synthesizeBird(ctx, destination) {
            const osc = ctx.createOscillator();
            osc.type = 'sine';
            const gain = ctx.createGain();
            gain.gain.setValueAtTime(0.0, ctx.currentTime);

            const startTime = ctx.currentTime;
            for (let i = 0; i < 5; i++) {
                const chirpStart = startTime + i * 0.5;
                const chirpEnd = chirpStart + 0.25;

                osc.frequency.setValueAtTime(2000, chirpStart);
                osc.frequency.exponentialRampToValueAtTime(4500, chirpEnd);

                gain.gain.setValueAtTime(0.0, chirpStart);
                gain.gain.linearRampToValueAtTime(0.15, chirpStart + 0.05);
                gain.gain.exponentialRampToValueAtTime(0.001, chirpEnd);
            }

            osc.connect(gain);
            gain.connect(destination);
            osc.start(startTime);
            osc.stop(startTime + 3.0);

            return [osc, gain];
        }

        function synthesizeRanger(ctx, destination) {
            const bufferSize = ctx.sampleRate * 4.0;
            const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
            const data = buffer.getChannelData(0);
            for (let i = 0; i < bufferSize; i++) {
                data[i] = Math.random() * 2 - 1;
            }
            const noise = ctx.createBufferSource();
            noise.buffer = buffer;

            const filter = ctx.createBiquadFilter();
            filter.type = 'lowpass';
            filter.frequency.setValueAtTime(150, ctx.currentTime);

            const gain = ctx.createGain();
            gain.gain.setValueAtTime(0.0, ctx.currentTime);

            const startTime = ctx.currentTime;
            for (let i = 0; i < 6; i++) {
                const stepStart = startTime + i * 0.6;
                const stepEnd = stepStart + 0.15;

                gain.gain.setValueAtTime(0.0, stepStart);
                gain.gain.linearRampToValueAtTime(0.4, stepStart + 0.02);
                gain.gain.exponentialRampToValueAtTime(0.001, stepEnd);
            }

            noise.connect(filter);
            filter.connect(gain);
            gain.connect(destination);

            noise.start(startTime);
            noise.stop(startTime + 4.0);

            return [noise, filter, gain];
        }

        async function playPresetSound(type) {
            stopActiveSynth();

            if (!dropAudioCtx) {
                dropAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (dropAudioCtx.state === 'suspended') {
                await dropAudioCtx.resume();
            }

            dropAnalyser = dropAudioCtx.createAnalyser();
            dropAnalyser.fftSize = 128;
            dropBufferLength = dropAnalyser.frequencyBinCount;
            dropDataArray = new Uint8Array(dropBufferLength);

            let nodes = [];
            let duration = 3.0;

            if (type === 'gunshot') {
                nodes = synthesizeGunshot(dropAudioCtx, dropAnalyser);
                duration = 1.5;
                waveformMode = 'gunshot';
                gunshotDecay = 1.0;
            } else if (type === 'chainsaw') {
                nodes = synthesizeChainsaw(dropAudioCtx, dropAnalyser);
                duration = 4.0;
                waveformMode = 'chainsaw';
            } else if (type === 'bird') {
                nodes = synthesizeBird(dropAudioCtx, dropAnalyser);
                duration = 3.0;
                waveformMode = 'bird';
            } else if (type === 'ranger') {
                nodes = synthesizeRanger(dropAudioCtx, dropAnalyser);
                duration = 4.0;
                waveformMode = 'ranger';
            } else {
                return;
            }

            const dropVolumeGain = dropAudioCtx.createGain();
            dropVolumeGain.gain.setValueAtTime(globalMuted ? 0 : globalVolume, dropAudioCtx.currentTime);
            dropAnalyser.connect(dropVolumeGain);
            dropVolumeGain.connect(dropAudioCtx.destination);
            dropVolumeNode = dropVolumeGain;
            activeSynthNodes = nodes;
            dropIsPlaying = true;

            // Add system chat announcement
            const chatFeed = document.getElementById("chat-feed");
            if (chatFeed) {
                const div = document.createElement("div");
                div.className = "chat-msg system";
                div.innerHTML = `
                    <div class="chat-header-line"><span class="chat-sender">🔊 System</span></div>
                    <div class="chat-text">Synthesizing live audio stream: <strong>${type.toUpperCase()}</strong> (${duration}s). Feeding FFT bins into classifier.</div>
                `;
                chatFeed.appendChild(div);
                chatFeed.scrollTop = chatFeed.scrollHeight;
            }

            setTimeout(() => {
                if (waveformMode === type || waveformMode === 'custom') {
                    dropIsPlaying = false;
                    waveformMode = 'idle';
                }
            }, duration * 1000);
        }

        async function toggleMicrophone() {
            const micBtn = document.getElementById("mic-btn");
            if (isMicStreaming) {
                stopMicrophone();
                return;
            }

            stopActiveSynth();

            if (!dropAudioCtx) {
                dropAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (dropAudioCtx.state === 'suspended') {
                await dropAudioCtx.resume();
            }

            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
                micStream = stream;

                dropAnalyser = dropAudioCtx.createAnalyser();
                dropAnalyser.fftSize = 128;
                dropBufferLength = dropAnalyser.frequencyBinCount;
                dropDataArray = new Uint8Array(dropBufferLength);

                const micSource = dropAudioCtx.createMediaStreamSource(stream);
                micSource.connect(dropAnalyser);

                dropIsPlaying = true;
                isMicStreaming = true;
                waveformMode = "custom";

                micBtn.innerHTML = "🛑 STOP MIC";
                micBtn.style.background = "rgba(220,38,38,0.2)";
                micBtn.style.color = "#f87171";
                micBtn.style.borderColor = "rgba(220,38,38,0.4)";

                const chatFeed = document.getElementById("chat-feed");
                if (chatFeed) {
                    const div = document.createElement("div");
                    div.className = "chat-msg system";
                    div.innerHTML = `
                        <div class="chat-header-line"><span class="chat-sender">🎙️ System</span></div>
                        <div class="chat-text">Live Microphone Stream connected. Visualizing ambient reserve frequencies in real-time.</div>
                    `;
                    chatFeed.appendChild(div);
                    chatFeed.scrollTop = chatFeed.scrollHeight;
                }

                updateMicDecibelsLoop();

            } catch (err) {
                console.error("Error accessing microphone:", err);
                alert("Microphone access denied or unavailable: " + err.message);
            }
        }

        function stopMicrophone() {
            if (micStream) {
                micStream.getTracks().forEach(track => track.stop());
                micStream = null;
            }
            isMicStreaming = false;
            dropIsPlaying = false;
            waveformMode = "idle";

            const micBtn = document.getElementById("mic-btn");
            if (micBtn) {
                micBtn.innerHTML = "🎙️ STREAM MIC";
                micBtn.style.background = "rgba(234,179,8,0.12)";
                micBtn.style.color = "#facc15";
                micBtn.style.borderColor = "rgba(234,179,8,0.25)";
            }

            const chatFeed = document.getElementById("chat-feed");
            if (chatFeed) {
                const div = document.createElement("div");
                div.className = "chat-msg system";
                div.innerHTML = `
                    <div class="chat-header-line"><span class="chat-sender">🎙️ System</span></div>
                    <div class="chat-text">Live Microphone Stream disconnected.</div>
                `;
                chatFeed.appendChild(div);
                chatFeed.scrollTop = chatFeed.scrollHeight;
            }
        }

        function updateMicDecibelsLoop() {
            if (!isMicStreaming || !dropAnalyser || !dropDataArray) return;

            dropAnalyser.getByteFrequencyData(dropDataArray);
            let sum = 0;
            for (let i = 0; i < dropBufferLength; i++) {
                sum += dropDataArray[i];
            }
            const average = sum / dropBufferLength;
            const dbVal = Math.round(30 + (average / 255) * 80);

            if (dbVal > 35) {
                document.getElementById("decibel_level").value = dbVal;
            }

            requestAnimationFrame(updateMicDecibelsLoop);
        }

        function handleAudioInputChange(event) {
            if (event.target.files && event.target.files[0]) {
                handleAudioUpload(event.target.files[0]);
            }
        }

        function handleAudioUpload(file) {
            if (!file) return;
            stopActiveSynth();
            const nameWithoutExt = file.name.substring(0, file.name.lastIndexOf('.')) || file.name;

            let guessedSig = "Unknown Sound Profile";
            let guessedDb = 75.0;

            if (file.name.toLowerCase().includes("chainsaw") || file.name.toLowerCase().includes("saw")) {
                guessedSig = "Chainsaw Revving";
                guessedDb = 95.5;
                document.getElementById("template-select").value = "chainsaw";
                waveformMode = "chainsaw";
            } else if (file.name.toLowerCase().includes("gunshot") || file.name.toLowerCase().includes("shot")) {
                guessedSig = "Possible Gunshot";
                guessedDb = 110.0;
                document.getElementById("template-select").value = "gunshot";
                waveformMode = "gunshot";
                gunshotDecay = 1.0;
            } else if (file.name.toLowerCase().includes("footstep") || file.name.toLowerCase().includes("walk")) {
                guessedSig = "Ranger Footsteps";
                guessedDb = 68.0;
                document.getElementById("template-select").value = "ranger";
                waveformMode = "ranger";
            } else if (file.name.toLowerCase().includes("bird") || file.name.toLowerCase().includes("chirp")) {
                guessedSig = "Chirping Bird";
                guessedDb = 42.0;
                document.getElementById("template-select").value = "bird";
                waveformMode = "bird";
            } else {
                guessedSig = nameWithoutExt.split('_').join(' ').split('-').join(' ');
                waveformMode = "custom";
            }

            document.getElementById("sensor_id").value = "sensor-upl-" + Math.floor(Math.random()*899 + 100);
            document.getElementById("acoustic_signature").value = guessedSig;
            document.getElementById("decibel_level").value = guessedDb;

            const reader = new FileReader();
            reader.onload = function(e) {
                const arrayBuffer = e.target.result;
                if (dropSource) { try { dropSource.stop(); } catch(err) {} }

                if (!dropAudioCtx) {
                    dropAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
                }

                dropAudioCtx.decodeAudioData(arrayBuffer, function(audioBuffer) {
                    dropSource = dropAudioCtx.createBufferSource();
                    dropSource.buffer = audioBuffer;

                    dropAnalyser = dropAudioCtx.createAnalyser();
                    dropAnalyser.fftSize = 128;
                    dropBufferLength = dropAnalyser.frequencyBinCount;
                    dropDataArray = new Uint8Array(dropBufferLength);

                    const dropVolumeGain = dropAudioCtx.createGain();
                    dropVolumeGain.gain.setValueAtTime(globalMuted ? 0 : globalVolume, dropAudioCtx.currentTime);
                    dropAnalyser.connect(dropVolumeGain);
                    dropVolumeGain.connect(dropAudioCtx.destination);
                    dropVolumeNode = dropVolumeGain;

                    dropSource.start(0);
                    dropIsPlaying = true;

                    // Add system chat announcement
                    setTimeout(() => {
                        const chatFeed = document.getElementById("chat-feed");
                        const div = document.createElement("div");
                        div.className = "chat-msg system";
                        div.innerHTML = `
                            <div class="chat-header-line"><span class="chat-sender">🔊 System</span></div>
                            <div class="chat-text">Processing raw audio file upload: <strong>${file.name}</strong> (${Math.round(audioBuffer.duration*10)/10}s). Feeding FFT bins into classifier.</div>
                        `;
                        chatFeed.appendChild(div);
                        chatFeed.scrollTop = chatFeed.scrollHeight;
                    }, 100);

                    // Auto-trigger the ADK graph run!
                    triggerEvent();

                    dropSource.onended = function() {
                        dropIsPlaying = false;
                        waveformMode = "idle";
                    };
                }, function(err) {
                    console.error("Error decoding audio buffer:", err);
                });
            };
            reader.readAsArrayBuffer(file);
        }

        // Drag & Drop event bindings
        const dropzoneCard = document.getElementById("spectrogram-card");
        const dropOverlay = document.getElementById("drop-zone-overlay");

        dropzoneCard.addEventListener("dragenter", (e) => {
            e.preventDefault();
            dropOverlay.style.display = "flex";
        });
        dropOverlay.addEventListener("dragleave", (e) => {
            e.preventDefault();
            dropOverlay.style.display = "none";
        });
        dropOverlay.addEventListener("dragover", (e) => {
            e.preventDefault();
        });
        dropOverlay.addEventListener("drop", (e) => {
            e.preventDefault();
            dropOverlay.style.display = "none";
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleAudioUpload(e.dataTransfer.files[0]);
            }
        });

        async function triggerEvent() {
            const sensorVal = document.getElementById("sensor_id").value;
            const locVal = document.getElementById("location").value;
            const dbVal = parseFloat(document.getElementById("decibel_level").value);
            const sigVal = document.getElementById("acoustic_signature").value;
            const resilienceMode = document.getElementById("resilience-mode").value;

            // Auto-synthesize sound if not already playing or streaming mic and not in LoRa mode
            if (!dropIsPlaying && !isMicStreaming && resilienceMode !== "lora") {
                const modes = ["chainsaw", "gunshot", "bird", "ranger"];
                if (modes.includes(waveformMode)) {
                    playPresetSound(waveformMode);
                } else {
                    const templateType = document.getElementById("template-select").value;
                    playPresetSound(templateType);
                }
            }

            highlightTrace("intake", "active");
            highlightMapSector(locVal, null);

            const body = {
                sensor_id: sensorVal,
                location: locVal,
                decibel_level: dbVal,
                acoustic_signature: sigVal,
                timestamp: new Date().toISOString(),
                resilience_mode: resilienceMode
            };

            setTimeout(() => {
                highlightTrace("triage", "active");
                setTimeout(() => {
                    highlightTrace("fusion", "active");
                    setTimeout(async () => {
                        const res = await fetch("/api/trigger", {
                            method: "POST",
                            headers: { 
                                "Content-Type": "application/json",
                                "X-API-Key": "conservation-secure-key"
                            },
                            body: JSON.stringify(body)
                        });
                        const data = await res.json();

                        await loadChats(); // Reload chats instantly

                        if (data.interrupted) {
                            highlightTrace("assessor", "alert");
                            highlightMapSector(locVal, "alert");
                            if (resilienceMode !== "lora") {
                                playSiren();
                            }
                            setTimeout(() => {
                                highlightTrace("policy", "paused");
                                loadLogs();
                                loadPending();
                            }, 500);
                        } else {
                            highlightTrace("dispatcher", "active");
                            if (resilienceMode !== "lora") {
                                playSonarPing();
                            }
                            loadLogs();
                            loadPending();
                            updateFlywheelMetrics("LOGGED");
                            setTimeout(() => { highlightTrace(null); if(!dropIsPlaying) waveformMode = "idle"; }, 2000);
                        }
                    }, 600);
                }, 500);
            }, 400);
        }

        // Load logs
        async function loadLogs() {
            const res = await fetch("/api/logs");
            const logs = await res.json();
            const tbody = document.getElementById("logs-tbody");
            tbody.innerHTML = "";
            if (logs.length === 0) { tbody.innerHTML = `<tr><td colspan="4" style="text-align:center;color:var(--text-muted);">No incidents recorded.</td></tr>`; return; }
            logs.slice(-8).reverse().forEach(log => {
                const tr = document.createElement("tr");
                let bc = "status-logged", icon = "✓";
                if (log.status === "DRONE_DISPATCHED") { bc = "status-dispatched"; icon = "⊕"; }
                if (log.status === "RANGER_DIVERTED" || log.status === "RANGER_DISPATCHED") { bc = "status-ranger-dispatched"; icon = "◇"; }
                if (log.status === "FALSE_ALARM") { bc = "status-false"; icon = "✕"; }
                tr.innerHTML = `
                    <td><div style="font-weight:700;">${log.sensor_id}</div><span style="font-size:0.62rem;color:var(--text-muted);">${log.location}</span></td>
                    <td>${log.acoustic_signature}</td>
                    <td style="font-weight:700;color:${log.threat_level > 60 ? 'var(--alert)' : 'var(--primary-bright)'};">${log.threat_level}%</td>
                    <td><span class="status-badge ${bc}">${icon} ${log.status.replace("_", " ")}</span></td>
                `;
                tbody.appendChild(tr);
            });
        }

        // currentActiveSession declared at top
        async function loadPending() {
            const res = await fetch("/api/pending");
            const pending = await res.json();
            const alertCard = document.getElementById("alert-card");
            const alertDetails = document.getElementById("alert-details");
            const evidenceList = document.getElementById("alert-evidence-list");
            const warningsBox = document.getElementById("policy-warnings");
            const approveBtn = document.getElementById("approve-dispatch-btn");

            // Populate the dropdown select options with all pending session IDs
            const selectEl = document.getElementById("chat-session-select");
            if (selectEl) {
                const prevVal = selectEl.value;
                selectEl.innerHTML = '<option value="">Auto</option>';
                pending.forEach(p => {
                    const opt = document.createElement("option");
                    opt.value = p.session_id;
                    opt.innerText = p.session_id.substring(0, 8);
                    selectEl.appendChild(opt);
                });
                if ([...selectEl.options].some(o => o.value === prevVal)) {
                    selectEl.value = prevVal;
                }
            }

            if (pending.length > 0) {
                currentActiveSession = pending[0];
                alertCard.style.display = "flex";
                alertDetails.innerHTML = `
                    <div style="margin-bottom:4px;"><span style="color:var(--alert);font-weight:700;">INCIDENT:</span> ${currentActiveSession.sensor_id} (${currentActiveSession.acoustic_signature})</div>
                    <div style="margin-bottom:4px;"><span style="color:var(--info);font-weight:700;">LOCATION:</span> ${currentActiveSession.location} (${currentActiveSession.decibel_level}dB)</div>
                    <div style="margin-bottom:4px;"><span style="color:var(--warning);font-weight:700;">ANALYSIS:</span> ${currentActiveSession.explanation || "Suspected poaching signature."}</div>
                    <div><span style="color:var(--primary-bright);font-weight:700;">FUSION:</span> Wind: ${currentActiveSession.weather?.wind_speed_kmh}km/h | Pop: ${currentActiveSession.human_presence}/10</div>
                `;
                evidenceList.innerHTML = "";
                if (currentActiveSession.top_evidence?.length > 0) {
                    currentActiveSession.top_evidence.forEach(ev => { const li = document.createElement("li"); li.innerText = ev; evidenceList.appendChild(li); });
                } else {
                    evidenceList.innerHTML = "<li>Suspicious signature match</li>";
                }
                if (currentActiveSession.recommended_action?.includes("Ranger")) {
                    approveBtn.innerHTML = `<svg class="svg-icon icon-inline" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg> Divert Ranger`;
                } else {
                    approveBtn.innerHTML = `<svg class="svg-icon icon-inline" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"></circle><path d="M12 12L7 7m10 10l-5-5m0 0l5-5m-5 5l-5 5"></path><circle cx="6" cy="6" r="2"></circle><circle cx="18" cy="18" r="2"></circle><circle cx="18" cy="6" r="2"></circle><circle cx="6" cy="18" r="2"></circle></svg> Deploy Drone`;
                }
                if (currentActiveSession.warnings?.length > 0) {
                    warningsBox.style.display = "block";
                    warningsBox.innerHTML = `<svg class="svg-icon icon-inline" style="color:var(--alert);width:14px;height:14px;vertical-align:middle;margin-right:4px;" viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0zM12 9v4M12 17h.01"></path></svg> Safety Notice: ` + currentActiveSession.warnings.join(" | ");
                } else {
                    warningsBox.style.display = "none";
                }
                highlightMapSector(currentActiveSession.location, "alert");
                highlightTrace("policy", "paused");
            } else {
                currentActiveSession = null;
                alertCard.style.display = "none";
            }
        }

        async function resolveAlert(decision) {
            if (!currentActiveSession) return;
            const targetLoc = currentActiveSession.location;
            const resilienceMode = document.getElementById("resilience-mode").value;
            const body = {
                session_id: currentActiveSession.session_id,
                interrupt_id: currentActiveSession.interrupt_id,
                decision: decision
            };
            highlightTrace("dispatcher", "active");
            highlightMapSector(targetLoc, null);
            const isApproved = decision === "approve";
            updateFlywheelMetrics(isApproved ? "DISPATCHED" : "FALSE_ALARM");
            if (isApproved) {
                const recAction = currentActiveSession.recommended_action || "Deploy Drone";
                if (recAction.includes("Ranger")) {
                    const rl = document.getElementById("ranger-alpha-status");
                    if (rl) { rl.innerText = `Diverted to ${targetLoc}`; rl.style.color = "#7bafd4"; }
                    if (resilienceMode !== "lora") {
                        playSonarPing();
                    }
                    setTimeout(() => { if (rl) { rl.innerText = "Patrolling Elk Prairie"; rl.style.color = ""; } }, 8000);
                } else {
                    animateDrone(targetLoc);
                    if (resilienceMode !== "lora") {
                        playSonarPing();
                    }
                    setTimeout(returnDroneToHQ, 6000);
                }
            } else {
                if (resilienceMode !== "lora") {
                    playSonarPing();
                }
            }
            await fetch("/api/resolve", { 
                method: "POST", 
                headers: { 
                    "Content-Type": "application/json",
                    "X-API-Key": "conservation-secure-key"
                }, 
                body: JSON.stringify(body) 
            });
            setTimeout(() => { loadLogs(); loadPending(); highlightTrace(null); if(!dropIsPlaying) waveformMode = "idle"; }, 2000);
        }

        // Periodic ranger chat simulation
        setInterval(async () => {
            if (!currentActiveSession) {
                try {
                    await fetch("/api/chats/simulate", { method: "POST" });
                    await loadChats();
                } catch(e) {}
            }
        }, 22000);

        // Auto telemetry simulation
        let autoTelemetryInterval = null;
        function toggleAutoTelemetry() {
            const cb = document.getElementById("auto-telemetry-toggle");
            cb.checked = !cb.checked;
            const slider = document.getElementById("auto-telemetry-slider");
            const knob = document.getElementById("auto-telemetry-knob");
            const label = document.getElementById("auto-telemetry-label");

            if (cb.checked) {
                slider.style.background = "var(--primary-bright)";
                knob.style.left = "13px";
                knob.style.background = "var(--bg-base)";
                label.style.color = "var(--primary-bright)";

                triggerAutoEvent();
                autoTelemetryInterval = setInterval(triggerAutoEvent, 16000);
            } else {
                slider.style.background = "rgba(255,255,255,0.1)";
                knob.style.left = "1px";
                knob.style.background = "var(--text-muted)";
                label.style.color = "var(--text-muted)";

                if (autoTelemetryInterval) {
                    clearInterval(autoTelemetryInterval);
                    autoTelemetryInterval = null;
                }
            }
        }

        function triggerAutoEvent() {
            if (currentActiveSession) return;

            const keys = ["bird", "chainsaw", "gunshot", "ranger"];
            const type = keys[Math.floor(Math.random() * keys.length)];
            const template = templates[type];

            const sId = "sensor-live-" + Math.floor(Math.random() * 89 + 10);
            const dbVariance = (Math.random() * 8 - 4);
            const finalDb = Math.max(20, Math.min(120, template.decibel_level + dbVariance)).toFixed(1);

            document.getElementById("sensor_id").value = sId;
            document.getElementById("location").value = template.location;
            document.getElementById("decibel_level").value = finalDb;
            document.getElementById("acoustic_signature").value = template.acoustic_signature;

            waveformMode = type;
            if (type === "gunshot") {
                gunshotDecay = 1.0;
            }

            triggerEvent();
        }

        // Init
        applyTemplate();
        loadLogs();
        loadPending();
        loadChats();
        loadReports();
        toggleResilienceMode();

        // Periodically poll for updates (logs, pending alerts, chats, and reports) to keep UI synchronized in real-time
        setInterval(() => {
            loadLogs();
            loadPending();
            loadChats();
            loadReports();
        }, 4000);

        // Start auto telemetry simulation by default on load
        if (document.getElementById("auto-telemetry-toggle").checked) {
            autoTelemetryInterval = setInterval(triggerAutoEvent, 16000);
            setTimeout(triggerAutoEvent, 1500); // initial trigger
        }
    </script>
</body>
</html>
"""
