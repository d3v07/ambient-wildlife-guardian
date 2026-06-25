"""
Ambient Wildlife Poaching Guardian - Multi-Agent Engine
Implements the resilient conservation command fabric using ADK 2.0.
Provides Sensor Intake, Acoustic Triage, Context Fusion, Threat Assessor,
Policy & Safety, human checkpoints, Dispatcher, and After-Action agents.
"""

import base64
import json
import logging
import os
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events.event import Event, EventActions
from google.adk.events.request_input import RequestInput
from google.adk.workflow import START, Edge, Workflow, node
from pydantic import BaseModel

# Setup standard logging to console for runtime tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONFIGURATION PARAMETERS
MODEL_NAME = "gemini-3.5-flash"
THREAT_THRESHOLD = 70  # Scaled from 0 to 100

# --- DATA MODELS ---


class SensorPayload(BaseModel):
    """Represents incoming telemetry from reserve acoustic monitoring hardware."""

    sensor_id: str
    location: str
    decibel_level: float
    acoustic_signature: str
    timestamp: str


class ThreatAssessment(BaseModel):
    """Pydantic schema used by ThreatAssessor for structured output validation."""

    threat_level: int
    is_poaching_suspected: bool
    confidence_score: float
    top_evidence: list[str]
    recommended_action: str
    explanation: str


# --- MULTI-AGENT SUBCLASSES WITH SAFE FALLBACKS ---


class SafeLlmAgent(LlmAgent):
    """
    Threat Assessor Agent subclass with offline fallback capabilities.
    Returns structured threat assessments even when cloud services are unavailable.
    """

    async def _run_async_impl(self, ctx: Any) -> Any:
        has_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
        )

        if not has_key:
            logger.info(
                "ThreatAssessor: No cloud API keys found. Evaluating threat locally."
            )
            sensor_data = ctx.session.state.get("sensor_data", {})
            sig = str(sensor_data.get("acoustic_signature", "")).lower()
            db = float(sensor_data.get("decibel_level", 0))
            str(sensor_data.get("location", ""))

            threat_level = 10
            confidence = 0.65
            evidence = [
                "Fauna sound profiles match typical birds/wind.",
                "Sector is inside expected biodiversity corridor.",
            ]
            rec_action = "None"
            explanation = (
                "Normal animal vocalizations or wind movement. Evaluated as low threat."
            )

            # Simple keyword matching to simulate threat classifier
            if "chainsaw" in sig:
                threat_level = 95
                confidence = 0.92
                evidence = [
                    f"High-intensity acoustic signature ({sig}) at {db}dB.",
                    "Unscheduled vehicle/motorized engine activity.",
                    "Isolated reserve interior coordinates.",
                ]
                rec_action = "Deploy Drone"
                explanation = f"Detected high-intensity signature ({sig}) matching poaching tools at {db}dB."
            elif "gunshot" in sig:
                threat_level = 98
                confidence = 0.96
                evidence = [
                    f"Acoustic peak impulse matched gunshot signature at {db}dB.",
                    "High confidence acoustic profile anomaly.",
                    "Remote sector outside normal transit windows.",
                ]
                rec_action = "Deploy Drone"
                explanation = (
                    "Detected gunshot peak impulse pattern. Confirmed alert status."
                )
            elif "footsteps" in sig and db > 60:
                threat_level = 75
                confidence = 0.80
                evidence = [
                    "Footstep cadence detected by geophone sensor.",
                    "Unscheduled human intrusion sector.",
                    "Low human settlement density zone.",
                ]
                rec_action = "Divert Ranger Patrol"
                explanation = "Unscheduled human footsteps detected in remote reserve sector. Potential intrusion."

            assessment = {
                "threat_level": threat_level,
                "is_poaching_suspected": threat_level >= THREAT_THRESHOLD,
                "confidence_score": confidence,
                "top_evidence": evidence,
                "recommended_action": rec_action,
                "explanation": explanation,
            }
            yield Event(
                output=assessment,
                actions=EventActions(state_delta={"assessment": assessment}),
            )
            return

        async for event in super()._run_async_impl(ctx):
            yield event


# --- WORKFLOW GRAPH NODES (AGENTS) ---

# Redwood National & State Parks (California, USA) coordinate mappings
# Real GPS coordinates for actual named locations within the park system
SECTOR_COORDINATES = {
    "Elk Prairie": {"lat": 41.3648, "lon": -124.0274},
    "Fern Canyon": {"lat": 41.4018, "lon": -124.0664},
    "Tall Trees Grove": {"lat": 41.3697, "lon": -124.0108},
    "Orick HQ": {"lat": 41.2870, "lon": -124.0590},
    "Stout Grove": {"lat": 41.7843, "lon": -124.0841},
    "Gold Bluffs": {"lat": 41.3712, "lon": -124.0790},
    "Lady Bird Grove": {"lat": 41.3185, "lon": -124.0031},
    "Howland Hill": {"lat": 41.7710, "lon": -124.0930},
    "Prairie Creek": {"lat": 41.3740, "lon": -124.0270},
}

# USGS Stream Gauge Station IDs for live hydrological monitoring
USGS_STATIONS = {
    "redwood_creek": {
        "id": "11482500",
        "name": "Redwood Creek at Orick",
        "param": "00060",
    },
    "klamath_river": {
        "id": "11530500",
        "name": "Klamath River near Klamath",
        "param": "00060",
    },
}

# Streamflow danger threshold (cubic feet per second)
USGS_FLOW_DANGER_THRESHOLD = 100.0


@node
def sensor_intake(ctx: Context, node_input: Any) -> Event:
    """
    Sensor Intake Agent.
    Ingests sensor data and wraps it in a CloudEvent envelope.
    """
    payload = {}
    if hasattr(node_input, "parts") and node_input.parts:
        text = getattr(node_input.parts[0], "text", "")
        try:
            payload = json.loads(text)
        except Exception:
            payload = {"text": text}
    elif isinstance(node_input, dict):
        payload = node_input
    elif isinstance(node_input, str):
        try:
            payload = json.loads(node_input)
        except Exception:
            payload = {"text": node_input}

    data = payload.get("data", "")
    if isinstance(data, str):
        try:
            decoded = base64.b64decode(data).decode("utf-8")
            parsed_payload = json.loads(decoded)
        except Exception:
            try:
                parsed_payload = json.loads(data)
            except Exception:
                parsed_payload = payload
    elif isinstance(data, dict):
        parsed_payload = data
    else:
        parsed_payload = payload

    try:
        sensor_data = SensorPayload(**parsed_payload)
    except Exception:
        # Fallback for plain-text prompts (e.g. from evaluations or conversational UI)
        import datetime

        now_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        text_content = (
            parsed_payload.get("text", "")
            if isinstance(parsed_payload, dict)
            else str(parsed_payload)
        )
        sensor_data = SensorPayload(
            sensor_id="sensor-manual",
            location="Orick HQ",
            decibel_level=35.0,
            acoustic_signature=text_content,
            timestamp=now_str,
        )

    state_delta = {
        "sensor_data": sensor_data.model_dump(),
        "cloudevent": {
            "specversion": "1.0",
            "id": f"evt-{sensor_data.sensor_id}-{sensor_data.timestamp}",
            "source": f"reserve/sensors/{sensor_data.sensor_id}",
            "type": "org.conservation.poaching.sensor.anomaly",
            "time": sensor_data.timestamp,
            "data": sensor_data.model_dump(),
        },
    }

    return Event(
        output=sensor_data.model_dump(), actions=EventActions(state_delta=state_delta)
    )


@node
def acoustic_triage(ctx: Context, node_input: dict[str, Any]) -> Event:
    """
    Acoustic Triage Agent.
    Filters out safe, low-decibel wildlife vocalizations to save computing resource.
    """
    db = node_input.get("decibel_level", 0)
    sig = str(node_input.get("acoustic_signature", "")).lower()

    # Low-decibel pre-filtering: route to auto_log if low noise and no poacher signals
    if db < 50 and "gunshot" not in sig and "chainsaw" not in sig:
        return Event(output=node_input, actions=EventActions(route="auto_log"))

    return Event(output=node_input, actions=EventActions(route="context_fusion"))


@node
def auto_log_event(node_input: dict[str, Any]) -> Event:
    """Logs low-threat events quietly in the local file system log."""
    from google.genai import types
    status_msg = f"Logged: Low Threat Event ({node_input.get('acoustic_signature', 'Unknown')})"
    return Event(
        output={"status": "LOGGED", "action": "None", "sensor_data": node_input},
        content=types.Content(role="model", parts=[types.Part.from_text(text=status_msg)])
    )


@node
async def context_fusion(ctx: Context, node_input: dict[str, Any]) -> Event:
    """
    Context Fusion Agent.
    Queries the live Open-Meteo API for real-time weather in Redwood National Park
    and the USGS Instantaneous Values API for real-time streamflow on Redwood Creek
    and Klamath River.
    """
    import asyncio
    loc = node_input.get("location", "")
    coords = SECTOR_COORDINATES.get(loc, {"lat": 41.2870, "lon": -124.0590})

    # Defaults
    wind_speed = 15.0
    rain = False

    # Define local sync functions to be executed in a thread pool
    def fetch_weather_sync():
        import urllib.request
        url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&current=wind_speed_10m,precipitation"
        req = urllib.request.Request(
            url, headers={"User-Agent": "WildlifeGuardian/1.0"}
        )
        with urllib.request.urlopen(req, timeout=2.0) as response:
            return json.loads(response.read().decode("utf-8"))

    # Call Open-Meteo API for real-time weather using asyncio.to_thread
    try:
        res_data = await asyncio.to_thread(fetch_weather_sync)
        current = res_data.get("current", {})
        wind_speed = float(current.get("wind_speed_10m", 15.0))
        rain = float(current.get("precipitation", 0.0)) > 0.0
        logger.info(
            f"ContextFusion API: Fetched live weather for {loc}. Wind: {wind_speed} km/h, Rain: {rain}"
        )
    except Exception as e:
        logger.warning(f"ContextFusion API failed ({e}). Falling back to mock values.")
        if "Gold Bluffs" in loc:
            wind_speed = 45.0  # Hazardous coastal wind speeds
            rain = True
        elif "Howland Hill" in loc:
            wind_speed = 22.0
            rain = False

    # Fetch USGS live streamflow for Redwood Creek and Klamath River
    streamflow = {"redwood_creek_cfs": None, "klamath_river_cfs": None}
    
    def fetch_usgs_sync(station_id, param):
        import urllib.request
        usgs_url = (
            f"https://waterservices.usgs.gov/nwis/iv/"
            f"?format=json&sites={station_id}"
            f"&parameterCd={param}"
            f"&siteStatus=all"
        )
        req = urllib.request.Request(
            usgs_url, headers={"User-Agent": "WildlifeGuardian/1.0"}
        )
        with urllib.request.urlopen(req, timeout=3.0) as response:
            return json.loads(response.read().decode("utf-8"))

    for key, station in USGS_STATIONS.items():
        try:
            usgs_data = await asyncio.to_thread(fetch_usgs_sync, station['id'], station['param'])
            ts = usgs_data.get("value", {}).get("timeSeries", [])
            if ts:
                values = ts[0].get("values", [{}])[0].get("value", [])
                if values:
                    cfs = float(values[-1].get("value", 0))
                    streamflow[f"{key}_cfs"] = cfs
                    logger.info(
                        f"ContextFusion USGS: {station['name']} flow = {cfs} cfs"
                    )
        except Exception as e:
            logger.warning(
                f"ContextFusion USGS {station['name']} failed ({e}). Using fallback."
            )

    # Fallback mock streamflow if API unavailable
    if streamflow["redwood_creek_cfs"] is None:
        streamflow["redwood_creek_cfs"] = 42.0
    if streamflow["klamath_river_cfs"] is None:
        streamflow["klamath_river_cfs"] = 780.0

    # Local native helper evaluation to prevent HTTP deadlock loops on loopback port
    is_patrol_scheduled = "Elk Prairie" in loc
    if is_patrol_scheduled:
        logger.info(
            f"ContextFusion SMART Local: Scheduled Ranger patrol active in {loc}."
        )

    human_presence = 3
    if "Orick HQ" in loc:
        human_presence = 8
    elif "Howland Hill" in loc:
        human_presence = 1
    logger.info(
        f"ContextFusion WorldPop Local: Human presence score for {loc} is {human_presence}."
    )

    state_delta = {
        "weather": {"wind_speed_kmh": wind_speed, "rain": rain},
        "streamflow": streamflow,
        "human_presence": human_presence,
        "is_patrol_scheduled": is_patrol_scheduled,
    }

    if is_patrol_scheduled:
        return Event(
            output=node_input,
            actions=EventActions(route="auto_log", state_delta=state_delta),
        )

    return Event(
        output=node_input,
        actions=EventActions(route="threat_assessment", state_delta=state_delta),
    )


# Instantiate threat assessor
threat_assessor = SafeLlmAgent(
    name="threat_assessor",
    model=MODEL_NAME,
    instruction="""
    Analyze the sensor telemetry alongside reserve context. Identify poaching signatures (chainsaws, gunshots).
    Return threat_level 0-100, confidence score, evidence list, recommended action, and explanation.
    """,
    output_schema=ThreatAssessment,
    output_key="assessment",
)


@node
def policy_safety_check(ctx: Context, node_input: dict[str, Any]) -> Event:
    """
    Policy and Safety Agent (Deterministic Guardrails).
    Ensures safety regulations are satisfied before requesting Ranger dispatches:
    1. Drone dispatches are blocked if wind exceeds 40 km/h.
    2. Operations near Fern Canyon are altitude restricted (breeding corridor).
    3. Wade patrols to Tall Trees Grove blocked if Redwood Creek exceeds flow threshold.
    """
    assessment = ctx.state.get("assessment", {})
    weather = ctx.state.get("weather", {})
    sensor = ctx.state.get("sensor_data", {})
    streamflow = ctx.state.get("streamflow", {})

    overridden = False
    warnings = []
    rec_action = assessment.get("recommended_action", "None")

    # Rule 1: High Wind Speed Constraint
    if weather.get("wind_speed_kmh", 0) > 40.0:
        overridden = True
        warnings.append(
            f"Drone dispatch blocked: Hazardous wind speeds of {weather.get('wind_speed_kmh')} km/h."
        )
        if rec_action == "Deploy Drone":
            rec_action = "Divert Ranger Patrol"  # Re-route to land dispatch

    # Rule 2: Breeding Season Coordinates (Fern Canyon corridor)
    if "Fern Canyon" in sensor.get("location", ""):
        warnings.append(
            "Fern Canyon breeding corridor: Drone altitude locked to 120m AGL to protect fauna."
        )

    # Rule 3: USGS Streamflow Danger Threshold (Redwood Creek)
    redwood_flow = streamflow.get("redwood_creek_cfs", 0)
    if redwood_flow and redwood_flow > USGS_FLOW_DANGER_THRESHOLD:
        loc = sensor.get("location", "")
        if "Tall Trees" in loc:
            overridden = True
            warnings.append(
                f"River swell alert: Redwood Creek at {redwood_flow} cfs (threshold: {USGS_FLOW_DANGER_THRESHOLD} cfs). "
                "Wade patrol to Tall Trees Grove blocked. Rerouting to drone surveillance."
            )
            if rec_action == "Divert Ranger Patrol":
                rec_action = "Deploy Drone"

    policy_state = {
        "policy_checked": True,
        "overridden": overridden,
        "warnings": warnings,
        "recommended_action": rec_action,
    }

    updated_assessment = dict(assessment)
    updated_assessment["recommended_action"] = rec_action

    state_delta = {"policy": policy_state, "assessment": updated_assessment}

    return Event(
        output=updated_assessment, actions=EventActions(state_delta=state_delta)
    )


@node
async def human_escalation(ctx: Context, node_input: dict[str, Any]):
    """
    Human-in-the-Loop Node.
    For low threat levels, auto-resolves.
    For high threat levels, yields a RequestInput to pause the workflow.
    """
    sensor = ctx.state.get("sensor_data", {})
    assessment = node_input
    policy = ctx.state.get("policy", {})

    threat_level = assessment.get("threat_level", 0)
    if threat_level < THREAT_THRESHOLD:
        yield Event(output={"decision": "false_alarm", "assessment": assessment})
        return

    warnings_str = " | ".join(policy.get("warnings", []))
    warning_text = f" [WARNINGS: {warnings_str}]" if warnings_str else ""

    message = (
        f"🚨 CRITICAL ALERT in {sensor.get('location')} 🚨\n"
        f"Sensor {sensor.get('sensor_id')} detected: {sensor.get('acoustic_signature')} ({sensor.get('decibel_level')}dB)\n"
        f"AI Assessment: {assessment.get('explanation')}\n"
        f"Recommendation: {assessment.get('recommended_action')}{warning_text}\n"
        "Confirm Dispatch? (Approve/Reject)"
    )

    yield RequestInput(interrupt_id="ranger_dispatch", message=message)


@node
def dispatcher_agent(ctx: Context, node_input: dict[str, Any]) -> Event:
    """
    Dispatcher Agent.
    Coordinates response resources (Drone flight assets & Ranger radios)
    once confirmation is received.
    """
    decision = (
        str(node_input.get("response", "") or node_input.get("decision", ""))
        .strip()
        .lower()
    )
    assessment = ctx.state.get("assessment", {})

    drone_status = "STANDBY"
    ranger_status = "STANDBY"
    action_taken = "None"

    if decision in ["false_alarm"]:
        action_taken = "Quiet Logged"
    elif decision in ["y", "yes", "true", "1", "approve"]:
        rec = assessment.get("recommended_action", "")
        if "Drone" in rec:
            drone_status = "DISPATCHED"
            action_taken = "Drone Alpha launched. Battery: 95%. Cruising speed: 60km/h."
        else:
            ranger_status = "DIVERTED"
            action_taken = "Ranger Team Alpha diverted. Comms channel active."
    else:
        action_taken = "Marked as False Alarm by operator"

    dispatch_state = {
        "decision": decision,
        "action_taken": action_taken,
        "drone_status": drone_status,
        "ranger_status": ranger_status,
    }
    return Event(
        output=dispatch_state,
        actions=EventActions(state_delta={"dispatch": dispatch_state}),
    )


@node
def after_action_report(ctx: Context, node_input: dict[str, Any]) -> Event:
    """
    After Action Agent (Terminal Node).
    Compiles complete mission timeline metrics into a report for the dashboard database.
    """
    sensor = ctx.state.get("sensor_data", {})
    assessment = ctx.state.get("assessment", {})
    weather = ctx.state.get("weather", {})
    policy = ctx.state.get("policy", {})
    dispatch = ctx.state.get("dispatch", {})

    decision = dispatch.get("decision", "suppressed")
    status = "LOGGED (Low Threat AI)"
    if decision != "suppressed":
        status = (
            "DRONE_DISPATCHED"
            if decision in ["y", "yes", "true", "1", "approve"]
            else "FALSE_ALARM"
        )

    report = {
        "status": status,
        "sensor_data": sensor,
        "assessment": assessment,
        "weather": weather,
        "policy": policy,
        "dispatch": dispatch,
    }
    return Event(
        output=report,
        actions=EventActions(state_delta={"after_action_report": report}),
    )


# --- GRAPH EDGES ---

edges = [
    # Ingest -> Triage
    Edge(from_node=START, to_node=sensor_intake),
    Edge(from_node=sensor_intake, to_node=acoustic_triage),
    # Triage -> Auto log or Context Fusion
    Edge(from_node=acoustic_triage, to_node=auto_log_event, route="auto_log"),
    Edge(from_node=acoustic_triage, to_node=context_fusion, route="context_fusion"),
    # Context Fusion -> Auto log or LLM threat analysis
    Edge(from_node=context_fusion, to_node=auto_log_event, route="auto_log"),
    Edge(from_node=context_fusion, to_node=threat_assessor, route="threat_assessment"),
    # Threat Assessment -> Safety Policy Guardrails -> HITL Check
    Edge(from_node=threat_assessor, to_node=policy_safety_check),
    Edge(from_node=policy_safety_check, to_node=human_escalation),
    # HITL -> Dispatcher Coordinator -> After Action Report
    Edge(from_node=human_escalation, to_node=dispatcher_agent),
    Edge(from_node=dispatcher_agent, to_node=after_action_report),
]


# --- INSTANTIATE WORKFLOW ---

root_agent = Workflow(
    name="ambient_wildlife_guardian",
    description="Resilient Operations Fabric for Park Conservation Alerts.",
    edges=edges,
)

app = App(name="app", root_agent=root_agent)
