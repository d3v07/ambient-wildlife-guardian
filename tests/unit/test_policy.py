import pytest
from app.agent import policy_safety_check

class MockContext:
    def __init__(self, state):
        self.state = state

def test_policy_high_wind_speed() -> None:
    # Rule 1: High Wind Speed override (Drone -> Ranger)
    state = {
        "assessment": {
            "threat_level": 95,
            "recommended_action": "Deploy Drone",
            "explanation": "Chainsaw revving detected."
        },
        "weather": {
            "wind_speed_kmh": 45.0, # Exceeds 40km/h limit
            "rain": False
        },
        "sensor_data": {
            "location": "Howland Hill"
        },
        "streamflow": {
            "redwood_creek_cfs": 30.0
        }
    }
    
    ctx = MockContext(state)
    event = policy_safety_check._func(ctx, state["assessment"])
    
    assert event.output["recommended_action"] == "Divert Ranger Patrol"
    assert event.actions.state_delta["policy"]["overridden"] is True
    assert any("Drone dispatch blocked: Hazardous wind speeds" in w for w in event.actions.state_delta["policy"]["warnings"])


def test_policy_fern_canyon_corridor() -> None:
    # Rule 2: Fern Canyon Corridor breeding lock altitude lock
    state = {
        "assessment": {
            "threat_level": 95,
            "recommended_action": "Deploy Drone",
            "explanation": "Chainsaw revving detected."
        },
        "weather": {
            "wind_speed_kmh": 10.0,
            "rain": False
        },
        "sensor_data": {
            "location": "Fern Canyon"
        },
        "streamflow": {
            "redwood_creek_cfs": 30.0
        }
    }
    
    ctx = MockContext(state)
    event = policy_safety_check._func(ctx, state["assessment"])
    
    assert event.actions.state_delta["policy"]["overridden"] is False
    assert any("Fern Canyon breeding corridor: Drone altitude locked" in w for w in event.actions.state_delta["policy"]["warnings"])


def test_policy_streamflow_danger() -> None:
    # Rule 3: USGS streamflow danger (Ranger -> Drone)
    state = {
        "assessment": {
            "threat_level": 75,
            "recommended_action": "Divert Ranger Patrol",
            "explanation": "Footsteps detected."
        },
        "weather": {
            "wind_speed_kmh": 10.0,
            "rain": False
        },
        "sensor_data": {
            "location": "Tall Trees Grove"
        },
        "streamflow": {
            "redwood_creek_cfs": 120.0 # Exceeds 100.0 cfs danger threshold
        }
    }
    
    ctx = MockContext(state)
    event = policy_safety_check._func(ctx, state["assessment"])
    
    assert event.output["recommended_action"] == "Deploy Drone"
    assert event.actions.state_delta["policy"]["overridden"] is True
    assert any("River swell alert: Redwood Creek" in w for w in event.actions.state_delta["policy"]["warnings"])
