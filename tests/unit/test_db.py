import os
import pytest
from app.db import (
    init_db,
    add_chat,
    get_chats,
    add_incident,
    update_incident,
    get_incidents,
    get_pending_incidents,
    add_report,
    get_reports,
    DB_PATH
)

@pytest.fixture(autouse=True)
def setup_test_db():
    # Back up existing db if exists
    backup_path = DB_PATH + ".bak"
    exists = os.path.exists(DB_PATH)
    if exists:
        if os.path.exists(backup_path):
            os.remove(backup_path)
        os.rename(DB_PATH, backup_path)
        
    init_db()
    yield
    
    # Restore original db
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    if exists and os.path.exists(backup_path):
        os.rename(backup_path, DB_PATH)


def test_database_chats() -> None:
    # Initially empty or pre-populated. Let's add custom chats
    add_chat("Ranger Test", "This is a unit test message.", "2026-06-24T12:00:00Z")
    chats = get_chats()
    assert len(chats) >= 1
    assert chats[-1]["sender"] == "Ranger Test"
    assert chats[-1]["message"] == "This is a unit test message."


def test_database_incidents() -> None:
    log_entry = {
        "session_id": "test-session-123",
        "sensor_id": "sensor-test",
        "location": "Fern Canyon",
        "decibel_level": 75.0,
        "acoustic_signature": "Gunshot",
        "timestamp": "2026-06-24T12:05:00Z",
        "status": "PENDING_DECISION",
        "threat_level": 98,
        "confidence_score": 0.95,
        "top_evidence": ["Evidence 1", "Evidence 2"],
        "recommended_action": "Deploy Drone",
        "explanation": "Gunshot sound peak detected.",
        "warnings": ["Warning 1"],
        "weather": {"wind_speed_kmh": 10.0, "rain": False},
        "human_presence": 1,
        "interrupted": True,
        "interrupt_id": "ranger_dispatch",
        "interrupt_message": "Confirm dispatch?",
        "final_outcome": "Awaiting Confirmation (Deploy Drone)",
        "resilience_mode": "lora"
    }
    
    add_incident(log_entry)
    
    incidents = get_incidents()
    assert len(incidents) == 1
    assert incidents[0]["session_id"] == "test-session-123"
    assert incidents[0]["threat_level"] == 98
    assert incidents[0]["top_evidence"] == ["Evidence 1", "Evidence 2"]
    assert incidents[0]["resilience_mode"] == "lora"
    
    pending = get_pending_incidents()
    assert len(pending) == 1
    assert pending[0]["session_id"] == "test-session-123"
    
    # Update incident
    update_incident("test-session-123", {"status": "DRONE_DISPATCHED", "final_outcome": "Drone Deployed", "interrupted": False})
    
    incidents = get_incidents()
    assert incidents[0]["status"] == "DRONE_DISPATCHED"
    assert incidents[0]["final_outcome"] == "Drone Deployed"
    assert not incidents[0]["interrupted"]
    
    pending = get_pending_incidents()
    assert len(pending) == 0


def test_database_reports() -> None:
    add_report("Illegal_Snare_Trap", "Gold Bluffs", 150.0, 200.0, "Snare found near coordinates.")
    reports = get_reports()
    assert len(reports) == 1
    assert reports[0]["report_type"] == "Illegal_Snare_Trap"
    assert reports[0]["location"] == "Gold Bluffs"
    assert reports[0]["x"] == 150.0
    assert reports[0]["y"] == 200.0
