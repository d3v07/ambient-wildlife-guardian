import pytest
from fastapi import HTTPException
from app.server import verify_api_key, final_outcome_label

def test_verify_api_key():
    # Test valid key
    assert verify_api_key("conservation-secure-key") == "conservation-secure-key"
    
    # Test invalid key raises HTTPException with 403
    with pytest.raises(HTTPException) as excinfo:
        verify_api_key("wrong-key")
    assert excinfo.value.status_code == 403

def test_final_outcome_label():
    assert final_outcome_label(90, True, "Deploy Drone") == "Awaiting Confirmation (Deploy Drone)"
    assert final_outcome_label(30, False) == "Logged: Wildlife Activity"
    assert final_outcome_label(80, False) == "Unescalated Threat"
