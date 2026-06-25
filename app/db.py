import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "guardian.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def db_session():
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with db_session() as conn:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS incidents (
                    session_id TEXT PRIMARY KEY,
                    sensor_id TEXT,
                    location TEXT,
                    decibel_level REAL,
                    acoustic_signature TEXT,
                    timestamp TEXT,
                    status TEXT,
                    threat_level INTEGER,
                    confidence_score REAL,
                    top_evidence TEXT,
                    recommended_action TEXT,
                    explanation TEXT,
                    warnings TEXT,
                    weather TEXT,
                    human_presence INTEGER,
                    interrupted INTEGER,
                    interrupt_id TEXT,
                    interrupt_message TEXT,
                    final_outcome TEXT,
                    resilience_mode TEXT
                )
            """)
            try:
                conn.execute("ALTER TABLE incidents ADD COLUMN resilience_mode TEXT")
            except sqlite3.OperationalError:
                pass
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender TEXT,
                    message TEXT,
                    timestamp TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_type TEXT,
                    location TEXT,
                    x REAL,
                    y REAL,
                    details TEXT
                )
            """)

# --- INCIDENT LOGS CRUD ---

def add_incident(entry: dict[str, Any]) -> None:
    with db_session() as conn:
        with conn:
            conn.execute("""
                INSERT OR REPLACE INTO incidents (
                    session_id, sensor_id, location, decibel_level, acoustic_signature,
                    timestamp, status, threat_level, confidence_score, top_evidence,
                    recommended_action, explanation, warnings, weather, human_presence,
                    interrupted, interrupt_id, interrupt_message, final_outcome, resilience_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry["session_id"],
                entry["sensor_id"],
                entry["location"],
                entry["decibel_level"],
                entry["acoustic_signature"],
                entry["timestamp"],
                entry["status"],
                entry["threat_level"],
                entry["confidence_score"],
                json.dumps(entry.get("top_evidence", [])),
                entry["recommended_action"],
                entry["explanation"],
                json.dumps(entry.get("warnings", [])),
                json.dumps(entry.get("weather", {})),
                entry["human_presence"],
                1 if entry.get("interrupted") else 0,
                entry.get("interrupt_id"),
                entry.get("interrupt_message"),
                entry["final_outcome"],
                entry.get("resilience_mode", "primary")
            ))

def update_incident(session_id: str, updates: dict[str, Any]) -> None:
    with db_session() as conn:
        with conn:
            cursor = conn.cursor()
            
            # Build dynamic query based on key fields
            query_parts = []
            params = []
            for key, val in updates.items():
                if key in ["top_evidence", "warnings", "weather"]:
                    query_parts.append(f"{key} = ?")
                    params.append(json.dumps(val))
                elif key in ["interrupted"]:
                    query_parts.append(f"{key} = ?")
                    params.append(1 if val else 0)
                else:
                    query_parts.append(f"{key} = ?")
                    params.append(val)
                    
            params.append(session_id)
            query = f"UPDATE incidents SET {', '.join(query_parts)} WHERE session_id = ?"
            cursor.execute(query, params)

def get_incidents() -> list[dict[str, Any]]:
    with db_session() as conn:
        rows = conn.execute("SELECT * FROM incidents").fetchall()
        result = []
        for row in rows:
            entry = dict(row)
            entry["top_evidence"] = json.loads(entry["top_evidence"] or "[]")
            entry["warnings"] = json.loads(entry["warnings"] or "[]")
            entry["weather"] = json.loads(entry["weather"] or "{}")
            entry["interrupted"] = bool(entry["interrupted"])
            result.append(entry)
        return result

def get_pending_incidents() -> list[dict[str, Any]]:
    with db_session() as conn:
        rows = conn.execute("SELECT * FROM incidents WHERE status = 'PENDING_DECISION'").fetchall()
        result = []
        for row in rows:
            entry = dict(row)
            entry["top_evidence"] = json.loads(entry["top_evidence"] or "[]")
            entry["warnings"] = json.loads(entry["warnings"] or "[]")
            entry["weather"] = json.loads(entry["weather"] or "{}")
            entry["interrupted"] = bool(entry["interrupted"])
            result.append(entry)
        return result

# --- CHATS CRUD ---

def add_chat(sender: str, message: str, timestamp: str) -> None:
    with db_session() as conn:
        with conn:
            conn.execute(
                "INSERT INTO chats (sender, message, timestamp) VALUES (?, ?, ?)",
                (sender, message, timestamp)
            )

def get_chats() -> list[dict[str, Any]]:
    with db_session() as conn:
        rows = conn.execute("SELECT sender, message, timestamp FROM chats ORDER BY id ASC").fetchall()
        return [dict(row) for row in rows]

# --- REPORTS CRUD ---

def add_report(report_type: str, location: str, x: float, y: float, details: str) -> None:
    with db_session() as conn:
        with conn:
            conn.execute(
                "INSERT INTO reports (report_type, location, x, y, details) VALUES (?, ?, ?, ?, ?)",
                (report_type, location, x, y, details)
            )

def get_reports() -> list[dict[str, Any]]:
    with db_session() as conn:
        rows = conn.execute("SELECT report_type, location, x, y, details FROM reports ORDER BY id ASC").fetchall()
        return [dict(row) for row in rows]
