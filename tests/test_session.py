import pytest
import time
from edge_hippo.session import SessionManager

def test_session_singleton():
    s1 = SessionManager()
    s2 = SessionManager()
    assert s1 is s2
    assert s1.sessions is s2.sessions

def test_session_lifecycle():
    manager = SessionManager()
    session_id = "test_sess_1"
    
    # 1. New session should be empty
    assert manager.get_context(session_id) == []
    
    # 2. Update context
    entities = ["EntityA", "EntityB"]
    manager.update_context(session_id, entities)
    
    # 3. Verify update
    assert manager.get_context(session_id) == entities
    assert session_id in manager.sessions
    assert manager.sessions[session_id].last_updated > 0

    # 4. Clear or overwrite (simulate drift flush)
    manager.update_context(session_id, [])
    assert manager.get_context(session_id) == []

def test_multiple_sessions():
    manager = SessionManager()
    id1 = "u1"
    id2 = "u2"
    
    manager.update_context(id1, ["A"])
    manager.update_context(id2, ["B"])
    
    assert manager.get_context(id1) == ["A"]
    assert manager.get_context(id2) == ["B"]
