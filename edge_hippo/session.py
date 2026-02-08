from typing import Dict, List, Optional
import time
from dataclasses import dataclass, field

@dataclass
class SessionState:
    last_query_entities: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

class SessionManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
            cls._instance.sessions: Dict[str, SessionState] = {}
        return cls._instance

    def get_context(self, session_id: str) -> List[str]:
        """
        Retrieve the last query entities for a given session.
        Returns empty list if session doesn't exist or has no context.
        """
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id].last_query_entities

    def update_context(self, session_id: str, entities: List[str]):
        """
        Update the context for a session with the latest query entities.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState()
        
        self.sessions[session_id].last_query_entities = entities
        self.sessions[session_id].last_updated = time.time()

    def clear_context(self, session_id: str):
        """
        Explicitly clear context for a session (e.g., on topic shift).
        """
        if session_id in self.sessions:
            self.sessions[session_id].last_query_entities = []
            self.sessions[session_id].last_updated = time.time()

session_manager = SessionManager()
