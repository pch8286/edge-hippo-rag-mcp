from typing import List
from .storage import GraphStorage

async def check_drift(storage: GraphStorage, current_entities: List[str], history_entities: List[str]) -> bool:
    """
    Check if there is a topological drift between current and history entities.
    Returns True if DRIFT DETECTED (Disconnected).
    Returns False if CONNECTED.
    """
    if not history_entities:
        return False
        
    if not current_entities:
        return False
    
    current_ids = []
    for name in current_entities:
        nid = await storage.get_node_by_name(name, 'phrase')
        if nid:
            current_ids.append(nid)
            
    history_ids = []
    for name in history_entities:
        nid = await storage.get_node_by_name(name, 'phrase')
        if nid:
            history_ids.append(nid)
            
    if not current_ids or not history_ids:
        return True
        
    is_connected = await storage.check_connectivity(current_ids, history_ids)
    
    return not is_connected
