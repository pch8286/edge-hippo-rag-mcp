
import asyncio
import sys
import os
import logging
from unittest.mock import MagicMock, AsyncMock

# Ensure we can import edge_hippo
sys.path.append(os.getcwd())

from edge_hippo.hippo_engine import HippoEngine
from edge_hippo.session import SessionManager
from edge_hippo.storage import GraphStorage
from edge_hippo.extraction import EntityExtractor
from edge_hippo.retrieval import PPRRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_e2e_sim")

async def run_simulation():
    logger.info("Starting E2E Simulation (Mocked Storage)...")
    
    # 1. Setup Mocks
    mock_storage = MagicMock(spec=GraphStorage)
    mock_storage.initialize = AsyncMock()
    mock_storage.add_node = AsyncMock(return_value=1)
    mock_storage.add_edge = AsyncMock()
    mock_storage.get_node_by_name = AsyncMock(return_value=10) # Concept found
    # Mock retrieval logic calls
    mock_storage.search_vectors = AsyncMock(return_value=[(10, 0.9)])
    mock_storage.get_node_content = AsyncMock(return_value="Valid Content")
    
    # Mock Graph Structure for PPR
    nodes_data = [
        {"id": 10, "name": "Python", "type": "phrase", "is_hub": 0},
        {"id": 20, "name": "Passage1", "type": "passage", "content": "Python is a language...", "metadata": "{}"}
    ]
    edges_data = [{"source": 10, "target": 20, "weight": 1.0}]
    mock_storage.get_ego_subgraph = AsyncMock(return_value={"nodes": nodes_data, "edges": edges_data})

    mock_extractor = MagicMock(spec=EntityExtractor)
    mock_extractor.extract_entities = AsyncMock(return_value=[{"text": "Python", "label": "tech"}])
    mock_extractor.load_model = MagicMock()

    # 2. Initialize Engine with Mocks
    # We need to manually set components because __init__ creates them
    engine = HippoEngine()
    engine.storage = mock_storage
    engine.extractor = mock_extractor
    
    # Mock Retriever explicitly if needed, but let's try to let engine instantiate it using our mocks
    # engine.initialize() creates self.retriever = PPRRetriever(self.extractor, self.storage)
    # This should work fine with our mock storage/extractor!
    
    # We need to patch SentenceTransformer creation in initialize if it happens
    # But initialize() only calls self.extractor.load_model() and self.storage.initialize() 
    # and sets up self.encoder via SentenceTransformer.
    
    mock_encoder = MagicMock()
    mock_encoder.encode.return_value = [0.1]*384
    
    # Patch SentenceTransformer globally for this script scope logic? 
    # Hard to patch contextually without unittest.mock.patch
    # So we'll just mock engine.encoder manually after initialize?
    # No, initialize() will fail if we don't patch it.
    
    # Let's bypass engine.initialize() logic for encoder creation by mocking it?
    # Or just wrap in a patch context.
    from unittest.mock import patch
    with patch("sentence_transformers.SentenceTransformer", return_value=mock_encoder):
        await engine.initialize()

    session_id = "sim_user"
    
    # 3. Turn 1: Query "Python"
    logger.info(f"--- Turn 1: Query 'Python' (Session: {session_id}) ---")
    
    # engine.search calls retriever.search
    # retriever.search calls extract_entities, get_node_by_name, get_ego_subgraph
    
    res1 = await engine.search("Tell me about Python", session_id)
    print("Result 1:", res1)
    
    # Verify Context Update
    # engine.search uses the global session_manager imported in hippo_engine.py
    # We can access the singleton via SessionManager()
    manager = SessionManager()
    ctx1 = manager.get_context(session_id)
    logger.info(f"Context after Turn 1: {ctx1}")
    
    if "Python" in ctx1:
        logger.info("[SUCCESS] Context updated with 'Python'.")
    else:
        logger.error("[FAILURE] Context not updated.")

    # 4. Turn 2: Query "Pandas" (Context Check)
    logger.info(f"--- Turn 2: Query 'Pandas' (Session: {session_id}) ---")
    
    # Update mock extractor to return Pandas for 2nd query
    mock_extractor.extract_entities = AsyncMock(return_value=[{"text": "Pandas", "label": "tech"}])
    mock_storage.get_node_by_name = AsyncMock(return_value=11) # Pandas found
    
    # We need subgraph to contain Pandas (11) -> Passage2
    # But engine.search calls retrieval logic which considers History (Python=10).
    # So both 10 and 11 will be seeds.
    
    # retrieval.py: 
    # current_entities = ["Pandas"]
    # history_entities = ["Python"]
    # check_drift -> Connected?
    # We need to ensure check_drift returns True/False as desired.
    # Logic: shortest_paths. 
    # If we mock storage.get_shortest_paths? No, it uses igraph on subgraph or global graph?
    # algorithms.check_drift uses storage.get_shortest_paths if implemented? 
    # No, check_drift builds a graph from storage?
    # Actually check_drift in algorithms.py takes (storage, current, history). 
    # It calls storage.get_shortest_paths presumably?
    
    # If algorithm uses real graph logic, we might need a real graph or mock the drift function.
    
    # Let's mock check_drift to return False (Connected)
    with patch("edge_hippo.retrieval.check_drift", new_callable=AsyncMock) as mock_drift:
        mock_drift.return_value = False # Connected
        
        # update subgraph to return nodes for both
        nodes_data_2 = [
            {"id": 10, "name": "Python", "type": "phrase"},
            {"id": 11, "name": "Pandas", "type": "phrase"},
            {"id": 21, "name": "Passage2", "type": "passage", "content": "Pandas is a Python lib..."}
        ]
        edges_data_2 = [
            {"source": 10, "target": 21, "weight": 0.5},
            {"source": 11, "target": 21, "weight": 1.0}
        ]
        
        mock_storage.get_ego_subgraph = AsyncMock(return_value={"nodes": nodes_data_2, "edges": edges_data_2})
        
        res2 = await engine.search("Pandas", session_id)
        print("Result 2:", res2)
        
        # Verify Context has both?
        ctx2 = manager.get_context(session_id)
        logger.info(f"Context after Turn 2: {ctx2}")
        
        # Session should accumulate: Python, Pandas?
        # get_context returns list. 
        # Logic: manager.update_context(session_id, current_entities)
        # It replaces context? 
        # SessionManager.update_context:
        # self.sessions[session_id].context_entities = entities
        
        # retrieval.py Logic:
        # history_entities = session_manager.get_context(session_id)
        # ... logic ...
        # current_entities = extractor.extract(...)
        # ...
        # session_manager.update_context(session_id, current_entities)  <-- It stores CURRENT entities only?
        # Wait, if it only stores current entities, then history is only 1 turn deep?
        # Yes, SessionManager usually stores the *last turn's* entities.
        
        if "Pandas" in ctx2:
             logger.info("[SUCCESS] Context updated to 'Pandas'.")


if __name__ == "__main__":
    asyncio.run(run_simulation())
