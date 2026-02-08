import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from edge_hippo.hippo_engine import HippoEngine, SearchResult

@pytest.mark.asyncio
async def test_hybrid_ppr_weighting():
    """
    Verify that seeds with higher vector similarity get higher initial mass in PPR.
    
    Scenario:
    - Query: "Q"
    - Node A: Matches "Q" exactly (Sim 1.0) -> High Weight
    - Node B: Matches "Q" via Vector (Sim 0.8) -> Medium Weight
    - Node C: Matches "Q" via Vector (Sim 0.2) -> Low Weight
    
    - Target 1 connected to A
    - Target 2 connected to B
    - Target 3 connected to C
    
    Expectation: Target 1 > Target 2 > Target 3 in ranking.
    """
    engine = HippoEngine()
    engine.storage = AsyncMock()
    # vital: update retriever's storage reference to the new mock
    engine.retriever.storage = engine.storage
    engine.extractor = AsyncMock()
    engine.encoder = MagicMock()
    
    # Mock Models
    import numpy as np
    engine.extractor.extract_entities.return_value = [] # No exact keyword matches for simplicity/control
    engine.encoder.encode.return_value = np.array([0.1]*384)
    
    # Mock Storage
    # Simulate 3 seeds found via vector search
    # Seed A (ID 1): Dist 0.1 (Similar)
    # Seed B (ID 2): Dist 0.5
    # Seed C (ID 3): Dist 1.0
    
    engine.storage.search_vectors.return_value = [
        (1, 0.1),
        (2, 0.5),
        (3, 1.0)
    ]
    
    # Mock Graph Construction
    # Ego subgraph must return these nodes + targets
    # Graph Topology:
    # 1 -> 10 (Target A)
    # 2 -> 20 (Target B)
    # 3 -> 30 (Target C)
    
    mock_subgraph = {
        "nodes": [
            {"id": 1, "type": "phrase", "name": "A", "is_hub": False, "embedding": None},
            {"id": 2, "type": "phrase", "name": "B", "is_hub": False, "embedding": None},
            {"id": 3, "type": "phrase", "name": "C", "is_hub": False, "embedding": None},
            {"id": 10, "type": "passage", "name": "TA", "is_hub": False, "embedding": None},
            {"id": 20, "type": "passage", "name": "TB", "is_hub": False, "embedding": None},
            {"id": 30, "type": "passage", "name": "TC", "is_hub": False, "embedding": None},
        ],
        "edges": [
            (1, 10, 1.0), (10, 1, 1.0),
            (2, 20, 1.0), (20, 2, 1.0),
            (3, 30, 1.0), (30, 3, 1.0)
        ]
    }
    engine.storage.get_ego_subgraph.return_value = mock_subgraph
    engine.storage.get_node_content.return_value = "Content"
    
    # Run Search
    # Check retrieval order by inspecting calls to storage.get_node_content
    await engine.search("query")
    
    # Check call history of get_node_content
    # Expected order: 10, 20, 30
    calls = engine.storage.get_node_content.call_args_list
    assert len(calls) >= 3
    
    # IDs passed to get_node_content
    # calls[0] is first result, etc.
    first_id = calls[0].args[0]
    second_id = calls[1].args[0]
    third_id = calls[2].args[0]
    
    assert first_id == 10, f"Expected Target A (10) first (Seed A dist 0.1), got {first_id}"
    assert second_id == 20, f"Expected Target B (20) second (Seed B dist 0.5), got {second_id}"
    assert third_id == 30, f"Expected Target C (30) third (Seed C dist 1.0), got {third_id}"
