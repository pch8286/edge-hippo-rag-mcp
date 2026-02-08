import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from edge_hippo.hippo_engine import HippoEngine

@pytest.mark.asyncio
async def test_synonym_optimization(tmp_path, vector_search_supported):
    if not vector_search_supported:
        pytest.skip("Vector search not supported in this environment")

    # Setup Engine with Temp DB
    engine = HippoEngine()
    db_file = tmp_path / "test_consolidation.db"
    engine.storage.db_path = db_file
    
    # Initialize schema
    await engine.initialize()
    
    # Mock search_vectors to return a match for a specific node
    # We need to populate DB with nodes first
    
    # Add Node A
    # create vector A
    emb_a = [0.1] * 384
    id_a = await engine.storage.add_node("phrase", "Node A", embedding=emb_a)
    
    # Add Node B
    emb_b = [0.11] * 384 # slightly different
    id_b = await engine.storage.add_node("phrase", "Node B", embedding=emb_b)
    
    # We want optimize_synonyms to find Node B when querying for Node A
    # Mock search_vectors call inside optimize_synonyms
    # optimize_synonyms iterates all phrases.
    # When filtering A: it queries vector(A). We want it to find B with distance < threshold.
    
    # Since we are using real DB + sqlite-vec, we might not need to mock if we can rely on proper vector search.
    # But for unit test stability given random vectors, let's mock the `search_vectors` method of storage?
    # Or just trust sqlite-vec works (we verified in test_vector_search).
    
    # Let's rely on integration test capability.
    # Insert two very similar vectors.
    
    # Need to be careful with packing.
    # If sqlite-vec is working, exact duplicate vector should have dist 0.
    
    # Add Node C (identical to A)
    id_c = await engine.storage.add_node("phrase", "Node C", embedding=emb_a)
    
    # Run optimization
    # Threshold 0.1 (very strict)
    # Distance between A and C should be 0.
    # They should be linked.
    
    links = await engine.optimize_synonyms(threshold=0.1)
    
    assert links >= 1, "Should have linked Node A and Node C"
    
    # Verify Edge
    # We don't have get_edge method, but we can query raw edges or use get_ego_subgraph
    subgraph = await engine.storage.get_ego_subgraph([id_a], depth=1)
    edges = subgraph['edges']
    
    # Look for edge (id_a, id_c) or (id_c, id_a)
    found = False
    for u, v, w in edges:
        if (u == id_a and v == id_c) or (u == id_c and v == id_a):
            found = True
            break
            
    assert found, "Edge A <-> C should exist"
