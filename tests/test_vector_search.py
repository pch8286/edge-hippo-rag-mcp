import pytest
import sqlite3
import numpy as np
from edge_hippo.storage import GraphStorage

@pytest.mark.asyncio
async def test_vector_storage_and_search(tmp_path, vector_search_supported):
    if not vector_search_supported:
        pytest.skip("Vector search not supported via built-in SQLite")
    
    storage = GraphStorage()
    # Use valid temp path
    db_file = tmp_path / "test_graph.db"
    storage.db_path = db_file
    
    # Initialize schema
    await storage.initialize()
    
    # Check if table exists (Whitebox test)
    async with storage._get_conn() as db:
        # Check for vec_nodes
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vec_nodes'")
        row = await cursor.fetchone()
        assert row is not None, "Table 'vec_nodes' should exist"

    # Add dummy node with embedding
    embedding = [0.1] * 384
    node_id = await storage.add_node(
        node_type="phrase",
        name="test_concept",
        embedding=embedding
    )
    assert node_id is not None
    
    # Search
    # Exact match query
    results = await storage.search_vectors(query_vec=embedding, top_k=5)
    
    assert len(results) > 0
    found_id, distance = results[0]
    # In sqlite-vec, distance is usually cosine distance? Or L2?
    # vec0 defaults to L2 or Cosine depending on creation? 
    # Actually sqlite-vec default is usually L2 if not specified, 
    # but let's check what we get.
    # With identical vector, distance should be close to 0.
    
    assert found_id == node_id
    assert distance < 0.0001
