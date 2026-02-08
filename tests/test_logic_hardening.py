import pytest
import pytest_asyncio
import aiosqlite
from edge_hippo.hippo_engine import HippoEngine
from edge_hippo.storage import GraphStorage
from edge_hippo.config import settings

@pytest_asyncio.fixture
async def engine(tmp_path):
    eng = HippoEngine()
    eng.storage.db_path = str(tmp_path / "test_logic.db")
    await eng.initialize()
    return eng

@pytest.mark.asyncio
async def test_Hub_Explosion_Prevention(engine):
    """
    Test that the Ego-Graph extraction stops at Hub Nodes and respects Hard Limit.
    """
    # 1. Setup: Create a Hub Node ("Python") and many connections
    #    Passage_0 -> Python -> [1000 sub-nodes]
    #    If we query Passage_0, depth 2 should theoretically hit 1000 nodes.
    #    But if Python is a HUB, it should NOT traverse *through* it (Conditional Recursion).
    
    # Manually insert nodes/edges to simulate state
    store = engine.storage
    
    # Create Seed Passage
    p_seed_id = await store.add_node("passage", "p_seed", "contains python")
    
    # Create Hub Node "Python"
    hub_id = await store.add_node("phrase", "Python")
    
    # Flag as Hub manually
    async with store._get_conn() as db:
        await db.execute("UPDATE nodes SET is_hub = 1 WHERE id = ?", (hub_id,))
        await db.commit()
    
    # Connect Seed <-> Hub
    await store.add_edge(p_seed_id, hub_id)
    await store.add_edge(hub_id, p_seed_id)
    
    # Create 50 neighbors for Hub (small scale explosion)
    # If recursion barrier works, we should NOT see these in local subgraph of p_seed 
    # (because path is p_seed -> Hub -> neighbor, depth 2).
    # If barrier fails, we see them.
    for i in range(50):
        n_id = await store.add_node("phrase", f"lib_{i}")
        await store.add_edge(hub_id, n_id)
        await store.add_edge(n_id, hub_id)
        
    # 2. Execute: Get Subgraph for Seed
    subgraph = await store.get_ego_subgraph([p_seed_id], depth=2)
    nodes = subgraph['nodes']
    node_ids = {n['id'] for n in nodes}
    
    # 3. Assert: Hub should be present, but its neighbors should be BLOCKED
    assert hub_id in node_ids, "Hub itself should be reachable as direct neighbor"
    
    # Check if neighbors leaked
    neighbor_count = sum(1 for n in nodes if n['name'].startswith("lib_"))
    
    # Current behavior (Failure expected until fix): Neighbors are fetched.
    # Desired behavior: Neighbor count should be 0 (Stop traversal at Hub).
    # We write assertions for DESIRED behavior to fail first (TDD).
    assert neighbor_count == 0, f"Recursion leaked through Hub! Found {neighbor_count} neighbors."

@pytest.mark.asyncio
async def test_Synonym_Linking(engine, vector_search_supported):
    """
    Test that optimize_synonyms links disconnected nodes with similar embeddings.
    """
    if not vector_search_supported:
        pytest.skip("Synonym linking requires vector search")

    store = engine.storage
    
    # 1. Create two synonymous nodes manually (Phrase A and Phrase B)
    # We provide manual embeddings to ensure they are close.
    # sqlite-vec uses 384 dims by default.
    vec_a = [0.0] * 384
    vec_a[0] = 1.0 # Unit vector on axis 0
    
    vec_b = [0.0] * 384
    vec_b[0] = 0.99
    vec_b[1] = 0.1 # Slightly different
    # Distance approx sqrt((0.01)^2 + (0.1)^2) = sqrt(0.0001 + 0.01) = sqrt(0.0101) ~= 0.1
    # Threshold 0.55 should catch this.
    
    id_a = await store.add_node("phrase", "RPi 5", embedding=vec_a)
    id_b = await store.add_node("phrase", "Raspberry Pi 5", embedding=vec_b)
    
    # Verify no edge initially
    edges_before = await store.get_all_edges()
    assert len(edges_before) == 0
    
    # 2. Run Optimization
    links = await engine.optimize_synonyms(threshold=0.85)
    
    # 3. Assert
    # Should create 2 edges (A->B, B->A)
    assert links > 0, "Should have found links"
    
    edges_after = await store.get_all_edges()
    # It might create edges (A->B) and (B->A).
    # add_edge helper uses INSERT... ON CONFLICT UPDATE.
    # Our optimized loop adds (pid, other) and (other, pid).
    # Since edges count might duplicate if we scan both A and B?
    # Iterate A: finds B. Adds A->B, B->A.
    # Iterate B: finds A. Adds B->A, A->B.
    # Since UNIQUE(source, target) constraints exist, duplicates are ignored.
    # So total edges should be 2.
    
    assert len(edges_after) == 2
    assert (id_a, id_b, 1.0) in edges_after or (id_a, id_b, 1.0) in [(s,t,w) for s,t,w in edges_after]
 
