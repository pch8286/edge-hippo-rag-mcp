
import pytest
import aiosqlite
import sqlite_vec
import struct
from edge_hippo.storage import GraphStorage

@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")

@pytest.mark.asyncio
async def test_schema_creation(db_path):
    storage = GraphStorage()
    storage.db_path = db_path
    
    await storage.initialize()
    
    async with storage._get_conn() as db:
        async with db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cursor:
            tables = [row[0] for row in await cursor.fetchall()]
            print(f"Tables: {tables}")
            assert "nodes" in tables
            assert "edges" in tables
            # vec_nodes is virtual
            
            # Check vec_nodes specifically
            try:
                await db.execute("SELECT count(*) FROM vec_nodes")
            except Exception as e:
                pytest.fail(f"vec_nodes table query failed: {e}")

        async with db.execute("PRAGMA table_info(nodes)") as cursor:
            cols = [row[1] for row in await cursor.fetchall()]
            assert "is_hub" in cols
            assert "embedding" not in cols

@pytest.mark.asyncio
async def test_add_node_embedding(db_path):
    storage = GraphStorage()
    storage.db_path = db_path
    await storage.initialize()
    
    vec = [0.1] * 384
    node_id = await storage.add_node("phrase", "test_entity", embedding=vec)
    assert node_id is not None
    
    # Verify embedding insertion
    async with storage._get_conn() as db:
        async with db.execute("SELECT rowid FROM vec_nodes WHERE rowid = ?", (node_id,)) as cursor:
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == node_id

@pytest.mark.asyncio
async def test_add_edge_no_relation(db_path):
    storage = GraphStorage()
    storage.db_path = db_path
    await storage.initialize()
    
    id1 = await storage.add_node("phrase", "n1")
    id2 = await storage.add_node("phrase", "n2")
    
    await storage.add_edge(id1, id2, weight=0.5)
    
    edges = await storage.get_all_edges()
    assert len(edges) == 1
    # source, target, weight
    assert edges[0] == (id1, id2, 0.5)
