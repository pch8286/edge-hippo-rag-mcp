import pytest
import os
import aiosqlite
from edge_hippo.storage import GraphStorage

@pytest.mark.asyncio
async def test_storage_init(temp_data_dir):
    storage = GraphStorage()
    await storage.initialize()
    assert (temp_data_dir / "knowledge_graph.db").exists()

@pytest.mark.asyncio
async def test_add_nodes_and_edges(temp_data_dir):
    storage = GraphStorage()
    await storage.initialize()
    
    # Add Passage
    pid = await storage.add_node("passage", "p1", "content 1", {"source": "test"})
    assert pid is not None
    
    # Add Phrase
    eid = await storage.add_node("phrase", "Python", "Python", {"label": "tech"})
    assert eid is not None
    
    # Add Duplicate Phrase (should return same ID)
    eid2 = await storage.add_node("phrase", "Python")
    assert eid == eid2

    # Add Edge
    await storage.add_edge(pid, eid, "contains")
    
    # Verify
    edges = await storage.get_all_edges()
    assert len(edges) == 1
    assert edges[0][0] == pid
    assert edges[0][1] == eid
    
    # Verify Content
    content = await storage.get_node_content(pid)
    assert content == "content 1"
    
@pytest.mark.asyncio
async def test_get_node_by_name(temp_data_dir):
    storage = GraphStorage()
    await storage.initialize()
    
    await storage.add_node("phrase", "TestEntity")
    nid = await storage.get_node_by_name("TestEntity", "phrase")
    assert nid is not None
    
    nid_none = await storage.get_node_by_name("NonExistent", "phrase")
    assert nid_none is None
