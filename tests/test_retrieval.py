import pytest
import pytest_asyncio
import tempfile
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from edge_hippo.storage import GraphStorage
from edge_hippo.retrieval import PPRRetriever
from edge_hippo.extraction import EntityExtractor
from edge_hippo.config import settings

# Mock Extractor
class MockExtractor(EntityExtractor):
    def __init__(self):
        pass
    async def load_model(self):
        pass
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        # Simple exact match for test
        if "EntityA" in text:
            return [{"text": "EntityA", "label": "Mock", "score": 1.0}]
        if "EntityB" in text:
            return [{"text": "EntityB", "label": "Mock", "score": 1.0}]
        return []

@pytest_asyncio.fixture
async def storage():
    original_data_dir = settings.DATA_DIR
    with tempfile.TemporaryDirectory() as temp_dir:
        settings.DATA_DIR = Path(temp_dir)
        
        class MockGraphStorage(GraphStorage):
            @asynccontextmanager
            async def _get_conn(self):
                import aiosqlite
                async with aiosqlite.connect(self.db_path) as db:
                     yield db
            
            async def initialize(self):
                async with self._get_conn() as db:
                    await db.execute("CREATE TABLE IF NOT EXISTS nodes (id INTEGER PRIMARY KEY, type TEXT, name TEXT, content TEXT, metadata TEXT, is_hub INTEGER DEFAULT 0)")
                    await db.execute("CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name)")
                    await db.execute("CREATE TABLE IF NOT EXISTS edges (source INTEGER, target INTEGER, weight REAL, FOREIGN KEY(source) REFERENCES nodes(id), FOREIGN KEY(target) REFERENCES nodes(id), UNIQUE(source, target))")
                    await db.execute("CREATE TABLE IF NOT EXISTS vec_nodes (rowid INTEGER PRIMARY KEY, embedding BLOB)")
                    await db.commit()
            
            async def add_node(self, node_type, name, content="", metadata=None, embedding=None):
                return await super().add_node(node_type, name, content, metadata, embedding=None)
                
        store = MockGraphStorage()
        await store.initialize()
        yield store
    settings.DATA_DIR = original_data_dir

@pytest.mark.asyncio
async def test_ppr_context_influence(storage):
    # Setup Graph
    # P1 <-> E1
    # P2 <-> E2
    # E1 <-> E2
    
    id_e1 = await storage.add_node("phrase", "EntityA")
    id_e2 = await storage.add_node("phrase", "EntityB")
    id_p1 = await storage.add_node("passage", "P1", content="Content about EntityA")
    id_p2 = await storage.add_node("passage", "P2", content="Content about EntityB")
    
    # Edges
    await storage.add_edge(id_p1, id_e1, weight=1.0)
    await storage.add_edge(id_e1, id_p1, weight=1.0)
    
    await storage.add_edge(id_p2, id_e2, weight=1.0)
    await storage.add_edge(id_e2, id_p2, weight=1.0)
    
    await storage.add_edge(id_e1, id_e2, weight=0.5)
    await storage.add_edge(id_e2, id_e1, weight=0.5)
    
    retriever = PPRRetriever(MockExtractor(), storage)
    
    # 1. Search "EntityA" WITHOUT context
    result_no_ctx, _ = await retriever.search("query EntityA", top_k=5)
    
    # Parse result to find Score of P2
    # Output format: --- [Score: 0.1234] ---\nContent...
    # We can inspect internal logic or parse string.
    # Let's inspect raw object if we expose it, or parse.
    
    def get_score(res_str, content_marker):
        lines = res_str.split('\n')
        for i, line in enumerate(lines):
            if content_marker in line:
                # Score is in previous line usually?
                # Format:
                # --- [Score: 0.0573] ---
                # Content about EntityB
                # So if we find content, look back 1 line
                prev = lines[i-1]
                import re
                m = re.search(r"Score: ([\d\.]+)", prev)
                if m:
                    return float(m.group(1))
        return 0.0

    score_p2_no_ctx = get_score(result_no_ctx, "Content about EntityB")
    
    # 2. Search "EntityA" WITH context "EntityB"
    # They are connected (E1-E2), so Drift Check should Pass.
    result_ctx, _ = await retriever.search("query EntityA", top_k=5, history_entities=["EntityB"])
    score_p2_ctx = get_score(result_ctx, "Content about EntityB")
    
    print(f"P2 Score No Context: {score_p2_no_ctx}")
    print(f"P2 Score With Context: {score_p2_ctx}")
    
    assert score_p2_ctx > score_p2_no_ctx, "Context 'EntityB' should boost P2 score"
