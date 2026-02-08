
import pytest
import aiosqlite
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np

from edge_hippo.hippo_engine import HippoEngine
from edge_hippo.config import settings

@pytest.fixture
def temp_env(tmp_path):
    """Setup a temporary environment with mocked settings."""
    db_file = tmp_path / "integration_test.db"
    with patch.object(settings, "DATA_DIR", tmp_path):
        yield tmp_path

@pytest.mark.asyncio
async def test_paul_graham_integration(temp_env, vector_search_supported):
    """
    Integration test based on the Paul Graham example.
    Mocks the heavy ML models (Extractor/Encoder) but exercises the full
    Storage -> Engine -> Retrieval flow including sqlite-vec and graph logic.
    """
    if not vector_search_supported:
        pytest.skip("Vector search integration test requires sqlite-vec support")
    
    # 1. Setup Engine with Mocked Models
    engine = HippoEngine()
    
    # Mock Extractor
    # We want it to "work" for specific sentences to simulate real behavior.
    engine.extractor = MagicMock()
    
    async def mock_extract(text):
        entities = []
        if "Paul Graham" in text:
            entities.append({"text": "Paul Graham", "label": "PERSON", "score": 0.9})
        if "Lisp" in text:
            entities.append({"text": "Lisp", "label": "tech", "score": 0.9})
        if "Viaweb" in text:
            entities.append({"text": "Viaweb", "label": "ORG", "score": 0.9})
        return entities
        
    engine.extractor.extract_entities.side_effect = mock_extract
    engine.extractor.load_model = MagicMock()
    
    # Mock Encoder (SentenceTransformer)
    # Return random vectors but deterministic enough for "Lisp" matches if we wanted
    # For this test, simpler to just return zeros or specific patterns.
    engine.encoder = MagicMock()
    engine.encoder.encode.return_value = np.zeros((384,), dtype=np.float32)
    
    # Initialize implementation
    await engine.initialize()
    
    # 2. Ingest Data
    text = """
    Paul Graham is an English computer scientist, essayist, and venture capitalist. 
    He is best known for his work on the programming language Lisp, his former startup Viaweb.
    """
    await engine.add_document(text, source="wiki_pg")
    
    # 3. Verify Storage Integrity
    # Nodes: Paul Graham, Lisp, Viaweb, Passage(s)
    # Edges: Passage <-> Entities
    
    pg_id = await engine.storage.get_node_by_name("Paul Graham", "phrase")
    lisp_id = await engine.storage.get_node_by_name("Lisp", "phrase")
    
    assert pg_id is not None, "Paul Graham entity not stored"
    assert lisp_id is not None, "Lisp entity not stored"
    
    edges = await engine.storage.get_all_edges()
    assert len(edges) >= 4 # At least bidirectional edges for 2 entities
    
    # 4. Run Retrieval
    # Query: "Tell me about Lisp"
    # Extractor -> "Lisp"
    # Seed -> Lisp Node
    # Graph -> Lisp <-> Passage
    
    result = await engine.search("Tell me about Lisp")
    
    print(f"\nSearch Result:\n{result}")
    
    assert "Found 1 seed entities" in result
    assert "expanded to" in result
    assert "Lisp" in result
    assert "Paul Graham" in result # Passage contains Paul Graham
    assert "Score:" in result

    # 5. Verify Vector Search (Mocked, but flow check)
    # Query vector is zeros. Nodes have zeros. Should match everything.
    # The 'expanded to' in output confirms vector search added candidates.
