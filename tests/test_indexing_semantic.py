import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from edge_hippo.hippo_engine import HippoEngine
from edge_hippo.config import settings

import numpy as np

# Mock Encoder
class MockEncoder:
    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.array([0.1] * 384)
        return np.array([[0.1] * 384 for _ in texts])

@pytest.mark.asyncio
async def test_indexing_prefixes_and_coverage(tmp_path):
    # Setup Engine with Temp DB
    engine = HippoEngine()
    db_file = tmp_path / "test_index.db"
    engine.storage.db_path = db_file
    
    # Mock Extractor to return dummy entities
    engine.extractor.extract_entities = AsyncMock(return_value=[
        {"text": "Quantum Computing", "label": "concept", "score": 0.9}
    ])
    engine.extractor.load_model = MagicMock()
    
    # Mock Encoder to track calls
    mock_encoder = MockEncoder()
    mock_encoder.encode = MagicMock(side_effect=mock_encoder.encode)
    engine.encoder = mock_encoder
    
    # Prevent background task from overwriting our mock
    engine._warmup_models = AsyncMock()
    # We also don't need ensure_models to wait
    engine._ensure_models = AsyncMock()
    
    await engine.initialize()
    
    # 1. Test Indexing (add_document)
    doc_text = "This is a document about Quantum Computing."
    await engine.add_document(doc_text, source="test")
    
    # Assert Encoder called with "passage: " prefix for Passage
    # The passage text is chunked. 
    # Check checks calls.
    # We expect calls:
    # 1. Call for Phrases: "passage: Quantum Computing" (or batch)
    # 2. Call for Passages: "passage: This is a document about Quantum Computing." (or batch)
    
    encode_calls = mock_encoder.encode.call_args_list
    assert len(encode_calls) > 0
    
    # Flatten arguments to check prefixes
    all_encoded_texts = []
    for call in encode_calls:
        args, _ = call
        arg = args[0]
        if isinstance(arg, list):
            all_encoded_texts.extend(arg)
        else:
            all_encoded_texts.append(arg)
            
    # Check prefixes
    for text in all_encoded_texts:
        assert text.startswith("passage: "), f"Text '{text}' missing 'passage: ' prefix"
        
    # Check if Passage ID has embedding in DB
    # We know passage name format: p_{source}_{i}
    passage_node_id = await engine.storage.get_node_by_name("p_test_0", "passage")
    assert passage_node_id is not None
    
    # Check embedding existence
    async with engine.storage._get_conn() as db:
        cursor = await db.execute("SELECT rowid FROM vec_nodes WHERE rowid = ?", (passage_node_id,))
        row = await cursor.fetchone()
        assert row is not None, "Passage node should be indexed in vector store"
        
    # 2. Test Search (search)
    query = "Find Quantum"
    # We need to mock extractor again for query extraction if needed
    engine.extractor.extract_entities = AsyncMock(return_value=[])
    
    await engine.search(query)
    
    # Verify "query: " prefix on search encoding
    # Latest call to encode should be the query
    last_call = mock_encoder.encode.call_args_list[-1]
    args, _ = last_call
    assert args[0].startswith("query: "), f"Query '{args[0]}' missing 'query: ' prefix"
