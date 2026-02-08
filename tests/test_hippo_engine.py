import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np
from edge_hippo.hippo_engine import HippoEngine

@pytest.mark.asyncio
async def test_engine_add_document(temp_data_dir):
    # Mock Extractor
    with patch("edge_hippo.hippo_engine.EntityExtractor") as MockExtractorCls:
        mock_extractor = MockExtractorCls.return_value
        mock_extractor.extract_entities = AsyncMock(return_value=[
            {"text": "Python", "label": "tech", "score": 1.0}
        ])
        mock_extractor.load_model = MagicMock()
        mock_extractor._executor = MagicMock() 

        # Mock SentenceTransformer logic if engine uses it
        # Since we mocked sys.modules['sentence_transformers'], HippoEngine.initialize 
        # will create self.encoder as a MagicMock. 
        # We need to ensure self.encoder.encode returns something with .tobytes() -> bytes
        mock_encoder = MagicMock()
        mock_numpy = MagicMock()
        mock_numpy.tolist.return_value = [0.1] * 384
        mock_encoder.encode.return_value = mock_numpy

        # We can't easily reach into engine.encoder before initialize, 
        # but initialize creates it from SentenceTransformer().
        # So we patch SentenceTransformer constructor in the library.
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_encoder):
            engine = HippoEngine()
            await engine.initialize()
            
            # Test Ingest
            await engine.add_document("Python is great.")
        
        # Check if nodes added
        edges = await engine.storage.get_all_edges()
        assert len(edges) == 2 # bidirectional
        passages = await engine.storage.get_all_passage_ids()
        assert len(passages) == 1

@pytest.mark.asyncio
async def test_engine_search(temp_data_dir):
    with patch("edge_hippo.hippo_engine.EntityExtractor") as MockExtractorCls:
        mock_extractor = MockExtractorCls.return_value
        # Setup: Query "Python" -> finds entity "Python"
        mock_extractor.extract_entities = AsyncMock(return_value=[
            {"text": "Python", "label": "tech", "score": 1.0}
        ])
        mock_extractor.load_model = MagicMock()

        # Search needs encoder too (even if mocked/loaded)
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.array([0.1]*384)
        
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_encoder):
            engine = HippoEngine()
            await engine.initialize()
            
            # Manually inject data to ensure graph exists
            # p1 contains Python
            pid = await engine.storage.add_node("passage", "p1", "Python rules")
            eid = await engine.storage.add_node("phrase", "Python")
            await engine.storage.add_edge(pid, eid, weight=1.0)
            await engine.storage.add_edge(eid, pid, weight=1.0)
            
            # Search
            result = await engine.search("What about Python?")
            
            # New format includes "Found 1 seed entities" in result
            assert "Found 1 seed entities" in result
            # Vector expansion info matches if found, but we only have 1 entity here
            # so we just check that Python results are present.
            assert "Python" in result
            assert "Python rules" in result
            assert "Score:" in result

@pytest.mark.asyncio
async def test_engine_add_documents_batch(temp_data_dir):
    with patch("edge_hippo.hippo_engine.EntityExtractor") as MockExtractorCls:
        mock_extractor = MockExtractorCls.return_value
        # Batch extract returns list of lists
        mock_extractor.extract_entities_batch = AsyncMock(return_value=[
            [{"text": "Python", "label": "tech", "score": 1.0}],
            [{"text": "Raspberry Pi", "label": "tech", "score": 1.0}]
        ])
        mock_extractor.load_model = MagicMock()

        # Mock Encoder
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.array([[0.1]*384, [0.2]*384])
        
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_encoder):
            engine = HippoEngine()
            await engine.initialize()
            
            docs = ["Doc 1 about Python", "Doc 2 about Raspberry Pi"]
            await engine.add_documents(docs, source="test_batch")
            
            # Verify Extractor called
            mock_extractor.extract_entities_batch.assert_called_once()
            
            # Verify Storage
            # Should have 2 passages + 2 phrases (Python, RPi)
            passages = await engine.storage.get_all_passage_ids()
            assert len(passages) == 2

