import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from edge_hippo.hippo_engine import HippoEngine
from edge_hippo.retrieval import PPRRetriever
from edge_hippo.session import SessionManager
from edge_hippo.storage import GraphStorage

# Unit tests to verify logic flow without relying on DB
# This ensures high coverage of Python code.

@pytest.mark.asyncio
async def test_hippo_engine_search_flow():
    with patch("edge_hippo.hippo_engine.GraphStorage") as MockStorage, \
         patch("edge_hippo.hippo_engine.EntityExtractor") as MockExtractor, \
         patch("edge_hippo.hippo_engine.PPRRetriever") as MockRetrieverCls:
        
        # Setup Storage Mock (Even if not used by search directly, engine init uses it)
        mock_storage = MockStorage.return_value
        mock_storage.initialize = AsyncMock()

        # Setup Extractor Mock
        mock_extractor = MockExtractor.return_value
        mock_extractor.load_model = MagicMock()
        
        # Setup Retriever Mock
        mock_retriever = MockRetrieverCls.return_value
        mock_retriever.search = AsyncMock(return_value=("Result String", ["E1"]))

        engine = HippoEngine()
        engine.storage = mock_storage
        engine.extractor = mock_extractor
        
        # We need to ensure engine.retriever is set to our mock.
        # But engine.retriever is created in initialize?
        # No, engine.retriever is likely created in __init__ or initialize?
        # Actually retrieval.py says PPRRetriever is instantiated in search?
        # Let's check hippo_engine.py source code.
        # It creates self.retriever in initialize() usually.
        # "self.retriever = PPRRetriever(...)"
        
        # Since we patched PPRRetriever class, 
        # when engine calls PPRRetriever(), it gets mock_retriever.
        
        await engine.initialize()
        
        # Verify retriever was created
        # engine.retriever should be our mock instance
        
        # Test Search
        res = await engine.search("test query", "session_1")
        assert res == "Result String"
        
        # Verify Session Update
        # We can check if update_context was called if we mock SessionManager, 
        # or check the real state if we use real SessionManager.
        assert SessionManager().get_context("session_1") == ["E1"]


@pytest.mark.asyncio
async def test_hippo_engine_add_document_flow():
    with patch("edge_hippo.hippo_engine.GraphStorage") as MockStorage, \
         patch("edge_hippo.hippo_engine.EntityExtractor") as MockExtractor:
         
        mock_storage = MockStorage.return_value
        mock_storage.initialize = AsyncMock()
        mock_storage.add_node = AsyncMock(return_value=1)
        mock_storage.add_edge = AsyncMock()
        mock_storage.get_node_by_name = AsyncMock(return_value=None) 
        
        mock_extractor_instance = MockExtractor.return_value
        mock_extractor_instance.extract_entities = AsyncMock(return_value=[
            {"text": "Python", "label": "tech"}
        ])
        mock_extractor_instance.load_model = MagicMock()
        
        mock_encoder = MagicMock()
        # encode returns numpy array or list. run_in_executor returns it.
        mock_encoder.encode.return_value = [0.1]*384
        
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_encoder):
            engine = HippoEngine()
            engine.storage = mock_storage
            engine.extractor = mock_extractor_instance
            
            await engine.initialize() 
            
            await engine.add_document("Python is great.")
            
            assert mock_storage.add_node.call_count >= 2 
            mock_storage.add_edge.assert_called()


@pytest.mark.asyncio
async def test_ppr_retriever_logic():
    with patch("edge_hippo.retrieval.GraphStorage") as MockStorage, \
         patch("edge_hippo.retrieval.EntityExtractor") as MockExtractor:
         
        mock_storage = MockStorage.return_value
        mock_storage.get_node_by_name = AsyncMock(return_value=10) 
        mock_storage.search_vectors = AsyncMock(return_value=[(10, 0.9)])
        # Mock get_node_content needed for result formatting
        mock_storage.get_node_content = AsyncMock(return_value="Content of passage")
        
        nodes_data = [
            {"id": 10, "name": "Python", "type": "phrase", "is_hub": 0},
            {"id": 20, "name": "Passage1", "type": "passage", "content": "Python info", "metadata": "{}"}
        ]
        edges_data = [{"source": 10, "target": 20, "weight": 1.0}]
        
        mock_storage.get_ego_subgraph = AsyncMock(return_value={"nodes": nodes_data, "edges": edges_data})
        
        mock_extractor_instance = MockExtractor.return_value
        mock_extractor_instance.extract_entities = AsyncMock(return_value=[
            {"text": "Python", "label": "tech"}
        ])
        
        retriever = PPRRetriever(mock_extractor_instance, mock_storage)
        
        result, entities = await retriever.search("query: Python")
        
        assert "Content of passage" in result
        assert "Python" in entities


        
