"""Unit tests for seahorse.hippo_engine (HippoEngine).

Merged from: test_hippo_engine.py, test_hybrid_retrieval.py,
             test_consolidation.py, engine parts of test_unit_logic.py
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from seahorse.hippo_engine import HippoEngine


# --- Shared helpers ---

def _mock_encoder():
    encoder = MagicMock()
    encoder.encode.return_value = np.array([0.1] * 384)
    return encoder


# --- Add Document ---

class TestAddDocument:
    @pytest.mark.asyncio
    async def test_add_single_document(self, temp_data_dir):
        with patch("seahorse.hippo_engine.EntityExtractor") as MockExtractorCls:
            mock_ext = MockExtractorCls.return_value
            mock_ext.extract_entities = AsyncMock(return_value=[
                {"text": "Python", "label": "tech", "score": 1.0},
            ])
            mock_ext.load_model = MagicMock()
            mock_ext._executor = MagicMock()

            mock_numpy = MagicMock()
            mock_numpy.tolist.return_value = [0.1] * 384
            mock_enc = MagicMock()
            mock_enc.encode.return_value = mock_numpy

            with patch("sentence_transformers.SentenceTransformer", return_value=mock_enc):
                engine = HippoEngine()
                await engine.initialize()
                await engine.add_document("Python is great.")

            edges = await engine.storage.get_all_edges()
            assert len(edges) == 2  # bidirectional
            passages = await engine.storage.get_all_passage_ids()
            assert len(passages) == 1

    @pytest.mark.asyncio
    async def test_add_documents_batch(self, temp_data_dir):
        with patch("seahorse.hippo_engine.EntityExtractor") as MockExtractorCls:
            mock_ext = MockExtractorCls.return_value
            mock_ext.extract_entities_batch = AsyncMock(return_value=[
                [{"text": "Python", "label": "tech", "score": 1.0}],
                [{"text": "Raspberry Pi", "label": "tech", "score": 1.0}],
            ])
            mock_ext.load_model = MagicMock()

            mock_enc = MagicMock()
            mock_enc.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])

            with patch("sentence_transformers.SentenceTransformer", return_value=mock_enc):
                engine = HippoEngine()
                await engine.initialize()
                await engine.add_documents(
                    ["Doc about Python", "Doc about RPi"], source="batch"
                )

            mock_ext.extract_entities_batch.assert_called_once()
            passages = await engine.storage.get_all_passage_ids()
            assert len(passages) == 2

    @pytest.mark.asyncio
    async def test_add_document_flow_mocked(self):
        """Fully mocked flow: no real DB."""
        with (
            patch("seahorse.hippo_engine.GraphStorage") as MockStorage,
            patch("seahorse.hippo_engine.EntityExtractor") as MockExtractor,
        ):
            mock_storage = MockStorage.return_value
            mock_storage.initialize = AsyncMock()
            mock_storage.add_node = AsyncMock(return_value=1)
            mock_storage.add_edge = AsyncMock()
            mock_storage.get_node_by_name = AsyncMock(return_value=None)

            mock_ext = MockExtractor.return_value
            mock_ext.extract_entities = AsyncMock(return_value=[
                {"text": "Python", "label": "tech"},
            ])
            mock_ext.load_model = MagicMock()

            mock_enc = MagicMock()
            mock_enc.encode.return_value = [0.1] * 384

            with patch("sentence_transformers.SentenceTransformer", return_value=mock_enc):
                engine = HippoEngine()
                engine.storage = mock_storage
                engine.extractor = mock_ext
                await engine.initialize()
                await engine.add_document("Python is great.")

            assert mock_storage.add_node.call_count >= 2
            mock_storage.add_edge.assert_called()


# --- Search ---

class TestSearch:
    @pytest.mark.asyncio
    async def test_search_returns_result(self, temp_data_dir):
        with patch("seahorse.hippo_engine.EntityExtractor") as MockExtractorCls:
            mock_ext = MockExtractorCls.return_value
            mock_ext.extract_entities = AsyncMock(return_value=[
                {"text": "Python", "label": "tech", "score": 1.0},
            ])
            mock_ext.load_model = MagicMock()

            mock_enc = MagicMock()
            mock_enc.encode.return_value = np.array([0.1] * 384)

            with patch("sentence_transformers.SentenceTransformer", return_value=mock_enc):
                engine = HippoEngine()
                await engine.initialize()

                pid = await engine.storage.add_node("passage", "p1", "Python rules")
                eid = await engine.storage.add_node("phrase", "Python")
                await engine.storage.add_edge(pid, eid, weight=1.0)
                await engine.storage.add_edge(eid, pid, weight=1.0)

                result = await engine.search("What about Python?")

            assert "Found 1 seed entities" in result
            assert "Python" in result
            assert "Score:" in result

    @pytest.mark.asyncio
    async def test_search_flow_fully_mocked(self):
        """Fully mocked: verifies retriever call and session update."""
        from seahorse.session import SessionManager

        with (
            patch("seahorse.hippo_engine.GraphStorage") as MockStorage,
            patch("seahorse.hippo_engine.EntityExtractor") as MockExtractor,
            patch("seahorse.hippo_engine.PPRRetriever") as MockRetrieverCls,
        ):
            mock_storage = MockStorage.return_value
            mock_storage.initialize = AsyncMock()

            mock_ext = MockExtractor.return_value
            mock_ext.load_model = MagicMock()

            mock_retriever = MockRetrieverCls.return_value
            mock_retriever.search = AsyncMock(return_value=("Result String", ["E1"]))

            engine = HippoEngine()
            engine.storage = mock_storage
            engine.extractor = mock_ext
            await engine.initialize()

            res = await engine.search("test query", "session_1")
            assert res == "Result String"
            assert SessionManager().get_context("session_1") == ["E1"]


# --- Hybrid PPR Weighting ---

class TestHybridPPR:
    @pytest.mark.asyncio
    async def test_seed_weights_ordered_by_distance(self):
        """Seeds with lower vector distance must score higher."""
        engine = HippoEngine()
        engine.storage = AsyncMock()
        engine.retriever.storage = engine.storage
        engine.extractor = AsyncMock()
        engine.encoder = MagicMock()

        engine.extractor.extract_entities.return_value = []
        engine.encoder.encode.return_value = np.array([0.1] * 384)

        engine.storage.search_vectors.return_value = [
            (1, 0.1), (2, 0.5), (3, 1.0),
        ]

        mock_subgraph = {
            "nodes": [
                {"id": 1, "type": "phrase", "name": "A", "is_hub": False, "embedding": None},
                {"id": 2, "type": "phrase", "name": "B", "is_hub": False, "embedding": None},
                {"id": 3, "type": "phrase", "name": "C", "is_hub": False, "embedding": None},
                {"id": 10, "type": "passage", "name": "TA", "is_hub": False, "embedding": None},
                {"id": 20, "type": "passage", "name": "TB", "is_hub": False, "embedding": None},
                {"id": 30, "type": "passage", "name": "TC", "is_hub": False, "embedding": None},
            ],
            "edges": [
                (1, 10, 1.0), (10, 1, 1.0),
                (2, 20, 1.0), (20, 2, 1.0),
                (3, 30, 1.0), (30, 3, 1.0),
            ],
        }
        engine.storage.get_ego_subgraph.return_value = mock_subgraph
        engine.storage.get_node_content.return_value = "Content"

        await engine.search("query")

        calls = engine.storage.get_node_content.call_args_list
        assert len(calls) >= 3
        assert calls[0].args[0] == 10
        assert calls[1].args[0] == 20
        assert calls[2].args[0] == 30
