"""Unit tests for seahorse.retrieval (PPRRetriever).

Merged from: test_retrieval.py, retrieval parts of test_unit_logic.py
"""

import pytest
import re
import tempfile
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from unittest.mock import MagicMock, AsyncMock

from seahorse.storage import GraphStorage
from seahorse.retrieval import PPRRetriever
from seahorse.extraction import EntityExtractor
from seahorse.config import settings
from seahorse.reranker import CrossEncoderReranker


class MockExtractor(EntityExtractor):
    """Simple mock extractor with keyword-based extraction."""

    def __init__(self):
        pass

    async def load_model(self):
        pass

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        if "EntityA" in text:
            return [{"text": "EntityA", "label": "Mock", "score": 1.0}]
        if "EntityB" in text:
            return [{"text": "EntityB", "label": "Mock", "score": 1.0}]
        return []


@pytest.fixture
async def storage():
    """Real storage with simplified schema (no sqlite-vec)."""
    original_data_dir = settings.DATA_DIR
    with tempfile.TemporaryDirectory() as temp_dir:
        settings.DATA_DIR = Path(temp_dir)

        class LiteGraphStorage(GraphStorage):
            @asynccontextmanager
            async def _get_conn(self):
                import aiosqlite
                async with aiosqlite.connect(self.db_path) as db:
                    yield db

            async def initialize(self):
                async with self._get_conn() as db:
                    await db.execute(
                        "CREATE TABLE IF NOT EXISTS nodes "
                        "(id INTEGER PRIMARY KEY, type TEXT, name TEXT, "
                        "content TEXT, metadata TEXT, is_hub INTEGER DEFAULT 0)"
                    )
                    await db.execute(
                        "CREATE INDEX IF NOT EXISTS idx_nodes_name "
                        "ON nodes(name)"
                    )
                    await db.execute(
                        "CREATE TABLE IF NOT EXISTS edges "
                        "(source INTEGER, target INTEGER, weight REAL, "
                        "FOREIGN KEY(source) REFERENCES nodes(id), "
                        "FOREIGN KEY(target) REFERENCES nodes(id), "
                        "UNIQUE(source, target))"
                    )
                    await db.execute(
                        "CREATE TABLE IF NOT EXISTS vec_nodes "
                        "(rowid INTEGER PRIMARY KEY, embedding BLOB)"
                    )
                    await db.commit()

            async def add_node(self, node_type, name, content="",
                               metadata=None, embedding=None):
                return await super().add_node(
                    node_type, name, content, metadata, embedding=None
                )

        store = LiteGraphStorage()
        await store.initialize()
        yield store
    settings.DATA_DIR = original_data_dir


class TestPPRRetriever:
    def test_seed_distribution_softmax_uniform_mix(self):
        dist = PPRRetriever._build_seed_distribution(
            {5: 0.9, 1: 0.2, 9: 0.5},
            beta=3.0,
            uniform_mix=0.10,
        )
        assert set(dist.keys()) == {1, 5, 9}
        assert abs(sum(dist.values()) - 1.0) < 1e-6
        # higher score should still rank higher after mixing
        assert dist[5] > dist[9] > dist[1]
        # uniform mix keeps strictly positive mass on all seeds
        assert min(dist.values()) > 0.0

    def test_seed_hub_self_mass_suppression(self):
        seed_dist = {1: 0.7, 2: 0.3}
        out = PPRRetriever._suppress_seed_hub_self_mass(
            seed_dist,
            seed_hub_indices={1},
            passage_indices=[10, 11],
            total_nodes=12,
        )
        assert abs(sum(out.values()) - 1.0) < 1e-6
        assert out.get(1, 0.0) == 0.0
        assert out[2] > 0.0

    def test_schedule_damping_bounds(self):
        low = PPRRetriever._schedule_damping(
            base=0.8, seed_count=1, seed_hub_ratio=1.0, node_count=50
        )
        high = PPRRetriever._schedule_damping(
            base=0.8, seed_count=8, seed_hub_ratio=0.0, node_count=5000
        )
        assert 0.55 <= low <= 0.92
        assert 0.55 <= high <= 0.92
        assert low < high

    def test_fanout_cap_hub_phrase_to_passage(self):
        retriever = PPRRetriever(MagicMock(), MagicMock())
        nodes = [{"id": 1, "type": "phrase", "name": "hub", "is_hub": 1}] + [
            {"id": i + 2, "type": "passage", "name": f"p{i}", "is_hub": 0}
            for i in range(80)
        ]
        edges = [
            {
                "src_idx": 0,
                "dst_idx": i + 1,
                "etype": PPRRetriever.ET_PP,
                "score": float(100 - i),
            }
            for i in range(80)
        ]
        kept = retriever._apply_fanout_caps(edges, nodes)
        assert len(kept) == 48
        assert min(e["score"] for e in kept) >= 53.0

    def test_residual_teleport_excludes_seed_hub_self(self):
        retriever = PPRRetriever(MagicMock(), MagicMock())
        nodes = [
            {"id": 1, "type": "phrase", "name": "H", "is_hub": 1},
            {"id": 2, "type": "phrase", "name": "S2", "is_hub": 0},
            {"id": 3, "type": "passage", "name": "P1", "is_hub": 0},
        ]
        pruned_edges = [
            {"src_idx": 0, "dst_idx": 2, "etype": PPRRetriever.ET_PP, "score": 1.0}
        ]
        node_stats = {
            1: {"global_phrase_deg": 10.0, "global_passage_phrase_deg": 0.0, "idf": 1.0},
            2: {"global_phrase_deg": 1.0, "global_passage_phrase_deg": 0.0, "idf": 1.0},
            3: {"global_phrase_deg": 0.0, "global_passage_phrase_deg": 5.0, "idf": 0.0},
        }
        global_meta = {"AVG_PHRASE_DEG": 5.0, "AVG_PASSAGE_PHRASE_DEG": 4.0}
        g, _ = retriever._build_residual_graph(
            nodes=nodes,
            pruned_edges=pruned_edges,
            node_stats=node_stats,
            global_meta=global_meta,
            base_seed_dist={0: 0.6, 1: 0.4},
            seed_hub_indices={0},
        )
        ew = {}
        for e, w in zip(g.get_edgelist(), g.es["weight"]):
            ew[e] = ew.get(e, 0.0) + float(w)

        # Hub residual is teleported away from itself (to seed peer), self keeps p_self only.
        p_self = 0.8 * (10.0 / (10.0 + 5.0))
        assert ew.get((0, 1), 0.0) > 0.0
        assert abs(ew.get((0, 0), 0.0) - p_self) < 1e-6

    @pytest.mark.asyncio
    async def test_search_returns_result(self, storage):
        """Build a small graph and verify search finds content."""
        id_e1 = await storage.add_node("phrase", "EntityA")
        id_p1 = await storage.add_node("passage", "P1", content="Content about EntityA")

        await storage.add_edge(id_p1, id_e1, weight=1.0)
        await storage.add_edge(id_e1, id_p1, weight=1.0)

        retriever = PPRRetriever(MockExtractor(), storage)
        result, entities = await retriever.search("query EntityA", top_k=5)

        assert "Content about EntityA" in result
        assert "EntityA" in entities

    @pytest.mark.asyncio
    async def test_context_boosts_connected_passage(self, storage):
        """Context entity should boost the score of connected passages."""
        id_e1 = await storage.add_node("phrase", "EntityA")
        id_e2 = await storage.add_node("phrase", "EntityB")
        id_p1 = await storage.add_node("passage", "P1", content="Content about EntityA")
        id_p2 = await storage.add_node("passage", "P2", content="Content about EntityB")

        await storage.add_edge(id_p1, id_e1, weight=1.0)
        await storage.add_edge(id_e1, id_p1, weight=1.0)
        await storage.add_edge(id_p2, id_e2, weight=1.0)
        await storage.add_edge(id_e2, id_p2, weight=1.0)
        await storage.add_edge(id_e1, id_e2, weight=0.5)
        await storage.add_edge(id_e2, id_e1, weight=0.5)

        retriever = PPRRetriever(MockExtractor(), storage)

        def get_score(res_str, content_marker):
            lines = res_str.split("\n")
            for i, line in enumerate(lines):
                if content_marker in line:
                    m = re.search(r"Score: ([\d.]+)", lines[i - 1])
                    if m:
                        return float(m.group(1))
            return 0.0

        result_no_ctx, _ = await retriever.search("query EntityA", top_k=5)
        score_p2_no_ctx = get_score(result_no_ctx, "Content about EntityB")

        result_ctx, _ = await retriever.search(
            "query EntityA", top_k=5, history_entities=["EntityB"]
        )
        score_p2_ctx = get_score(result_ctx, "Content about EntityB")

        assert score_p2_ctx > score_p2_no_ctx

    @pytest.mark.asyncio
    async def test_fully_mocked_flow(self):
        """Fully mocked: no real DB, verify data flows correctly."""
        mock_storage = MagicMock()
        mock_storage.get_node_by_name = AsyncMock(return_value=10)
        mock_storage.search_vectors = AsyncMock(return_value=[(10, 0.9)])
        mock_storage.get_node_content = AsyncMock(return_value="Content of passage")
        mock_storage.get_ego_subgraph = AsyncMock(return_value={
            "nodes": [
                {"id": 10, "name": "Python", "type": "phrase", "is_hub": 0},
                {"id": 20, "name": "Passage1", "type": "passage",
                 "content": "Python info", "metadata": "{}"},
            ],
            "edges": [{"source": 10, "target": 20, "weight": 1.0}],
        })

        mock_ext = MagicMock()
        mock_ext.extract_entities = AsyncMock(return_value=[
            {"text": "Python", "label": "tech"},
        ])

        retriever = PPRRetriever(mock_ext, mock_storage)
        result, entities = await retriever.search("query: Python")

        assert "Content of passage" in result
        assert "Python" in entities

    @pytest.mark.asyncio
    async def test_no_seed_returns_no_matching(self):
        mock_storage = MagicMock()
        mock_ext = MagicMock()
        mock_ext.extract_entities = AsyncMock(return_value=[])
        retriever = PPRRetriever(mock_ext, mock_storage)

        result, entities = await retriever.search("query: unknown")
        assert result == "No matching entities found."
        assert entities == []

    @pytest.mark.asyncio
    async def test_empty_subgraph_returns_no_context(self):
        mock_storage = MagicMock()
        mock_storage.get_node_by_name = AsyncMock(return_value=1)
        mock_storage.get_ego_subgraph = AsyncMock(return_value={"nodes": [], "edges": []})

        mock_ext = MagicMock()
        mock_ext.extract_entities = AsyncMock(return_value=[{"text": "Python", "label": "tech"}])
        retriever = PPRRetriever(mock_ext, mock_storage)

        result, entities = await retriever.search("query: Python")
        assert result == "No context found for entities."
        assert entities == ["Python"]

    @pytest.mark.asyncio
    async def test_encoder_vector_expansion_and_reranker_path(self):
        class FakeReranker:
            top_n = 2

            def rerank_and_fuse(self, query, candidates, top_k):
                assert query == "query: Python"
                assert top_k == 2
                assert len(candidates) >= 1
                return candidates[:2]

        mock_storage = MagicMock()
        # current entity id: 10, expanded id: 30
        mock_storage.get_node_by_name = AsyncMock(side_effect=[10, None])
        mock_storage.search_vectors = AsyncMock(return_value=[(30, 0.2)])
        mock_storage.get_node_content = AsyncMock(return_value="Content of passage")
        mock_storage.get_ego_subgraph = AsyncMock(return_value={
            "nodes": [
                {"id": 10, "name": "Python", "type": "phrase", "is_hub": 1},
                {"id": 20, "name": "P20", "type": "passage", "is_hub": 0},
                {"id": 30, "name": "ExtraSeed", "type": "phrase", "is_hub": 0},
                {"id": 40, "name": "Dangling", "type": "phrase", "is_hub": 0},
            ],
            "edges": [
                {"source": 10, "target": 20, "weight": 1.0},
                {"source": 30, "target": 10, "weight": 1.0},
                {"source": 999, "target": 10, "weight": 1.0},  # filtered edge
            ],
        })

        mock_ext = MagicMock()
        mock_ext.extract_entities = AsyncMock(return_value=[{"text": "Python", "label": "tech"}])
        mock_enc = MagicMock()
        mock_enc.encode.return_value = np.array([0.1] * 384)

        retriever = PPRRetriever(mock_ext, mock_storage, encoder=mock_enc, reranker=FakeReranker())
        result, entities = await retriever.search("query: Python", top_k=2)
        assert "expanded to" in result
        assert entities == ["Python"]

    @pytest.mark.asyncio
    async def test_search_uses_dynamic_node_budget(self, monkeypatch):
        mock_storage = MagicMock()
        mock_storage.get_node_by_name = AsyncMock(return_value=10)
        mock_storage.get_ego_subgraph = AsyncMock(
            return_value={
                "nodes": [
                    {"id": 10, "name": "Python", "type": "phrase", "is_hub": 0},
                    {"id": 20, "name": "P20", "type": "passage", "is_hub": 0},
                ],
                "edges": [{"source": 10, "target": 20, "weight": 1.0}],
            }
        )
        mock_storage.get_node_content = AsyncMock(return_value="Content of passage")

        mock_ext = MagicMock()
        mock_ext.extract_entities = AsyncMock(return_value=[{"text": "Python", "label": "tech"}])

        monkeypatch.setattr(
            "seahorse.retrieval.resource_manager.calculate_node_budget",
            lambda: 777,
        )
        retriever = PPRRetriever(mock_ext, mock_storage)
        await retriever.search("query: Python", top_k=1)
        mock_storage.get_ego_subgraph.assert_awaited_once_with([10], depth=2, limit=777)

    @pytest.mark.asyncio
    async def test_build_candidates_empty_and_stable_dedup_warning(self, caplog):
        mock_storage = MagicMock()
        mock_storage.get_node_content = AsyncMock(return_value="X")
        retriever = PPRRetriever(MagicMock(), mock_storage)

        empty = await retriever._build_candidates([])
        assert empty == []

        caplog.clear()
        deduped = retriever._stable_dedup_ranked([(1, 0.9), (1, 0.8), (2, 0.7)])
        assert deduped == [(1, 0.9), (2, 0.7)]
        assert "duplicate passage candidates" in caplog.text

    def test_from_settings_disabled_returns_none(self, monkeypatch):
        monkeypatch.setattr(settings, "RERANK_ENABLED", False)
        assert CrossEncoderReranker.from_settings() is None
