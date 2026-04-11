"""Integration tests — end-to-end flows using real storage.

Merged from: test_integration_flow.py, test_indexing.py,
             test_indexing_semantic.py, test_drift_control.py,
             test_logic_hardening.py, test_consolidation.py
"""

import pytest
import pytest_asyncio
import aiosqlite
import numpy as np
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, AsyncMock, patch

from seahorse.hippo_engine import HippoEngine
from seahorse.storage import GraphStorage
from seahorse.algorithms import check_drift
from seahorse.config import settings


# ──────────────────────────────────────────────
# Shared Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def temp_env(tmp_path):
    """Temporary settings.DATA_DIR."""
    with patch.object(settings, "DATA_DIR", tmp_path):
        yield tmp_path


class _LiteStorage(GraphStorage):
    """GraphStorage without sqlite-vec (for portability)."""

    @asynccontextmanager
    async def _get_conn(self):
        async with aiosqlite.connect(self.db_path) as db:
            yield db

    async def initialize(self):
        async with self._get_conn() as db:
            await db.execute(
                "CREATE TABLE IF NOT EXISTS nodes "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT NOT NULL, "
                "name TEXT, content TEXT, metadata TEXT, "
                "is_hub INTEGER DEFAULT 0)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_name "
                "ON nodes(name) WHERE type='phrase'"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)"
            )
            await db.execute(
                "CREATE TABLE IF NOT EXISTS edges "
                "(source INTEGER, target INTEGER, weight REAL DEFAULT 1.0, "
                "FOREIGN KEY(source) REFERENCES nodes(id), "
                "FOREIGN KEY(target) REFERENCES nodes(id), "
                "UNIQUE(source, target))"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target)"
            )
            await db.commit()

    async def add_node(self, node_type, name, content="",
                       metadata=None, embedding=None):
        return await super().add_node(
            node_type, name, content, metadata, embedding=None
        )


@pytest_asyncio.fixture
async def lite_storage(tmp_path):
    """Fixture providing a LiteStorage (no sqlite-vec dependency)."""
    original = settings.DATA_DIR
    settings.DATA_DIR = tmp_path
    store = _LiteStorage()
    await store.initialize()
    yield store
    settings.DATA_DIR = original


# ──────────────────────────────────────────────
# Paul Graham Full Flow
# ──────────────────────────────────────────────

class TestFullFlow:
    @pytest.mark.asyncio
    async def test_paul_graham_integration(self, temp_env, vector_search_supported):
        """Ingest → verify storage → search → validate results."""
        if not vector_search_supported:
            pytest.skip("Requires sqlite-vec support")

        engine = HippoEngine()
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
        engine.encoder = MagicMock()
        engine.encoder.encode.return_value = np.zeros((384,), dtype=np.float32)

        await engine.initialize()

        text = (
            "Paul Graham is an English computer scientist, essayist, "
            "and venture capitalist. He is best known for his work on "
            "the programming language Lisp, his former startup Viaweb."
        )
        await engine.add_document(text, source="wiki_pg")

        pg_id = await engine.storage.get_node_by_name("Paul Graham", "phrase")
        lisp_id = await engine.storage.get_node_by_name("Lisp", "phrase")
        assert pg_id is not None
        assert lisp_id is not None

        result = await engine.search("Tell me about Lisp")
        assert "Found 1 seed entities" in result
        assert "Lisp" in result
        assert "Score:" in result


# ──────────────────────────────────────────────
# Indexing Pipeline
# ──────────────────────────────────────────────

class TestIndexingPipeline:
    @pytest.mark.asyncio
    async def test_indexing_creates_nodes_and_edges(self, temp_env):
        engine = HippoEngine()
        engine.extractor = MagicMock()

        async def mock_extract(text):
            return [
                {"text": "Python", "label": "technology", "score": 0.9},
                {"text": "HippoRAG", "label": "concept", "score": 0.8},
            ]

        engine.extractor.extract_entities.side_effect = mock_extract

        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda texts, **kw: (
            np.random.rand(384).astype(np.float32)
            if isinstance(texts, str)
            else np.random.rand(len(texts), 384).astype(np.float32)
        )
        engine.encoder = mock_enc
        await engine.initialize()

        await engine.add_document("Python is great. HippoRAG uses Python.", source="test_doc")

        db_path = settings.db_path
        async with aiosqlite.connect(db_path) as db:
            async with db.execute(
                "SELECT count(*) FROM nodes WHERE type='passage'"
            ) as c:
                passages = (await c.fetchone())[0]
                assert passages >= 1
            async with db.execute(
                "SELECT count(*) FROM nodes WHERE type='phrase'"
            ) as c:
                phrases = (await c.fetchone())[0]
                assert phrases == 2

    @pytest.mark.asyncio
    async def test_embedding_prefix_passage(self, temp_env):
        """Verify 'passage:' prefix is used when encoding passages."""
        engine = HippoEngine()
        engine.extractor = MagicMock()
        engine.extractor.extract_entities = AsyncMock(return_value=[
            {"text": "Quantum Computing", "label": "concept", "score": 0.9},
        ])
        engine.extractor.load_model = MagicMock()

        mock_enc = MagicMock()

        def _encode(texts, **kw):
            if isinstance(texts, str):
                return np.random.rand(384).astype(np.float32)
            return np.random.rand(len(texts), 384).astype(np.float32)

        mock_enc.encode = MagicMock(side_effect=_encode)
        engine.encoder = mock_enc
        engine._warmup_models = AsyncMock()
        engine._ensure_models = AsyncMock()
        await engine.initialize()

        await engine.add_document("About Quantum Computing.", source="test")

        all_texts = []
        for call in mock_enc.encode.call_args_list:
            args, _ = call
            arg = args[0]
            if isinstance(arg, list):
                all_texts.extend(arg)
            else:
                all_texts.append(arg)

        for text in all_texts:
            assert text.startswith("passage: "), f"Missing prefix: '{text}'"


# ──────────────────────────────────────────────
# Drift Control
# ──────────────────────────────────────────────

class TestDriftControl:
    @pytest.mark.asyncio
    async def test_connected_direct(self, lite_storage):
        id_a = await lite_storage.add_node("phrase", "entity_a")
        id_b = await lite_storage.add_node("phrase", "entity_b")
        await lite_storage.add_edge(id_a, id_b)

        connected = await lite_storage.check_connectivity([id_a], [id_b])
        assert connected is True
        assert await check_drift(lite_storage, ["entity_a"], ["entity_b"]) is False

    @pytest.mark.asyncio
    async def test_connected_common_neighbor(self, lite_storage):
        id_a = await lite_storage.add_node("phrase", "entity_a")
        id_b = await lite_storage.add_node("phrase", "entity_b")
        id_c = await lite_storage.add_node("phrase", "entity_c")
        await lite_storage.add_edge(id_a, id_c)
        await lite_storage.add_edge(id_b, id_c)

        connected = await lite_storage.check_connectivity([id_a], [id_b])
        assert connected is True

    @pytest.mark.asyncio
    async def test_disconnected_islands(self, lite_storage):
        id_a = await lite_storage.add_node("phrase", "entity_a")
        id_b = await lite_storage.add_node("phrase", "entity_b")
        await lite_storage.add_edge(id_a, id_b)

        id_c = await lite_storage.add_node("phrase", "entity_c")
        id_d = await lite_storage.add_node("phrase", "entity_d")
        await lite_storage.add_edge(id_c, id_d)

        assert await lite_storage.check_connectivity([id_a], [id_c]) is False
        assert await check_drift(lite_storage, ["entity_a"], ["entity_c"]) is True

    @pytest.mark.asyncio
    async def test_missing_entity_is_drift(self, lite_storage):
        await lite_storage.add_node("phrase", "entity_a")
        assert await check_drift(lite_storage, ["entity_b"], ["entity_a"]) is True


# ──────────────────────────────────────────────
# Hub Explosion Prevention
# ──────────────────────────────────────────────

class TestHubExplosion:
    @pytest.mark.asyncio
    async def test_hub_blocks_traversal(self, tmp_path):
        engine = HippoEngine()
        engine.storage.db_path = str(tmp_path / "test_logic.db")
        await engine.initialize()
        store = engine.storage

        p_seed = await store.add_node("passage", "p_seed", "contains python")
        hub_id = await store.add_node("phrase", "Python")

        async with store._get_conn() as db:
            await db.execute("UPDATE nodes SET is_hub = 1 WHERE id = ?", (hub_id,))
            await db.commit()

        await store.add_edge(p_seed, hub_id)
        await store.add_edge(hub_id, p_seed)

        for i in range(50):
            n_id = await store.add_node("phrase", f"lib_{i}")
            await store.add_edge(hub_id, n_id)
            await store.add_edge(n_id, hub_id)

        subgraph = await store.get_ego_subgraph([p_seed], depth=2)
        nodes = subgraph["nodes"]
        neighbor_count = sum(1 for n in nodes if n["name"].startswith("lib_"))
        assert neighbor_count == 0, f"Recursion leaked through Hub! ({neighbor_count})"


# ──────────────────────────────────────────────
# Synonym Linking
# ──────────────────────────────────────────────

class TestSynonymLinking:
    @pytest.mark.asyncio
    async def test_optimize_synonyms(self, tmp_path, vector_search_supported):
        if not vector_search_supported:
            pytest.skip("Requires vector search")

        engine = HippoEngine()
        engine.storage.db_path = str(tmp_path / "test_syn.db")
        await engine.initialize()
        store = engine.storage

        vec_a = [0.0] * 384
        vec_a[0] = 1.0
        vec_b = [0.0] * 384
        vec_b[0] = 0.99
        vec_b[1] = 0.1

        await store.add_node("phrase", "RPi 5", embedding=vec_a)
        await store.add_node("phrase", "Raspberry Pi 5", embedding=vec_b)

        assert len(await store.get_all_edges()) == 0

        links = await engine.optimize_synonyms(threshold=0.85)
        assert links > 0

        edges = await store.get_all_edges()
        assert len(edges) == 2
