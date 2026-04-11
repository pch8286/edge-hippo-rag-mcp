"""Unit tests for seahorse.storage (GraphStorage).

Merged from: test_storage.py, test_storage_schema.py, test_vector_search.py
"""

import pytest
from seahorse.storage import GraphStorage


# ──────────────────────────────────────────────
# Schema & Init
# ──────────────────────────────────────────────

class TestSchema:
    @pytest.mark.asyncio
    async def test_initializes_db(self, temp_data_dir):
        storage = GraphStorage()
        await storage.initialize()
        assert (temp_data_dir / "knowledge_graph.db").exists()

    @pytest.mark.asyncio
    async def test_tables_created(self, tmp_path):
        storage = GraphStorage()
        storage.db_path = str(tmp_path / "test.db")
        await storage.initialize()

        async with storage._get_conn() as db:
            async with db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ) as cursor:
                tables = [row[0] for row in await cursor.fetchall()]
            assert "nodes" in tables
            assert "edges" in tables

    @pytest.mark.asyncio
    async def test_columns_include_is_hub(self, tmp_path):
        storage = GraphStorage()
        storage.db_path = str(tmp_path / "test.db")
        await storage.initialize()

        async with storage._get_conn() as db:
            async with db.execute("PRAGMA table_info(nodes)") as cursor:
                cols = [row[1] for row in await cursor.fetchall()]
        assert "is_hub" in cols


# ──────────────────────────────────────────────
# Node Operations
# ──────────────────────────────────────────────

class TestNodes:
    @pytest.mark.asyncio
    async def test_add_node(self, temp_data_dir):
        storage = GraphStorage()
        await storage.initialize()
        pid = await storage.add_node("passage", "p1", "content 1", {"source": "test"})
        assert pid is not None

    @pytest.mark.asyncio
    async def test_add_phrase_node(self, temp_data_dir):
        storage = GraphStorage()
        await storage.initialize()
        eid = await storage.add_node("phrase", "Python", "Python", {"label": "tech"})
        assert eid is not None

    @pytest.mark.asyncio
    async def test_add_duplicate_returns_same_id(self, temp_data_dir):
        storage = GraphStorage()
        await storage.initialize()
        eid1 = await storage.add_node("phrase", "Python")
        eid2 = await storage.add_node("phrase", "Python")
        assert eid1 == eid2

    @pytest.mark.asyncio
    async def test_get_node_by_name(self, temp_data_dir):
        storage = GraphStorage()
        await storage.initialize()
        await storage.add_node("phrase", "TestEntity")
        nid = await storage.get_node_by_name("TestEntity", "phrase")
        assert nid is not None

    @pytest.mark.asyncio
    async def test_get_node_by_name_nonexistent(self, temp_data_dir):
        storage = GraphStorage()
        await storage.initialize()
        nid = await storage.get_node_by_name("NonExistent", "phrase")
        assert nid is None

    @pytest.mark.asyncio
    async def test_get_node_content(self, temp_data_dir):
        storage = GraphStorage()
        await storage.initialize()
        pid = await storage.add_node("passage", "p1", "content 1")
        content = await storage.get_node_content(pid)
        assert content == "content 1"

    @pytest.mark.asyncio
    async def test_get_node_content_nonexistent(self, temp_data_dir):
        storage = GraphStorage()
        await storage.initialize()
        content = await storage.get_node_content(99999)
        assert content is None

    @pytest.mark.asyncio
    async def test_add_node_with_metadata_json(self, temp_data_dir):
        storage = GraphStorage()
        await storage.initialize()
        meta = {"source": "wiki", "score": 0.9}
        pid = await storage.add_node("passage", "p_meta", "text", meta)
        async with storage._get_conn() as db:
            async with db.execute(
                "SELECT metadata FROM nodes WHERE id = ?", (pid,)
            ) as cursor:
                row = await cursor.fetchone()
        import json
        assert json.loads(row[0]) == meta

    @pytest.mark.asyncio
    async def test_add_node_no_metadata(self, temp_data_dir):
        storage = GraphStorage()
        await storage.initialize()
        pid = await storage.add_node("passage", "p_no_meta", "text")
        async with storage._get_conn() as db:
            async with db.execute(
                "SELECT metadata FROM nodes WHERE id = ?", (pid,)
            ) as cursor:
                row = await cursor.fetchone()
        assert row[0] == "{}"


# ──────────────────────────────────────────────
# Edge Operations
# ──────────────────────────────────────────────

class TestEdges:
    @pytest.mark.asyncio
    async def test_add_and_get_edges(self, temp_data_dir):
        storage = GraphStorage()
        await storage.initialize()
        pid = await storage.add_node("passage", "p1", "content")
        eid = await storage.add_node("phrase", "Python")
        await storage.add_edge(pid, eid, "contains")
        edges = await storage.get_all_edges()
        assert len(edges) == 1
        assert edges[0][0] == pid
        assert edges[0][1] == eid

    @pytest.mark.asyncio
    async def test_add_edge_with_weight(self, tmp_path):
        storage = GraphStorage()
        storage.db_path = str(tmp_path / "test.db")
        await storage.initialize()
        id1 = await storage.add_node("phrase", "n1")
        id2 = await storage.add_node("phrase", "n2")
        await storage.add_edge(id1, id2, weight=0.5)
        edges = await storage.get_all_edges()
        assert len(edges) == 1
        assert edges[0] == (id1, id2, 0.5)

    @pytest.mark.asyncio
    async def test_add_edge_duplicate_upserts(self, temp_data_dir):
        """ON CONFLICT should update weight, not raise."""
        storage = GraphStorage()
        await storage.initialize()
        id1 = await storage.add_node("phrase", "a")
        id2 = await storage.add_node("phrase", "b")
        await storage.add_edge(id1, id2, weight=1.0)
        await storage.add_edge(id1, id2, weight=2.0)  # upsert
        edges = await storage.get_all_edges()
        assert len(edges) == 1  # no duplicate


# ──────────────────────────────────────────────
# Embedding & Vector Search
# ──────────────────────────────────────────────

class TestVectorSearch:
    @pytest.mark.asyncio
    async def test_add_node_with_embedding(self, tmp_path):
        storage = GraphStorage()
        storage.db_path = str(tmp_path / "test.db")
        await storage.initialize()
        vec = [0.1] * 384
        node_id = await storage.add_node("phrase", "test_entity", embedding=vec)
        assert node_id is not None

        async with storage._get_conn() as db:
            async with db.execute(
                "SELECT rowid FROM vec_nodes WHERE rowid = ?", (node_id,)
            ) as cursor:
                row = await cursor.fetchone()
                assert row is not None

    @pytest.mark.asyncio
    async def test_vector_search_exact_match(self, tmp_path, vector_search_supported):
        if not vector_search_supported:
            pytest.skip("Vector search not supported via built-in SQLite")

        storage = GraphStorage()
        storage.db_path = str(tmp_path / "test_graph.db")
        await storage.initialize()

        embedding = [0.1] * 384
        node_id = await storage.add_node(
            node_type="phrase", name="test_concept", embedding=embedding
        )
        results = await storage.search_vectors(query_vec=embedding, top_k=5)
        assert len(results) > 0
        found_id, distance = results[0]
        assert found_id == node_id
        assert distance < 0.0001

    @pytest.mark.asyncio
    async def test_search_vectors_extension_not_loaded(self, temp_data_dir):
        """When extension_loaded=False, search_vectors returns []."""
        storage = GraphStorage()
        storage.extension_loaded = False
        await storage.initialize()
        results = await storage.search_vectors([0.1] * 384, top_k=5)
        assert results == []


# ──────────────────────────────────────────────
# Connectivity
# ──────────────────────────────────────────────

class TestConnectivity:
    @pytest.mark.asyncio
    async def test_direct_connection(self, temp_data_dir):
        storage = GraphStorage()
        await storage.initialize()
        id_a = await storage.add_node("phrase", "a")
        id_b = await storage.add_node("phrase", "b")
        await storage.add_edge(id_a, id_b)
        connected = await storage.check_connectivity([id_a], [id_b])
        assert connected is True

    @pytest.mark.asyncio
    async def test_disconnected(self, temp_data_dir):
        storage = GraphStorage()
        await storage.initialize()
        id_a = await storage.add_node("phrase", "a")
        id_b = await storage.add_node("phrase", "b")
        id_c = await storage.add_node("phrase", "c")
        id_d = await storage.add_node("phrase", "d")
        await storage.add_edge(id_a, id_b)
        await storage.add_edge(id_c, id_d)
        connected = await storage.check_connectivity([id_a], [id_c])
        assert connected is False

    @pytest.mark.asyncio
    async def test_common_neighbor_connection(self, temp_data_dir):
        """A->C and B->C, then A and B should be connected (common neighbor)."""
        storage = GraphStorage()
        await storage.initialize()
        id_a = await storage.add_node("phrase", "a")
        id_b = await storage.add_node("phrase", "b")
        id_c = await storage.add_node("phrase", "c")
        await storage.add_edge(id_a, id_c)
        await storage.add_edge(id_b, id_c)
        connected = await storage.check_connectivity([id_a], [id_b])
        assert connected is True

    @pytest.mark.asyncio
    async def test_empty_groups_returns_false(self, temp_data_dir):
        """Empty group_a or group_b => False."""
        storage = GraphStorage()
        await storage.initialize()
        id_a = await storage.add_node("phrase", "a")
        assert await storage.check_connectivity([], [id_a]) is False
        assert await storage.check_connectivity([id_a], []) is False
        assert await storage.check_connectivity([], []) is False


# ──────────────────────────────────────────────
# flag_hub_nodes
# ──────────────────────────────────────────────

class TestFlagHubNodes:
    @pytest.mark.asyncio
    async def test_flags_highest_degree_node(self, temp_data_dir):
        """The node with most incoming edges should be flagged as hub."""
        storage = GraphStorage()
        await storage.initialize()

        hub_id = await storage.add_node("phrase", "Python")
        for i in range(10):
            pid = await storage.add_node("passage", f"p{i}", f"content {i}")
            await storage.add_edge(pid, hub_id)

        other_id = await storage.add_node("phrase", "Rust")
        p_single = await storage.add_node("passage", "p_rust", "rust doc")
        await storage.add_edge(p_single, other_id)

        await storage.flag_hub_nodes(percentile=0.99)

        async with storage._get_conn() as db:
            async with db.execute(
                "SELECT is_hub FROM nodes WHERE id = ?", (hub_id,)
            ) as cur:
                row = await cur.fetchone()
                assert row[0] == 1
            async with db.execute(
                "SELECT is_hub FROM nodes WHERE id = ?", (other_id,)
            ) as cur:
                row = await cur.fetchone()
                assert row[0] == 0

    @pytest.mark.asyncio
    async def test_no_edges_does_not_crash(self, temp_data_dir):
        """flag_hub_nodes on empty graph should not raise."""
        storage = GraphStorage()
        await storage.initialize()
        await storage.flag_hub_nodes()  # should not raise

    @pytest.mark.asyncio
    async def test_resets_previous_hubs(self, temp_data_dir):
        """Calling flag_hub_nodes should clear previously set hubs."""
        storage = GraphStorage()
        await storage.initialize()
        n_id = await storage.add_node("phrase", "X")
        async with storage._get_conn() as db:
            await db.execute("UPDATE nodes SET is_hub = 1 WHERE id = ?", (n_id,))
            await db.commit()

        await storage.flag_hub_nodes()  # resets all then recalculates

        async with storage._get_conn() as db:
            async with db.execute(
                "SELECT is_hub FROM nodes WHERE id = ?", (n_id,)
            ) as cur:
                row = await cur.fetchone()
                assert row[0] == 0  # reset because no edges


# ──────────────────────────────────────────────
# get_ego_subgraph
# ──────────────────────────────────────────────

class TestEgoSubgraph:
    @pytest.mark.asyncio
    async def test_empty_seeds_returns_empty(self, temp_data_dir):
        storage = GraphStorage()
        await storage.initialize()
        result = await storage.get_ego_subgraph([])
        assert result == {"nodes": [], "edges": []}

    @pytest.mark.asyncio
    async def test_subgraph_contains_neighbor(self, temp_data_dir):
        storage = GraphStorage()
        await storage.initialize()
        id_a = await storage.add_node("phrase", "A")
        id_b = await storage.add_node("phrase", "B")
        await storage.add_edge(id_a, id_b)

        result = await storage.get_ego_subgraph([id_a], depth=1)
        node_ids = {n["id"] for n in result["nodes"]}
        assert id_a in node_ids
        assert id_b in node_ids

    @pytest.mark.asyncio
    async def test_subgraph_edge_format(self, temp_data_dir):
        storage = GraphStorage()
        await storage.initialize()
        id_a = await storage.add_node("phrase", "A")
        id_b = await storage.add_node("phrase", "B")
        await storage.add_edge(id_a, id_b, weight=0.5)

        result = await storage.get_ego_subgraph([id_a], depth=1)
        assert len(result["edges"]) >= 1
        edge = result["edges"][0]
        assert len(edge) == 3  # (source, target, weight)

    @pytest.mark.asyncio
    async def test_subgraph_node_shape(self, temp_data_dir):
        """Each node dict should have required keys."""
        storage = GraphStorage()
        await storage.initialize()
        id_a = await storage.add_node("phrase", "A")
        result = await storage.get_ego_subgraph([id_a], depth=0)
        node = result["nodes"][0]
        assert "id" in node
        assert "type" in node
        assert "name" in node
        assert "is_hub" in node
        assert "embedding" in node

    @pytest.mark.asyncio
    async def test_handles_nonexistent_seed(self, temp_data_dir):
        """Nonexistent seed_id should return empty."""
        storage = GraphStorage()
        await storage.initialize()
        result = await storage.get_ego_subgraph([99999])
        assert result == {"nodes": [], "edges": []}


# ──────────────────────────────────────────────
# verify_integrity
# ──────────────────────────────────────────────

class TestVerifyIntegrity:
    @pytest.mark.asyncio
    async def test_returns_stats_dict(self, temp_data_dir):
        storage = GraphStorage()
        await storage.initialize()
        stats = await storage.verify_integrity()
        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert "hub_nodes" in stats
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0

    @pytest.mark.asyncio
    async def test_counts_match_inserted_data(self, temp_data_dir):
        storage = GraphStorage()
        await storage.initialize()
        id1 = await storage.add_node("phrase", "X")
        id2 = await storage.add_node("passage", "P", "content")
        await storage.add_edge(id1, id2)

        stats = await storage.verify_integrity()
        assert stats["total_nodes"] == 2
        assert stats["total_edges"] == 1
        assert stats["phrase_nodes"] == 1
        assert stats["passage_nodes"] == 1

    @pytest.mark.asyncio
    async def test_hub_count_in_stats(self, temp_data_dir):
        storage = GraphStorage()
        await storage.initialize()
        nid = await storage.add_node("phrase", "hub")
        async with storage._get_conn() as db:
            await db.execute("UPDATE nodes SET is_hub = 1 WHERE id = ?", (nid,))
            await db.commit()

        stats = await storage.verify_integrity()
        assert stats["hub_nodes"] == 1


# ──────────────────────────────────────────────
# get_all_passage_ids
# ──────────────────────────────────────────────

class TestPassageIds:
    @pytest.mark.asyncio
    async def test_returns_only_passages(self, temp_data_dir):
        storage = GraphStorage()
        await storage.initialize()
        await storage.add_node("phrase", "X")
        pid = await storage.add_node("passage", "P1", "text")
        ids = await storage.get_all_passage_ids()
        assert ids == [pid]

    @pytest.mark.asyncio
    async def test_empty_db(self, temp_data_dir):
        storage = GraphStorage()
        await storage.initialize()
        assert await storage.get_all_passage_ids() == []
