import json
import sqlite3

import pytest

from seahorse.memory_adapter import MemoryAdapter
from seahorse.storage import GraphStorage
from memory_crud.schema import (
    MEM_KV_INDEX_DDL_SQL,
    MEM_KV_INDEX_INDEX_SQL,
    STRENGTH_EPS,
    TRUST_EPS,
    run_immediate_transaction_with_retry,
)


@pytest.fixture
async def memory_stack(temp_data_dir):
    storage = GraphStorage()
    await storage.initialize()
    adapter = MemoryAdapter(storage)
    await adapter.initialize()
    return storage, adapter


async def _fetch_mem_row(storage: GraphStorage, node_id: int):
    async with storage._get_conn() as db:
        async with db.execute(
            """
            SELECT node_id, mem_key, scope, status, trust, strength, last_retrieved
            FROM mem_kv_index
            WHERE node_id = ?
            """,
            (node_id,),
        ) as cursor:
            return await cursor.fetchone()


async def _fetch_metadata(storage: GraphStorage, node_id: int):
    async with storage._get_conn() as db:
        async with db.execute("SELECT metadata FROM nodes WHERE id = ?", (node_id,)) as cursor:
            row = await cursor.fetchone()
    return json.loads(row[0])


def test_mem_ddl_snapshot_exact():
    expected_ddl = """CREATE TABLE IF NOT EXISTS mem_kv_index (
    node_id INTEGER PRIMARY KEY,
    mem_key TEXT NOT NULL,
    status TEXT NOT NULL, -- 'active', 'superseded', 'deleted'
    kind TEXT,
    scope TEXT,
    updated_at INTEGER,
    last_seen INTEGER,
    last_retrieved INTEGER,
    trust REAL,
    strength REAL,
    FOREIGN KEY(node_id) REFERENCES nodes(id)
);"""
    expected_indexes = [
        """CREATE UNIQUE INDEX IF NOT EXISTS ux_mem_key_active
ON mem_kv_index(mem_key)
WHERE status = 'active';""",
        "CREATE INDEX IF NOT EXISTS ix_mem_updated_at ON mem_kv_index(updated_at);",
        "CREATE INDEX IF NOT EXISTS ix_mem_last_seen ON mem_kv_index(last_seen);",
        "CREATE INDEX IF NOT EXISTS ix_mem_last_retrieved ON mem_kv_index(last_retrieved);",
    ]
    assert MEM_KV_INDEX_DDL_SQL == expected_ddl
    assert MEM_KV_INDEX_INDEX_SQL == expected_indexes


class TestMemoryCrud:
    @pytest.mark.asyncio
    async def test_mem_schema_is_applied(self, memory_stack):
        storage, _ = memory_stack
        async with storage._get_conn() as db:
            async with db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='mem_kv_index'"
            ) as cursor:
                table = await cursor.fetchone()
            async with db.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='ux_mem_key_active'"
            ) as cursor:
                index = await cursor.fetchone()
        assert table is not None
        assert index is not None

    @pytest.mark.asyncio
    async def test_create_uses_eps_and_is_immediately_retrievable(self, memory_stack):
        _, adapter = memory_stack
        decision = {
            "action": "create",
            "memory": {
                "scope": "User",
                "key": "Favorite Language",
                "kind": "fact",
                "value": "Python",
                "init": {"trust": 0.0, "strength": 0.0},
            },
        }
        result = await adapter.apply_decision(decision, now_ts=1_000)
        assert result["status"] == "created"

        rows = await adapter.query_memory(scope="user", now_ts=1_000, top_k=10)
        assert len(rows) == 1
        assert rows[0]["trust"] == pytest.approx(TRUST_EPS)
        assert rows[0]["strength_base"] == pytest.approx(STRENGTH_EPS)
        assert rows[0]["mem_key"] == "user:favorite language"
        assert rows[0]["scope"] == "user"

    @pytest.mark.asyncio
    async def test_apply_decision_does_not_overwrite_metadata_last_retrieved(self, memory_stack):
        storage, adapter = memory_stack
        create = {
            "action": "create",
            "memory": {
                "scope": "chat",
                "key": "topic",
                "kind": "fact",
                "value": "graphs",
                "init": {"trust": 0.4, "strength": 0.5},
            },
        }
        created = await adapter.apply_decision(create, now_ts=100)
        node_id = created["node_id"]

        await adapter.mark_retrieved([node_id], now_ts=150)
        metadata_before = await _fetch_metadata(storage, node_id)
        assert metadata_before["mem"]["last_retrieved"] == 150

        candidates = await adapter.query_memory(scope="chat", now_ts=160)
        update = {
            "action": "update",
            "target_node_id": node_id,
            "affected_keys": ["chat:topic"],
            "memory": {
                "scope": "chat",
                "key": "topic",
                "kind": "fact",
                "value": "graphs-v2",
                "init": {"trust": 0.7, "strength": 0.6},
            },
        }
        updated = await adapter.apply_decision(update, candidates=candidates, now_ts=170)
        assert updated["status"] == "updated"

        metadata_after = await _fetch_metadata(storage, node_id)
        assert metadata_after["mem"]["last_retrieved"] == 150

    @pytest.mark.asyncio
    async def test_last_retrieved_null_falls_back_to_created_at(self, memory_stack):
        _, adapter = memory_stack
        create = {
            "action": "create",
            "memory": {
                "scope": "chat",
                "key": "fallback",
                "kind": "fact",
                "init": {"trust": 0.5, "strength": 0.5},
            },
        }
        created = await adapter.apply_decision(create, now_ts=1_000)
        assert created["status"] == "created"

        rows = await adapter.query_memory(scope="chat", now_ts=1_100)
        assert len(rows) == 1
        row = rows[0]
        assert row["last_retrieved"] is None
        assert row["last_retrieved_effective"] == 1_000

    @pytest.mark.asyncio
    async def test_update_key_immutability_violation_is_noop(self, memory_stack):
        storage, adapter = memory_stack
        create = {
            "action": "create",
            "memory": {
                "scope": "chat",
                "key": "alpha",
                "kind": "fact",
                "init": {"trust": 0.7, "strength": 0.7},
            },
        }
        created = await adapter.apply_decision(create, now_ts=200)
        node_id = created["node_id"]
        candidates = await adapter.query_memory(scope="chat", now_ts=210)

        update = {
            "action": "update",
            "target_node_id": node_id,
            "affected_keys": ["chat:alpha"],
            "memory": {
                "scope": "chat",
                "key": "beta",
                "kind": "fact",
                "init": {"trust": 0.9, "strength": 0.9},
            },
        }
        result = await adapter.apply_decision(update, candidates=candidates, now_ts=220)
        assert result["status"] == "noop"
        assert result["reason"] == "immutable_key_violation"

        row = await _fetch_mem_row(storage, node_id)
        assert row[1] == "chat:alpha"

    @pytest.mark.asyncio
    async def test_unique_active_key_conflict_converges_to_update(self, memory_stack):
        storage, adapter = memory_stack
        first = {
            "action": "create",
            "memory": {
                "scope": "chat",
                "key": "conflict",
                "kind": "fact",
                "init": {"trust": 0.2, "strength": 0.2},
            },
        }
        second = {
            "action": "create",
            "memory": {
                "scope": "chat",
                "key": "conflict",
                "kind": "fact",
                "init": {"trust": 0.8, "strength": 0.9},
            },
        }
        first_result = await adapter.apply_decision(first, now_ts=300)
        second_result = await adapter.apply_decision(second, now_ts=310)
        assert first_result["status"] == "created"
        assert second_result["status"] == "updated"
        assert second_result["reason"] == "create_conflict_converged"
        assert first_result["node_id"] == second_result["node_id"]

        async with storage._get_conn() as db:
            async with db.execute(
                "SELECT COUNT(*) FROM mem_kv_index WHERE mem_key = 'chat:conflict' AND status = 'active'"
            ) as cursor:
                count = (await cursor.fetchone())[0]
            async with db.execute(
                "SELECT trust, strength FROM mem_kv_index WHERE node_id = ?",
                (first_result["node_id"],),
            ) as cursor:
                trust, strength = await cursor.fetchone()
        assert count == 1
        assert trust == pytest.approx(0.8)
        assert strength == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_retry_utility_retries_on_locked_error(self, memory_stack):
        storage, _ = memory_stack
        attempts = 0

        async def operation(_db):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise sqlite3.OperationalError("database is locked")
            return "ok"

        result = await run_immediate_transaction_with_retry(storage._get_conn, operation, retries=5)
        assert result == "ok"
        assert attempts == 3

    @pytest.mark.asyncio
    async def test_substring_like_is_disabled_for_short_terms(self, memory_stack):
        storage, adapter = memory_stack
        created = await adapter.apply_decision(
            {
                "action": "create",
                "memory": {
                    "scope": "chat",
                    "key": "entity-link",
                    "kind": "fact",
                    "init": {"trust": 0.6, "strength": 0.6},
                },
            },
            now_ts=500,
        )
        memory_node_id = created["node_id"]

        hub_node_id = await storage.add_node("phrase", "hub candidate", "hub candidate")
        async with storage._get_conn() as db:
            await db.execute("UPDATE nodes SET is_hub = 1 WHERE id = ?", (hub_node_id,))
            await db.commit()

        link_result = await adapter.link_entities(memory_node_id, ["hub"])
        assert link_result["linked"] == 1

        async with storage._get_conn() as db:
            async with db.execute(
                "SELECT id FROM nodes WHERE type = 'phrase' AND name = 'hub'"
            ) as cursor:
                created_phrase_row = await cursor.fetchone()
            assert created_phrase_row is not None
            created_phrase_id = created_phrase_row[0]
            async with db.execute(
                "SELECT COUNT(*) FROM edges WHERE source = ? AND target = ?",
                (memory_node_id, hub_node_id),
            ) as cursor:
                hub_edge_count = (await cursor.fetchone())[0]
            async with db.execute(
                "SELECT COUNT(*) FROM edges WHERE source = ? AND target = ? AND weight = 2.0",
                (memory_node_id, created_phrase_id),
            ) as cursor:
                created_edge_count = (await cursor.fetchone())[0]
        assert hub_edge_count == 0
        assert created_edge_count == 1
