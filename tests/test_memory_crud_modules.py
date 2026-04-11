import math
from contextlib import asynccontextmanager

import aiosqlite
import pytest

from seahorse.memory_adapter import MemoryAdapter
from seahorse.storage import GraphStorage
from memory_crud import prompts
from memory_crud.judge import (
    build_known_keys_payload,
    split_mem_key,
    validate_decision,
)
from memory_crud.maintenance import (
    _load_metadata as maintenance_load_metadata,
    list_candidates,
    mark_retrieved,
    mark_used,
    purge_stale_memories,
    query_memory,
    resolve_last_retrieved_effective,
)
from memory_crud.normalize import (
    canonicalize_key,
    canonicalize_scope,
    is_phrase_candidate,
    normalize_text,
    supports_substring_like,
)
from memory_crud.schema import (
    STRENGTH_EPS,
    TRUST_EPS,
    is_lock_error,
    run_immediate_transaction_with_retry,
    sanitize_strength,
    sanitize_trust,
)
from memory_crud.store import _load_metadata as store_load_metadata
from memory_crud.store import apply_decision


@pytest.fixture
async def memory_stack(temp_data_dir):
    storage = GraphStorage()
    await storage.initialize()
    adapter = MemoryAdapter(storage)
    await adapter.initialize()
    return storage, adapter


class TestNormalizeAndPrompt:
    def test_normalize_helpers(self):
        assert normalize_text("  A\x00Ｂ\tc  ") == "a b c"
        assert canonicalize_scope("User:Profile") == "user_profile"
        assert canonicalize_scope("  ") == "global"
        assert canonicalize_key("  MixEd Key  ") == "mixed key"
        assert is_phrase_candidate("a") is False
        assert is_phrase_candidate("12") is False
        assert is_phrase_candidate("ab") is True
        assert supports_substring_like("abc") is False
        assert supports_substring_like("abcd") is True

    def test_prompt_contract_string(self):
        assert "Decision JSON v3 contract" in prompts.JUDGE_OUTPUT_GUIDE
        assert "create MUST include memory.init.trust and memory.init.strength" in prompts.JUDGE_OUTPUT_GUIDE


class TestJudge:
    def test_known_key_payload_and_split(self):
        payload = build_known_keys_payload(
            [
                {"mem_key": "chat:user.profile"},
                {"mem_key": "chat:user.profile"},
                {"mem_key": "chat:task_today"},
            ],
            max_keys=3,
            max_prefixes=2,
        )
        assert payload["known_keys"] == ["chat:user.profile", "chat:task_today"]
        assert payload["known_prefixes"] == ["user", "task"]
        assert split_mem_key("no_scope") == ("global", "no_scope")

    @pytest.mark.parametrize(
        "decision,candidates,reason",
        [
            ({"action": "bad"}, None, "unsupported_action"),
            ({"action": "create", "memory": {"key": "", "init": {"trust": 0.1, "strength": 0.1}}}, None, "missing_key"),
            ({"action": "create", "memory": {"key": "k", "init": {"trust": 0.1}}}, None, "missing_init_values"),
            (
                {
                    "action": "update",
                    "target_node_id": 1,
                    "affected_keys": ["chat:k"],
                    "memory": {"scope": "chat", "key": "k"},
                },
                [{"mem_key": "chat:k"}],
                "candidate_missing_node_id",
            ),
            (
                {
                    "action": "update",
                    "target_node_id": 2,
                    "affected_keys": ["chat:k"],
                    "memory": {"scope": "chat", "key": "k"},
                },
                [{"node_id": 1, "mem_key": "chat:k", "scope": "chat"}],
                "target_not_in_candidates",
            ),
            (
                {
                    "action": "update",
                    "target_node_id": 1,
                    "affected_keys": [],
                    "memory": {"scope": "chat", "key": "k"},
                },
                [{"node_id": 1, "mem_key": "chat:k", "scope": "chat"}],
                "empty_affected_keys",
            ),
            (
                {
                    "action": "update",
                    "target_node_id": 1,
                    "affected_keys": ["chat:k"],
                    "memory": {"scope": "user", "key": "k"},
                },
                [{"node_id": 1, "mem_key": "chat:k", "scope": "chat"}],
                "immutable_scope_violation",
            ),
        ],
    )
    def test_validate_decision_invalid(self, decision, candidates, reason):
        result = validate_decision(decision, candidates)
        assert result.valid is False
        assert result.action == "noop"
        assert result.reason == reason

    def test_validate_decision_noop_and_valid_delete(self):
        noop = validate_decision({"action": "noop"}, [])
        assert noop.valid is True
        assert noop.action == "noop"
        valid_delete = validate_decision(
            {
                "action": "delete",
                "target_node_id": 1,
                "affected_keys": ["chat:k"],
                "memory": {"scope": "chat", "key": "k"},
            },
            [{"node_id": 1, "mem_key": "chat:k", "scope": "chat"}],
        )
        assert valid_delete.valid is True
        assert valid_delete.action == "delete"


class TestSchema:
    def test_sanitize_and_lock_error_helpers(self):
        assert sanitize_trust(-1.0) == TRUST_EPS
        assert sanitize_strength(-1.0) == STRENGTH_EPS
        assert sanitize_trust(1.5) == 1.0
        assert sanitize_strength(1.5) == 1.0
        assert is_lock_error(RuntimeError("database is locked")) is True
        assert is_lock_error(RuntimeError("database is busy")) is True
        assert is_lock_error(RuntimeError("other error")) is False
        with pytest.raises(ValueError):
            sanitize_trust(math.inf)

    @pytest.mark.asyncio
    async def test_retry_runtimeerror_when_retries_zero(self):
        @asynccontextmanager
        async def conn_factory():
            async with aiosqlite.connect(":memory:") as db:
                yield db

        async def operation(_db):
            return "ok"

        with pytest.raises(RuntimeError, match="transaction retry failed unexpectedly"):
            await run_immediate_transaction_with_retry(conn_factory, operation, retries=0)

    @pytest.mark.asyncio
    async def test_retry_non_lock_error_raises_immediately(self):
        @asynccontextmanager
        async def conn_factory():
            async with aiosqlite.connect(":memory:") as db:
                yield db

        async def operation(_db):
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await run_immediate_transaction_with_retry(conn_factory, operation, retries=3)


class TestMaintenanceAndStore:
    @pytest.mark.asyncio
    async def test_metadata_loaders_and_fallback_helper(self):
        assert maintenance_load_metadata("{bad") == {}
        assert store_load_metadata("{bad") == {}
        assert store_load_metadata("[]") == {}
        assert resolve_last_retrieved_effective(None, 10, 20) == 10
        assert resolve_last_retrieved_effective(None, None, 20) == 20

    @pytest.mark.asyncio
    async def test_store_value_error_and_target_not_active(self, memory_stack):
        storage, _ = memory_stack
        invalid_create = {
            "action": "create",
            "memory": {
                "scope": "chat",
                "key": "finite-check",
                "kind": "fact",
                "init": {"trust": float("inf"), "strength": 0.2},
            },
        }
        result = await apply_decision(storage._get_conn, invalid_create, now_ts=1)
        assert result["status"] == "noop"
        assert "finite" in result["reason"]

        create = {
            "action": "create",
            "memory": {
                "scope": "chat",
                "key": "gone",
                "kind": "fact",
                "init": {"trust": 0.4, "strength": 0.4},
            },
        }
        created = await apply_decision(storage._get_conn, create, now_ts=2)
        node_id = created["node_id"]
        delete = {
            "action": "delete",
            "target_node_id": node_id,
            "affected_keys": ["chat:gone"],
            "memory": {"scope": "chat", "key": "gone"},
        }
        candidates = [{"node_id": node_id, "mem_key": "chat:gone", "scope": "chat"}]
        await apply_decision(storage._get_conn, delete, candidates=candidates, now_ts=3)
        deleted_again = await apply_decision(storage._get_conn, delete, candidates=candidates, now_ts=4)
        assert deleted_again["status"] == "noop"
        assert deleted_again["reason"] == "target_not_active"

    @pytest.mark.asyncio
    async def test_maintenance_paths_and_scope_consistency_filter(self, memory_stack):
        storage, adapter = memory_stack
        first = await adapter.apply_decision(
            {
                "action": "create",
                "memory": {
                    "scope": "chat",
                    "key": "m1",
                    "kind": "fact",
                    "value": "v1",
                    "init": {"trust": 0.7, "strength": 0.7},
                },
            },
            now_ts=100,
        )
        second = await adapter.apply_decision(
            {
                "action": "create",
                "memory": {
                    "scope": "team",
                    "key": "m2",
                    "kind": "task",
                    "value": "v2",
                    "init": {"trust": 0.7, "strength": 0.7},
                },
            },
            now_ts=110,
        )
        assert first["status"] == "created"
        assert second["status"] == "created"

        scoped = await list_candidates(storage._get_conn, scope="chat", limit=10)
        assert len(scoped) == 1
        assert scoped[0]["scope"] == "chat"

        assert await mark_retrieved(storage._get_conn, [], now_ts=120) == {"updated": 0}
        assert await mark_used(storage._get_conn, [], now_ts=120) == {"updated": 0}
        assert await mark_used(storage._get_conn, [999999], now_ts=120) == {"updated": 0}

        used = await mark_used(storage._get_conn, [first["node_id"]], now_ts=121)
        assert used["updated"] == 1

        # Force scope inconsistency and verify strict skip behavior.
        async with storage._get_conn() as db:
            await db.execute(
                "UPDATE mem_kv_index SET scope = 'wrong' WHERE node_id = ?",
                (first["node_id"],),
            )
            await db.commit()
        rows = await query_memory(storage._get_conn, scope="wrong", top_k=10, now_ts=130)
        assert rows == []

    @pytest.mark.asyncio
    async def test_purge_last_seen_null_fallbacks(self, memory_stack):
        storage, adapter = memory_stack
        created_with_created_at = await adapter.apply_decision(
            {
                "action": "create",
                "memory": {
                    "scope": "chat",
                    "key": "old-created-at",
                    "kind": "fact",
                    "init": {"trust": 0.8, "strength": 0.8},
                },
            },
            now_ts=10,
        )
        created_with_updated_at = await adapter.apply_decision(
            {
                "action": "create",
                "memory": {
                    "scope": "chat",
                    "key": "old-updated-at",
                    "kind": "task",
                    "init": {"trust": 0.8, "strength": 0.8},
                },
            },
            now_ts=20,
        )

        async with storage._get_conn() as db:
            await db.execute(
                "UPDATE mem_kv_index SET last_seen = NULL WHERE node_id = ?",
                (created_with_created_at["node_id"],),
            )
            await db.execute(
                "UPDATE mem_kv_index SET last_seen = NULL, updated_at = 30 WHERE node_id = ?",
                (created_with_updated_at["node_id"],),
            )
            await db.execute(
                "UPDATE nodes SET metadata = '{}' WHERE id = ?",
                (created_with_updated_at["node_id"],),
            )
            await db.commit()

        purged = await purge_stale_memories(storage._get_conn, now_ts=35 * 24 * 60 * 60)
        assert purged["deleted"] >= 2
        assert created_with_created_at["node_id"] in purged["node_ids"]
        assert created_with_updated_at["node_id"] in purged["node_ids"]


class TestMemoryAdapter:
    @pytest.mark.asyncio
    async def test_adapter_validate_known_keys_and_entity_link(self, memory_stack):
        storage, adapter = memory_stack
        decision = {
            "action": "create",
            "memory": {
                "scope": "chat",
                "key": "entity-link-test",
                "kind": "fact",
                "init": {"trust": 0.6, "strength": 0.6},
                "entities": ["EntityX"],
            },
        }
        phrase_id = await storage.add_node("phrase", "EntityX", "EntityX")
        created = await adapter.apply_decision(decision, now_ts=1_000)
        assert created["status"] == "created"
        node_id = created["node_id"]

        # Proxy helpers
        known = await adapter.build_known_keys(scope="chat", limit=16)
        assert "chat:entity-link-test" in known["known_keys"]
        validation = await adapter.validate({"action": "noop"}, [])
        assert validation.valid is True

        # link_entities should ignore non-candidate tokens and keep exact match.
        ignored = await adapter.link_entities(node_id, ["1", "!!"])
        assert ignored["linked"] == 0

        async with storage._get_conn() as db:
            async with db.execute(
                "SELECT COUNT(*) FROM edges WHERE source = ? AND target = ? AND weight = 2.0",
                (node_id, phrase_id),
            ) as cursor:
                exact_edge_count = (await cursor.fetchone())[0]
        assert exact_edge_count == 1
