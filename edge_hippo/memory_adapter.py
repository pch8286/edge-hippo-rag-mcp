import json
from typing import Any, Dict, Iterable, List, Optional

import aiosqlite

from memory_crud.judge import build_known_keys_payload, validate_decision
from memory_crud.maintenance import (
    list_candidates,
    mark_retrieved,
    mark_used,
    purge_stale_memories,
    query_memory,
)
from memory_crud.normalize import (
    is_phrase_candidate,
    normalize_text,
    supports_substring_like,
)
from memory_crud.schema import (
    configure_connection,
    configure_wal,
    ensure_mem_schema,
    run_immediate_transaction_with_retry,
)
from memory_crud.store import apply_decision


class MemoryAdapter:
    def __init__(self, storage) -> None:
        self.storage = storage

    async def initialize(self) -> None:
        async with self.storage._get_conn() as db:
            await configure_connection(db)
            await configure_wal(db)
            await ensure_mem_schema(db)
            await db.commit()

    async def build_known_keys(
        self, scope: Optional[str] = None, limit: int = 128
    ) -> Dict[str, List[str]]:
        candidates = await list_candidates(self.storage._get_conn, scope=scope, limit=limit)
        return build_known_keys_payload(candidates)

    async def validate(
        self, decision: Dict[str, Any], candidates: Optional[Iterable[Dict[str, Any]]] = None
    ):
        return validate_decision(decision, candidates)

    async def apply_decision(
        self,
        decision: Dict[str, Any],
        candidates: Optional[Iterable[Dict[str, Any]]] = None,
        now_ts: Optional[int] = None,
    ) -> Dict[str, Any]:
        result = await apply_decision(
            self.storage._get_conn,
            decision=decision,
            candidates=candidates,
            now_ts=now_ts,
        )
        if result.get("status") in {"created", "updated"}:
            memory = decision.get("memory") or {}
            entities = memory.get("entities") or memory.get("link_entities") or []
            if entities and result.get("node_id"):
                await self.link_entities(int(result["node_id"]), entities)
        return result

    async def query_memory(
        self,
        scope: Optional[str] = None,
        kind: Optional[str] = None,
        status: str = "active",
        min_trust: Optional[float] = None,
        min_strength: Optional[float] = None,
        top_k: int = 20,
        now_ts: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        kwargs = {
            "scope": scope,
            "kind": kind,
            "status": status,
            "top_k": top_k,
            "now_ts": now_ts,
        }
        if min_trust is not None:
            kwargs["min_trust"] = min_trust
        if min_strength is not None:
            kwargs["min_strength"] = min_strength
        return await query_memory(self.storage._get_conn, **kwargs)

    async def mark_retrieved(
        self, node_ids: Iterable[int], now_ts: Optional[int] = None
    ) -> Dict[str, Any]:
        return await mark_retrieved(self.storage._get_conn, node_ids=node_ids, now_ts=now_ts)

    async def mark_used(self, node_ids: Iterable[int], now_ts: Optional[int] = None) -> Dict[str, Any]:
        return await mark_used(self.storage._get_conn, node_ids=node_ids, now_ts=now_ts)

    async def purge(self, now_ts: Optional[int] = None) -> Dict[str, Any]:
        return await purge_stale_memories(self.storage._get_conn, now_ts=now_ts)

    async def link_entities(self, memory_node_id: int, entities: Iterable[str]) -> Dict[str, Any]:
        normalized_terms: List[str] = []
        for entity in entities:
            term = normalize_text(str(entity))
            if not term or not is_phrase_candidate(term):
                continue
            normalized_terms.append(term)

        if not normalized_terms:
            return {"linked": 0, "node_ids": []}

        async def _resolve_phrase_node(db: aiosqlite.Connection, term: str) -> int:
            async with db.execute(
                """
                SELECT id
                FROM nodes
                WHERE type = 'phrase' AND lower(name) = ?
                ORDER BY id ASC
                LIMIT 1
                """,
                (term,),
            ) as cursor:
                row = await cursor.fetchone()
            if row:
                return int(row[0])

            if supports_substring_like(term):
                async with db.execute(
                    """
                    SELECT id
                    FROM nodes
                    WHERE type = 'phrase'
                      AND is_hub = 1
                      AND lower(name) LIKE ?
                    ORDER BY length(name) ASC, id ASC
                    LIMIT 1
                    """,
                    (f"%{term}%",),
                ) as cursor:
                    row = await cursor.fetchone()
                if row:
                    return int(row[0])

            meta = json.dumps({"source": "memory_pseudo_edge"}, ensure_ascii=True)
            cursor = await db.execute(
                """
                INSERT INTO nodes (type, name, content, metadata, is_hub)
                VALUES ('phrase', ?, ?, ?, 0)
                """,
                (term, term, meta),
            )
            return int(cursor.lastrowid)

        async def _add_edge(db: aiosqlite.Connection, source: int, target: int) -> None:
            await db.execute(
                """
                INSERT INTO edges (source, target, weight)
                VALUES (?, ?, 2.0)
                ON CONFLICT(source, target) DO UPDATE SET weight = MAX(weight, excluded.weight)
                """,
                (source, target),
            )

        async def _operation(db: aiosqlite.Connection) -> Dict[str, Any]:
            db.row_factory = aiosqlite.Row
            await ensure_mem_schema(db)
            linked_ids: List[int] = []
            for term in normalized_terms:
                phrase_node_id = await _resolve_phrase_node(db, term)
                await _add_edge(db, memory_node_id, phrase_node_id)
                await _add_edge(db, phrase_node_id, memory_node_id)
                linked_ids.append(phrase_node_id)
            return {"linked": len(linked_ids), "node_ids": linked_ids}

        return await run_immediate_transaction_with_retry(self.storage._get_conn, _operation)
