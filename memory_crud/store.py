import json
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

import aiosqlite

from .judge import build_mem_key, validate_decision
from .normalize import canonicalize_key, canonicalize_scope
from .schema import (
    ensure_mem_schema,
    run_immediate_transaction_with_retry,
    sanitize_strength,
    sanitize_trust,
)

SSOT_FIELDS = (
    "mem_key",
    "scope",
    "kind",
    "status",
    "updated_at",
    "last_seen",
    "last_retrieved",
    "trust",
    "strength",
)


def _load_metadata(raw_metadata: Optional[str]) -> Dict[str, Any]:
    if not raw_metadata:
        return {}
    try:
        payload = json.loads(raw_metadata)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def _dump_metadata(metadata: Dict[str, Any]) -> str:
    return json.dumps(metadata, ensure_ascii=True, separators=(",", ":"))


def _merge_snapshot_preserving_last_retrieved(
    metadata: Dict[str, Any],
    snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(metadata or {})
    existing_mem = merged.get("mem") if isinstance(merged.get("mem"), dict) else {}
    extras = {k: v for k, v in existing_mem.items() if k not in SSOT_FIELDS}
    mem_payload = dict(extras)
    for field in SSOT_FIELDS:
        if field == "last_retrieved":
            continue
        if field in snapshot:
            mem_payload[field] = snapshot[field]
    if "last_retrieved" in existing_mem:
        mem_payload["last_retrieved"] = existing_mem["last_retrieved"]
    merged["mem"] = mem_payload
    if "created_at" not in merged:
        merged["created_at"] = snapshot.get("updated_at")
    return merged


async def _fetch_mem_row_by_node_id(
    db: aiosqlite.Connection, node_id: int, active_only: bool = False
) -> Optional[aiosqlite.Row]:
    sql = """
    SELECT
        m.node_id, m.mem_key, m.status, m.kind, m.scope, m.updated_at, m.last_seen,
        m.last_retrieved, m.trust, m.strength, n.metadata, n.content
    FROM mem_kv_index m
    JOIN nodes n ON n.id = m.node_id
    WHERE m.node_id = ?
    """
    params: List[Any] = [node_id]
    if active_only:
        sql += " AND m.status = 'active'"
    async with db.execute(sql, params) as cursor:
        return await cursor.fetchone()


async def _fetch_active_row_by_mem_key(
    db: aiosqlite.Connection, mem_key: str
) -> Optional[aiosqlite.Row]:
    async with db.execute(
        """
        SELECT
            m.node_id, m.mem_key, m.status, m.kind, m.scope, m.updated_at, m.last_seen,
            m.last_retrieved, m.trust, m.strength, n.metadata, n.content
        FROM mem_kv_index m
        JOIN nodes n ON n.id = m.node_id
        WHERE m.mem_key = ? AND m.status = 'active'
        """,
        (mem_key,),
    ) as cursor:
        return await cursor.fetchone()


async def _refresh_metadata_snapshot_without_last_retrieved(
    db: aiosqlite.Connection, node_id: int
) -> None:
    row = await _fetch_mem_row_by_node_id(db, node_id, active_only=False)
    if not row:
        return
    metadata = _load_metadata(row["metadata"])
    snapshot = {field: row[field] for field in SSOT_FIELDS}
    merged = _merge_snapshot_preserving_last_retrieved(metadata, snapshot)
    await db.execute(
        "UPDATE nodes SET metadata = ? WHERE id = ?",
        (_dump_metadata(merged), node_id),
    )


def _resolve_metric(
    existing: float,
    memory: Dict[str, Any],
    field_name: str,
    sanitizer: Callable[[Any], float],
) -> float:
    if field_name in memory:
        return sanitizer(memory[field_name])
    init_payload = memory.get("init")
    if isinstance(init_payload, dict) and field_name in init_payload:
        return sanitizer(init_payload[field_name])
    return float(existing)


async def apply_decision(
    conn_factory: Callable[[], Any],
    decision: Dict[str, Any],
    candidates: Optional[Iterable[Dict[str, Any]]] = None,
    now_ts: Optional[int] = None,
) -> Dict[str, Any]:
    validation = validate_decision(decision, candidates)
    if not validation.valid:
        return {"status": "noop", "reason": validation.reason}

    now = int(now_ts if now_ts is not None else time.time())
    payload = validation.decision
    action = validation.action

    async def _operation(db: aiosqlite.Connection) -> Dict[str, Any]:
        db.row_factory = aiosqlite.Row
        await ensure_mem_schema(db)
        memory = payload.get("memory") or {}
        scope = canonicalize_scope(str(memory.get("scope") or "global"))
        canonical_key = canonicalize_key(str(memory.get("key") or ""))
        mem_key = build_mem_key(scope, canonical_key)
        kind = memory.get("kind")

        if action == "create":
            trust = sanitize_trust((memory.get("init") or {}).get("trust"))
            strength = sanitize_strength((memory.get("init") or {}).get("strength"))
            existing = await _fetch_active_row_by_mem_key(db, mem_key)
            if existing:
                await db.execute(
                    """
                    UPDATE mem_kv_index
                    SET kind = COALESCE(?, kind),
                        scope = ?,
                        updated_at = ?,
                        trust = ?,
                        strength = ?,
                        status = 'active'
                    WHERE node_id = ?
                    """,
                    (
                        kind,
                        scope,
                        now,
                        trust,
                        strength,
                        existing["node_id"],
                    ),
                )
                if memory.get("value") is not None:
                    await db.execute(
                        "UPDATE nodes SET content = ? WHERE id = ?",
                        (str(memory.get("value")), existing["node_id"]),
                    )
                await _refresh_metadata_snapshot_without_last_retrieved(db, existing["node_id"])
                return {"status": "updated", "reason": "create_conflict_converged", "node_id": existing["node_id"]}

            node_metadata = {"created_at": now}
            if "provenance" in memory:
                node_metadata["provenance"] = memory["provenance"]
            node_name = mem_key
            node_content = str(memory.get("value") or canonical_key)
            cursor = await db.execute(
                "INSERT INTO nodes (type, name, content, metadata, is_hub) VALUES (?, ?, ?, ?, 0)",
                ("memory", node_name, node_content, _dump_metadata(node_metadata)),
            )
            node_id = cursor.lastrowid

            try:
                await db.execute(
                    """
                    INSERT INTO mem_kv_index (
                        node_id, mem_key, status, kind, scope, updated_at,
                        last_seen, last_retrieved, trust, strength
                    ) VALUES (?, ?, 'active', ?, ?, ?, ?, NULL, ?, ?)
                    """,
                    (node_id, mem_key, kind, scope, now, now, trust, strength),
                )
            except aiosqlite.IntegrityError:
                # Unique active key conflict: converge to update within same transaction.
                await db.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
                existing = await _fetch_active_row_by_mem_key(db, mem_key)
                if not existing:
                    raise
                await db.execute(
                    """
                    UPDATE mem_kv_index
                    SET kind = COALESCE(?, kind),
                        scope = ?,
                        updated_at = ?,
                        trust = ?,
                        strength = ?,
                        status = 'active'
                    WHERE node_id = ?
                    """,
                    (kind, scope, now, trust, strength, existing["node_id"]),
                )
                await _refresh_metadata_snapshot_without_last_retrieved(db, existing["node_id"])
                return {"status": "updated", "reason": "create_conflict_converged", "node_id": existing["node_id"]}

            await _refresh_metadata_snapshot_without_last_retrieved(db, node_id)
            return {"status": "created", "node_id": node_id}

        target_node_id = int(payload["target_node_id"])
        existing = await _fetch_mem_row_by_node_id(db, target_node_id, active_only=True)
        if not existing:
            return {"status": "noop", "reason": "target_not_active"}

        if action == "delete":
            await db.execute(
                "UPDATE mem_kv_index SET status = 'deleted', updated_at = ? WHERE node_id = ?",
                (now, target_node_id),
            )
            await _refresh_metadata_snapshot_without_last_retrieved(db, target_node_id)
            return {"status": "deleted", "node_id": target_node_id}

        # update
        trust = _resolve_metric(existing["trust"], memory, "trust", sanitize_trust)
        strength = _resolve_metric(existing["strength"], memory, "strength", sanitize_strength)
        next_kind = memory.get("kind", existing["kind"])
        next_scope = existing["scope"] or canonicalize_scope(str(memory.get("scope") or "global"))
        await db.execute(
            """
            UPDATE mem_kv_index
            SET kind = ?, scope = ?, updated_at = ?, trust = ?, strength = ?, status = 'active'
            WHERE node_id = ?
            """,
            (next_kind, next_scope, now, trust, strength, target_node_id),
        )
        if memory.get("value") is not None:
            await db.execute(
                "UPDATE nodes SET content = ? WHERE id = ?",
                (str(memory.get("value")), target_node_id),
            )
        await _refresh_metadata_snapshot_without_last_retrieved(db, target_node_id)
        return {"status": "updated", "node_id": target_node_id}

    try:
        return await run_immediate_transaction_with_retry(conn_factory, _operation)
    except ValueError as exc:
        return {"status": "noop", "reason": str(exc)}
