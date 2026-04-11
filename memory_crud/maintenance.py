import json
import math
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

import aiosqlite

from .judge import split_mem_key
from .schema import (
    DEFAULT_MIN_STRENGTH,
    DEFAULT_MIN_TRUST,
    STRENGTH_EPS,
    ensure_mem_schema,
    run_immediate_transaction_with_retry,
)
from .store import SSOT_FIELDS

KIND_DECAY_LAMBDA_PER_SEC = {
    "fact": 1.0 / (30.0 * 24.0 * 60.0 * 60.0),
    "preference": 1.0 / (14.0 * 24.0 * 60.0 * 60.0),
    "task": 1.0 / (7.0 * 24.0 * 60.0 * 60.0),
}
DEFAULT_DECAY_LAMBDA_PER_SEC = 1.0 / (21.0 * 24.0 * 60.0 * 60.0)

KIND_PURGE_GRACE_SECONDS = {
    "fact": 30 * 24 * 60 * 60,
    "preference": 14 * 24 * 60 * 60,
    "task": 7 * 24 * 60 * 60,
}
DEFAULT_PURGE_GRACE_SECONDS = 14 * 24 * 60 * 60


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


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_last_retrieved_effective(
    last_retrieved: Optional[int],
    created_at: Optional[int],
    updated_at: Optional[int],
) -> int:
    if last_retrieved is not None:
        return last_retrieved
    if created_at is not None:
        return created_at
    return updated_at or 0


def compute_strength_eff(
    strength_base: float,
    now_ts: int,
    last_retrieved_effective: int,
    kind: Optional[str],
) -> float:
    decay_lambda = KIND_DECAY_LAMBDA_PER_SEC.get(str(kind or "").casefold(), DEFAULT_DECAY_LAMBDA_PER_SEC)
    delta_t = max(0, now_ts - last_retrieved_effective)
    strength_eff = float(strength_base) * math.exp(-decay_lambda * delta_t)
    return max(strength_eff, STRENGTH_EPS)


def _metadata_with_snapshot(
    metadata: Dict[str, Any],
    snapshot: Dict[str, Any],
    write_last_retrieved: bool,
) -> Dict[str, Any]:
    merged = dict(metadata or {})
    existing_mem = merged.get("mem") if isinstance(merged.get("mem"), dict) else {}
    extras = {k: v for k, v in existing_mem.items() if k not in SSOT_FIELDS}
    mem_payload = dict(extras)
    for field in SSOT_FIELDS:
        if field == "last_retrieved" and not write_last_retrieved:
            continue
        mem_payload[field] = snapshot.get(field)
    if not write_last_retrieved and "last_retrieved" in existing_mem:
        mem_payload["last_retrieved"] = existing_mem["last_retrieved"]
    merged["mem"] = mem_payload
    if "created_at" not in merged:
        merged["created_at"] = snapshot.get("updated_at")
    return merged


async def _fetch_row_with_node(
    db: aiosqlite.Connection, node_id: int, active_only: bool = True
) -> Optional[aiosqlite.Row]:
    sql = """
    SELECT
        m.node_id, m.mem_key, m.status, m.kind, m.scope, m.updated_at, m.last_seen,
        m.last_retrieved, m.trust, m.strength, n.metadata, n.content
    FROM mem_kv_index m
    JOIN nodes n ON n.id = m.node_id
    WHERE m.node_id = ?
    """
    if active_only:
        sql += " AND m.status = 'active'"
    async with db.execute(sql, (node_id,)) as cursor:
        return await cursor.fetchone()


async def _refresh_metadata(
    db: aiosqlite.Connection,
    node_id: int,
    write_last_retrieved: bool,
) -> None:
    row = await _fetch_row_with_node(db, node_id, active_only=False)
    if not row:
        return
    snapshot = {field: row[field] for field in SSOT_FIELDS}
    metadata = _load_metadata(row["metadata"])
    merged = _metadata_with_snapshot(metadata, snapshot, write_last_retrieved=write_last_retrieved)
    await db.execute(
        "UPDATE nodes SET metadata = ? WHERE id = ?",
        (_dump_metadata(merged), node_id),
    )


async def list_candidates(
    conn_factory: Callable[[], Any],
    scope: Optional[str] = None,
    limit: int = 128,
) -> List[Dict[str, Any]]:
    async with conn_factory() as db:
        db.row_factory = aiosqlite.Row
        await ensure_mem_schema(db)
        sql = """
        SELECT node_id, mem_key, scope, kind, status, updated_at
        FROM mem_kv_index
        """
        params: List[Any] = []
        if scope is not None:
            sql += " WHERE scope = ?"
            params.append(scope)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def query_memory(
    conn_factory: Callable[[], Any],
    scope: Optional[str] = None,
    kind: Optional[str] = None,
    status: str = "active",
    min_trust: float = DEFAULT_MIN_TRUST,
    min_strength: float = DEFAULT_MIN_STRENGTH,
    top_k: int = 20,
    now_ts: Optional[int] = None,
) -> List[Dict[str, Any]]:
    now = int(now_ts if now_ts is not None else time.time())
    async with conn_factory() as db:
        db.row_factory = aiosqlite.Row
        await ensure_mem_schema(db)
        sql = """
        SELECT
            m.node_id, m.mem_key, m.status, m.kind, m.scope, m.updated_at, m.last_seen,
            m.last_retrieved, m.trust, m.strength, n.metadata, n.content
        FROM mem_kv_index m
        JOIN nodes n ON n.id = m.node_id
        WHERE m.status = ?
        """
        params: List[Any] = [status]
        if scope is not None:
            sql += " AND m.scope = ?"
            params.append(scope)
        if kind is not None:
            sql += " AND m.kind = ?"
            params.append(kind)
        sql += " ORDER BY m.updated_at DESC LIMIT ?"
        params.append(max(top_k * 8, top_k))

        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()

    enriched: List[Dict[str, Any]] = []
    for row in rows:
        trust = float(row["trust"] or 0.0)
        if trust < min_trust:
            continue

        stored_scope = str(row["scope"] or "")
        parsed_scope, canonical_key = split_mem_key(str(row["mem_key"] or ""))
        if stored_scope and parsed_scope and stored_scope != parsed_scope:
            # Keep SSOT consistency strict: skip inconsistent rows.
            continue

        metadata = _load_metadata(row["metadata"])
        created_at = _as_int(metadata.get("created_at"))
        updated_at = _as_int(row["updated_at"]) or 0
        last_retrieved = _as_int(row["last_retrieved"])
        last_retrieved_effective = resolve_last_retrieved_effective(last_retrieved, created_at, updated_at)
        strength_eff = compute_strength_eff(
            strength_base=float(row["strength"] or 0.0),
            now_ts=now,
            last_retrieved_effective=last_retrieved_effective,
            kind=row["kind"],
        )
        if strength_eff < min_strength:
            continue

        enriched.append(
            {
                "node_id": row["node_id"],
                "mem_key": row["mem_key"],
                "canonical_key": canonical_key,
                "scope": parsed_scope or stored_scope,
                "kind": row["kind"],
                "status": row["status"],
                "updated_at": row["updated_at"],
                "last_seen": row["last_seen"],
                "last_retrieved": row["last_retrieved"],
                "last_retrieved_effective": last_retrieved_effective,
                "trust": trust,
                "strength_base": float(row["strength"] or 0.0),
                "strength_eff": strength_eff,
                "content": row["content"],
                "metadata": metadata,
            }
        )

    enriched.sort(key=lambda item: (-item["strength_eff"], -item["trust"], -(item["updated_at"] or 0)))
    return enriched[:top_k]


async def mark_retrieved(
    conn_factory: Callable[[], Any],
    node_ids: Iterable[int],
    now_ts: Optional[int] = None,
) -> Dict[str, Any]:
    now = int(now_ts if now_ts is not None else time.time())
    target_ids = sorted({int(node_id) for node_id in node_ids})
    if not target_ids:
        return {"updated": 0}

    async def _operation(db: aiosqlite.Connection) -> Dict[str, Any]:
        db.row_factory = aiosqlite.Row
        await ensure_mem_schema(db)
        updated = 0
        for node_id in target_ids:
            cursor = await db.execute(
                """
                UPDATE mem_kv_index
                SET last_retrieved = ?, updated_at = ?
                WHERE node_id = ? AND status = 'active'
                """,
                (now, now, node_id),
            )
            if cursor.rowcount:
                updated += cursor.rowcount
                await _refresh_metadata(db, node_id, write_last_retrieved=True)
        return {"updated": updated}

    return await run_immediate_transaction_with_retry(conn_factory, _operation)


async def mark_used(
    conn_factory: Callable[[], Any],
    node_ids: Iterable[int],
    now_ts: Optional[int] = None,
) -> Dict[str, Any]:
    now = int(now_ts if now_ts is not None else time.time())
    target_ids = sorted({int(node_id) for node_id in node_ids})
    if not target_ids:
        return {"updated": 0}

    async def _operation(db: aiosqlite.Connection) -> Dict[str, Any]:
        db.row_factory = aiosqlite.Row
        await ensure_mem_schema(db)
        updated = 0
        for node_id in target_ids:
            row = await _fetch_row_with_node(db, node_id, active_only=True)
            if not row:
                continue
            metadata = _load_metadata(row["metadata"])
            created_at = _as_int(metadata.get("created_at"))
            updated_at = _as_int(row["updated_at"]) or 0
            last_retrieved = _as_int(row["last_retrieved"])
            last_retrieved_effective = resolve_last_retrieved_effective(last_retrieved, created_at, updated_at)
            strength_eff = compute_strength_eff(
                strength_base=float(row["strength"] or 0.0),
                now_ts=now,
                last_retrieved_effective=last_retrieved_effective,
                kind=row["kind"],
            )
            next_strength = min(1.0, max(strength_eff, STRENGTH_EPS) + 0.1)
            await db.execute(
                """
                UPDATE mem_kv_index
                SET strength = ?, last_seen = ?, updated_at = ?
                WHERE node_id = ? AND status = 'active'
                """,
                (next_strength, now, now, node_id),
            )
            updated += 1
            await _refresh_metadata(db, node_id, write_last_retrieved=True)
        return {"updated": updated}

    return await run_immediate_transaction_with_retry(conn_factory, _operation)


async def purge_stale_memories(
    conn_factory: Callable[[], Any],
    now_ts: Optional[int] = None,
) -> Dict[str, Any]:
    now = int(now_ts if now_ts is not None else time.time())

    async def _operation(db: aiosqlite.Connection) -> Dict[str, Any]:
        db.row_factory = aiosqlite.Row
        await ensure_mem_schema(db)
        async with db.execute(
            """
            SELECT
                m.node_id, m.kind, m.last_seen, m.updated_at, n.metadata
            FROM mem_kv_index m
            JOIN nodes n ON n.id = m.node_id
            WHERE m.status = 'active'
            """
        ) as cursor:
            rows = await cursor.fetchall()

        deleted: List[int] = []
        for row in rows:
            metadata = _load_metadata(row["metadata"])
            created_at = _as_int(metadata.get("created_at"))
            updated_at = _as_int(row["updated_at"]) or 0
            last_seen = _as_int(row["last_seen"])
            reference_ts = last_seen if last_seen is not None else (created_at if created_at is not None else updated_at)
            grace = KIND_PURGE_GRACE_SECONDS.get(
                str(row["kind"] or "").casefold(), DEFAULT_PURGE_GRACE_SECONDS
            )
            if now - reference_ts >= grace:
                await db.execute(
                    "UPDATE mem_kv_index SET status = 'deleted', updated_at = ? WHERE node_id = ?",
                    (now, row["node_id"]),
                )
                deleted.append(int(row["node_id"]))
                await _refresh_metadata(db, int(row["node_id"]), write_last_retrieved=True)
        return {"deleted": len(deleted), "node_ids": deleted}

    return await run_immediate_transaction_with_retry(conn_factory, _operation)
