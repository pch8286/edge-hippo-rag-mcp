import asyncio
import math
import random
from typing import Any, Awaitable, Callable, Optional, TypeVar

import aiosqlite

TRUST_EPS = 0.05
STRENGTH_EPS = 0.01
DEFAULT_MIN_TRUST = 0.05
DEFAULT_MIN_STRENGTH = 0.01

if DEFAULT_MIN_TRUST > TRUST_EPS:
    raise RuntimeError("DEFAULT_MIN_TRUST must be <= TRUST_EPS")
if DEFAULT_MIN_STRENGTH > STRENGTH_EPS:
    raise RuntimeError("DEFAULT_MIN_STRENGTH must be <= STRENGTH_EPS")

BUSY_TIMEOUT_MS = 5_000
MAX_LOCK_RETRIES = 5
LOCK_BACKOFF_BASE_SECONDS = 0.05

MEM_KV_INDEX_DDL_SQL = """CREATE TABLE IF NOT EXISTS mem_kv_index (
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

MEM_KV_INDEX_INDEX_SQL = [
    """CREATE UNIQUE INDEX IF NOT EXISTS ux_mem_key_active
ON mem_kv_index(mem_key)
WHERE status = 'active';""",
    "CREATE INDEX IF NOT EXISTS ix_mem_updated_at ON mem_kv_index(updated_at);",
    "CREATE INDEX IF NOT EXISTS ix_mem_last_seen ON mem_kv_index(last_seen);",
    "CREATE INDEX IF NOT EXISTS ix_mem_last_retrieved ON mem_kv_index(last_retrieved);",
]

_LOCK_ERROR_TOKENS = (
    "database is locked",
    "database table is locked",
    "database schema is locked",
    "database is busy",
)

T = TypeVar("T")


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _as_finite_float(raw: Any, field_name: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def sanitize_trust(raw: Any) -> float:
    return max(clamp01(_as_finite_float(raw, "trust")), TRUST_EPS)


def sanitize_strength(raw: Any) -> float:
    return max(clamp01(_as_finite_float(raw, "strength")), STRENGTH_EPS)


def is_lock_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(token in message for token in _LOCK_ERROR_TOKENS)


async def configure_connection(db: aiosqlite.Connection) -> None:
    await db.execute("PRAGMA foreign_keys=ON")
    await db.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")


async def configure_wal(db: aiosqlite.Connection) -> None:
    # WAL is required for better writer/read concurrency.
    await db.execute("PRAGMA journal_mode=WAL")


async def ensure_mem_schema(db: aiosqlite.Connection) -> None:
    await db.execute(MEM_KV_INDEX_DDL_SQL)
    for index_sql in MEM_KV_INDEX_INDEX_SQL:
        await db.execute(index_sql)


async def run_immediate_transaction_with_retry(
    conn_factory: Callable[[], Any],
    operation: Callable[[aiosqlite.Connection], Awaitable[T]],
    retries: int = MAX_LOCK_RETRIES,
) -> T:
    last_exc: Optional[BaseException] = None
    for attempt in range(retries):
        async with conn_factory() as db:
            await configure_connection(db)
            try:
                await db.execute("BEGIN IMMEDIATE")
                result = await operation(db)
                await db.commit()
                return result
            except Exception as exc:  # pragma: no cover - rollback branch is covered through tests
                last_exc = exc
                await db.rollback()
                if not is_lock_error(exc) or attempt == retries - 1:
                    raise
                backoff = LOCK_BACKOFF_BASE_SECONDS * (2**attempt)
                backoff += random.uniform(0.0, LOCK_BACKOFF_BASE_SECONDS / 2.0)
                await asyncio.sleep(backoff)

    if last_exc:
        raise last_exc
    raise RuntimeError("transaction retry failed unexpectedly")
