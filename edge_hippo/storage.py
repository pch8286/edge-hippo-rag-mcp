import aiosqlite
import sqlite3
import json
import logging
import math
import sqlite_vec
import ctypes
import os
import ctypes.util
from typing import List, Optional, Tuple, Dict, Any
from contextlib import asynccontextmanager
from .config import settings
from memory_crud.schema import configure_connection, configure_wal, ensure_mem_schema

logger = logging.getLogger(__name__)

def _register_sqlite_vec():
    try:
        lib_path = ctypes.util.find_library('sqlite3')
        if not lib_path:
            lib_path = 'libsqlite3.so.0'
        sqlite_lib = ctypes.CDLL(lib_path)
        
        auto_ext = sqlite_lib.sqlite3_auto_extension
        auto_ext.argtypes = [ctypes.c_void_p]
        auto_ext.restype = ctypes.c_int
        
        vec_lib_path = sqlite_vec.loadable_path()
        if not vec_lib_path.endswith(".so"):
            vec_lib_path += ".so"
            
        if not os.path.exists(vec_lib_path):
            vec_lib_path = vec_lib_path.replace(".so", ".so") 

        vec_lib = ctypes.CDLL(vec_lib_path)
        vec_init = vec_lib.sqlite3_vec_init
        
        res = auto_ext(ctypes.cast(vec_init, ctypes.c_void_p))
        if res == 0:
            logger.info("Successfully registered sqlite-vec via auto-extension.")
            return True
        else:
            logger.warning(f"Failed to register sqlite-vec: Result code {res}")
    except Exception as e:
        logger.warning(f"SQLite extension workaround failed: {e}")
    return False

LOAD_EXTENSION_HACK_SUCCESS = _register_sqlite_vec()

class GraphStorage:
    def __init__(self):
        self.db_path = settings.db_path
        self.extension_loaded = LOAD_EXTENSION_HACK_SUCCESS

    @asynccontextmanager
    async def _get_conn(self):
        async with aiosqlite.connect(self.db_path) as db:
            await configure_connection(db)
            yield db

    async def initialize(self):
        async with self._get_conn() as db:
            await configure_wal(db)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    name TEXT,
                    content TEXT,
                    metadata TEXT,
                    is_hub INTEGER DEFAULT 0
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name) WHERE type='phrase'")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)")

            try:
                await db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vec_nodes USING vec0(embedding float[384])")
            except Exception as e:
                logger.info(f"Using fallback standard table for vec_nodes (Extension unavailable): {e}")
                await db.execute("CREATE TABLE IF NOT EXISTS vec_nodes (rowid INTEGER PRIMARY KEY, embedding BLOB)")

            await db.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    source INTEGER,
                    target INTEGER,
                    weight REAL DEFAULT 1.0,
                    FOREIGN KEY(source) REFERENCES nodes(id),
                    FOREIGN KEY(target) REFERENCES nodes(id),
                    UNIQUE(source, target)
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target)")
            await ensure_mem_schema(db)
            await db.commit()

    async def add_node(self, node_type: str, name: str, content: str = "", metadata: Dict = None, embedding: List[float] = None) -> int:
        """
        Add a node. 
        For phrases, 'name' is the unique identifier.
        """
        async with self._get_conn() as db:
            if node_type == 'phrase':
                async with db.execute("SELECT id FROM nodes WHERE name = ? AND type = 'phrase'", (name,)) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        return row[0]
            
            meta_json = json.dumps(metadata) if metadata else "{}"
            cursor = await db.execute(
                "INSERT INTO nodes (type, name, content, metadata) VALUES (?, ?, ?, ?)",
                (node_type, name, content, meta_json)
            )
            node_id = cursor.lastrowid
            

            if embedding:
                logger.debug(f"Inserting embedding for node_id={node_id}")
                try:
                    await db.execute("DELETE FROM vec_nodes WHERE rowid = ?", (node_id,))
                    
                    try:
                        blob = sqlite_vec.serialize_float32(embedding)
                    except AttributeError:
                        import struct
                        blob = struct.pack(f'{len(embedding)}f', *embedding)
                    
                    await db.execute("INSERT INTO vec_nodes(rowid, embedding) VALUES (?, ?)", (node_id, blob))
                    
                except Exception as e:
                    logger.error(f"Failed to insert embedding for node {node_id}: {e}")
                    raise e

            await db.commit()
            return node_id

    async def add_edge(self, source: int, target: int, weight: float = 1.0):
        async with self._get_conn() as db:
            try:
                await db.execute(
                    "INSERT INTO edges (source, target, weight) VALUES (?, ?, ?) ON CONFLICT(source, target) DO UPDATE SET weight=weight",
                    (source, target, weight)
                )
                await db.commit()
            except Exception as e:
                logger.error(f"Failed to add edge {source}->{target}: {e}")

    async def get_node_by_name(self, name: str, node_type: str = 'phrase') -> Optional[int]:
        async with self._get_conn() as db:
            async with db.execute("SELECT id FROM nodes WHERE name = ? AND type = ?", (name, node_type)) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else None

    async def get_node_content(self, node_id: int) -> Optional[str]:
         async with self._get_conn() as db:
            async with db.execute("SELECT content FROM nodes WHERE id = ?", (node_id,)) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else None
                
    async def get_all_edges(self) -> List[Tuple[int, int, float]]:
        """Fetch all edges for graph construction."""
        async with self._get_conn() as db:
            async with db.execute("SELECT source, target, weight FROM edges") as cursor:
                return await cursor.fetchall()
    
    async def get_all_passage_ids(self) -> List[int]:
        async with self._get_conn() as db:
            async with db.execute("SELECT id FROM nodes WHERE type='passage'") as cursor:
                rows = await cursor.fetchall()
                return [r[0] for r in rows]

    async def flag_hub_nodes(self, percentile: float = 0.99):
        """
        Identify high-degree nodes and flag them as hubs.
        """
        async with self._get_conn() as db:
            await db.execute("UPDATE nodes SET is_hub = 0")
            rows = await db.execute("""
                SELECT target, COUNT(*) as degree 
                FROM edges 
                JOIN nodes ON edges.target = nodes.id 
                WHERE nodes.type = 'phrase' 
                GROUP BY target
                ORDER BY degree DESC
            """)
            degrees = await rows.fetchall()
            if not degrees:
                await db.commit()
                return

            cutoff_idx = int(len(degrees) * (1.0 - percentile))
            # E.g. top 1%: cutoff is index (N * 0.01).
            # If 100 items, top 1% = 1 item. (1 - 0.99) = 0.01. 100 * 0.01 = 1.
            # Slice top k.
            
            cutoff_idx = max(1, cutoff_idx) # At least 1 if exists? Or 0 if empty?
            # If len < 100, might flag top 1.
            
            top_hubs = [r[0] for r in degrees[:cutoff_idx]]
            
            if top_hubs:
                placeholders = ",".join("?" * len(top_hubs))
                await db.execute(f"UPDATE nodes SET is_hub = 1 WHERE id IN ({placeholders})", top_hubs)
                await db.commit()
                logger.info(f"Flagged {len(top_hubs)} nodes as Hubs (Top {(1-percentile)*100:.1f}%).")

    async def search_vectors(self, query_vec: List[float], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for similar entities using sqlite-vec.
        Returns list of (node_id, distance).
        """
        if not self.extension_loaded:
            logger.debug("Extension not loaded, skipping vector search.")
            return []
            
        async with self._get_conn() as db:
            try:
                try:
                    query_blob = sqlite_vec.serialize_float32(query_vec)
                except AttributeError:
                    import struct
                    query_blob = struct.pack(f'{len(query_vec)}f', *query_vec)
                
                async with db.execute("""
                    SELECT rowid, distance 
                    FROM vec_nodes 
                    WHERE embedding MATCH ? 
                    AND k = ? 
                    ORDER BY distance
                """, (query_blob, top_k)) as cursor:
                    results = await cursor.fetchall()
                    return results # [(rowid, distance), ...]
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                return []

    async def get_ego_subgraph(
        self,
        seed_ids: List[int],
        depth: int = 2,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve k-hop subgraph around seeds.
        Returns: {
            "nodes": [{id, type, name, is_hub, embedding}, ...],
            "edges": [(source, target, weight), ...]
        }
        Uses Vector Pre-fetching via subquery/join.
        ``limit`` is intentionally variable. If omitted, it uses
        ``settings.HIPPO_NODE_MAX`` when configured, otherwise a conservative
        default (5000).
        """
        if not seed_ids:
            return {"nodes": [], "edges": []}
        resolved_limit = int(limit if limit is not None else (settings.HIPPO_NODE_MAX or 5000))
        resolved_limit = max(1, resolved_limit)

        async with self._get_conn() as db:
            seed_str = ",".join(map(str, seed_ids))

            query = f"""
            WITH RECURSIVE visited(id, depth) AS (
                SELECT id, 0 as depth FROM nodes WHERE id IN ({seed_str})
                UNION
                SELECT e.target, v.depth + 1
                FROM edges e
                JOIN visited v ON e.source = v.id
                JOIN nodes n ON v.id = n.id
                WHERE v.depth < ?
                AND n.is_hub = 0
            )
            SELECT DISTINCT id FROM visited
            UNION
            SELECT e.target 
            FROM edges e
            JOIN visited v ON e.source = v.id
            JOIN nodes n ON e.target = n.id
            WHERE n.type = 'passage'
            LIMIT ?
            """
            
            async with db.execute(query, (depth, resolved_limit)) as cursor:
                node_ids = [r[0] for r in await cursor.fetchall()]

            
            if not node_ids:
                return {"nodes": [], "edges": []}
            
            node_ids_str = ",".join(map(str, node_ids))
            
            async with db.execute(f"""
                SELECT n.id, n.type, n.name, n.is_hub, v.embedding
                FROM nodes n
                LEFT JOIN vec_nodes v ON n.id = v.rowid
                WHERE n.id IN ({node_ids_str})
            """) as cursor:
                 raw_nodes = await cursor.fetchall()
            
            nodes = []
            import struct
            for row in raw_nodes:
                nid, ntype, nname, is_hub, emb_blob = row
                emb_list = None
                if emb_blob:
                    try:
                        count = len(emb_blob) // 4
                        emb_list = list(struct.unpack(f'{count}f', emb_blob))
                    except Exception as e:
                        logger.warning(f"Failed to unpack embedding for node {nid}: {e}")
                
                nodes.append({
                    "id": nid,
                    "type": ntype,
                    "name": nname,
                    "is_hub": bool(is_hub),
                    "embedding": emb_list
                })

            async with db.execute(f"""
                SELECT source, target, weight 
                FROM edges 
                WHERE source IN ({node_ids_str}) 
                AND target IN ({node_ids_str})
            """) as cursor:
                edges = await cursor.fetchall()

            return {"nodes": nodes, "edges": edges}

    async def get_transition_stats(
        self, node_ids: List[int]
    ) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
        """Fetch global degree/IDF stats needed by retrieval-time transition logic."""
        if not node_ids:
            return {}, {"AVG_PHRASE_DEG": 1.0, "AVG_PASSAGE_PHRASE_DEG": 1.0}

        placeholders = ",".join(["?"] * len(node_ids))
        node_params = tuple(int(n) for n in node_ids)

        async with self._get_conn() as db:
            async with db.execute(
                f"""
                SELECT source, COUNT(*)
                FROM edges
                WHERE source IN ({placeholders})
                GROUP BY source
                """,
                node_params,
            ) as cursor:
                global_deg_rows = await cursor.fetchall()
            global_phrase_deg = {int(node_id): float(cnt) for node_id, cnt in global_deg_rows}

            async with db.execute(
                f"""
                SELECT e.source, COUNT(*)
                FROM edges e
                JOIN nodes s ON s.id = e.source
                JOIN nodes t ON t.id = e.target
                WHERE e.source IN ({placeholders})
                AND s.type = 'passage'
                AND t.type = 'phrase'
                GROUP BY e.source
                """,
                node_params,
            ) as cursor:
                passage_deg_rows = await cursor.fetchall()
            global_passage_phrase_deg = {
                int(node_id): float(cnt) for node_id, cnt in passage_deg_rows
            }

            async with db.execute(
                f"""
                SELECT e.target, COUNT(DISTINCT e.source)
                FROM edges e
                JOIN nodes s ON s.id = e.source
                WHERE e.target IN ({placeholders})
                AND s.type = 'passage'
                GROUP BY e.target
                """,
                node_params,
            ) as cursor:
                df_rows = await cursor.fetchall()
            df_map = {int(node_id): float(df) for node_id, df in df_rows}

            async with db.execute("SELECT COUNT(*) FROM nodes WHERE type='passage'") as cursor:
                total_passages_row = await cursor.fetchone()
                total_passages = float(total_passages_row[0] if total_passages_row else 0.0)

            async with db.execute(
                """
                SELECT AVG(cnt)
                FROM (
                  SELECT e.source AS sid, COUNT(*) AS cnt
                  FROM edges e
                  JOIN nodes n ON n.id = e.source
                  WHERE n.type = 'phrase'
                  GROUP BY e.source
                )
                """
            ) as cursor:
                avg_phrase_row = await cursor.fetchone()
            avg_phrase_deg = float(avg_phrase_row[0] or 1.0) if avg_phrase_row else 1.0

            async with db.execute(
                """
                SELECT AVG(cnt)
                FROM (
                  SELECT e.source AS sid, COUNT(*) AS cnt
                  FROM edges e
                  JOIN nodes s ON s.id = e.source
                  JOIN nodes t ON t.id = e.target
                  WHERE s.type = 'passage' AND t.type = 'phrase'
                  GROUP BY e.source
                )
                """
            ) as cursor:
                avg_passage_row = await cursor.fetchone()
            avg_passage_phrase_deg = (
                float(avg_passage_row[0] or 1.0) if avg_passage_row else 1.0
            )

            async with db.execute(
                f"""
                SELECT id, type FROM nodes WHERE id IN ({placeholders})
                """,
                node_params,
            ) as cursor:
                type_rows = await cursor.fetchall()

        stats: Dict[int, Dict[str, float]] = {}
        n_passages = max(total_passages, 1.0)
        for node_id, node_type in type_rows:
            node_id = int(node_id)
            g_phrase = global_phrase_deg.get(node_id, 0.0) if node_type == "phrase" else 0.0
            g_passage = (
                global_passage_phrase_deg.get(node_id, 0.0)
                if node_type == "passage"
                else 0.0
            )
            if node_type == "phrase":
                df = df_map.get(node_id, 0.0)
                idf = math.log((n_passages + 1.0) / (df + 1.0)) + 1.0
                idf = float(min(8.0, max(0.1, idf)))
            else:
                idf = 0.0
            stats[node_id] = {
                "global_phrase_deg": float(g_phrase),
                "global_passage_phrase_deg": float(g_passage),
                "idf": float(idf),
            }

        return stats, {
            "AVG_PHRASE_DEG": float(max(avg_phrase_deg, 1.0)),
            "AVG_PASSAGE_PHRASE_DEG": float(max(avg_passage_phrase_deg, 1.0)),
        }

    async def check_connectivity(self, group_a: List[int], group_b: List[int]) -> bool:
        """
        Check if two groups of nodes are topologically related.
        Criteria:
        1. Direct Connection: Edge between any a in A and b in B.
        2. Common Neighbor: Any node n s.t. A->n and B->n.
        """
        if not group_a or not group_b:
            return False

        async with self._get_conn() as db:
            str_a = ",".join(map(str, group_a))
            str_b = ",".join(map(str, group_b))
            
            query_direct = f"""
                SELECT 1 FROM edges 
                WHERE (source IN ({str_a}) AND target IN ({str_b}))
                LIMIT 1
            """
            async with db.execute(query_direct) as cursor:
                if await cursor.fetchone():
                    return True
            
            query_common = f"""
                SELECT 1 
                FROM edges e1
                JOIN edges e2 ON e1.target = e2.target
                WHERE e1.source IN ({str_a}) 
                AND e2.source IN ({str_b})
                LIMIT 1
            """
            async with db.execute(query_common) as cursor:
                if await cursor.fetchone():
                    return True
                    
        return False

    async def verify_integrity(self) -> Dict[str, Any]:
        """Verify graph integrity and return stats."""
        stats = {}
        async with self._get_conn() as db:
            async with db.execute("SELECT count(*) FROM nodes") as cursor:
                stats["total_nodes"] = (await cursor.fetchone())[0]
            async with db.execute("SELECT count(*) FROM edges") as cursor:
                stats["total_edges"] = (await cursor.fetchone())[0]
            async with db.execute("SELECT count(*) FROM nodes WHERE is_hub = 1") as cursor:
                stats["hub_nodes"] = (await cursor.fetchone())[0]
            async with db.execute("SELECT type, count(*) FROM nodes GROUP BY type") as cursor:
                async for row in cursor:
                    stats[f"{row[0]}_nodes"] = row[1]
        return stats
