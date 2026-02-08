import aiosqlite
import sqlite3
import json
import logging
import sqlite_vec
from typing import List, Optional, Tuple, Dict, Any
from contextlib import asynccontextmanager
from .config import settings

logger = logging.getLogger(__name__)

class GraphStorage:
    def __init__(self):
        self.db_path = settings.db_path
        self.ext_path = sqlite_vec.loadable_path()
        self.extension_loaded = False

    @asynccontextmanager
    async def _get_conn(self):
        async with aiosqlite.connect(self.db_path) as db:
            try:
                await db.enable_load_extension(True)
                await db.load_extension(self.ext_path)
                self.extension_loaded = True
            except (AttributeError, Exception) as e:
                self.extension_loaded = False
                logger.warning(f"Failed to load SQLite extension: {e}. Vector search will be disabled.")
            yield db

    async def initialize(self):
        async with self._get_conn() as db:
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
            
            # Find degree threshold
            # Subquery gets counts.
            # Window function ntile or just math?
            # SQLite doesn't have percentile_cont easily without extensions.
            # We can select counts, fetch all, determine threshold in python for simplicity on RPi.
            
            # Count degrees for phrases (type='phrase')
            # Assuming edges are stored directionally or bidirectionally? 
            # add_document adds (passage->phrase) and (phrase->passage).
            # So degree = count(*) in 'edges' where target=id (incoming from passages).
            
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

    async def get_ego_subgraph(self, seed_ids: List[int], depth: int = 2, limit: int = 1500) -> Dict[str, Any]:
        """
        Retrieve k-hop subgraph around seeds.
        Returns: {
            "nodes": [{id, type, name, is_hub, embedding}, ...],
            "edges": [(source, target, weight), ...]
        }
        Uses Vector Pre-fetching via subquery/join.
        """
        if not seed_ids:
            return {"nodes": [], "edges": []}

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
            
            async with db.execute(query, (depth, limit)) as cursor:
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

            # Fetch Edges within subgraph
            async with db.execute(f"""
                SELECT source, target, weight 
                FROM edges 
                WHERE source IN ({node_ids_str}) 
                AND target IN ({node_ids_str})
            """) as cursor:
                edges = await cursor.fetchall()

            return {"nodes": nodes, "edges": edges}

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
            
            # 1. Direct Connection Check
            # Check A->B or B->A
            query_direct = f"""
                SELECT 1 FROM edges 
                WHERE (source IN ({str_a}) AND target IN ({str_b}))
                LIMIT 1
            """
            async with db.execute(query_direct) as cursor:
                if await cursor.fetchone():
                    return True
            
            # 2. Common Neighbor Check (1-hop shared)
            # Find n where A->n AND B->n
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
