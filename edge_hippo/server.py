import json
import logging

from fastmcp import FastMCP

from .hippo_engine import HippoEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")
mcp = FastMCP("SeahorseRAGMCP")

engine = HippoEngine()
_initialized = False

async def ensure_initialized():
    global _initialized
    if not _initialized:
        await engine.initialize()
        _initialized = True

@mcp.tool()
async def add_document(text: str) -> str:
    """Add a text document to the Knowledge Graph.
    Extracts entities and updates the graph structure.
    """
    await ensure_initialized()
    try:
        await engine.add_document(text)
        return "Document processed and graph updated successfully."
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        return f"Error adding document: {str(e)}"

@mcp.tool()
async def search(query: str, session_id: str = "default") -> str:
    """Search the Knowledge Graph using Personalized PageRank.
    finds relevant passages based on entities in the query.
    """
    await ensure_initialized()
    try:
        result = await engine.search(query, session_id)
        return result
    except Exception as e:
        logger.error(f"Error searching: {e}")
        return f"Error searching: {str(e)}"

@mcp.tool()
async def upsert_memory(
    key: str, 
    value: str, 
    scope: str = "global", 
    kind: str = "fact", 
    trust: float = 1.0, 
    strength: float = 1.0,
    provenance: str = "user_direct"
) -> str:
    """
    Create or update a high-integrity memory node in the knowledge graph.
    Useful for storing specific facts, preferences, or technical decisions.
    
    Args:
        key: Unique identifier for the memory (e.g. 'user_preference_trading_style').
        value: The actual content or fact to remember.
        scope: Context scope ('global' or specific session/user).
        kind: Type of memory ('fact', 'rule', 'preference', etc.).
        trust: Confidence level (0.0 to 1.0).
        strength: Importance/weight (0.0 to 1.0).
        provenance: Source of the information.
    """
    from memory_crud.store import apply_decision
    
    await ensure_initialized()
    
    decision = {
        "action": "create", # apply_decision handles create-as-upsert
        "memory": {
            "key": key,
            "value": value,
            "scope": scope,
            "kind": kind,
            "init": {
                "trust": trust,
                "strength": strength
            },
            "provenance": provenance
        }
    }
    
    try:
        # HippoEngine uses a connection pool, but apply_decision needs a factory or connection.
        # HippoEngine.storage._get_conn() is an async context manager.
        # We'll pass a factory that returns the connection.
        def conn_factory():
            return engine.storage._get_conn()

        result = await apply_decision(conn_factory, decision)
        return f"Memory upserted successfully: {json.dumps(result)}"
    except Exception as e:
        logger.error(f"Error upserting memory: {e}")
        return f"Error upserting memory: {str(e)}"

@mcp.tool()
async def delete_memory(key: str, scope: str = "global") -> str:
    """
    Mark a memory node as deleted.
    
    Args:
        key: The key of the memory to remove.
        scope: The scope of the memory.
    """
    from memory_crud.store import apply_decision, _fetch_active_row_by_mem_key
    from memory_crud.judge import build_mem_key
    
    await ensure_initialized()
    
    def conn_factory():
        return engine.storage._get_conn()
    
    try:
        # Find the node_id first
        mem_key = build_mem_key(scope, key)
        async with engine.storage._get_conn() as db:
            row = await _fetch_active_row_by_mem_key(db, mem_key)
            if not row:
                return f"Memory not found: {mem_key}"
            node_id = row["node_id"]

        decision = {
            "action": "delete",
            "target_node_id": node_id
        }
        
        result = await apply_decision(conn_factory, decision)
        return f"Memory deleted successfully: {json.dumps(result)}"
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        return f"Error deleting memory: {str(e)}"

@mcp.resource("graph://stats")
async def graph_stats() -> str:
    """Get the current graph statistics (node/edge counts)."""
    await ensure_initialized()
    try:
        stats = await engine.storage.verify_integrity()
        return str(stats)
    except Exception as e:
        return f"Error fetching stats: {str(e)}"

def main():
    mcp.run()

if __name__ == "__main__":
    main()
