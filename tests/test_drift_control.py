import pytest
import pytest_asyncio
import tempfile
import os
from pathlib import Path
from contextlib import asynccontextmanager
from edge_hippo.storage import GraphStorage
from edge_hippo.algorithms import check_drift
from edge_hippo.config import settings

@pytest.fixture
def temp_db_path():
    # Create temp DB file
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.remove(path)

@pytest_asyncio.fixture
async def storage(temp_db_path):
    # Override settings to use temp DB
    original_data_dir = settings.DATA_DIR
    
    with tempfile.TemporaryDirectory() as temp_dir:
        settings.DATA_DIR = Path(temp_dir)
        
        # Subclass to bypass vector extension loading for this test
        class MockGraphStorage(GraphStorage):
            @asynccontextmanager
            async def _get_conn(self):
                # Standard aiosqlite connect without loading extension
                import aiosqlite
                async with aiosqlite.connect(self.db_path) as db:
                     # Skip extension loading
                     yield db

            async def initialize(self):
                async with self._get_conn() as db:
                    # Create ONLY standard tables, skip vec0
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

                    # Skip Vec0 table

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
            
            # Override add_node to skip vector insertion
            async def add_node(self, node_type: str, name: str, content: str = "", metadata: dict = None, embedding: list = None) -> int:
                # We ignore embedding here
                return await super().add_node(node_type, name, content, metadata, embedding=None)

        store = MockGraphStorage()
        await store.initialize()
        yield store
    
    # Restore settings
    settings.DATA_DIR = original_data_dir

@pytest.mark.asyncio
async def test_drift_control_connected_direct(storage):
    # Setup: A -> B (Directly Connected)
    # A is current, B is history (or vice versa)
    
    # Add nodes
    # Using 'phrase' type so we can look them up by name
    id_a = await storage.add_node('phrase', 'entity_a', content='content_a')
    id_b = await storage.add_node('phrase', 'entity_b', content='content_b')
    
    # Add edge A -> B
    await storage.add_edge(id_a, id_b)
    
    # Test specific method first
    connected = await storage.check_connectivity([id_a], [id_b])
    assert connected is True, "Should detect direct connection"
    
    # Test high-level drift check
    # History: [entity_a], Current: [entity_b]
    # They are connected, so Drift should be FALSE
    is_drift = await check_drift(storage, ['entity_a'], ['entity_b'])
    assert is_drift is False, "Should NOT detect drift when connected"

@pytest.mark.asyncio
async def test_drift_control_connected_common_neighbor(storage):
    # Setup: A -> C <- B (Shared Neighbor C)
    
    id_a = await storage.add_node('phrase', 'entity_a')
    id_b = await storage.add_node('phrase', 'entity_b')
    id_c = await storage.add_node('phrase', 'entity_c') # Common
    
    # A -> C
    await storage.add_edge(id_a, id_c)
    # B -> C
    await storage.add_edge(id_b, id_c)
    
    # Check connectivity between A and B
    connected = await storage.check_connectivity([id_a], [id_b])
    assert connected is True, "Should detect common neighbor"
    
    is_drift = await check_drift(storage, ['entity_a'], ['entity_b'])
    assert is_drift is False

@pytest.mark.asyncio
async def test_drift_control_disconnected(storage):
    # Setup: A -> B,  C -> D (Two separate islands)
    
    id_a = await storage.add_node('phrase', 'entity_a')
    id_b = await storage.add_node('phrase', 'entity_b')
    await storage.add_edge(id_a, id_b)
    
    id_c = await storage.add_node('phrase', 'entity_c')
    id_d = await storage.add_node('phrase', 'entity_d')
    await storage.add_edge(id_c, id_d)
    
    # Check A vs C
    connected = await storage.check_connectivity([id_a], [id_c])
    assert connected is False, "Should NOT detect connection"
    
    is_drift = await check_drift(storage, ['entity_a'], ['entity_c'])
    assert is_drift is True, "Should detect drift"

@pytest.mark.asyncio
async def test_drift_control_missing_entities(storage):
    # History exists but Current is new/unknown
    id_a = await storage.add_node('phrase', 'entity_a')
    
    # entity_b does not exist in graph
    is_drift = await check_drift(storage, ['entity_b'], ['entity_a'])
    
    # Should default to Drift=True (since connection checks will fail)
    # The check_drift logic does `get_node_by_name`. If None, list is empty.
    # If list is empty, it returns True.
    assert is_drift is True
