import pytest
import shutil
import sys
from unittest.mock import MagicMock
from pathlib import Path

# Mock heavy dependencies if missing
for mod_name in ["gliner", "igraph", "sentence_transformers", "fastmcp"]:
    try:
        __import__(mod_name)
    except ImportError:
        mock_mod = MagicMock()
        if mod_name == "fastmcp":
            # FastMCP needs to handle decorators @mcp.tool()
            mock_instance = MagicMock()
            def decorator(*args, **kwargs):
                def wrapper(f): return f
                return wrapper
            mock_instance.tool.side_effect = lambda *args, **kwargs: decorator
            mock_instance.resource.side_effect = lambda *args, **kwargs: decorator
            mock_mod.FastMCP.return_value = mock_instance
            
        sys.modules[mod_name] = mock_mod

from edge_hippo.config import settings

@pytest.fixture
def temp_data_dir(tmp_path):
    """Fixture to provide a temporary data directory."""
    original_data_dir = settings.DATA_DIR
    settings.DATA_DIR = tmp_path
    if not settings.DATA_DIR.exists():
        settings.DATA_DIR.mkdir()
    
    yield settings.DATA_DIR
    
    # Cleanup
    if settings.DATA_DIR.exists():
        shutil.rmtree(settings.DATA_DIR)
    settings.DATA_DIR = original_data_dir

@pytest.fixture
def vector_search_supported():
    """Check if SQLite vector compilation is supported."""
    import sqlite3
    conn = sqlite3.connect(":memory:")
    try:
        support = hasattr(conn, "enable_load_extension")
        return support
    finally:
        conn.close()
