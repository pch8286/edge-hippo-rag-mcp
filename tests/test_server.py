import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

# Need to ensure FastMCP mock behaves like a decorator
@pytest.fixture(autouse=True)
def mock_fastmcp_setup():
    # Setup FastMCP mock to handle decorators before importing server
    import sys
    # conftest already mocked fastmcp module.
    # We need to refine the mock in sys.modules['fastmcp'] specifically for FastMCP class.
    
    mock_mcp_instance = MagicMock()
    # Decorator passthrough: @mcp.tool() -> returns function
    def tool_decorator():
        def wrapper(func):
            return func
        return wrapper
    
    # Allow @mcp.tool() or @mcp.tool
    mock_mcp_instance.tool.side_effect = lambda *args, **kwargs: tool_decorator()
    mock_mcp_instance.resource.side_effect = lambda *args, **kwargs: tool_decorator()
    
    # Patch the FastMCP class to return our instance
    with patch("fastmcp.FastMCP", return_value=mock_mcp_instance):
        yield mock_mcp_instance

@pytest.mark.asyncio
async def test_server_tools():
    # Import server inside test to ensure patching applies
    from edge_hippo import server
    
    # Mock engine methods
    server.engine.add_document = AsyncMock(return_value=None)
    server.engine.search = AsyncMock(return_value="Result")
    server.engine.initialize = AsyncMock()
    server.engine.storage.verify_integrity = AsyncMock(return_value={"nodes": 1})

    # Test add_document
    res = await server.add_document("test text")
    assert "successfully" in res
    server.engine.add_document.assert_called_once()
    
    # Test search
    res = await server.search("query")
    assert res == "Result"
    server.engine.search.assert_called_once()
    
    # Test stats
    res = await server.graph_stats()
    assert "{'nodes': 1}" in res
