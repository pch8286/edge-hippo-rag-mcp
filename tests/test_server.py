"""MCP server tests for the preferred seahorse entrypoint."""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


@pytest.fixture(autouse=True)
def mock_fastmcp_setup():
    """Setup FastMCP mock to handle decorators before importing server."""
    mock_mcp_instance = MagicMock()

    def tool_decorator(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper

    mock_mcp_instance.tool.side_effect = tool_decorator
    mock_mcp_instance.resource.side_effect = tool_decorator

    with patch("fastmcp.FastMCP", return_value=mock_mcp_instance):
        yield mock_mcp_instance


class TestServerTools:
    @pytest.mark.asyncio
    async def test_add_document_success(self):
        from seahorse import server
        server.engine.add_document = AsyncMock(return_value=None)
        server.engine.initialize = AsyncMock()
        server._initialized = False

        res = await server.add_document("test text")
        assert "successfully" in res
        server.engine.add_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_document_error(self):
        from seahorse import server
        server.engine.add_document = AsyncMock(side_effect=Exception("boom"))
        server.engine.initialize = AsyncMock()
        server._initialized = False

        res = await server.add_document("bad text")
        assert "Error" in res

    @pytest.mark.asyncio
    async def test_search_success(self):
        from seahorse import server
        server.engine.search = AsyncMock(return_value="Result")
        server.engine.initialize = AsyncMock()
        server._initialized = False

        res = await server.search("query")
        assert res == "Result"
        server.engine.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_error(self):
        from seahorse import server
        server.engine.search = AsyncMock(side_effect=Exception("fail"))
        server.engine.initialize = AsyncMock()
        server._initialized = False

        res = await server.search("query")
        assert "Error" in res

    @pytest.mark.asyncio
    async def test_graph_stats_success(self):
        from seahorse import server
        server.engine.initialize = AsyncMock()
        server.engine.storage.verify_integrity = AsyncMock(
            return_value={"nodes": 1}
        )
        server._initialized = False

        res = await server.graph_stats()
        assert "nodes" in res

    @pytest.mark.asyncio
    async def test_graph_stats_error(self):
        from seahorse import server
        server.engine.initialize = AsyncMock()
        server.engine.storage.verify_integrity = AsyncMock(
            side_effect=Exception("db error")
        )
        server._initialized = False

        res = await server.graph_stats()
        assert "Error" in res

    @pytest.mark.asyncio
    async def test_ensure_initialized_idempotent(self):
        from seahorse import server
        server.engine.initialize = AsyncMock()
        server._initialized = False

        await server.ensure_initialized()
        await server.ensure_initialized()
        # Only called once since _initialized is set to True after first call
        server.engine.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_memory_uses_connection_context_manager(self):
        from seahorse import server

        sentinel_db = object()
        state = {"entered": False, "exited": False}

        class DummyConnManager:
            async def __aenter__(self):
                state["entered"] = True
                return sentinel_db

            async def __aexit__(self, exc_type, exc, tb):
                state["exited"] = True
                return False

        server.engine.initialize = AsyncMock()
        server.engine.storage._get_conn = MagicMock(return_value=DummyConnManager())
        server._initialized = False

        async def fake_apply_decision(conn_factory, decision):
            assert decision["action"] == "create"
            assert decision["memory"]["key"] == "user_pref"
            async with conn_factory() as db:
                assert db is sentinel_db
            return {"status": "ok"}

        with patch("memory_crud.store.apply_decision", new=AsyncMock(side_effect=fake_apply_decision)):
            res = await server.upsert_memory("user_pref", "value")

        assert "successfully" in res
        assert json.loads(res.split(": ", 1)[1]) == {"status": "ok"}
        assert state == {"entered": True, "exited": True}

    @pytest.mark.asyncio
    async def test_delete_memory_uses_connection_context_manager(self):
        from seahorse import server

        sentinel_db = object()
        state = {"entered": 0, "exited": 0}

        class DummyConnManager:
            async def __aenter__(self):
                state["entered"] += 1
                return sentinel_db

            async def __aexit__(self, exc_type, exc, tb):
                state["exited"] += 1
                return False

        server.engine.initialize = AsyncMock()
        server.engine.storage._get_conn = MagicMock(return_value=DummyConnManager())
        server._initialized = False

        async def fake_fetch_active_row_by_mem_key(db, mem_key):
            assert db is sentinel_db
            assert mem_key == "global:user_pref"
            return {"node_id": 7}

        async def fake_apply_decision(conn_factory, decision):
            assert decision == {"action": "delete", "target_node_id": 7}
            async with conn_factory() as db:
                assert db is sentinel_db
            return {"status": "ok"}

        with patch("memory_crud.store._fetch_active_row_by_mem_key", new=AsyncMock(side_effect=fake_fetch_active_row_by_mem_key)):
            with patch("memory_crud.store.apply_decision", new=AsyncMock(side_effect=fake_apply_decision)):
                res = await server.delete_memory("user_pref")

        assert "successfully" in res
        assert json.loads(res.split(": ", 1)[1]) == {"status": "ok"}
        assert state["entered"] == 2
        assert state["exited"] == 2

    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self):
        from seahorse import server

        sentinel_db = object()

        class DummyConnManager:
            async def __aenter__(self):
                return sentinel_db

            async def __aexit__(self, exc_type, exc, tb):
                return False

        server.engine.initialize = AsyncMock()
        server.engine.storage._get_conn = MagicMock(return_value=DummyConnManager())
        server._initialized = False

        with patch("memory_crud.store._fetch_active_row_by_mem_key", new=AsyncMock(return_value=None)):
            res = await server.delete_memory("missing_key")

        assert res == "Memory not found: global:missing_key"
