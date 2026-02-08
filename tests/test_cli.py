import pytest
from edge_hippo.cli import main
import sys
from unittest.mock import patch, MagicMock, AsyncMock
import io

@pytest.fixture
def mock_engine():
    with patch("edge_hippo.cli.HippoEngine") as mock:
        engine_instance = mock.return_value
        engine_instance.initialize = AsyncMock()
        engine_instance.add_document = AsyncMock()
        engine_instance.search = AsyncMock(return_value="mock result")
        
        storage_mock = MagicMock()
        storage_mock.verify_integrity = AsyncMock(return_value={"nodes": 10})
        engine_instance.storage = storage_mock
        
        yield engine_instance

def test_cli_help():
    with patch.object(sys, 'argv', ['hippo', '--help']):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0

def test_cli_stats(mock_engine):
    with patch.object(sys, 'argv', ['hippo', 'stats']):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            main()
            output = fake_out.getvalue()
            assert "Graph Statistics:" in output
            assert "nodes: 10" in output
            mock_engine.initialize.assert_called_once()

def test_cli_search(mock_engine):
    with patch.object(sys, 'argv', ['hippo', 'search', 'test query']):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            main()
            output = fake_out.getvalue()
            assert "Searching for: test query" in output
            assert "Results:" in output
            assert "mock result" in output
            mock_engine.search.assert_called_once_with("test query")

def test_cli_index_text(mock_engine):
    with patch.object(sys, 'argv', ['hippo', 'index', '--text', 'hello world']):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            main()
            output = fake_out.getvalue()
            assert "Indexing content..." in output
            assert "Indexing complete." in output
            mock_engine.add_document.assert_called_once_with("hello world")
