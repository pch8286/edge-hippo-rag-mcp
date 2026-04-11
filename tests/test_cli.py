"""CLI tests for the preferred seahorse entrypoint."""

import pytest
import sys
import io
from unittest.mock import patch, MagicMock, AsyncMock
from seahorse.cli import main

CLI_PROG = "seahorse"


@pytest.fixture
def mock_engine():
    with patch("seahorse.cli.HippoEngine") as mock:
        inst = mock.return_value
        inst.initialize = AsyncMock()
        inst.add_document = AsyncMock()
        inst.search = AsyncMock(return_value="mock result")
        inst.optimize_synonyms = AsyncMock(return_value=5)

        storage_mock = MagicMock()
        storage_mock.verify_integrity = AsyncMock(return_value={"nodes": 10})
        inst.storage = storage_mock

        yield inst


class TestCLI:
    def test_help(self):
        with patch.object(sys, "argv", [CLI_PROG, "--help"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0

    def test_no_command_shows_help(self):
        with patch.object(sys, "argv", [CLI_PROG]):
            with patch("sys.stdout", new=io.StringIO()):
                main()  # should not crash

    def test_stats(self, mock_engine):
        with patch.object(sys, "argv", [CLI_PROG, "stats"]):
            with patch("sys.stdout", new=io.StringIO()) as out:
                main()
                output = out.getvalue()
                assert "Graph Statistics:" in output
                assert "nodes: 10" in output
                mock_engine.initialize.assert_called_once()

    def test_search(self, mock_engine):
        with patch.object(sys, "argv", [CLI_PROG, "search", "test query"]):
            with patch("sys.stdout", new=io.StringIO()) as out:
                main()
                output = out.getvalue()
                assert "Searching for: test query" in output
                assert "mock result" in output
                mock_engine.search.assert_called_once_with("test query")

    def test_index_text(self, mock_engine):
        with patch.object(sys, "argv", [CLI_PROG, "index", "--text", "hello"]):
            with patch("sys.stdout", new=io.StringIO()) as out:
                main()
                output = out.getvalue()
                assert "Indexing content..." in output
                assert "Indexing complete." in output
                mock_engine.add_document.assert_called_once_with("hello")

    def test_index_file(self, mock_engine, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("file content")
        with patch.object(sys, "argv", [CLI_PROG, "index", "--file", str(f)]):
            with patch("sys.stdout", new=io.StringIO()) as out:
                main()
                output = out.getvalue()
                assert "Indexing complete." in output
                mock_engine.add_document.assert_called_once_with("file content")

    def test_optimize(self, mock_engine):
        with patch.object(
            sys, "argv", [CLI_PROG, "optimize-graph", "--threshold", "0.7"]
        ):
            with patch("sys.stdout", new=io.StringIO()) as out:
                main()
                output = out.getvalue()
                assert "optimization" in output.lower() or "Optimization" in output
                mock_engine.optimize_synonyms.assert_called_once_with(threshold=0.7)

    def test_models_setup_command(self):
        with patch.object(sys, "argv", [CLI_PROG, "models"]):
            with patch("seahorse.cli.ensure_gliner_onnx_model", return_value=True) as mock_setup:
                with patch("sys.stdout", new=io.StringIO()) as out:
                    main()
                    output = out.getvalue()
                    assert "Preparing ONNX GLiNER model" in output
                    assert "complete" in output.lower()
                    mock_setup.assert_called_once()
