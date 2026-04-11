"""Unit tests for seahorse.extraction (EntityExtractor)."""

import pytest
from unittest.mock import MagicMock, patch
from seahorse.extraction import EntityExtractor


@pytest.fixture
def mock_model():
    """Provide an EntityExtractor with mocked PureOnnxGLiNER."""
    with patch("seahorse.extraction.PureOnnxGLiNER") as mock_cls:
        instance = MagicMock()
        instance.predict.return_value = [
            {"text": "entity", "label": "label", "score": 0.9},
        ]
        mock_cls.return_value = instance
        yield mock_cls


# --- Load Model ---

class TestLoadModel:
    def test_loads_onnx_model(self, mock_model):
        extractor = EntityExtractor()
        with patch("seahorse.extraction.ensure_gliner_onnx_model"):
            model = extractor.load_model()
        assert model is not None

    def test_auto_setup_uses_force_on_onnx_init_failure(self):
        extractor = EntityExtractor()
        repaired_model = MagicMock()
        with patch(
            "seahorse.extraction.PureOnnxGLiNER",
            side_effect=[Exception("init failed"), repaired_model],
        ):
            with patch(
                "seahorse.extraction.ensure_gliner_onnx_model",
                return_value=True,
            ) as mock_setup:
                model = extractor.load_model()
        assert model is repaired_model
        mock_setup.assert_called_once()
        assert mock_setup.call_args.kwargs["force"] is True

    def test_fallback_to_gliner(self):
        extractor = EntityExtractor()
        with patch(
            "seahorse.extraction.PureOnnxGLiNER",
            side_effect=Exception("Load failed"),
        ):
            with patch("seahorse.extraction.ensure_gliner_onnx_model", side_effect=Exception("setup failed")):
                import builtins
                original_import = builtins.__import__

                def mock_import(name, *args, **kwargs):
                    if name == "gliner":
                        raise ImportError("No gliner")
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=mock_import):
                    with pytest.raises(Exception, match="Load failed"):
                        extractor.load_model()


# --- Entity Extraction ---

class TestExtraction:
    @pytest.mark.asyncio
    async def test_single_text(self, mock_model):
        extractor = EntityExtractor()
        with patch("seahorse.extraction.ensure_gliner_onnx_model"):
            entities = await extractor.extract_entities("Test text")
        assert len(entities) == 1
        assert entities[0]["text"] == "entity"

    @pytest.mark.asyncio
    async def test_empty_text(self, mock_model):
        extractor = EntityExtractor()
        with patch("seahorse.extraction.ensure_gliner_onnx_model"):
            entities = await extractor.extract_entities("")
        assert entities == []

    @pytest.mark.asyncio
    async def test_batch(self, mock_model):
        extractor = EntityExtractor()
        with patch("seahorse.extraction.ensure_gliner_onnx_model"):
            results = await extractor.extract_entities_batch(["T1", "T2"])
        assert len(results) == 2
        assert len(results[0]) == 1


# --- Deduplication ---

class TestDedup:
    def test_removes_duplicates(self):
        entities = [
            {"text": "Apple", "label": "org", "score": 0.9},
            {"text": "apple", "label": "org", "score": 0.8},  # dup (case)
            {"text": "Google", "label": "org", "score": 0.7},
        ]
        result = EntityExtractor._dedup_entities(entities)
        assert len(result) == 2

    def test_removes_short_entities(self):
        entities = [
            {"text": "A", "label": "org", "score": 0.9},  # too short
            {"text": "", "label": "org", "score": 0.9},    # empty
            {"text": "OK", "label": "org", "score": 0.9},
        ]
        result = EntityExtractor._dedup_entities(entities)
        assert len(result) == 1
        assert result[0]["text"] == "OK"

    def test_strips_whitespace(self):
        entities = [
            {"text": "  Apple  ", "label": "org", "score": 0.9},
        ]
        result = EntityExtractor._dedup_entities(entities)
        assert result[0]["text"] == "Apple"
