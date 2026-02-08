import pytest
from unittest.mock import MagicMock, patch
from edge_hippo.extraction import EntityExtractor

@pytest.mark.asyncio
async def test_extraction_async():
    # Mock GLiNER model
    mock_model = MagicMock()
    # Mock predict_entities method
    mock_model.predict_entities.return_value = [
        {"text": "Apple", "label": "organization", "score": 0.95},
        {"text": "Cupertino", "label": "location", "score": 0.88},
        {"text": "  Apple  ", "label": "organization", "score": 0.95} # Duplicate/Spaces
    ]

    # Patch where GLiNER is defined (gliner package), or where it is imported.
    # Since extraction.py does `from gliner import GLiNER` inside the function,
    # we should patch `gliner.GLiNER` globally or `edge_hippo.extraction.GLiNER` if it was top-level.
    # But it is local. So we must patch `gliner.GLiNER`.
    with patch("gliner.GLiNER.from_pretrained", return_value=mock_model):
        extractor = EntityExtractor()
        # Force load
        extractor.load_model()
        
        text = "Apple assumes Cupertino."
        entities = await extractor.extract_entities(text)
        
        assert len(entities) == 2 # Deduped
        assert entities[0]['text'] == "Apple"
        assert entities[0]['label'] == "organization"
        assert entities[1]['text'] == "Cupertino"
        
        extractor.close()

@pytest.mark.asyncio
async def test_extraction_empty():
    mock_model = MagicMock()
    mock_model.predict_entities.return_value = []
    
    with patch("gliner.GLiNER.from_pretrained", return_value=mock_model):
        extractor = EntityExtractor()
        extractor.load_model()
        res = await extractor.extract_entities("")
        assert res == []
        extractor.close()
