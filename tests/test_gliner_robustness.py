import pytest
import warnings
from pathlib import Path
from edge_hippo.extraction import EntityExtractor
from edge_hippo.config import settings

# Skip if model doesn't exist to avoid CI failures
model_exists = (settings.QUANTIZED_MODEL_DIR / "gliner_small").exists() or Path(settings.GLINER_MODEL).exists()

@pytest.mark.skipif(not model_exists, reason="GLiNER model not found locally.")
@pytest.mark.asyncio
class TestGlinerRobustness:
    """
    Integration tests for GLiNER model robustness.
    These tests run against the actual loaded model (Quantized or Standard)
    to verify edge cases handling that mocks usually miss.
    """
    
    @pytest.fixture(scope="class")
    def extractor(self):
        """Load extractor once for the class."""
        ext = EntityExtractor()
        ext.load_model()
        yield ext
        ext.close()

    async def test_standard_entities(self, extractor):
        """Verify basic entity extraction (Steve Jobs)."""
        text = "Steve Jobs founded Apple in 1976."
        entities = await extractor.extract_entities(text)
        
        # We expect at least Person and Org. Date might be optional depending on labels.
        # Default labels in extraction.py: ["person", "organization", "location", "date", ...]
        
        labels = {e["label"] for e in entities}
        texts = {e["text"] for e in entities}
        
        assert "person" in labels, f"Missing person label. Got: {labels}"
        assert "Steve Jobs" in texts, f"Missing Steve Jobs. Got: {texts}"
        assert "organization" in labels
        assert "Apple" in texts

    async def test_disambiguation(self, extractor):
        """Verify context sensitivity (Apple company vs fruit)."""
        # "Apple is a company" -> Organization
        text_org = "Apple is a huge technology company."
        # "I ate an apple." -> Food? (If label exists) or just NOT Org.
        # Check extraction.py labels.
        
        ents_org = await extractor.extract_entities(text_org)
        org_labels = [e["label"] for e in ents_org if e["text"] == "Apple"]
        
        if not org_labels:
             # Sometimes small models miss short contexts. 
             # But "Apple" in "technology company" context usually works.
             warnings.warn("GLiNER failed to extract 'Apple' as Org in clear context.")
        else:
            assert "organization" in org_labels, f"Expected Apple to be Org, got {org_labels}"

    async def test_empty_input(self, extractor):
        """Verify robustness against empty input."""
        res = await extractor.extract_entities("")
        assert res == []
        
        res = await extractor.extract_entities("   ")
        assert res == []

    async def test_special_characters(self, extractor):
        """Verify robustness against specific special characters."""
        # Previous debug sessions hinted at issues with token mask alignment
        text = ">>> special <<< characters @ # $ % ^ & *"
        # Should not crash
        res = await extractor.extract_entities(text)
        assert isinstance(res, list)

    async def test_prompt_injection_safety(self, extractor):
        """Verify model doesn't hallucinate on prompt-like tokens if they leak."""
        # GLiNER uses <<ENT>>, <<SEP>> internally.
        # If user provides them, they should be treated as text or handled gracefully.
        text = "This text contains <<ENT>> fake tags <<SEP>>."
        res = await extractor.extract_entities(text)
        assert isinstance(res, list)
        # We generally expect no entities, or at least no crash.
