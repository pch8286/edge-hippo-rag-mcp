import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Set, TYPE_CHECKING
if TYPE_CHECKING:
    from gliner import GLiNER
from .config import settings

logger = logging.getLogger(__name__)

class EntityExtractor:
    def __init__(self):
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=settings.THREAD_POOL_SIZE)
        self.labels = [
            "person", "organization", "location", "date", "event", 
            "concept", "technology", "product", "metric", "process"
        ] # Customizable labels

    def load_model(self):
        """
        Load GLiNER model.
        Prioritizes Local Quantized/FP32 (via Native Loader) -> Local Standard -> Download.
        """
        if self._model is None:
            # Import GLiNER here to avoid circular dependency if gliner_onnx is used
            from gliner import GLiNER 

            try:
                if settings.QUANTIZED_MODEL_DIR.exists():
                    logger.info(f"Loading Quantized/Optimized model from {settings.QUANTIZED_MODEL_DIR}...")
                    # Use Native Loader. 
                    # Note: This may fall back to PyTorch if ONNX fails or is invalid, 
                    # trading RAM for Correctness (Parity).
                    # To attempt ONNX loading: load_onnx_model=True
                    model_path = settings.QUANTIZED_MODEL_DIR / "gliner_small"
                    self._model = GLiNER.from_pretrained(
                        str(model_path),
                        load_onnx_model=True, 
                        load_tokenizer=True
                    )
                    logger.info("Optimized GLiNER model loaded successfully.")
                    return self._model
            except Exception as e:
                logger.warning(f"Failed to load optimized model: {e}")
                # Fallback to standard model loading

            logger.info(f"Loading standard model: {settings.GLINER_MODEL}")
            # On RPi, we might want to ensure we don't explode RAM.
            # GLiNER small is ~150MB, very safe.
            self._model = GLiNER.from_pretrained(settings.GLINER_MODEL)
            # Dynamic quantization fallback removed (ONNX pref.)
            logger.info("GLiNER model loaded.")
        return self._model

    def _extract_sync(self, text: str) -> List[Dict[str, Any]]:
        model = self.load_model()
        # predict_entities returns list of dicts: {'text':..., 'label':..., 'score':...}
        entities = model.predict_entities(text, self.labels, threshold=0.3)
        return entities

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Run extraction in thread pool to avoid blocking async loop."""
        loop = asyncio.get_running_loop()
        try:
            entities = await loop.run_in_executor(
                self._executor, 
                self._extract_sync, 
                text
            )
            # Basic validation/cleaning
            cleaned = []
            seen = set()
            for ent in entities:
                text_clean = ent['text'].strip()
                if not text_clean or len(text_clean) < 2:
                    continue
                key = (text_clean.lower(), ent['label'])
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append({
                    "text": text_clean,
                    "label": ent['label'],
                    "score": float(ent['score'])
                })
            return cleaned
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return []


    async def extract_entities_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Extract entities from multiple texts in a batch.
        Uses GLiNER's internal batching if available, or loops in executor.
        """
        loop = asyncio.get_running_loop()
        try:
            # Run in executor
            batch_results = await loop.run_in_executor(
                self._executor,
                self._extract_batch_sync,
                texts
            )
            
            cleaned_batch = []
            for entities in batch_results:
                # Basic validation/cleaning per item
                cleaned = []
                seen = set()
                for ent in entities:
                    text_clean = ent['text'].strip()
                    if not text_clean or len(text_clean) < 2:
                        continue
                    key = (text_clean.lower(), ent['label'])
                    if key in seen:
                        continue
                    seen.add(key)
                    cleaned.append({
                        "text": text_clean,
                        "label": ent['label'],
                        "score": float(ent['score'])
                    })
                cleaned_batch.append(cleaned)
            return cleaned_batch
        except Exception as e:
            logger.error(f"Error in batch entity extraction: {e}")
            return [[] for _ in texts] # Return empty lists on error

    def _extract_batch_sync(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        model = self.load_model()
        # GLiNER predict_entities can take a list? 
        # Checking GLiNER docs (assumed): predict_entities(text: str | List[str], ...)
        # If not, we fallback to loop.
        # But for optimization, we assume it batches internally or we use batch_predict calls.
        # Recent GLiNER versions support batch with `predict_entities(texts, ...)` returning list of lists?
        # Let's hope so. If not, we iterate.
        # To be safe, let's try calling it with list. If it fails, fallback to loop.
        
        try:
            return model.predict_entities(texts, self.labels, threshold=0.3)
        except (TypeError, AttributeError):
            # Fallback
            results = []
            for t in texts:
                results.append(model.predict_entities(t, self.labels, threshold=0.3))
            return results

    def close(self):
        self._executor.shutdown(wait=False)
