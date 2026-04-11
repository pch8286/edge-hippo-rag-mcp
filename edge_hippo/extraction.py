import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

from .config import settings
from .model_setup import ensure_gliner_onnx_model
from .models import PureOnnxGLiNER

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Async wrapper around PureOnnxGLiNER for entity extraction."""

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._executor = ThreadPoolExecutor(max_workers=settings.THREAD_POOL_SIZE)
        self.labels = [
            "person", "organization", "location", "date", "event",
            "concept", "technology", "product", "metric", "process",
        ]

    def load_model(self) -> Any:
        """Load Pure ONNX GLiNER model with fallback to standard GLiNER.

        Resolution order for model directory:
        1. ``settings.GLINER_ONNX_PATH`` (explicit config)
        2. ``./models/gliner_onnx`` (convention default)
        """
        if self._model is None:
            model_dir = settings.GLINER_ONNX_PATH or os.path.join(os.getcwd(), "models", "gliner_onnx")

            logger.info("Loading Pure ONNX GLiNER from %s", model_dir)
            try:
                self._model = PureOnnxGLiNER(model_dir)
                logger.info("Pure ONNX GLiNER loaded successfully.")
            except Exception as e:
                logger.warning(
                    "Failed to load Pure ONNX GLiNER from %s: %s",
                    model_dir, e,
                )

                if settings.GLINER_ONNX_AUTO_SETUP:
                    try:
                        downloaded = ensure_gliner_onnx_model(
                            model_dir,
                            repo_id=settings.GLINER_ONNX_REPO_ID,
                            force=True,
                        )
                        if downloaded:
                            logger.info("ONNX GLiNER downloaded. Retrying initialization.")
                        self._model = PureOnnxGLiNER(model_dir)
                        logger.info("Pure ONNX GLiNER loaded after auto-setup.")
                        return self._model
                    except Exception as setup_err:
                        logger.warning("ONNX GLiNER auto-setup failed: %s", setup_err)

                # Fallback to standard GLiNER
                logger.info("Falling back to standard GLiNER library...")
                try:
                    from gliner import GLiNER

                    fallback_model = settings.GLINER_MODEL
                    self._model = GLiNER.from_pretrained(fallback_model)
                    logger.info(
                        "Fallback GLiNER model %s loaded.", fallback_model
                    )
                except Exception as e2:
                    logger.error("Failed to load fallback GLiNER: %s", e2)
                    raise e from e2
        return self._model

    def _extract_sync(self, text: str) -> List[Dict[str, Any]]:
        """Run synchronous entity extraction for a single text."""
        if not text.strip():
            return []
        model = self.load_model()
        return self._predict_single(model, text)

    def _predict_single(self, model: Any, text: str) -> List[Dict[str, Any]]:
        """Compatibility wrapper for ONNX model and multiple GLiNER APIs."""
        predict_fn = getattr(model, "predict", None)
        if callable(predict_fn):
            result = predict_fn(text, self.labels, threshold=0.3)
            if isinstance(result, list):
                return result

        predict_entities_fn = getattr(model, "predict_entities", None)
        if callable(predict_entities_fn):
            result = predict_entities_fn(text, self.labels, threshold=0.3)
            if isinstance(result, list):
                return result

        if callable(model):
            try:
                result = model(text, self.labels, threshold=0.3)
            except TypeError:
                result = model(text, labels=self.labels, threshold=0.3)
            if isinstance(result, list):
                return result
        raise AttributeError("Loaded extractor model has no supported predict API.")

    def _predict_batch(self, model: Any, texts: List[str]) -> List[List[Dict[str, Any]]]:
        batch_predict_fn = getattr(model, "batch_predict_entities", None)
        if callable(batch_predict_fn):
            try:
                result = batch_predict_fn(texts, self.labels, threshold=0.3)
                if isinstance(result, list) and (
                    len(result) == 0 or isinstance(result[0], list)
                ):
                    return result
            except Exception:
                pass

        inference_fn = getattr(model, "inference", None)
        if callable(inference_fn):
            try:
                result = inference_fn(texts, self.labels, threshold=0.3)
                if isinstance(result, list) and (
                    len(result) == 0 or isinstance(result[0], list)
                ):
                    return result
            except Exception:
                pass

        return [self._predict_single(model, t) for t in texts]

    @staticmethod
    def _dedup_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate and trivially short entities."""
        cleaned: list[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for ent in entities:
            text_clean = ent["text"].strip()
            if not text_clean or len(text_clean) < 2:
                continue
            key = (text_clean.lower(), ent["label"])
            if key in seen:
                continue
            seen.add(key)
            cleaned.append({
                "text": text_clean,
                "label": ent["label"],
                "score": float(ent["score"]),
            })
        return cleaned

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Run extraction in thread pool to avoid blocking async loop."""
        loop = asyncio.get_running_loop()
        try:
            entities = await loop.run_in_executor(
                self._executor,
                self._extract_sync,
                text,
            )
            return self._dedup_entities(entities)
        except Exception as e:
            logger.error("Error in entity extraction: %s", e)
            return []

    def _extract_batch_sync(
        self, texts: List[str]
    ) -> List[List[Dict[str, Any]]]:
        model = self.load_model()
        return self._predict_batch(model, texts)

    async def extract_entities_batch(
        self, texts: List[str]
    ) -> List[List[Dict[str, Any]]]:
        """Extract entities from multiple texts in a batch.

        Runs the synchronous loop in an executor.
        """
        loop = asyncio.get_running_loop()
        try:
            batch_results = await loop.run_in_executor(
                self._executor,
                self._extract_batch_sync,
                texts,
            )
            return [self._dedup_entities(r) for r in batch_results]
        except Exception as e:
            logger.error("Error in batch entity extraction: %s", e)
            return [[] for _ in texts]

    def close(self) -> None:
        """Shutdown the thread pool executor."""
        self._executor.shutdown(wait=False)
