import logging
import math
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Local ONNX cross-encoder reranker with deterministic fusion policies."""

    _session_singleton: Any = None
    _session_model_path: Optional[str] = None
    _session_lock = threading.Lock()

    def __init__(
        self,
        *,
        enabled: bool,
        model_path: Optional[str],
        tokenizer_path: Optional[str],
        top_n: int,
        fusion_method: str,
        w_ppr: float,
        w_rerank: float,
        rrf_k: int,
        model_max_len: int,
        query_max_len: int,
        passage_max_len: int,
        apply_zscore: bool = False,
        logit_clip: Optional[float] = None,
        apply_sigmoid: bool = False,
        tokenizer: Optional[Any] = None,
        session: Optional[Any] = None,
    ) -> None:
        self.enabled = enabled
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.top_n = int(top_n)
        self.fusion_method = fusion_method.lower().strip()
        self.w_ppr = float(w_ppr)
        self.w_rerank = float(w_rerank)
        self.rrf_k = int(rrf_k)
        self.model_max_len = int(model_max_len)
        self.query_max_len = int(query_max_len)
        self.passage_max_len = int(passage_max_len)
        self.apply_zscore = bool(apply_zscore)
        self.logit_clip = logit_clip
        self.apply_sigmoid = bool(apply_sigmoid)

        self._tokenizer = tokenizer
        self._session = session
        self._input_names = self._extract_input_names(session)
        self._weights_warned = False
        self._runtime_warned = False

    @classmethod
    def from_settings(cls) -> Optional["CrossEncoderReranker"]:
        from .config import settings

        if not settings.RERANK_ENABLED:
            return None

        return cls(
            enabled=True,
            model_path=settings.RERANK_MODEL_PATH,
            tokenizer_path=settings.RERANK_TOKENIZER_PATH,
            top_n=settings.RERANK_TOP_N,
            fusion_method=settings.RERANK_FUSION_METHOD,
            w_ppr=settings.RERANK_W_PPR,
            w_rerank=settings.RERANK_W_RERANK,
            rrf_k=settings.RERANK_RRF_K,
            model_max_len=settings.RERANK_MODEL_MAX_LEN,
            query_max_len=settings.RERANK_QUERY_MAX_LEN,
            passage_max_len=settings.RERANK_PASSAGE_MAX_LEN,
            apply_zscore=settings.RERANK_LOGIT_ZSCORE,
            logit_clip=settings.RERANK_LOGIT_CLIP,
            apply_sigmoid=settings.RERANK_LOGIT_SIGMOID,
        )

    def rerank_and_fuse(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        *,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        deduped = self._stable_dedup(candidates)
        if not deduped:
            return []

        for item in deduped:
            item["final_score"] = float(item["ppr_score"])

        if not self.enabled:
            return deduped[:top_k]

        effective_top_n = self._resolve_top_n(top_k, len(deduped))
        if effective_top_n == 0:
            logger.warning("Reranker fallback: top_n=0, returning PPR-only results.")
            return deduped[:top_k]

        normalized_weights = self._normalize_weights(self.w_ppr, self.w_rerank)
        if normalized_weights is None:
            logger.error("Reranker fallback: invalid fusion weights, returning PPR-only results.")
            return deduped[:top_k]
        w_ppr, w_rerank = normalized_weights

        top_slice = [dict(x) for x in deduped[:effective_top_n]]
        rest_slice = [dict(x) for x in deduped[effective_top_n:]]

        rerank_scores = self._score_top_candidates(query, top_slice)
        if rerank_scores is None:
            logger.error("Reranker fallback: scoring failed, returning PPR-only results.")
            return deduped[:top_k]

        if len(rerank_scores) != len(top_slice):
            logger.error("Reranker fallback: score length mismatch, returning PPR-only results.")
            return deduped[:top_k]

        finite_mask = np.isfinite(rerank_scores)
        for idx, item in enumerate(top_slice):
            item["rerank_score"] = float(rerank_scores[idx])
            item["rerank_valid"] = bool(finite_mask[idx])

        if self.fusion_method == "weighted_sum":
            fused_top = self._fuse_weighted_sum(top_slice, w_ppr=w_ppr, w_rerank=w_rerank)
        elif self.fusion_method == "rrf":
            fused_top = self._fuse_rrf(
                top_slice,
                w_ppr=w_ppr,
                w_rerank=w_rerank,
                rrf_k=self.rrf_k,
            )
        else:
            logger.error("Unknown fusion method '%s'. Falling back to PPR-only.", self.fusion_method)
            return deduped[:top_k]

        final = fused_top + rest_slice
        return final[:top_k]

    def _score_top_candidates(
        self,
        query: str,
        top_candidates: Sequence[Dict[str, Any]],
    ) -> Optional[np.ndarray]:
        if not top_candidates:
            return np.array([], dtype=np.float32)

        if not self._ensure_runtime():
            return None

        inputs = [self._build_inputs(query, c.get("text", "")) for c in top_candidates]
        input_ids = np.stack([x["input_ids"] for x in inputs]).astype(np.int64, copy=False)
        attention_mask = np.stack([x["attention_mask"] for x in inputs]).astype(np.int64, copy=False)
        token_type_ids = np.stack([x["token_type_ids"] for x in inputs]).astype(np.int64, copy=False)

        ort_inputs: Dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._input_names:
            ort_inputs["token_type_ids"] = token_type_ids

        outputs = self._session.run(None, ort_inputs)
        logits = self._reduce_onnx_output(outputs[0], batch_size=len(top_candidates))
        if logits is None:
            return None
        return self._postprocess_logits(logits)

    def _build_inputs(self, query: str, passage: str) -> Dict[str, np.ndarray]:
        token_ids = self._get_special_token_ids()
        cls_id = token_ids["cls_id"]
        sep_id = token_ids["sep_id"]
        pad_id = token_ids["pad_id"]

        if self.model_max_len < 3:
            raise ValueError(f"model_max_len must be >= 3, got {self.model_max_len}")

        effective_query_max = min(self.query_max_len, self.model_max_len - 3)
        if effective_query_max < 0:
            effective_query_max = 0

        query_tokens = self._encode_no_special(query)[:effective_query_max]
        passage_tokens = self._encode_no_special(passage)[: self.passage_max_len]

        max_passage_by_model = self.model_max_len - 3 - len(query_tokens)
        if max_passage_by_model < 0:
            max_passage_by_model = 0
        if len(passage_tokens) > max_passage_by_model:
            passage_tokens = passage_tokens[:max_passage_by_model]

        # Standard pair packing with "only_second" overflow handling.
        token_sequence = [cls_id] + query_tokens + [sep_id] + passage_tokens + [sep_id]
        if len(token_sequence) > self.model_max_len:
            raise ValueError("Packed token sequence exceeded model_max_len.")

        prefix_len = 1 + len(query_tokens) + 1
        passage_len_with_sep = len(passage_tokens) + 1
        token_type_ids = ([0] * prefix_len) + ([1] * passage_len_with_sep)
        attention_mask = [1] * len(token_sequence)

        pad_len = self.model_max_len - len(token_sequence)
        if pad_len > 0:
            token_sequence += [pad_id] * pad_len
            token_type_ids += [0] * pad_len
            attention_mask += [0] * pad_len

        return {
            "input_ids": np.asarray(token_sequence, dtype=np.int64),
            "attention_mask": np.asarray(attention_mask, dtype=np.int64),
            "token_type_ids": np.asarray(token_type_ids, dtype=np.int64),
        }

    def _reduce_onnx_output(self, raw_output: Any, batch_size: int) -> Optional[np.ndarray]:
        arr = np.asarray(raw_output)
        if arr.ndim != 2:
            logger.error("Unsupported ONNX output rank=%d. Fallback to PPR-only.", arr.ndim)
            return None
        if arr.shape[0] != batch_size:
            logger.error(
                "ONNX output batch mismatch: expected=%d got=%d. Fallback to PPR-only.",
                batch_size,
                arr.shape[0],
            )
            return None
        if arr.shape[1] == 1:
            return np.asarray(arr.squeeze(1), dtype=np.float32)
        if arr.shape[1] == 2:
            return np.asarray(arr[:, 1] - arr[:, 0], dtype=np.float32)
        logger.error("Unsupported ONNX output shape=%s. Fallback to PPR-only.", arr.shape)
        return None

    def _postprocess_logits(self, logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits, dtype=np.float32)
        finite_mask = np.isfinite(values)

        if self.apply_zscore and np.any(finite_mask):
            finite_values = values[finite_mask]
            mean = float(np.mean(finite_values))
            std = float(np.std(finite_values))
            if std > 1e-6:
                values[finite_mask] = (finite_values - mean) / std

        if self.logit_clip is not None and self.logit_clip >= 0:
            values = np.clip(values, -self.logit_clip, self.logit_clip).astype(np.float32, copy=False)

        if self.apply_sigmoid:
            finite_values = values[finite_mask]
            values[finite_mask] = 1.0 / (1.0 + np.exp(-finite_values))

        return np.asarray(values, dtype=np.float32)

    def _fuse_weighted_sum(
        self,
        top_candidates: List[Dict[str, Any]],
        *,
        w_ppr: float,
        w_rerank: float,
    ) -> List[Dict[str, Any]]:
        top_n = len(top_candidates)
        rerank_scores = np.asarray([x["rerank_score"] for x in top_candidates], dtype=np.float32)
        finite_mask = np.isfinite(rerank_scores)
        finite_values = rerank_scores[finite_mask]

        if finite_values.size == 0:
            rerank_norm = np.zeros(top_n, dtype=np.float32)
        else:
            lo = float(np.min(finite_values))
            hi = float(np.max(finite_values))
            if hi - lo < 1e-9:
                rerank_norm = np.where(finite_mask, 1.0, 0.0).astype(np.float32)
            else:
                rerank_norm = np.zeros(top_n, dtype=np.float32)
                rerank_norm[finite_mask] = (rerank_scores[finite_mask] - lo) / (hi - lo)

        for i, item in enumerate(top_candidates):
            rank = i + 1
            if top_n <= 1:
                s_ppr = 1.0
            else:
                s_ppr = 1.0 - (rank - 1.0) / (top_n - 1.0)
            s_rer = float(rerank_norm[i])
            fused = (w_ppr * s_ppr) + (w_rerank * s_rer)
            if not item["rerank_valid"]:
                # NaN/Inf logits must sink within top-N.
                fused = -math.inf
            item["final_score"] = float(fused)

        top_candidates.sort(
            key=lambda x: (
                0 if x.get("rerank_valid", False) else 1,
                -float(x["final_score"]) if np.isfinite(x["final_score"]) else math.inf,
                int(x["ppr_rank"]),
                int(x["passage_id"]),
            )
        )
        return top_candidates

    def _fuse_rrf(
        self,
        top_candidates: List[Dict[str, Any]],
        *,
        w_ppr: float,
        w_rerank: float,
        rrf_k: int,
    ) -> List[Dict[str, Any]]:
        rerank_order = sorted(
            range(len(top_candidates)),
            key=lambda idx: (
                0 if top_candidates[idx].get("rerank_valid", False) else 1,
                -float(top_candidates[idx]["rerank_score"])
                if top_candidates[idx].get("rerank_valid", False)
                else math.inf,
                int(top_candidates[idx]["ppr_rank"]),
                int(top_candidates[idx]["passage_id"]),
            ),
        )
        rerank_rank: Dict[int, int] = {}
        for rank, idx in enumerate(rerank_order, start=1):
            rerank_rank[idx] = rank

        for idx, item in enumerate(top_candidates):
            ppr_rank = int(item["ppr_rank"])
            rr_rank = rerank_rank[idx]
            score = (w_ppr / (rrf_k + ppr_rank)) + (w_rerank / (rrf_k + rr_rank))
            if not item["rerank_valid"]:
                score = -math.inf
            item["final_score"] = float(score)
            item["rerank_rank"] = rr_rank

        top_candidates.sort(
            key=lambda x: (
                0 if x.get("rerank_valid", False) else 1,
                -float(x["final_score"]) if np.isfinite(x["final_score"]) else math.inf,
                int(x["ppr_rank"]),
                int(x.get("rerank_rank", 10**9)),
                int(x["passage_id"]),
            )
        )
        return top_candidates

    def _resolve_top_n(self, top_k: int, candidate_len: int) -> int:
        if self.top_n < 0:
            logger.error("Invalid top_n=%d. Fallback to PPR-only.", self.top_n)
            return 0
        if self.top_n == 0:
            return 0
        effective = self.top_n
        if effective < top_k:
            logger.warning("top_n=%d < top_k=%d. Adjusting top_n to top_k.", effective, top_k)
            effective = top_k
        return min(effective, candidate_len)

    def _normalize_weights(self, w_ppr: float, w_rerank: float) -> Optional[Tuple[float, float]]:
        w_sum = w_ppr + w_rerank
        if w_sum <= 0:
            return None

        n_ppr = w_ppr / w_sum
        n_rerank = w_rerank / w_sum
        if (abs(n_ppr - w_ppr) > 1e-9 or abs(n_rerank - w_rerank) > 1e-9) and not self._weights_warned:
            logger.warning(
                "Fusion weights normalized from (%.6f, %.6f) to (%.6f, %.6f).",
                w_ppr,
                w_rerank,
                n_ppr,
                n_rerank,
            )
            self._weights_warned = True
        return n_ppr, n_rerank

    def _ensure_runtime(self) -> bool:
        if self._tokenizer is not None and self._session is not None:
            if not self._input_names:
                self._input_names = self._extract_input_names(self._session)
            return True

        if not self.model_path or not self.tokenizer_path:
            if not self._runtime_warned:
                logger.warning(
                    "Reranker disabled at runtime: model/tokenizer path missing. Returning PPR-only."
                )
                self._runtime_warned = True
            return False

        try:
            from tokenizers import Tokenizer
            import onnxruntime as ort
        except Exception as exc:
            if not self._runtime_warned:
                logger.warning("Reranker runtime unavailable (%s). Returning PPR-only.", exc)
                self._runtime_warned = True
            return False

        try:
            self._tokenizer = Tokenizer.from_file(self.tokenizer_path)
        except Exception as exc:
            logger.error("Failed to load reranker tokenizer '%s': %s", self.tokenizer_path, exc)
            return False

        try:
            with self.__class__._session_lock:
                if self.__class__._session_singleton is None:
                    self.__class__._session_singleton = ort.InferenceSession(
                        self.model_path,
                        providers=["CPUExecutionProvider"],
                    )
                    self.__class__._session_model_path = self.model_path
                elif self.__class__._session_model_path != self.model_path:
                    logger.warning(
                        "Reranker singleton already initialized at '%s'; reusing that session.",
                        self.__class__._session_model_path,
                    )
                self._session = self.__class__._session_singleton
        except Exception as exc:
            logger.error("Failed to initialize reranker ONNX session '%s': %s", self.model_path, exc)
            return False

        self._input_names = self._extract_input_names(self._session)
        return True

    def _encode_no_special(self, text: str) -> List[int]:
        try:
            encoded = self._tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            encoded = self._tokenizer.encode(text)
        token_ids = getattr(encoded, "ids", encoded)
        return [int(x) for x in token_ids]

    def _get_special_token_ids(self) -> Dict[str, int]:
        cls_id = self._token_to_id("[CLS]")
        sep_id = self._token_to_id("[SEP]")
        pad_id = self._token_to_id("[PAD]")
        if pad_id is None:
            pad_id = self._token_to_id("<pad>")
        if pad_id is None:
            pad_id = 0
        if cls_id is None or sep_id is None:
            raise ValueError("Tokenizer is missing [CLS]/[SEP] token ids.")
        return {"cls_id": int(cls_id), "sep_id": int(sep_id), "pad_id": int(pad_id)}

    def _token_to_id(self, token: str) -> Optional[int]:
        if not hasattr(self._tokenizer, "token_to_id"):
            return None
        value = self._tokenizer.token_to_id(token)
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _stable_dedup(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for item in candidates:
            pid = int(item["passage_id"])
            if pid in seen:
                continue
            seen.add(pid)
            deduped.append(dict(item))
        return deduped

    @staticmethod
    def _extract_input_names(session: Optional[Any]) -> Tuple[str, ...]:
        if session is None:
            return tuple()
        try:
            return tuple(inp.name for inp in session.get_inputs())
        except Exception:
            return tuple()
