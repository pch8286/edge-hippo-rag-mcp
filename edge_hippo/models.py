import os
import logging
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


class PureOnnxGLiNER:
    """
    Pure ONNX implementation of GLiNER2 (Multi-Task) inference.
    Removes dependency on 'gliner' library and 'torch' in hot path.
    """

    def __init__(self, model_dir: str, threads: int = 4) -> None:
        """
        Initialize the model components.

        Args:
            model_dir: Directory containing ONNX files and config.
            threads: Number of intra-op threads for ONNX Runtime.
        """
        self.model_dir = model_dir
        self._tokenizer_backend = "tokenizers"
        try:
            tokenizer_path = os.path.join(model_dir, "tokenizer.json")
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        except Exception as e:
            logger.warning(
                "Tokenizer.from_file failed (%s), falling back to transformers AutoTokenizer.",
                e,
            )
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self._tokenizer_backend = "transformers"

        # Load Config/Special Tokens logic
        vocab = self.tokenizer.get_vocab()

        # Determine Entity Token
        if "[E]" in vocab:
            self.ent_token = "[E]"
        elif "<<ENT>>" in vocab:
            self.ent_token = "<<ENT>>"
        else:
            self.ent_token = "[ENT]"

        # Determine Separator
        if "[SEP_TEXT]" in vocab:
            self.sep_token = "[SEP_TEXT]"
        else:
            self.sep_token = "[SEP]"  # Standard BERT/DeBerta

        self.ent_token_id = vocab.get(
            self.ent_token,
            self._token_to_id(self.ent_token),
        )
        if self.ent_token_id is None:
            ids = self._encode_token_no_special(self.ent_token)
            if ids:
                self.ent_token_id = ids[0]

        logger.info(
            "Using ENT: %s (%s), SEP: %s",
            self.ent_token,
            self.ent_token_id,
            self.sep_token,
        )

        # Session Options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = threads
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Load Sessions — Hybrid Precision Strategy:
        # Encoder: FP16 preferred (INT8 degrades DeBERTa accuracy significantly)
        # span_rep/count_embed: INT8 preferred (75% smaller, accuracy preserved)
        int8_dir = os.path.join(model_dir, "onnx_int8")
        onnx_dir = os.path.join(model_dir, "onnx")

        def _onnx_requires_external_data(path: str) -> Any:
            try:
                import onnx

                model = onnx.load(path, load_external_data=False)
                for tensor in model.graph.initializer:
                    if tensor.data_location == onnx.TensorProto.EXTERNAL:
                        return True
                return False
            except Exception:
                return None

        def _artifact_ready(path: str) -> bool:
            if not os.path.exists(path):
                return False

            data_path = f"{path}.data"
            requires_external = _onnx_requires_external_data(path)
            if requires_external is True:
                return os.path.exists(data_path)
            if requires_external is False:
                return True
            if os.path.exists(data_path):
                return True
            # Fallback heuristic if ONNX metadata parsing is unavailable:
            # if directory clearly has external-data shards, require sibling.
            try:
                if any(name.endswith(".onnx.data") for name in os.listdir(os.path.dirname(path))):
                    return False
            except OSError:
                pass
            return True

        def _find_model(
            name: str, prefer_int8: bool = True
        ) -> tuple[str, str]:
            """Find best available model file with precision fallback."""
            candidates: list[tuple[str, str]] = []
            if prefer_int8:
                candidates.append(
                    (os.path.join(int8_dir, f"{name}_int8.onnx"), "INT8")
                )
            candidates.append(
                (os.path.join(onnx_dir, f"{name}_fp16.onnx"), "FP16")
            )
            candidates.append((os.path.join(onnx_dir, f"{name}.onnx"), "FP32"))
            for path, prec in candidates:
                if _artifact_ready(path):
                    return path, prec
            raise FileNotFoundError(f"No ONNX model found for '{name}'")

        # Encoder: prefer FP16 (INT8 degrades DeBERTa too much)
        encoder_path, enc_prec = _find_model("encoder", prefer_int8=False)
        # span_rep/count_embed: prefer INT8 (75% smaller, accuracy preserved)
        span_rep_path, span_prec = _find_model("span_rep", prefer_int8=True)
        count_embed_path, count_prec = _find_model(
            "count_embed", prefer_int8=True
        )

        logger.info(
            "Precision: encoder=%s, span_rep=%s, count_embed=%s",
            enc_prec,
            span_prec,
            count_prec,
        )
        self.encoder_session = ort.InferenceSession(encoder_path, sess_options)
        self.span_rep_session = ort.InferenceSession(
            span_rep_path, sess_options
        )
        self.count_embed_session = ort.InferenceSession(
            count_embed_path, sess_options
        )

    def _token_to_id(self, token: str) -> Any:
        if self._tokenizer_backend == "tokenizers":
            return self.tokenizer.token_to_id(token)
        return self.tokenizer.convert_tokens_to_ids(token)

    def _encode_token_no_special(self, token: str) -> List[int]:
        if self._tokenizer_backend == "tokenizers":
            return self.tokenizer.encode(token).ids
        return self.tokenizer.encode(token, add_special_tokens=False)

    def _encode_pretokenized(self, words: List[str]) -> tuple[np.ndarray, np.ndarray, List[Any]]:
        if self._tokenizer_backend == "tokenizers":
            enc = self.tokenizer.encode(words, is_pretokenized=True)
            word_ids = enc.word_ids if not callable(enc.word_ids) else enc.word_ids()
            return (
                np.asarray([enc.ids], dtype=np.int64),
                np.asarray([enc.attention_mask], dtype=np.int64),
                word_ids,
            )

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="np",
            padding=False,
            truncation=True,
        )
        word_ids = encoding.word_ids() if callable(encoding.word_ids) else encoding.word_ids
        return encoding["input_ids"], encoding["attention_mask"], word_ids

    def prepare_span_idx(
        self, num_tokens: int, max_width: int = 12
    ) -> np.ndarray:
        """Generate all valid (start, end) span index pairs.

        Args:
            num_tokens: Number of text tokens.
            max_width: Maximum span width.

        Returns:
            Array of shape (num_spans, 2) with [start, end] pairs.
        """
        span_idx: list[list[int]] = []
        for i in range(num_tokens):
            for j in range(max_width):
                if i + j < num_tokens:
                    span_idx.append([i, i + j])
        return np.array(span_idx, dtype=np.int64)

    def extract_word_embeddings(
        self,
        token_embeds: np.ndarray,
        words_mask: np.ndarray,
        max_text_length: int,
    ) -> np.ndarray:
        """Map subword token embeddings to word-level embeddings.

        Uses a 1-based ``words_mask`` where each position holds the target
        word index (or 0 for non-text tokens).

        Args:
            token_embeds: Encoder output, shape ``(1, seq_len, hidden)``.
            words_mask: Mask array, shape ``(1, seq_len)``.
            max_text_length: Number of text words.

        Returns:
            Word embeddings, shape ``(1, max_text_length, hidden)``.
        """
        hidden_dim = token_embeds.shape[-1]
        batch_idx, word_idx = np.where(words_mask > 0)
        target_word_idx = words_mask[batch_idx, word_idx] - 1

        words_embedding = np.zeros(
            (1, max_text_length, hidden_dim), dtype=np.float32
        )
        words_embedding[batch_idx, target_word_idx] = token_embeds[
            batch_idx, word_idx
        ]
        return words_embedding

    def extract_prompt_features(
        self, token_embeds: np.ndarray, input_ids: np.ndarray
    ) -> np.ndarray:
        """Extract label embeddings at entity-token positions.

        Args:
            token_embeds: Encoder output, shape ``(1, seq_len, hidden)``.
            input_ids: Token IDs, shape ``(1, seq_len)``.

        Returns:
            Prompt embeddings, shape ``(1, num_labels, hidden)``.
        """
        hidden_dim = token_embeds.shape[-1]
        mask = input_ids == self.ent_token_id
        batch_idx, token_idx = np.where(mask)

        if len(batch_idx) == 0:
            return np.zeros((1, 0, hidden_dim), dtype=np.float32)

        prompt_embeds = token_embeds[batch_idx, token_idx]
        return prompt_embeds.reshape(1, -1, hidden_dim)

    @staticmethod
    def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid that avoids exp overflow."""
        out = np.empty_like(x, dtype=np.float32)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        neg = ~pos
        exp_x = np.exp(x[neg])
        out[neg] = exp_x / (1.0 + exp_x)
        return out

    def predict(
        self,
        text: str,
        labels: List[str],
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Run full NER prediction pipeline.

        Args:
            text: Input text to extract entities from.
            labels: Entity label names (e.g. ``["person", "location"]``).
            threshold: Minimum sigmoid probability to accept.

        Returns:
            List of dicts with ``text``, ``label``, ``score``, ``start``,
            ``end`` keys.
        """
        if not text.strip():
            return []

        # 1. Prepare Inputs
        text_words = text.split()
        num_text_words = len(text_words)
        max_width = 12

        prompt_words: list[str] = []
        for label in labels:
            prompt_words.append(self.ent_token)
            prompt_words.append(label)
        prompt_words.append(self.sep_token)

        full_words = prompt_words + text_words

        input_ids, attention_mask, word_ids = self._encode_pretokenized(full_words)

        # Words Mask
        words_mask: list[int] = []
        prompt_len = len(prompt_words)

        current_word_id = None
        for _i, word_id in enumerate(word_ids):
            if word_id is None:
                words_mask.append(0)
            elif word_id < prompt_len:
                words_mask.append(0)
            else:
                if word_id != current_word_id:
                    relative_id = word_id - prompt_len + 1
                    words_mask.append(relative_id)
                else:
                    words_mask.append(0)
            current_word_id = word_id

        words_mask_arr = np.array(words_mask, dtype=np.int64).reshape(1, -1)

        # 2. Run Encoder
        enc_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        last_hidden_state = self.encoder_session.run(None, enc_inputs)[0]
        hidden_dim = last_hidden_state.shape[-1]

        # 3. Extract Features
        prompts_embeds = self.extract_prompt_features(
            last_hidden_state, input_ids
        )
        if prompts_embeds.shape[1] == 0:
            logger.warning("No prompt embeddings extracted.")
            return []

        word_embeds = self.extract_word_embeddings(
            last_hidden_state, words_mask_arr, num_text_words
        )

        # 4. Transform Prompts
        p_in = prompts_embeds.reshape(-1, hidden_dim)
        final_prompt_embeds = self.count_embed_session.run(
            None, {"label_embeddings": p_in}
        )[0]
        final_prompt_embeds = final_prompt_embeds.reshape(1, -1, hidden_dim)

        # 5. Span Rep
        span_idx = self.prepare_span_idx(num_text_words, max_width)
        if len(span_idx) == 0:
            return []

        span_start = span_idx[:, 0].reshape(1, -1)
        span_end = span_idx[:, 1].reshape(1, -1)

        span_rep_inputs = {
            "hidden_states": word_embeds,
            "span_start_idx": span_start,
            "span_end_idx": span_end,
        }
        span_reps = self.span_rep_session.run(None, span_rep_inputs)[0]

        # 6. Scores
        scores = np.einsum("bsd,bld->bsl", span_reps, final_prompt_embeds)
        scores = scores[0]  # (num_spans, num_labels)
        probs = self._stable_sigmoid(scores)

        # 7. Decode Spans
        predicted_entities: list[Dict[str, Any]] = []
        for i, span in enumerate(span_idx):
            start, end = span
            for label_idx in range(len(labels)):
                prob = probs[i, label_idx]
                if prob > threshold:
                    ent_text = " ".join(text_words[start : end + 1])
                    predicted_entities.append(
                        {
                            "text": ent_text,
                            "label": labels[label_idx],
                            "score": float(prob),
                            "start": start,
                            "end": end,
                        }
                    )

        return predicted_entities
