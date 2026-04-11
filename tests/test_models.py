"""Unit tests for seahorse.models (PureOnnxGLiNER)."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from seahorse.models import PureOnnxGLiNER

TEXT = "Apple released the iPhone."
LABELS = ["organization", "product"]


def _exists_fp32_only(p):
    """Mock os.path.exists: only FP32 .onnx files exist."""
    return p.endswith(".onnx") and "_fp16" not in p and "_int8" not in p


@pytest.fixture
def mock_onnx_model():
    """Create a PureOnnxGLiNER instance with mocked ONNX sessions."""
    with (
        patch("seahorse.models.Tokenizer.from_file") as mock_tok_from_file,
        patch("seahorse.models.ort.InferenceSession") as mock_sess_cls,
        patch("seahorse.models.os.path.exists", side_effect=_exists_fp32_only),
    ):
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {
            "[ENT]": 1, "[SEP]": 2, "Apple": 10, "released": 11,
        }
        mock_tokenizer.token_to_id.side_effect = (
            lambda x: 1 if x == "[ENT]" else (2 if x == "[SEP]" else 0)
        )
        mock_encoding = MagicMock()
        mock_encoding.ids = [1, 50, 1, 51, 2, 10, 11, 12]
        mock_encoding.attention_mask = [1, 1, 1, 1, 1, 1, 1, 1]
        mock_encoding.word_ids = [None, None, None, None, None, 0, 1, 2]
        mock_tokenizer.encode.return_value = mock_encoding
        mock_tok_from_file.return_value = mock_tokenizer

        # Session mocks
        mock_encoder = MagicMock()
        mock_span = MagicMock()
        mock_count = MagicMock()
        mock_encoder.run.return_value = [
            np.random.randn(1, 8, 768).astype(np.float32)
        ]
        mock_count.run.return_value = [
            np.random.randn(1, 2, 768).astype(np.float32)
        ]
        mock_span.run.return_value = [
            np.random.randn(1, 12, 768).astype(np.float32)
        ]

        def sess_side_effect(path, options=None):
            if "encoder" in str(path):
                return mock_encoder
            if "span_rep" in str(path):
                return mock_span
            if "count_embed" in str(path):
                return mock_count
            return MagicMock()

        mock_sess_cls.side_effect = sess_side_effect
        yield PureOnnxGLiNER("mock_path")


# --- Init & Config ---

class TestInit:
    def test_creates_instance(self, mock_onnx_model):
        assert mock_onnx_model is not None
        assert mock_onnx_model.ent_token == "[ENT]"

    def test_sessions_loaded(self, mock_onnx_model):
        assert mock_onnx_model.encoder_session is not None
        assert mock_onnx_model.span_rep_session is not None
        assert mock_onnx_model.count_embed_session is not None

    def test_fallback_tokenizer(self):
        with (
            patch("seahorse.models.Tokenizer.from_file", side_effect=ValueError("bad tokenizer")),
            patch("seahorse.models.ort.InferenceSession"),
            patch("seahorse.models.os.path.exists", side_effect=_exists_fp32_only),
        ):
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_auto_from_pretrained:
                mock_tok = MagicMock()
                mock_tok.get_vocab.return_value = {"[ENT]": 1, "[SEP]": 2}
                mock_tok.convert_tokens_to_ids.side_effect = lambda x: 1 if x == "[ENT]" else 2
                mock_encoding = MagicMock()
                mock_encoding.__getitem__.side_effect = lambda k: {
                    "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
                    "attention_mask": np.array([[1, 1, 1]], dtype=np.int64),
                }[k]
                mock_encoding.word_ids.side_effect = lambda: [None, 0, None]
                mock_tok.return_value = mock_encoding
                mock_tok.encode.return_value = [1]
                mock_auto_from_pretrained.return_value = mock_tok

                PureOnnxGLiNER("model_path")
                mock_auto_from_pretrained.assert_called_with("model_path")


# --- Span Index ---

class TestSpanIndex:
    def test_shape_and_count(self, mock_onnx_model):
        spans = mock_onnx_model.prepare_span_idx(5, max_width=3)
        assert isinstance(spans, np.ndarray)
        assert spans.shape[1] == 2
        expected = sum(1 for i in range(5) for j in range(3) if i + j < 5)
        assert len(spans) == expected

    def test_single_token(self, mock_onnx_model):
        spans = mock_onnx_model.prepare_span_idx(1, max_width=12)
        assert len(spans) == 1
        assert list(spans[0]) == [0, 0]


# --- Feature Extraction ---

class TestFeatureExtraction:
    def test_prompt_features_shape(self, mock_onnx_model):
        token_embeds = np.random.randn(1, 8, 768).astype(np.float32)
        input_ids = np.array([[1, 50, 1, 51, 2, 10, 11, 12]])
        prompts = mock_onnx_model.extract_prompt_features(token_embeds, input_ids)
        assert prompts.shape == (1, 2, 768)

    def test_word_embeddings_shape(self, mock_onnx_model):
        token_embeds = np.random.randn(1, 8, 768).astype(np.float32)
        words_mask = np.array([[0, 0, 0, 0, 0, 1, 2, 3]])
        words = mock_onnx_model.extract_word_embeddings(token_embeds, words_mask, 3)
        assert words.shape == (1, 3, 768)
        np.testing.assert_array_equal(words[0, 0], token_embeds[0, 5])

    def test_no_prompt_tokens(self, mock_onnx_model):
        token_embeds = np.random.randn(1, 8, 768).astype(np.float32)
        input_ids = np.array([[0, 0, 0, 0, 0, 0, 0, 0]])  # no ENT tokens
        prompts = mock_onnx_model.extract_prompt_features(token_embeds, input_ids)
        assert prompts.shape[1] == 0


# --- Predict ---

class TestPredict:
    def test_predict_calls_all_sessions(self, mock_onnx_model):
        entities = mock_onnx_model.predict(TEXT, LABELS)
        assert isinstance(entities, list)
        mock_onnx_model.encoder_session.run.assert_called()
        mock_onnx_model.span_rep_session.run.assert_called()
        mock_onnx_model.count_embed_session.run.assert_called()

    def test_empty_text_returns_empty(self, mock_onnx_model):
        assert mock_onnx_model.predict("", LABELS) == []
        assert mock_onnx_model.predict("   ", LABELS) == []


# --- Sigmoid ---

class TestStableSigmoid:
    def test_normal_values(self):
        x = np.array([0.0, 1.0, -1.0])
        result = PureOnnxGLiNER._stable_sigmoid(x)
        np.testing.assert_allclose(result[0], 0.5, atol=1e-6)
        assert result[1] > 0.5
        assert result[2] < 0.5

    def test_extreme_values_no_overflow(self):
        x = np.array([1000.0, -1000.0])
        result = PureOnnxGLiNER._stable_sigmoid(x)
        np.testing.assert_allclose(result[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(result[1], 0.0, atol=1e-6)
