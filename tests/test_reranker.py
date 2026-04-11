import numpy as np
import sys
import types
import pytest

from seahorse.reranker import CrossEncoderReranker


class DummyEncoding:
    def __init__(self, ids):
        self.ids = ids


class DummyTokenizer:
    def __init__(self):
        self._vocab = {"[CLS]": 101, "[SEP]": 102, "[PAD]": 0}
        self._next_id = 1000

    def token_to_id(self, token):
        return self._vocab.get(token)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        for token in text.split():
            if token not in self._vocab:
                self._vocab[token] = self._next_id
                self._next_id += 1
            ids.append(self._vocab[token])
        return DummyEncoding(ids)


class DummyInput:
    def __init__(self, name):
        self.name = name


class DummySession:
    def __init__(self, output):
        self.output = output
        self.captured_inputs = None
        self.run_called = 0

    def get_inputs(self):
        return [DummyInput("input_ids"), DummyInput("attention_mask"), DummyInput("token_type_ids")]

    def run(self, _output_names, ort_inputs):
        self.run_called += 1
        self.captured_inputs = ort_inputs
        if callable(self.output):
            return [self.output(ort_inputs)]
        return [self.output]


def make_candidates():
    return [
        {"passage_id": 11, "ppr_score": 0.9, "text": "p1 t1 t2 t3", "ppr_rank": 1},
        {"passage_id": 22, "ppr_score": 0.8, "text": "p2 t1 t2 t3", "ppr_rank": 2},
        {"passage_id": 33, "ppr_score": 0.7, "text": "p3 t1 t2 t3", "ppr_rank": 3},
    ]


def make_reranker(
    *,
    top_n=2,
    fusion_method="weighted_sum",
    w_ppr=0.7,
    w_rerank=0.3,
    output=None,
    model_max_len=16,
    query_max_len=6,
    passage_max_len=8,
):
    if output is None:
        output = np.asarray([[0.1], [0.2]], dtype=np.float32)
    return CrossEncoderReranker(
        enabled=True,
        model_path="model.onnx",
        tokenizer_path="tokenizer.json",
        top_n=top_n,
        fusion_method=fusion_method,
        w_ppr=w_ppr,
        w_rerank=w_rerank,
        rrf_k=60,
        model_max_len=model_max_len,
        query_max_len=query_max_len,
        passage_max_len=passage_max_len,
        tokenizer=DummyTokenizer(),
        session=DummySession(output),
    )


def _query_token_count(model_inputs):
    mask = model_inputs["attention_mask"]
    ttype = model_inputs["token_type_ids"]
    # [CLS] + query + [SEP] in segment 0
    return int(np.sum((mask == 1) & (ttype == 0))) - 2


def _passage_token_count(model_inputs):
    mask = model_inputs["attention_mask"]
    ttype = model_inputs["token_type_ids"]
    # passage + final [SEP] in segment 1
    return int(np.sum((mask == 1) & (ttype == 1))) - 1


def test_top_n_zero_immediate_fallback():
    rr = make_reranker(top_n=0)
    candidates = make_candidates()
    out = rr.rerank_and_fuse("q1 q2", candidates, top_k=2)

    assert [x["passage_id"] for x in out] == [11, 22]
    assert rr._session.run_called == 0


def test_top_n_one_boundary_weighted_sum_and_rrf():
    for fusion in ("weighted_sum", "rrf"):
        rr = make_reranker(
            top_n=1,
            fusion_method=fusion,
            output=np.asarray([[0.5]], dtype=np.float32),
        )
        one = [{"passage_id": 11, "ppr_score": 0.9, "text": "p1", "ppr_rank": 1}]
        out1 = rr.rerank_and_fuse("query", one, top_k=1)
        out2 = rr.rerank_and_fuse("query", one, top_k=1)
        assert [x["passage_id"] for x in out1] == [11]
        assert [x["passage_id"] for x in out2] == [11]


def test_weight_normalization_and_zero_sum_fallback():
    rr = make_reranker(w_ppr=7, w_rerank=3)
    n = rr._normalize_weights(7, 3)
    assert n == (0.7, 0.3)

    rr_zero = make_reranker(w_ppr=0, w_rerank=0)
    out = rr_zero.rerank_and_fuse("query", make_candidates(), top_k=2)
    assert [x["passage_id"] for x in out] == [11, 22]
    assert rr_zero._session.run_called == 0


def test_manual_truncation_only_second_and_per_field_limits():
    rr = make_reranker(model_max_len=12, query_max_len=6, passage_max_len=6)
    passage = "p1 p2 p3 p4 p5 p6 p7 p8 p9"

    short_q = rr._build_inputs("q1 q2 q3 q4", passage)
    long_q = rr._build_inputs("q1 q2 q3 q4 q5 q6 q7 q8", passage)

    q_short = _query_token_count(short_q)
    q_long = _query_token_count(long_q)
    p_short = _passage_token_count(short_q)
    p_long = _passage_token_count(long_q)

    assert q_short == 4
    assert q_long == 6  # query_max_len enforced
    assert p_short <= 6  # passage_max_len enforced
    assert p_long <= 6
    assert p_long < p_short  # only_second: longer query truncates passage
    assert 3 + q_long + p_long <= rr.model_max_len
    assert short_q["input_ids"].shape[0] == rr.model_max_len
    assert short_q["attention_mask"].shape[0] == rr.model_max_len
    assert short_q["token_type_ids"].shape[0] == rr.model_max_len


def test_dtype_policy_input_int64_and_output_float32():
    rr = make_reranker(output=np.asarray([[0.2], [0.1]], dtype=np.float16))
    candidates = make_candidates()[:2]
    out = rr._score_top_candidates("q1 q2 q3", candidates)

    assert rr._session.captured_inputs["input_ids"].dtype == np.int64
    assert rr._session.captured_inputs["attention_mask"].dtype == np.int64
    assert rr._session.captured_inputs["token_type_ids"].dtype == np.int64
    assert out.dtype == np.float32

    rr2 = make_reranker(output=np.asarray([[0.1, 0.6], [0.3, 0.7]], dtype=np.float32))
    out2 = rr2._score_top_candidates("q1 q2", candidates)
    assert out2.dtype == np.float32


def test_dedup_before_top_n_and_no_duplicate_output():
    rr = make_reranker(top_n=2, output=np.asarray([[0.4], [0.3]], dtype=np.float32))
    candidates = [
        {"passage_id": 11, "ppr_score": 0.9, "text": "A", "ppr_rank": 1},
        {"passage_id": 11, "ppr_score": 0.1, "text": "A-dup", "ppr_rank": 2},
        {"passage_id": 22, "ppr_score": 0.8, "text": "B", "ppr_rank": 3},
        {"passage_id": 33, "ppr_score": 0.7, "text": "C", "ppr_rank": 4},
    ]

    out = rr.rerank_and_fuse("query", candidates, top_k=3)
    ids = [x["passage_id"] for x in out]

    assert ids == [11, 22, 33]
    assert len(ids) == len(set(ids))
    assert rr._session.captured_inputs["input_ids"].shape[0] == 3


def test_invalid_onnx_output_shape_fallback_to_ppr():
    rr = make_reranker(top_n=2, output=np.ones((2, 3), dtype=np.float32))
    out = rr.rerank_and_fuse("query", make_candidates()[:2], top_k=2)
    assert [x["passage_id"] for x in out] == [11, 22]


def test_nan_inf_logits_sink_to_bottom_in_top_n():
    rr = make_reranker(top_n=2, output=np.asarray([[np.nan], [1.0]], dtype=np.float32))
    candidates = [
        {"passage_id": 11, "ppr_score": 0.9, "text": "A", "ppr_rank": 1},
        {"passage_id": 22, "ppr_score": 0.8, "text": "B", "ppr_rank": 2},
    ]
    out = rr.rerank_and_fuse("query", candidates, top_k=2)
    assert [x["passage_id"] for x in out] == [22, 11]


def test_rrf_deterministic_tie_break():
    rr = make_reranker(
        top_n=3,
        fusion_method="rrf",
        output=np.asarray([[0.0], [0.0], [0.0]], dtype=np.float32),
    )
    candidates = make_candidates()
    out1 = rr.rerank_and_fuse("query", candidates, top_k=3)
    out2 = rr.rerank_and_fuse("query", candidates, top_k=3)
    assert [x["passage_id"] for x in out1] == [x["passage_id"] for x in out2]


def test_empty_candidates_and_disabled_path():
    rr = make_reranker()
    assert rr.rerank_and_fuse("query", [], top_k=3) == []

    rr_disabled = CrossEncoderReranker(
        enabled=False,
        model_path="model.onnx",
        tokenizer_path="tokenizer.json",
        top_n=2,
        fusion_method="weighted_sum",
        w_ppr=0.7,
        w_rerank=0.3,
        rrf_k=60,
        model_max_len=16,
        query_max_len=4,
        passage_max_len=8,
    )
    out = rr_disabled.rerank_and_fuse("query", make_candidates(), top_k=2)
    assert [x["passage_id"] for x in out] == [11, 22]


def test_top_n_negative_and_unknown_fusion_fallback():
    rr_neg = make_reranker(top_n=-1)
    out_neg = rr_neg.rerank_and_fuse("query", make_candidates(), top_k=2)
    assert [x["passage_id"] for x in out_neg] == [11, 22]

    rr_bad = make_reranker(top_n=2, fusion_method="bad_method")
    out_bad = rr_bad.rerank_and_fuse("query", make_candidates(), top_k=2)
    assert [x["passage_id"] for x in out_bad] == [11, 22]


def test_score_length_mismatch_fallback(monkeypatch):
    rr = make_reranker(top_n=2)
    monkeypatch.setattr(rr, "_score_top_candidates", lambda _q, _c: np.asarray([0.5], dtype=np.float32))
    out = rr.rerank_and_fuse("query", make_candidates(), top_k=2)
    assert [x["passage_id"] for x in out] == [11, 22]


def test_score_empty_inputs_and_reduce_rank_error():
    rr = make_reranker()
    out = rr._score_top_candidates("query", [])
    assert out.dtype == np.float32
    assert out.size == 0

    assert rr._reduce_onnx_output(np.asarray([1.0, 2.0], dtype=np.float32), batch_size=2) is None


def test_build_input_invalid_model_max_len():
    rr = make_reranker(model_max_len=2)
    with pytest.raises(ValueError):
        rr._build_inputs("query", "passage")


def test_postprocess_zscore_clip_sigmoid_and_no_finite_weighted_sum():
    rr = make_reranker()
    rr.apply_zscore = True
    rr.logit_clip = 0.5
    rr.apply_sigmoid = True
    vals = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    out = rr._postprocess_logits(vals)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0) and np.all(out <= 1.0)

    rr2 = make_reranker(top_n=2, output=np.asarray([[np.nan], [np.inf]], dtype=np.float32))
    cands = [
        {"passage_id": 11, "ppr_score": 0.9, "text": "A", "ppr_rank": 1},
        {"passage_id": 22, "ppr_score": 0.8, "text": "B", "ppr_rank": 2},
    ]
    out2 = rr2.rerank_and_fuse("query", cands, top_k=2)
    assert [x["passage_id"] for x in out2] == [11, 22] or [x["passage_id"] for x in out2] == [22, 11]


def test_rrf_invalid_logit_sinks():
    rr = make_reranker(top_n=2, fusion_method="rrf", output=np.asarray([[np.nan], [0.2]], dtype=np.float32))
    cands = [
        {"passage_id": 11, "ppr_score": 0.9, "text": "A", "ppr_rank": 1},
        {"passage_id": 22, "ppr_score": 0.8, "text": "B", "ppr_rank": 2},
    ]
    out = rr.rerank_and_fuse("query", cands, top_k=2)
    assert [x["passage_id"] for x in out] == [22, 11]


def test_runtime_missing_path_and_token_id_fallbacks():
    rr = CrossEncoderReranker(
        enabled=True,
        model_path=None,
        tokenizer_path=None,
        top_n=2,
        fusion_method="weighted_sum",
        w_ppr=0.7,
        w_rerank=0.3,
        rrf_k=60,
        model_max_len=16,
        query_max_len=4,
        passage_max_len=8,
    )
    assert rr._ensure_runtime() is False

    class TokNoMethod:
        pass

    rr._tokenizer = TokNoMethod()
    assert rr._token_to_id("[CLS]") is None

    class TokNoPad:
        def token_to_id(self, token):
            if token == "[CLS]":
                return 101
            if token == "[SEP]":
                return 102
            return None

    rr._tokenizer = TokNoPad()
    ids = rr._get_special_token_ids()
    assert ids["pad_id"] == 0

    class TokMissingSep:
        def token_to_id(self, token):
            if token == "[CLS]":
                return 101
            return None

    rr._tokenizer = TokMissingSep()
    with pytest.raises(ValueError):
        rr._get_special_token_ids()


def test_encode_fallback_typeerror_path():
    class TokNoAddSpecial:
        def token_to_id(self, token):
            return {"[CLS]": 101, "[SEP]": 102, "[PAD]": 0}.get(token)

        def encode(self, text):
            return DummyEncoding([1001 for _ in text.split()])

    rr = CrossEncoderReranker(
        enabled=True,
        model_path="model.onnx",
        tokenizer_path="tokenizer.json",
        top_n=2,
        fusion_method="weighted_sum",
        w_ppr=0.7,
        w_rerank=0.3,
        rrf_k=60,
        model_max_len=12,
        query_max_len=4,
        passage_max_len=8,
        tokenizer=TokNoAddSpecial(),
        session=DummySession(np.asarray([[0.1], [0.2]], dtype=np.float32)),
    )
    built = rr._build_inputs("q1 q2", "p1 p2 p3")
    assert built["input_ids"].dtype == np.int64


def test_extract_input_names_and_runtime_singleton_loader(monkeypatch):
    class BadInputSession:
        def get_inputs(self):
            raise RuntimeError("bad input")

    assert CrossEncoderReranker._extract_input_names(None) == ()
    assert CrossEncoderReranker._extract_input_names(BadInputSession()) == ()

    class FlakySession:
        def __init__(self):
            self._count = 0

        def get_inputs(self):
            self._count += 1
            if self._count == 1:
                raise RuntimeError("first fail")
            return [DummyInput("input_ids"), DummyInput("attention_mask")]

        def run(self, _names, _inputs):
            return [np.asarray([[0.1]], dtype=np.float32)]

    flaky = FlakySession()
    rr = CrossEncoderReranker(
        enabled=True,
        model_path="model.onnx",
        tokenizer_path="tokenizer.json",
        top_n=1,
        fusion_method="weighted_sum",
        w_ppr=0.7,
        w_rerank=0.3,
        rrf_k=60,
        model_max_len=8,
        query_max_len=2,
        passage_max_len=3,
        tokenizer=DummyTokenizer(),
        session=flaky,
    )
    assert rr._input_names == ()
    assert rr._ensure_runtime() is True
    assert rr._input_names == ("input_ids", "attention_mask")

    # Cover runtime loading path with fake tokenizers/onnxruntime modules.
    class FakeTokenizerLoader:
        @staticmethod
        def from_file(_path):
            return DummyTokenizer()

    class FakeOrtModule:
        class InferenceSession:
            def __init__(self, _path, providers):
                del providers
                self.path = _path

            def get_inputs(self):
                return [DummyInput("input_ids"), DummyInput("attention_mask"), DummyInput("token_type_ids")]

            def run(self, _output_names, ort_inputs):
                batch = ort_inputs["input_ids"].shape[0]
                return [np.ones((batch, 1), dtype=np.float32)]

    old_singleton = CrossEncoderReranker._session_singleton
    old_model_path = CrossEncoderReranker._session_model_path
    CrossEncoderReranker._session_singleton = None
    CrossEncoderReranker._session_model_path = None

    monkeypatch.setitem(sys.modules, "tokenizers", types.SimpleNamespace(Tokenizer=FakeTokenizerLoader))
    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrtModule)

    rr_load_1 = CrossEncoderReranker(
        enabled=True,
        model_path="m1.onnx",
        tokenizer_path="tok.json",
        top_n=1,
        fusion_method="weighted_sum",
        w_ppr=0.7,
        w_rerank=0.3,
        rrf_k=60,
        model_max_len=8,
        query_max_len=2,
        passage_max_len=3,
    )
    rr_load_2 = CrossEncoderReranker(
        enabled=True,
        model_path="m2.onnx",
        tokenizer_path="tok.json",
        top_n=1,
        fusion_method="weighted_sum",
        w_ppr=0.7,
        w_rerank=0.3,
        rrf_k=60,
        model_max_len=8,
        query_max_len=2,
        passage_max_len=3,
    )

    assert rr_load_1._ensure_runtime() is True
    assert rr_load_2._ensure_runtime() is True
    assert rr_load_1._session is rr_load_2._session

    CrossEncoderReranker._session_singleton = old_singleton
    CrossEncoderReranker._session_model_path = old_model_path
