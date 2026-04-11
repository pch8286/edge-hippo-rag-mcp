from pathlib import Path

from seahorse.model_setup import ensure_gliner_onnx_model


def _create_min_bundle(base: Path):
    (base / "onnx").mkdir(parents=True, exist_ok=True)
    (base / "tokenizer.json").write_text("{}")
    (base / "onnx" / "encoder.onnx").write_text("x")
    (base / "onnx" / "span_rep.onnx").write_text("x")
    (base / "onnx" / "count_embed.onnx").write_text("x")


def _create_external_data_bundle(base: Path, *, missing_encoder_data: bool):
    (base / "onnx").mkdir(parents=True, exist_ok=True)
    (base / "tokenizer.json").write_text("{}")
    (base / "onnx" / "encoder_fp16.onnx").write_text("x")
    (base / "onnx" / "span_rep_fp16.onnx").write_text("x")
    (base / "onnx" / "count_embed_fp16.onnx").write_text("x")
    if not missing_encoder_data:
        (base / "onnx" / "encoder_fp16.onnx.data").write_text("d")
    (base / "onnx" / "span_rep_fp16.onnx.data").write_text("d")
    (base / "onnx" / "count_embed_fp16.onnx.data").write_text("d")


def _create_mixed_precision_bundle(base: Path):
    (base / "onnx").mkdir(parents=True, exist_ok=True)
    (base / "tokenizer.json").write_text("{}")
    for name in ("encoder", "span_rep", "count_embed"):
        (base / "onnx" / f"{name}_fp16.onnx").write_text("x")
        (base / "onnx" / f"{name}.onnx").write_text("x")


def test_skip_download_when_bundle_exists(monkeypatch, tmp_path):
    _create_min_bundle(tmp_path)

    def _should_not_be_called(**_kwargs):
        raise AssertionError("snapshot_download should not be called")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _should_not_be_called)
    changed = ensure_gliner_onnx_model(str(tmp_path), repo_id="repo/test", force=False)
    assert changed is False


def test_download_when_missing_bundle(monkeypatch, tmp_path):
    def _fake_download(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        _create_min_bundle(local_dir)
        return str(local_dir)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_download)
    changed = ensure_gliner_onnx_model(str(tmp_path), repo_id="repo/test", force=False)
    assert changed is True
    assert (tmp_path / "tokenizer.json").exists()


def test_redownload_when_external_data_is_incomplete(monkeypatch, tmp_path):
    _create_external_data_bundle(tmp_path, missing_encoder_data=True)
    monkeypatch.setattr(
        "seahorse.model_setup._onnx_requires_external_data",
        lambda p: p.name.endswith("_fp16.onnx"),
    )

    def _fake_hf_download(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        filename = kwargs["filename"]
        target = local_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("d")
        return str(target)

    def _snapshot_should_not_be_called(**_kwargs):
        raise AssertionError("snapshot_download should not be called for sidecar repair")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _fake_hf_download)
    monkeypatch.setattr("huggingface_hub.snapshot_download", _snapshot_should_not_be_called)
    changed = ensure_gliner_onnx_model(str(tmp_path), repo_id="repo/test", force=False)
    assert changed is True
    assert (tmp_path / "onnx" / "encoder_fp16.onnx.data").exists()


def test_component_falls_back_to_fp32_when_fp16_sidecar_missing(monkeypatch, tmp_path):
    _create_mixed_precision_bundle(tmp_path)
    monkeypatch.setattr(
        "seahorse.model_setup._onnx_requires_external_data",
        lambda p: p.name.endswith("_fp16.onnx"),
    )

    def _should_not_be_called(**_kwargs):
        raise AssertionError("download should not be called when fp32 fallback is valid")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _should_not_be_called)
    monkeypatch.setattr("huggingface_hub.snapshot_download", _should_not_be_called)
    changed = ensure_gliner_onnx_model(str(tmp_path), repo_id="repo/test", force=False)
    assert changed is False
