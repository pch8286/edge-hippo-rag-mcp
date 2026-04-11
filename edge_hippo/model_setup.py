import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def ensure_gliner_onnx_model(
    model_dir: str,
    *,
    repo_id: str = "lmo3/gliner2-multi-v1-onnx",
    force: bool = False,
) -> bool:
    """Ensure GLiNER ONNX bundle exists locally.

    Returns:
        True if download happened, False if already present.
    """
    target = Path(model_dir)
    if not force and _is_gliner_onnx_ready(target):
        return False

    target.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download, snapshot_download

    # Fast-path repair for interrupted downloads: fetch only missing sidecars.
    if not force:
        repaired = False
        for missing_file in _missing_external_data_files(target):
            logger.info("Repairing missing ONNX sidecar: %s", missing_file)
            hf_hub_download(
                repo_id=repo_id,
                filename=missing_file,
                local_dir=str(target),
            )
            repaired = True
        if repaired and _is_gliner_onnx_ready(target):
            return True

    logger.info("Downloading ONNX GLiNER bundle %s to %s", repo_id, target)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        force_download=force,
        allow_patterns=["*.onnx", "*.json", "*.txt", "*.data"],
    )

    if not _is_gliner_onnx_ready(target):
        raise RuntimeError(f"ONNX GLiNER bundle incomplete after download: {target}")
    return True


def _is_gliner_onnx_ready(path: Path) -> bool:
    if not (path / "tokenizer.json").exists():
        return False
    onnx = path / "onnx"
    has_encoder = _component_ready(onnx, "encoder")
    has_span = _component_ready(onnx, "span_rep")
    has_count = _component_ready(onnx, "count_embed")
    return has_encoder and has_span and has_count


def _component_ready(onnx_dir: Path, name: str) -> bool:
    candidates = [onnx_dir / f"{name}_fp16.onnx", onnx_dir / f"{name}.onnx"]
    for candidate in candidates:
        if not candidate.exists():
            continue
        if _onnx_file_ready(candidate):
            return True
    return False


def _onnx_file_ready(onnx_path: Path) -> bool:
    data_path = onnx_path.parent / f"{onnx_path.name}.data"
    requires_external = _onnx_requires_external_data(onnx_path)
    if requires_external is True:
        return data_path.exists()
    if requires_external is False:
        return True
    if data_path.exists():
        return True
    # Fallback heuristic when ONNX parsing fails:
    # mixed external-data bundles should have sibling .data per .onnx.
    if any(onnx_path.parent.glob("*.onnx.data")):
        return False
    return True


def _onnx_requires_external_data(onnx_path: Path) -> Optional[bool]:
    try:
        import onnx

        model = onnx.load(str(onnx_path), load_external_data=False)
        for tensor in model.graph.initializer:
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                return True
        return False
    except Exception:
        return None


def _missing_external_data_files(path: Path) -> list[str]:
    onnx_dir = path / "onnx"
    if not onnx_dir.exists():
        return []

    missing: list[str] = []
    for onnx_path in onnx_dir.glob("*.onnx"):
        if _onnx_requires_external_data(onnx_path) is True:
            data_path = onnx_dir / f"{onnx_path.name}.data"
            if not data_path.exists():
                missing.append(str(data_path.relative_to(path)).replace("\\", "/"))
    return missing
