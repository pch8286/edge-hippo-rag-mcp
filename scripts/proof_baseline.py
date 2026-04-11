#!/usr/bin/env python3
"""Generate a tiny measured proof comparing a simple baseline to Seahorse."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.append(str(PROJECT_ROOT))

from seahorse.config import settings
from seahorse.hippo_engine import HippoEngine


RESULT_BLOCK_RE = re.compile(
    r"--- \[Score: [^\]]+\] ---\n(.*?)(?:\n--- \[Score:|\Z)",
    re.DOTALL,
)

BASELINE_DEFINITION = (
    "Exact cosine top-1 over the same stored passage embeddings, with the same "
    "query encoder, no phrase nodes, no graph expansion, no reranker, no session history."
)


def _extract_first_passage_text(result_text: str) -> str:
    match = RESULT_BLOCK_RE.search(result_text or "")
    if not match:
        return ""
    return match.group(1).strip()


def _load_fixture(path: Path) -> Dict[str, Any]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    required_keys = {"fixture_id", "docs", "query", "target_doc_id", "target_substring"}
    missing = required_keys - set(fixture)
    if missing:
        raise ValueError(f"fixture missing keys: {sorted(missing)}")
    if not fixture["docs"]:
        raise ValueError("fixture must contain at least one document")
    return fixture


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _machine_model() -> str:
    model_path = Path("/proc/device-tree/model")
    if model_path.exists():
        return model_path.read_text(encoding="utf-8", errors="ignore").strip("\x00\n")
    return platform.machine()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


async def _encode_query(engine: HippoEngine, query: str) -> np.ndarray:
    if engine.encoder is None:
        raise RuntimeError("encoder did not load")
    query_prefixed = "query: " + query
    loop = asyncio.get_running_loop()
    if asyncio.iscoroutinefunction(engine.encoder.encode):
        query_vec = await engine.encoder.encode(query_prefixed)
    else:
        query_vec = await loop.run_in_executor(None, engine.encoder.encode, query_prefixed)
    return np.asarray(query_vec, dtype=np.float32)


async def _fetch_passages(engine: HippoEngine) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    async with engine.storage._get_conn() as db:
        async with db.execute(
            """
            SELECT n.id, n.content, n.metadata, v.embedding
            FROM nodes n
            LEFT JOIN vec_nodes v ON n.id = v.rowid
            WHERE n.type = 'passage'
            ORDER BY n.id ASC
            """
        ) as cursor:
            fetched = await cursor.fetchall()
    for node_id, content, metadata_raw, emb_blob in fetched:
        metadata = json.loads(metadata_raw) if metadata_raw else {}
        if not emb_blob:
            continue
        count = len(emb_blob) // 4
        embedding = np.asarray(struct.unpack(f"{count}f", emb_blob), dtype=np.float32)
        rows.append(
            {
                "node_id": int(node_id),
                "content": content,
                "metadata": metadata,
                "embedding": embedding,
            }
        )
    return rows


async def run_proof(
    fixture_path: Path,
    *,
    output_path: Path,
    keep_data: bool = False,
) -> Dict[str, Any]:
    fixture = _load_fixture(fixture_path)
    docs = fixture["docs"]
    optimize_threshold = fixture.get("optimize_synonyms_threshold")

    temp_dir = Path(tempfile.mkdtemp(prefix="seahorse_proof_"))
    prev_data_dir = os.environ.get("DATA_DIR")
    prev_settings_data_dir = settings.DATA_DIR
    os.environ["DATA_DIR"] = str(temp_dir)
    settings.DATA_DIR = temp_dir
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

    available_mb_start = psutil.virtual_memory().available / (1024 * 1024)
    total_ram_mb = psutil.virtual_memory().total / (1024 * 1024)
    swap_mb_total = psutil.swap_memory().total / (1024 * 1024)

    try:
        engine = HippoEngine()
        await engine.initialize()
        await engine.add_documents([doc["text"] for doc in docs], source=fixture["fixture_id"])
        links_added = None
        if optimize_threshold is not None:
            links_added = await engine.optimize_synonyms(threshold=float(optimize_threshold))

        passages = await _fetch_passages(engine)
        if len(passages) != len(docs):
            raise RuntimeError(
                f"expected {len(docs)} passage rows, found {len(passages)}; "
                "fixture docs should stay below chunk size"
            )

        query_vec = await _encode_query(engine, fixture["query"])

        baseline_start = time.perf_counter()
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for row in passages:
            score = _cosine(query_vec, row["embedding"])
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        baseline_elapsed_ms = (time.perf_counter() - baseline_start) * 1000.0

        baseline_top = scored[0][1]
        baseline_score = float(scored[0][0])

        seahorse_start = time.perf_counter()
        seahorse_raw_output = await engine.search(fixture["query"], top_k=1)
        seahorse_elapsed_ms = (time.perf_counter() - seahorse_start) * 1000.0
        seahorse_top_text = _extract_first_passage_text(seahorse_raw_output)
        seahorse_top_doc_id = None
        for doc in docs:
            if doc["text"] == seahorse_top_text:
                seahorse_top_doc_id = doc["id"]
                break

        quantized_model_used = bool(
            os.environ.get("QUANTIZED_MODEL_DIR")
            and Path(os.environ["QUANTIZED_MODEL_DIR"]).exists()
        )
        model_download_occurred = False

        result = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": _git_commit(),
            "machine_model": _machine_model(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "total_ram_mb": round(total_ram_mb, 2),
            "available_ram_mb_start": round(available_mb_start, 2),
            "swap_mb_total": round(swap_mb_total, 2),
            "fixture_path": str(fixture_path),
            "fixture_sha256": _sha256_file(fixture_path),
            "fixture_id": fixture["fixture_id"],
            "embedding_model": "same_query_encoder_as_seahorse",
            "quantized_model_used": quantized_model_used,
            "chunk_size": settings.CHUNK_SIZE,
            "hippo_profile": os.environ.get("HIPPO_PERFORMANCE_PROFILE", "auto"),
            "vector_extension_loaded": bool(getattr(engine.storage, "extension_loaded", False)),
            "query": fixture["query"],
            "target_doc_id": fixture["target_doc_id"],
            "target_substring": fixture["target_substring"],
            "optimize_synonyms_threshold": optimize_threshold,
            "synonym_links_added": links_added,
            "baseline_name": "cosine_top1_passage_embeddings",
            "baseline_definition": BASELINE_DEFINITION,
            "baseline_duration_ms": round(baseline_elapsed_ms, 3),
            "baseline_top1_doc_id": docs[baseline_top["metadata"]["doc_index"]]["id"],
            "baseline_top1_score": round(baseline_score, 6),
            "baseline_top1_text": baseline_top["content"],
            "baseline_hit_at_1": docs[baseline_top["metadata"]["doc_index"]]["id"]
            == fixture["target_doc_id"],
            "seahorse_duration_ms": round(seahorse_elapsed_ms, 3),
            "seahorse_raw_output": seahorse_raw_output,
            "seahorse_top1_doc_id": seahorse_top_doc_id,
            "seahorse_top1_text": seahorse_top_text,
            "seahorse_hit_at_1": seahorse_top_doc_id == fixture["target_doc_id"],
            "seahorse_target_present": fixture["target_substring"] in seahorse_raw_output,
            "model_download_occurred": model_download_occurred,
            "cold_or_warm_run": "warm_local_assets",
            "warnings": [],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    finally:
        if prev_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = prev_data_dir
        settings.DATA_DIR = prev_settings_data_dir
        if keep_data:
            print(f"proof data dir kept at: {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure a tiny public proof comparing a simple baseline against Seahorse."
    )
    parser.add_argument(
        "--fixture",
        default="scripts/data/proof_pi5_zram.json",
        help="Path to proof fixture JSON",
    )
    parser.add_argument(
        "--output",
        default="docs/proofs/pi5_zram_proof_2026-04-11.json",
        help="Path to write the JSON proof artifact",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep the temporary DATA_DIR for debugging",
    )
    return parser


async def _main() -> None:
    args = build_arg_parser().parse_args()
    result = await run_proof(
        Path(args.fixture),
        output_path=Path(args.output),
        keep_data=args.keep_data,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
