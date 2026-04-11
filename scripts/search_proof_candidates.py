#!/usr/bin/env python3
"""Search tiny proof fixtures for a clean baseline-vs-Seahorse separation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shutil
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
from seahorse.retrieval import PPRRetriever
from seahorse.session import session_manager
from seahorse.storage import GraphStorage

from scripts.proof_baseline import (
    BASELINE_DEFINITION,
    _cosine,
    _extract_first_passage_text,
    _fetch_passages,
    _git_commit,
    _load_fixture,
    _machine_model,
    _sha256_file,
    _encode_query,
)


def _load_candidates(path: Path) -> List[Dict[str, Any]]:
    candidates = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(candidates, list):
        raise ValueError("candidate file must contain a list")
    return candidates


async def _warm_engine() -> HippoEngine:
    engine = HippoEngine()
    await engine.initialize()
    await engine._ensure_models()
    return engine


async def _prepare_storage(engine: HippoEngine, data_dir: Path) -> None:
    settings.DATA_DIR = data_dir
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    engine.storage = GraphStorage()
    engine.retriever = PPRRetriever(engine.extractor, engine.storage, engine.encoder)
    engine.graph_cache = None
    engine._graph_dirty = True
    session_manager.sessions.clear()
    await engine.storage.initialize()


async def _evaluate_candidate(
    engine: HippoEngine,
    fixture: Dict[str, Any],
    fixture_path: Path,
    root_temp_dir: Path,
) -> Dict[str, Any]:
    fixture_dir = root_temp_dir / fixture["fixture_id"]
    if fixture_dir.exists():
        shutil.rmtree(fixture_dir)
    await _prepare_storage(engine, fixture_dir)

    docs = fixture["docs"]
    optimize_threshold = fixture.get("optimize_synonyms_threshold")

    await engine.add_documents([doc["text"] for doc in docs], source=fixture["fixture_id"])
    links_added = None
    if optimize_threshold is not None:
        links_added = await engine.optimize_synonyms(threshold=float(optimize_threshold))

    passages = await _fetch_passages(engine)
    query_vec = await _encode_query(engine, fixture["query"])

    baseline_start = time.perf_counter()
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for row in passages:
        score = _cosine(query_vec, row["embedding"])
        scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    baseline_elapsed_ms = (time.perf_counter() - baseline_start) * 1000.0

    baseline_top = scored[0][1]
    baseline_doc_id = docs[baseline_top["metadata"]["doc_index"]]["id"]

    seahorse_start = time.perf_counter()
    seahorse_raw_output = await engine.search(fixture["query"], top_k=1)
    seahorse_elapsed_ms = (time.perf_counter() - seahorse_start) * 1000.0
    seahorse_top_text = _extract_first_passage_text(seahorse_raw_output)
    seahorse_top_doc_id = None
    for doc in docs:
        if doc["text"] == seahorse_top_text:
            seahorse_top_doc_id = doc["id"]
            break

    return {
        "fixture_id": fixture["fixture_id"],
        "fixture_path": str(fixture_path),
        "fixture_sha256": _sha256_file(fixture_path),
        "query": fixture["query"],
        "target_doc_id": fixture["target_doc_id"],
        "target_substring": fixture["target_substring"],
        "optimize_synonyms_threshold": optimize_threshold,
        "synonym_links_added": links_added,
        "baseline_name": "cosine_top1_passage_embeddings",
        "baseline_definition": BASELINE_DEFINITION,
        "baseline_duration_ms": round(baseline_elapsed_ms, 3),
        "baseline_top1_doc_id": baseline_doc_id,
        "baseline_top1_score": round(float(scored[0][0]), 6),
        "baseline_top1_text": baseline_top["content"],
        "baseline_hit_at_1": baseline_doc_id == fixture["target_doc_id"],
        "seahorse_duration_ms": round(seahorse_elapsed_ms, 3),
        "seahorse_top1_doc_id": seahorse_top_doc_id,
        "seahorse_top1_text": seahorse_top_text,
        "seahorse_target_present": fixture["target_substring"] in seahorse_raw_output,
        "seahorse_hit_at_1": seahorse_top_doc_id == fixture["target_doc_id"],
        "seahorse_raw_output": seahorse_raw_output,
    }


async def run_candidate_search(
    candidates_path: Path,
    *,
    output_path: Path,
) -> Dict[str, Any]:
    candidates = _load_candidates(candidates_path)
    root_temp_dir = Path(tempfile.mkdtemp(prefix="seahorse_proof_search_"))
    prev_data_dir = os.environ.get("DATA_DIR")
    prev_settings_data_dir = settings.DATA_DIR
    settings.DATA_DIR = root_temp_dir / "warmup"
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["DATA_DIR"] = str(settings.DATA_DIR)

    try:
        engine = await _warm_engine()
        results = []
        for fixture in candidates:
            fixture_tmp = root_temp_dir / f"{fixture['fixture_id']}.json"
            fixture_tmp.write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(
                await _evaluate_candidate(engine, fixture, fixture_tmp, root_temp_dir)
            )

        output = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": _git_commit(),
            "machine_model": _machine_model(),
            "python_version": platform.python_version(),
            "total_ram_mb": round(psutil.virtual_memory().total / (1024 * 1024), 2),
            "available_ram_mb_start": round(psutil.virtual_memory().available / (1024 * 1024), 2),
            "swap_mb_total": round(psutil.swap_memory().total / (1024 * 1024), 2),
            "candidate_source": str(candidates_path),
            "results": results,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        return output
    finally:
        if prev_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = prev_data_dir
        settings.DATA_DIR = prev_settings_data_dir
        shutil.rmtree(root_temp_dir, ignore_errors=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search tiny proof fixtures for a clean baseline-vs-Seahorse separation."
    )
    parser.add_argument(
        "--candidates",
        default="scripts/data/proof_candidates.json",
        help="Path to candidate fixtures JSON",
    )
    parser.add_argument(
        "--output",
        default="docs/proofs/proof_candidate_search_latest.json",
        help="Path to write the search summary JSON",
    )
    return parser


async def _main() -> None:
    args = build_arg_parser().parse_args()
    output = await run_candidate_search(Path(args.candidates), output_path=Path(args.output))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
