#!/usr/bin/env python3
import argparse
import asyncio
import logging
import time
import json
import os
import resource
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path to ensure edge_hippo is importable
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from edge_hippo.hippo_engine import HippoEngine
from edge_hippo.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("benchmark_tool")

class ProfilingBenchmark:
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "platform": os.uname().sysname,
                "node": os.uname().nodename,
            },
            "metrics": {}
        }
        self.temp_dir = tempfile.mkdtemp(prefix="benchmark_hippo_")
        logger.info(f"Using temp dir for benchmark: {self.temp_dir}")
        
        # Override settings to use temp dir
        self.original_data_dir = settings.DATA_DIR
        settings.DATA_DIR = Path(self.temp_dir)
        settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

    def cleanup(self):
        logger.info(f"Cleaning up temp dir: {self.temp_dir}")
        shutil.rmtree(self.temp_dir)
        settings.DATA_DIR = self.original_data_dir

    def _measure_ram(self) -> float:
        """Returns peak RAM usage in MB."""
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: usage is in kb
        return usage / 1024.0

    async def run_indexing_benchmark(self, dataset_path: str, batch_size: int = 1):
        logger.info(f"Starting Indexing Benchmark with {dataset_path} (Batch Size: {batch_size})...")
        
        # Load dataset
        with open(dataset_path, "r") as f:
            data = json.load(f)
        
        engine = HippoEngine()
        await engine.initialize()
        
        start_time = time.time()
        start_ram = self._measure_ram()
        
        if batch_size > 1:
            documents = []
            for item in data:
                content = item.get("content", "") or item.get("text", "")
                if content:
                    documents.append(content)
            
            # Chunk into batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                if batch:
                    await engine.add_documents(batch, source="benchmark_batch")
        else:
            for item in data:
                content = item.get("content", "") or item.get("text", "")
                if content:
                    await engine.add_document(content, source="benchmark")
        
        # Ensure graph is flushed/optimized if needed (though add_document commits usually)
        # We might want to run finalize_index as part of benchmark?
        # Creating index is part of the process.
        await engine.finalize_index()

        end_time = time.time()
        peak_ram = self._measure_ram()
        
        self.results["metrics"]["indexing"] = {
             "duration_seconds": end_time - start_time,
             "peak_ram_mb": peak_ram,
             "dataset": dataset_path,
             "items_count": len(data),
             "batch_size": batch_size,
             "items_per_second": len(data) / (end_time - start_time) if (end_time - start_time) > 0 else 0
        }
        logger.info(f"Indexing completed in {end_time - start_time:.2f}s. Batch:{batch_size}, Speed: {self.results['metrics']['indexing']['items_per_second']:.2f} it/s")

    async def run_retrieval_benchmark(self, queries: List[str]):
        logger.info("Starting Retrieval Benchmark...")
        if not queries:
             queries = ["Raspberry Pi", "AI Model", "Edge Computing"]

        engine = HippoEngine()
        await engine.initialize()
        
        start_time = time.time()
        latencies = []
        
        for q in queries:
            t0 = time.time()
            await engine.search(q)
            t1 = time.time()
            latencies.append(t1 - t0)
            
        end_time = time.time()
        peak_ram = self._measure_ram()
        
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        self.results["metrics"]["retrieval"] = {
            "total_duration_seconds": end_time - start_time,
            "avg_latency_seconds": avg_latency,
            "peak_ram_mb": peak_ram,
            "queries_count": len(queries)
        }
        logger.info(f"Retrieval completed. Avg Latency: {avg_latency:.4f}s. Peak RAM: {peak_ram:.2f}MB")

    async def run_semantic_benchmark(self, dataset_path: str):
        logger.info(f"Starting Semantic Benchmark with {dataset_path}...")
        
        with open(dataset_path, "r") as f:
            data = json.load(f)
            
        engine = HippoEngine()
        await engine.initialize()
        
        # 1. Indexing (measure throughput)
        start_index = time.time()
        docs = [item["text"] for item in data]
        # Batch add for speed in setup, but we want to measure indexing if not already done?
        # Let's assume we index fresh for this benchmark.
        await engine.add_documents(docs, source="semantic_bench")
        # Optimization is CRITICAL for semantic search (synonym linking)
        t_opt_start = time.time()
        await engine.optimize_synonyms(threshold=0.6)
        t_opt_end = time.time()
        end_index = time.time()
        
        logger.info(f"Indexing + Optimization took {end_index - start_index:.2f}s (Opt: {t_opt_end - t_opt_start:.2f}s)")
        
        # 2. Retrieval (measure accuracy/hit rate implicitly via latency? No, just latency for now)
        # We search for the "expected_queries" and measure time.
        latencies = []
        hits = 0
        total_queries = 0
        
        for item in data:
            target_text = item["text"][:30] # First 30 chars acting as ID
            for query in item.get("expected_queries", []):
                total_queries += 1
                t0 = time.time()
                result_str = await engine.search(query)
                t1 = time.time()
                latencies.append(t1 - t0)
                
                # Simple check if target doc is in result (heuristic)
                # Engine result is a string, so we check for substring.
                if target_text in result_str:
                    hits += 1
        
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        hit_rate = hits / total_queries if total_queries else 0
        
        self.results["metrics"]["semantic"] = {
            "indexing_duration": end_index - start_index,
            "optimization_duration": t_opt_end - t_opt_start,
            "avg_latency": avg_latency,
            "hit_rate": hit_rate,
            "total_queries": total_queries
        }
        logger.info(f"Semantic Bench: Latency={avg_latency:.4f}s, Hit Rate={hit_rate:.1%}, Index+Opt={end_index - start_index:.2f}s")

    async def run_sqlite_benchmark(self):
        logger.info("Starting SQLite Benchmark...")
        # Simple FTS test
        engine = HippoEngine()
        await engine.initialize()
        
        start_time = time.time()
        async with engine.storage._get_conn() as db:
             async with db.execute("SELECT count(*) FROM nodes") as cursor:
                 await cursor.fetchone()
             # Run a heavy query
             async with db.execute("SELECT * FROM nodes WHERE type='phrase' ORDER BY id DESC LIMIT 10") as cursor:
                 await cursor.fetchall()
                 
        end_time = time.time()
        self.results["metrics"]["sqlite"] = {
            "duration_seconds": end_time - start_time
        }
        logger.info(f"SQLite benchmark completed in {end_time - start_time:.4f}s")

    async def run_startup_benchmark(self):
        logger.info("Starting Startup Benchmark...")
        start_time = time.time()
        engine = HippoEngine()
        await engine.initialize()
        interactive_time = time.time()
        peak_ram_init = self._measure_ram()
        
        # Wait for models to fully load
        if hasattr(engine, '_model_loading_task') and engine._model_loading_task:
            await engine._model_loading_task
            
        ready_time = time.time()
        peak_ram_ready = self._measure_ram()
        
        self.results["metrics"]["startup"] = {
            "time_to_interactive": interactive_time - start_time,
            "time_to_ready": ready_time - start_time,
            "peak_ram_mb_interactive": peak_ram_init,
            "peak_ram_mb_ready": peak_ram_ready
        }
        logger.info(f"Startup: TTI={interactive_time - start_time:.4f}s ({peak_ram_init:.2f}MB), TTR={ready_time - start_time:.4f}s ({peak_ram_ready:.2f}MB)")

    async def calculate_adaptive_recall(self, retrieved_chunks: List[str], relevant_chunks: List[str], used_nodes: int, max_nodes: int = 1500) -> float:
        """
        Score = (Relevant Intersect Retrieved / Total Relevant) * exp(-alpha * (Used / Max))
        """
        if not relevant_chunks:
            return 0.0
        
        # Simple exact match or substring match for chunks
        hits = 0
        for rel in relevant_chunks:
            for ret in retrieved_chunks:
                if rel in ret or ret in rel: # Loose matching
                    hits += 1
                    break
        
        recall = hits / len(relevant_chunks)
        alpha = 0.5
        import math
        penalty = math.exp(-alpha * (used_nodes / max_nodes))
        return recall * penalty

    def calculate_context_drift(self, result_vector: List[float], irrelevant_vectors: List[List[float]]) -> float:
        """
        Score = 1 - max(cos_sim(result, irrelevant_t))
        """
        if irrelevant_vectors is None or len(irrelevant_vectors) == 0:
            return 1.0
            
        import numpy as np
        
        def cos_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            
        max_sim = 0.0
        for irr_vec in irrelevant_vectors:
            sim = cos_sim(result_vector, irr_vec)
            if sim > max_sim:
                max_sim = sim
                
        return 1.0 - max_sim

    def calculate_token_compression(self, prompt_tokens: int, raw_doc_tokens: int) -> float:
        """
        Score = 1 - (Prompt Tokens / Raw Document Tokens)
        """
        if raw_doc_tokens == 0:
            return 0.0
        return 1.0 - (prompt_tokens / raw_doc_tokens)

    async def calculate_linkage_density(self, engine: HippoEngine) -> float:
        """
        Score = Valid Synonym Edges / Total Phrase Nodes
        Synonym Edges are edges where both source and target are 'phrase' type.
        """
        async with engine.storage._get_conn() as db:
            # Count phrase nodes
            async with db.execute("SELECT count(*) FROM nodes WHERE type='phrase'") as cursor:
                row = await cursor.fetchone()
                phrase_nodes = row[0] if row else 0
                
            if phrase_nodes == 0:
                return 0.0
                
            # Count synonym edges (Phrase -> Phrase)
            query = """
                SELECT count(*) 
                FROM edges e
                JOIN nodes ns ON e.source = ns.id
                JOIN nodes nt ON e.target = nt.id
                WHERE ns.type = 'phrase' AND nt.type = 'phrase'
            """
            async with db.execute(query) as cursor:
                row = await cursor.fetchone()
                synonym_edges = row[0] if row else 0
                
        return synonym_edges / phrase_nodes

    def generate_report(self):
        if not self.results["metrics"]:
            logger.warning("No metrics collected.")
            return

        report_path = self.output_dir / f"benchmark_report_{int(time.time())}.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Benchmark results saved to {report_path}")
        
        # Priority Report Markdown
        md_path = self.output_dir / "PRIORITY_REPORT.md"
        with open(md_path, "w") as f:
            f.write("# Profiling Priority Report\n\n")
            f.write("## Bottleneck Analysis\n")
            # Simple heuristic
            idx_time = self.results["metrics"].get("indexing", {}).get("duration_seconds", 0)
            ret_time = self.results["metrics"].get("retrieval", {}).get("avg_latency_seconds", 0)
            startup_time = self.results["metrics"].get("startup", {}).get("duration_seconds", 0)
            
            f.write(f"- **Indexing Time**: {idx_time:.2f}s\n")
            f.write(f"- **Avg Retrieval Latency**: {ret_time:.4f}s\n")
            f.write(f"- **Startup Time**: {startup_time:.4f}s\n\n")
            
            # Identify slowest
            f.write("## Optimization Recommendations\n")
            if idx_time > 10:
                f.write("- [HIGH] Optimize GLiNER batching or interaction.\n")
            if ret_time > 1.0:
                 f.write("- [HIGH] Optimize PPR calculation or subgraph extraction.\n")
            if startup_time > 2.0:
                 f.write("- [MEDIUM] Consider lazy loading of models.\n")
        
        logger.info(f"Priority Report saved to {md_path}")

    async def run_comprehensive_eval(self, dataset_path: str):
        logger.info(f"Starting Comprehensive Evaluation with {dataset_path}...")
        
        with open(dataset_path, "r") as f:
            data = json.load(f)
            
        engine = HippoEngine()
        await engine.initialize()
        
        # 1. Indexing
        start_index = time.time()
        docs = [item["text"] for item in data]
        await engine.add_documents(docs, source="eval_bench")
        await engine.optimize_synonyms(threshold=0.6)
        end_index = time.time()
        
        # Metrics buckets
        recall_scores = []
        raw_recall_scores = []
        drift_scores = []
        compression_scores = []
        
        total_queries = 0
        
        # Initialize embedding model for drift calculation (cosine sim)
        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer("intfloat/multilingual-e5-small")
        except Exception as e:
            logger.warning(f"Could not load independent embedder for drift calc: {e}")
            embedder = None

        for item in data:
            relevant_chunks = [item["text"]] 
            irrelevant_topics = item.get("irrelevant_topics", [])
            irrelevant_vectors = []
            if embedder and irrelevant_topics:
                 irrelevant_vectors = embedder.encode(irrelevant_topics)

            for query in item.get("expected_queries", []):
                total_queries += 1
                result_str = await engine.search(query)
                
                # 1. Adaptive Recall
                used_nodes_est = len(result_str.split()) / 5 
                
                # Calculate Raw Recall first
                hits = 0
                for rel in relevant_chunks:
                    for ret in [result_str]:
                        if rel in ret or ret in rel:
                            hits += 1
                            break
                raw_recall = hits / len(relevant_chunks) if relevant_chunks else 0.0
                raw_recall_scores.append(raw_recall)

                score_recall = await self.calculate_adaptive_recall(
                    retrieved_chunks=[result_str], 
                    relevant_chunks=relevant_chunks,
                    used_nodes=used_nodes_est,
                    max_nodes=1500
                )
                recall_scores.append(score_recall)
                
                # 2. Context Drift
                if embedder and irrelevant_vectors is not None:
                    result_vec = embedder.encode(result_str)
                    score_drift = self.calculate_context_drift(result_vec, irrelevant_vectors)
                    drift_scores.append(score_drift)
                else:
                    drift_scores.append(1.0)
                
                # 3. Token Compression
                prompt_tokens = len(result_str) / 4
                raw_tokens = len(item["text"]) / 4
                score_comp = self.calculate_token_compression(prompt_tokens, raw_tokens)
                compression_scores.append(score_comp)
        
        # 4. Linkage Density (Global)
        linkage_score = await self.calculate_linkage_density(engine)
        
        # Aggregation
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
        avg_raw_recall = sum(raw_recall_scores) / len(raw_recall_scores) if raw_recall_scores else 0
        avg_drift = sum(drift_scores) / len(drift_scores) if drift_scores else 0
        avg_compression = sum(compression_scores) / len(compression_scores) if compression_scores else 0
        
        self.results["metrics"]["evaluation"] = {
            "adaptive_recall": avg_recall,
            "raw_recall": avg_raw_recall,
            "context_drift_control": avg_drift,
            "token_compression": avg_compression,
            "cross_entity_linkage": linkage_score,
            "queries_processed": total_queries
        }
        
        logger.info("Evaluation Complete:")
        logger.info(f"- Raw Recall: {avg_raw_recall:.2%}")
        logger.info(f"- Adaptive Recall: {avg_recall:.2%}")
        logger.info(f"- Drift Control: {avg_drift:.2%}")
        logger.info(f"- Token Compression: {avg_compression:.2%}")
        logger.info(f"- Linkage Density: {linkage_score:.2f} (Edges/Node)")
        
        self.update_readme_metrics()

    def update_readme_metrics(self):
        readme_path = Path(__file__).parent.parent / "README.md"
        if not readme_path.exists():
            logger.warning("README.md not found, skipping auto-update.")
            return

        with open(readme_path, "r") as f:
            content = f.read()
            
        metrics = self.results["metrics"].get("evaluation", {})
        if not metrics:
            return
            
        updates = {
            "Adaptive Fact Recall": metrics["adaptive_recall"],
            "Context Drift Control": metrics["context_drift_control"],
            "Token Compression": metrics["token_compression"],
            "Cross-Entity Linkage": metrics["cross_entity_linkage"] 
        }
        
        import re
        
        def generate_bar(percentage):
            blocks = int(percentage * 20)
            bar = "█" * blocks + "░" * (20 - blocks)
            return f"|{bar}| {int(percentage * 100)}%"

        new_content = content
        for label, value in updates.items():
            if label == "Cross-Entity Linkage":
                norm_value = min(value / 2.5, 1.0)
                bar_str = generate_bar(norm_value)
            else:
                bar_str = generate_bar(value)
                
            regex_line = re.compile(rf"({re.escape(label)}\s*)\|[█░]+\|\s*\d+%")
            new_content = regex_line.sub(rf"\1 {bar_str}", new_content)
        
        if new_content != content:
            with open(readme_path, "w") as f:
                f.write(new_content)
            logger.info("Updated README.md with new metrics.")

async def main():
    parser = argparse.ArgumentParser(description="Edge-Hippo Profiling Benchmark Tool")
    parser.add_argument("--scenario", choices=["synthetic", "medium", "complex", "stress", "all", "semantic", "eval"], default="synthetic", help="Benchmark scenario to run")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for indexing")
    args = parser.parse_args()

    benchmark = ProfilingBenchmark()
    
    current_dir = Path(__file__).parent.parent

    try:
        data_file = "scripts/data/synthetic_small.json"
        
        # Special Eval Mode
        if args.scenario == "eval":
            data_file = "scripts/data/eval_scenarios.json"
            full_data_path = current_dir / data_file
            if full_data_path.exists():
                await benchmark.run_comprehensive_eval(str(full_data_path))
            else:
                logger.error(f"Eval dataset not found at {full_data_path}")
            return

        # 1. Startup
        await benchmark.run_startup_benchmark()
        
        # 2. Indexing (Populates DB for next steps)
        # Note: In real usage we might want to isolate tests, but here we need data for retrieval.
        current_dir = Path(__file__).parent.parent
        if args.scenario != "semantic":
            data_file = "scripts/data/synthetic_small.json"
            full_data_path = current_dir / data_file
            if full_data_path.exists():
                await benchmark.run_indexing_benchmark(str(full_data_path), batch_size=args.batch_size)
            else:
                 logger.warning(f"Default dataset {full_data_path} not found. Skipping default indexing.")
        
        # 3. Retrieval
        await benchmark.run_retrieval_benchmark(["Raspberry Pi", "MCP"])
        
        # 4. SQLite
        await benchmark.run_sqlite_benchmark()
        
        # 5. Semantic (New)
        if args.scenario in ["all", "semantic"]:
            data_file = "scripts/data/synthetic_semantic.json"
            full_data_path = current_dir / data_file
            if full_data_path.exists():
                await benchmark.run_semantic_benchmark(str(full_data_path))
            else:
                logger.warning(f"Semantic dataset not found at {full_data_path}")
                
        benchmark.generate_report()
    finally:
        benchmark.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

