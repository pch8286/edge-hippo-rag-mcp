# Performance Snapshot: Raspberry Pi 5 (4GB)

Date: 2026-04-11

## Known
- Machine: Raspberry Pi 5 Model B Rev 1.1
- CPU: Cortex-A76, 4 cores, 2.4 GHz
- RAM: 4 GB
- Swap/zram: 2 GB
- OS: Debian 13 (trixie), aarch64
- Python: 3.13.5
- GLiNER path used: local pre-downloaded ONNX GLiNER bundle
- Quantized embedding path used: local pre-downloaded quantized multilingual-e5-small bundle

## Unknown
- These numbers do not represent every Raspberry Pi 5 configuration.
- The tiny public eval set is too small to serve as a leaderboard benchmark.
- Thermal drift and long-run throttling were not profiled in detail during this pass.

## Assumptions
- Local ONNX and quantized model assets were reused to avoid re-downloading models.
- The goal of this snapshot is operational guidance for this exact machine, not a universal promise.

## Commands Used

Public smoke eval:

```bash
GLINER_ONNX_PATH=/path/to/gliner_onnx \
QUANTIZED_MODEL_DIR=/path/to/models_quantized \
/tmp/seahorse_bench_min_venv/bin/python scripts/benchmark_tool.py --scenario eval
```

Operational metrics snapshot:

```bash
GLINER_ONNX_PATH=/path/to/gliner_onnx \
QUANTIZED_MODEL_DIR=/path/to/models_quantized \
/tmp/seahorse_bench_min_venv/bin/python - <<'PY'
import asyncio, json
from pathlib import Path
from scripts.benchmark_tool import ProfilingBenchmark

dataset = Path('scripts/data/eval_scenarios.json')
with dataset.open() as f:
    data = json.load(f)
queries = []
for item in data:
    queries.extend(item.get('expected_queries', []))

async def main():
    bench = ProfilingBenchmark(output_dir='benchmark_results_public')
    try:
        await bench.run_startup_benchmark()
        await bench.run_indexing_benchmark(str(dataset), batch_size=1)
        await bench.run_retrieval_benchmark(queries)
        await bench.run_sqlite_benchmark()
        print(json.dumps(bench.results, indent=2))
    finally:
        bench.cleanup()

asyncio.run(main())
PY
```

## Public Smoke Eval Result

Dataset: `scripts/data/eval_scenarios.json`

| Metric | Result |
| :--- | :--- |
| Raw recall | **100.00%** |
| Adaptive recall | **99.17%** |
| Drift control | **100.00%** |
| Token compression | **-433.59%** |
| Linkage density | **5.33** |

Notes:
- `Token compression` is currently heuristic and should not be treated as a polished public KPI.
- This eval covers only 4 docs / 5 queries and is best used as a regression sanity check.

## Operational Metrics

| Metric | Result |
| :--- | :--- |
| Startup TTI | **0.014s** |
| Startup ready | **11.167s** |
| Peak RSS at ready | **1185.02 MB** |
| Indexing duration | **43.164s** |
| Indexing throughput | **0.093 docs/s** |
| Retrieval total duration | **30.491s** |
| Retrieval avg latency | **6.098s/query** |
| Retrieval peak RSS | **1476.06 MB** |
| SQLite smoke query | **10.600s** |

## Raw Metric JSON

```json
{
  "metadata": {
    "timestamp": "2026-04-11T04:17:04.815034",
    "platform": "Linux",
    "node": "raspberrypi"
  },
  "metrics": {
    "startup": {
      "time_to_interactive": 0.014309406280517578,
      "time_to_ready": 11.167035579681396,
      "peak_ram_mb_interactive": 66.015625,
      "peak_ram_mb_ready": 1185.015625
    },
    "indexing": {
      "duration_seconds": 43.16426205635071,
      "peak_ram_mb": 1476.0625,
      "dataset": "scripts/data/eval_scenarios.json",
      "items_count": 4,
      "batch_size": 1,
      "items_per_second": 0.09266925482886797
    },
    "retrieval": {
      "total_duration_seconds": 30.49095606803894,
      "avg_latency_seconds": 6.098188686370849,
      "peak_ram_mb": 1476.0625,
      "queries_count": 5
    },
    "sqlite": {
      "duration_seconds": 10.600196599960327
    }
  }
}
```
