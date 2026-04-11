import argparse
import asyncio
import sys
import os
from .hippo_engine import HippoEngine
from .config import settings
from .model_setup import ensure_gliner_onnx_model

async def run_cmd_run(args):
    """Start the MCP server."""
    # We use fastmcp run src/server.py:mcp
    # For now, we can just print the instructions or try to spawn it.
    # Actually, the requirement says "alias for fastmcp run".
    print("Starting Seahorse RAG MCP Server...")
    import subprocess
    cmd = [sys.executable, "-m", "fastmcp", "run", "seahorse.server:mcp"]
    completed = subprocess.run(cmd)
    raise SystemExit(completed.returncode)

async def run_cmd_index(args):
    """Index content from text or file."""
    engine = HippoEngine()
    await engine.initialize()
    
    content = ""
    if args.text:
        content = args.text
    elif args.file:
        with open(args.file, "r") as f:
            content = f.read()
    
    if not content:
        print("Error: No content provided. Use --text or --file.")
        return

    print(f"Indexing content...")
    await engine.add_document(content)
    print("Indexing complete.")

async def run_cmd_search(args):
    """Search the Knowledge Graph."""
    engine = HippoEngine()
    await engine.initialize()
    print(f"Searching for: {args.query}")
    result = await engine.search(args.query)
    print("\nResults:")
    print(result)

async def run_cmd_stats(args):
    """Show graph statistics."""
    engine = HippoEngine()
    await engine.initialize()
    stats = await engine.storage.verify_integrity()
    print("Graph Statistics:")
    for key, val in stats.items():
        print(f"  {key}: {val}")

async def run_cmd_optimize(args):
    """Run offline synonym optimization."""
    engine = HippoEngine()
    await engine.initialize()
    print(f"Starting optimization with threshold {args.threshold}...")
    links = await engine.optimize_synonyms(threshold=args.threshold)
    print(f"Optimization complete. Added {links} edges.")


async def run_cmd_models(args):
    """Download/setup required local model artifacts."""
    target_dir = args.dir or settings.GLINER_ONNX_PATH or "models/gliner_onnx"
    repo_id = args.repo or settings.GLINER_ONNX_REPO_ID
    print(f"Preparing ONNX GLiNER model from {repo_id} -> {target_dir}")
    changed = ensure_gliner_onnx_model(target_dir, repo_id=repo_id, force=args.force)
    if changed:
        print("Model download/setup complete.")
    else:
        print("Model already present. Nothing to do.")

def main():
    prog_name = os.path.basename(sys.argv[0]) or "seahorse"
    parser = argparse.ArgumentParser(prog=prog_name, description="Seahorse RAG MCP CLI")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Run
    run_parser = subparsers.add_parser("run", help="Start the MCP server")

    # Index
    index_parser = subparsers.add_parser("index", help="Index content")
    index_parser.add_validator = lambda x: None # Placeholder
    group = index_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Direct text input")
    group.add_argument("--file", help="Path to text file")

    # Search
    search_parser = subparsers.add_parser("search", help="Search the graph")
    search_parser.add_argument("query", help="Search query")

    # Stats
    stats_parser = subparsers.add_parser("stats", help="Show graph stats")

    # Optimize
    opt_parser = subparsers.add_parser("optimize-graph", help="Run offline synonym linking")
    opt_parser.add_argument("--threshold", type=float, default=0.55, help="Vector distance threshold (default 0.55)")

    # Models
    models_parser = subparsers.add_parser("models", help="Setup/download local ONNX model artifacts")
    models_parser.add_argument("--repo", help="Override ONNX GLiNER repo id")
    models_parser.add_argument("--dir", help="Override local model directory")
    models_parser.add_argument("--force", action="store_true", help="Re-download model artifacts")

    args = parser.parse_args()

    if args.command == "run":
        asyncio.run(run_cmd_run(args))
    elif args.command == "index":
        asyncio.run(run_cmd_index(args))
    elif args.command == "search":
        asyncio.run(run_cmd_search(args))
    elif args.command == "stats":
        asyncio.run(run_cmd_stats(args))
    elif args.command == "optimize-graph":
        asyncio.run(run_cmd_optimize(args))
    elif args.command == "models":
        asyncio.run(run_cmd_models(args))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
