import argparse
import asyncio
import sys
import os
from .hippo_engine import HippoEngine

async def run_cmd_run(args):
    """Start the MCP server."""
    # We use fastmcp run src/server.py:mcp
    # For now, we can just print the instructions or try to spawn it.
    # Actually, the requirement says "alias for fastmcp run".
    print("Starting Edge-Hippo RAG MCP Server...")
    import subprocess
    cmd = [sys.executable, "-m", "fastmcp", "run", "edge_hippo.server:mcp"]
    subprocess.run(cmd)

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

def main():
    parser = argparse.ArgumentParser(prog="hippo", description="Edge-Hippo RAG CLI")
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
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
