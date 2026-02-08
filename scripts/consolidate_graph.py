import asyncio
import logging
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from edge_hippo.hippo_engine import HippoEngine
from edge_hippo.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("consolidate")

async def main():
    parser = argparse.ArgumentParser(description="Offline Graph Consolidation")
    parser.add_argument("--threshold", type=float, default=0.55, help="Vector distance threshold (L2, lower is better. matching sim > 0.85)")
    args = parser.parse_args()
    
    engine = HippoEngine()
    
    logger.info("Initializing engine...")
    await engine.initialize()
    
    logger.info(f"Starting synonym optimization with threshold {args.threshold}...")
    try:
        links_added = await engine.optimize_synonyms(threshold=args.threshold)
        logger.info(f"Consolidation complete. Added {links_added} new edges.")
    except Exception as e:
        logger.error(f"Consolidation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
