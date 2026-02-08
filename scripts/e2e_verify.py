
import asyncio
import logging
import sys
from pathlib import Path
from src.hippo_engine import HippoEngine
from src.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("e2e")

async def main():
    logger.info("Starting E2E Verification...")
    
    # Use a temp db for verification? Or expected dev DB?
    # We'll use a specific test db file to avoid wiping user data if any.
    settings.DATA_DIR = Path("./data/e2e_test")
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize implementation
    engine = HippoEngine()
    await engine.initialize()
    
    # 1. Ingest Data
    docs = [
        "The Raspberry Pi is a small single-board computer developed in the UK.",
        "Edge computing optimizes processing by bringing computation closer to the source of data, like IoT devices.",
        "HippoRAG is a retrieval augmented generation system inspired by the hippocampus.",
        "Optimizing for edge devices requires managing RAM usage carefully."
    ]
    
    logger.info("Ingesting Documents...")
    for i, doc in enumerate(docs):
        await engine.add_document(doc, source=f"doc_{i}")
        
    # 2. Finalize (Hubs)
    # We expect "computer" or "system" might be hubs if frequent?
    # With 4 docs, tough. But let's run it.
    await engine.finalize_index()
    
    # 3. Retrieval Test
    queries = [
        ("What is the Raspberry Pi?", ["Raspberry Pi", "computer"]),
        ("How to optimize for edge?", ["Edge computing", "RAM", "optimize"]),
        ("What inspired HippoRAG?", ["hippocampus"])
    ]
    
    for q, keywords in queries:
        logger.info(f"\nQuerying: {q}")
        result = await engine.search(q, top_k=2)
        print(f"--- RESULT ---\n{result}\n--------------")
        
        # Verify keywords
        found = False
        for kw in keywords:
            if kw.lower() in result.lower():
                found = True
                break
        
        if found:
            logger.info("✅ Relevant content found.")
        else:
            logger.error("❌ Keywords not found!")

    logger.info("E2E Verification Complete.")

if __name__ == "__main__":
    asyncio.run(main())
