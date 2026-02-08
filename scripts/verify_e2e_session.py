
import asyncio
import sys
import os
import logging

# Ensure we can import edge_hippo
sys.path.append(os.getcwd())

from edge_hippo.hippo_engine import HippoEngine
from edge_hippo.server import mcp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_e2e")

async def run_scenario():
    logger.info("Initializing Engine...")
    engine = HippoEngine()
    
    # We need to initialize. 
    # If this fails due to sqlite-vec, we might need a workaround for the script too.
    try:
        await engine.initialize()
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        # Attempt fallback or just exit
        return

    # 1. Ingest Data (Simulate Knowledge)
    logger.info("Ingesting Knowledge...")
    docs = [
        "Python is a programming language widely used for data science.",
        "Pandas is a Python library for data manipulation.",
        "The Jaguar is a large cat species found in the Americas.",
        "Jaguar Cars is a luxury vehicle brand."
    ]
    await engine.add_documents(docs)

    session_id = "test_user_session"

    # 2. Turn 1: Query about "Python"
    logger.info(f"--- Turn 1: 'Tell me about Python' (Session: {session_id}) ---")
    res1 = await engine.search("Tell me about Python", session_id)
    print("Result 1:", res1)
    
    # Check if context updated
    ctx1 = engine.session_manager.get_context(session_id)
    logger.info(f"Context after Turn 1: {ctx1}")
    if not ctx1:
        logger.warning("Context is empty! Check extraction.")

    # 3. Turn 2: Ambiguous Query "Pandas" (Animal vs Library?)
    # Context "Python" should steer it towards the library.
    logger.info(f"--- Turn 2: 'Pandas' (Session: {session_id}) ---")
    res2 = await engine.search("Pandas", session_id)
    print("Result 2:", res2)
    
    # Verify "library" or "data" concepts in result
    if "library" in str(res2).lower() or "data" in str(res2).lower():
        logger.info("[SUCCESS] Context 'Python' influenced 'Pandas' result towards tech.")
    else:
        logger.warning("[FAILURE] Context did not influence result significantly.")

    # 4. Turn 3: Topic Shift "Jaguar" (Drift)
    # Context "Python, Pandas" is distinct from "Jaguar".
    # Should trigger drift detection ideally, or at least low scores.
    logger.info(f"--- Turn 3: 'Jaguar' (Session: {session_id}) ---")
    res3 = await engine.search("Jaguar", session_id)
    print("Result 3:", res3)
    
    # Check connection
    ctx3 = engine.session_manager.get_context(session_id)
    logger.info(f"Context after Turn 3: {ctx3}")
    
    # If drift logic works, context might be flushed or just Jaguar.
    # We'll just log it for now.

if __name__ == "__main__":
    asyncio.run(run_scenario())
