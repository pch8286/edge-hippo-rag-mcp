import asyncio
import aiosqlite
import sqlite_vec
import logging
from edge_hippo.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("migration")

async def migrate():
    db_path = settings.db_path
    logger.info(f"Migrating database at {db_path}...")
    
    # Load extension
    ext_path = sqlite_vec.loadable_path()
    
    async with aiosqlite.connect(db_path) as db:
        await db.enable_load_extension(True)
        await db.load_extension(ext_path)
        
        # Check if vec_items exists
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vec_items'")
        if await cursor.fetchone():
            logger.info("Found legacy table 'vec_items'. migrating to 'vec_nodes'...")
            
            # Since sqlite-vec virtual tables might not support standard RENAME easily or efficiently,
            # and they are just an index on rowid, valid check.
            # vec0 tables are somewhat special.
            # Easiest way: Create new table, copy data (if possible), drop old.
            # Or just Drop and Re-create if we assume re-indexing is fine or data is small.
            # Given this is "Edge-Hippo", let's try to preserve data.
            
            # Create vec_nodes
            await db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vec_nodes USING vec0(embedding float[384])")
            
            # Copy data
            # "INSERT INTO vec_nodes(rowid, embedding) SELECT rowid, embedding FROM vec_items"
            logger.info("Copying vectors...")
            try:
                await db.execute("INSERT INTO vec_nodes(rowid, embedding) SELECT rowid, embedding FROM vec_items")
                await db.execute("DROP TABLE vec_items")
                logger.info("Migration complete: vec_items -> vec_nodes")
            except Exception as e:
                logger.error(f"Migration failed: {e}")
                raise e
        else:
            # Check if vec_nodes exists
            cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vec_nodes'")
            if not await cursor.fetchone():
                logger.info("Creating 'vec_nodes' table...")
                await db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vec_nodes USING vec0(embedding float[384])")
                logger.info("Created 'vec_nodes'.")
            else:
                logger.info("'vec_nodes' already exists. No action needed.")

        await db.commit()

if __name__ == "__main__":
    asyncio.run(migrate())
