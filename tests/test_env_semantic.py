import pytest
import sqlite3
import sqlite_vec
from sentence_transformers import SentenceTransformer

def test_sqlite_vec_extension_loads(vector_search_supported):
    """Verify that sqlite-vec extension can be loaded into sqlite3."""
    if not vector_search_supported:
        pytest.skip("Vector search not supported via built-in SQLite")
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    
    cursor = db.cursor()
    cursor.execute("SELECT vec_version()")
    version = cursor.fetchone()[0]
    assert version is not None
    print(f"sqlite-vec version: {version}")
    db.close()

def test_embedding_model_generates():
    """Verify that the E5 model generates embeddings of correct dimension."""
    model_name = "intfloat/multilingual-e5-small"
    model = SentenceTransformer(model_name)
    
    # E5 requires "query: " or "passage: " prefix
    text = "passage: This is a test sentence."
    embedding = model.encode(text)
    
    assert len(embedding) == 384, f"Expected 384 dimensions, got {len(embedding)}"
    print("Embedding generation successful.")
