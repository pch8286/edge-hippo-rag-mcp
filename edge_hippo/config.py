import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Paths
    DATA_DIR: Path = Field(default=Path("./data"), description="Directory to store SQLite DB and other data")
    
    # Models
    GLINER_MODEL: str = Field(default="urchade/gliner_small-v2.1", description="GLiNER model for NER")
    EMBEDDING_MODEL: str = Field(default="intfloat/multilingual-e5-small", description="Sentence Transformer for synonyms")
    QUANTIZED_MODEL_DIR: Path = Field(default=Path("models_quantized"), description="Directory containing quantized models")
    
    # Engine Settings
    CHUNK_SIZE: int = Field(default=512, description="Text chunk size for passages")
    MAX_NEIGHBORS: int = Field(default=5, description="Max neighbors for expansion")
    PPR_DAMPING: float = Field(default=0.85, description="Damping factor for PageRank")
    
    # RPi Optimizations
    USE_QUANTIZATION: bool = Field(default=False, description="Prefer quantized models if available")
    THREAD_POOL_SIZE: int = Field(default=2, description="Threads for CPU bound tasks")
    
    # Performance & Budgeting
    HIPPO_PERFORMANCE_PROFILE: str = Field(default="auto", description="Performance profile: auto, low, mid, high")
    HIPPO_NODE_MAX: Optional[int] = Field(default=None, description="Override max nodes")
    HIPPO_MEMORY_ALPHA: Optional[float] = Field(default=None, description="Override memory alpha")


    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> Path:
        return self.DATA_DIR / "knowledge_graph.db"

settings = Settings()
