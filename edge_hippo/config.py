
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Paths
    DATA_DIR: Path = Field(default=Path("./data"), description="Directory to store SQLite DB and other data")
    
    # Models
    GLINER_MODEL: str = Field(default="fastino/gliner2-multi-v1", description="GLiNER model for NER")
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

    # ONNX GLiNER
    GLINER_ONNX_PATH: Optional[str] = Field(
        default="models/gliner_onnx",
        description="Path to ONNX GLiNER model directory",
    )
    GLINER_ONNX_REPO_ID: str = Field(
        default="lmo3/gliner2-multi-v1-onnx",
        description="Hugging Face repo id for ONNX GLiNER bundle",
    )
    GLINER_ONNX_AUTO_SETUP: bool = Field(
        default=True,
        description="Auto-download ONNX GLiNER bundle when missing",
    )

    # Cross-Encoder Reranker (Edge INT8 ONNX)
    RERANK_ENABLED: bool = Field(
        default=False,
        description="Enable local INT8 cross-encoder reranker for top-N fusion",
    )
    RERANK_MODEL_PATH: Optional[str] = Field(
        default=None,
        description="Path to INT8 cross-encoder ONNX model",
    )
    RERANK_TOKENIZER_PATH: Optional[str] = Field(
        default=None,
        description="Path to tokenizer.json for cross-encoder reranker",
    )
    RERANK_TOP_N: int = Field(default=20, description="Top-N passages for rerank/fusion")
    RERANK_FUSION_METHOD: str = Field(
        default="weighted_sum",
        description="Fusion method: weighted_sum or rrf",
    )
    RERANK_W_PPR: float = Field(default=0.7, description="PPR weight for fusion")
    RERANK_W_RERANK: float = Field(default=0.3, description="Reranker weight for fusion")
    RERANK_RRF_K: int = Field(default=60, description="RRF constant k")
    RERANK_MODEL_MAX_LEN: int = Field(default=512, description="Cross-encoder max input length")
    RERANK_QUERY_MAX_LEN: int = Field(default=64, description="Max query tokens before packing")
    RERANK_PASSAGE_MAX_LEN: int = Field(default=448, description="Max passage tokens before packing")
    RERANK_LOGIT_ZSCORE: bool = Field(default=False, description="Apply z-score to reranker logits")
    RERANK_LOGIT_CLIP: Optional[float] = Field(
        default=None,
        description="Optional absolute clip value for reranker logits",
    )
    RERANK_LOGIT_SIGMOID: bool = Field(
        default=False,
        description="Apply sigmoid to reranker logits",
    )


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> Path:
        return self.DATA_DIR / "knowledge_graph.db"

settings = Settings()
