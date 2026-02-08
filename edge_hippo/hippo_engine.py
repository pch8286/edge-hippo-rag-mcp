import logging
import asyncio
from typing import List, Dict, Any, Tuple, Optional, TypedDict
import igraph
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
import numpy as np

from .config import settings
from .extraction import EntityExtractor
from .storage import GraphStorage
from .retrieval import PPRRetriever
from .session import session_manager

logger = logging.getLogger(__name__)

class SearchResult(TypedDict):
    """Structured search result."""
    seed_entities: List[str]
    passages: List[Dict[str, Any]]


class PureOnnxEmbeddingWrapper:
    def __init__(self, session, tokenizer):
        self.session = session
        self.tokenizer = tokenizer
        self.input_names = [node.name for node in session.get_inputs()]
        self.use_token_type_ids = "token_type_ids" in self.input_names
    
    def encode(self, sentences, batch_size=32, show_progress_bar=False, **kwargs):
        is_single = False
        if isinstance(sentences, str):
            sentences = [sentences]
            is_single = True
        
        encodings = self.tokenizer.encode_batch(sentences)
        
        batch_ids = [e.ids for e in encodings]
        batch_mask = [e.attention_mask for e in encodings]
        
        input_ids = np.array(batch_ids, dtype=np.int64)
        attention_mask = np.array(batch_mask, dtype=np.int64)
        
        ort_inputs = {
            "input_ids": input_ids, 
            "attention_mask": attention_mask
        }
        
        if self.use_token_type_ids:
            ort_inputs["token_type_ids"] = np.zeros_like(input_ids)
        
        outputs = self.session.run(None, ort_inputs)
        last_hidden_state = outputs[0]
        
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        input_mask_expanded = np.broadcast_to(input_mask_expanded, last_hidden_state.shape).astype(np.float32)
        
        sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
        
        sum_mask = np.sum(input_mask_expanded, axis=1)
        sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)
        
        embeddings = sum_embeddings / sum_mask
        
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norm
        
        if is_single:
            return embeddings[0]
        
        return embeddings

class HippoEngine:
    """
    Core engine for HippoRAG 2 (Edge-Optimized).
    
    Handles indexing of documents into a SQLite-backed graph (Passage -> Phrase)
    and retrieval using Personalized PageRank (PPR) on the topology.
    """

    def __init__(self) -> None:
        """Initialize the engine with extraction, storage, and models."""
        self.extractor: EntityExtractor = EntityExtractor()
        self.storage: GraphStorage = GraphStorage()
        self.encoder: Optional[SentenceTransformer] = None
        self.graph_cache: Optional[igraph.Graph] = None
        self._graph_dirty: bool = True
        self._model_loading_task: Optional[asyncio.Task] = None
        self.retriever: PPRRetriever = PPRRetriever(self.extractor, self.storage, self.encoder)

    async def initialize(self) -> None:
        """
        Async initialization of storage and models.
        Must be called before usage.
        """
        await self.storage.initialize()
        logger.info("Starting background model loading...")
        self._model_loading_task = asyncio.create_task(self._warmup_models())

    async def _warmup_models(self) -> None:
        """Background task to load heavy ML models."""
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self.extractor.load_model)
            
            if settings.EMBEDDING_MODEL:
                logger.info(f"Loading Embedding model: {settings.EMBEDDING_MODEL}")
                
                def _load_model_safe(model_name: str):
                    try:
                        from pathlib import Path
                        model_safe_name = model_name.split('/')[-1]
                        quantized_path = settings.QUANTIZED_MODEL_DIR / model_safe_name / "quantized"
                        
                        logger.info(f"Checking path: {quantized_path} (Absolute: {quantized_path.absolute()})")
                        
                        if quantized_path.exists():
                            logger.info(f"⚡ Loading Quantized INT8 Model from {quantized_path}...")
                            import onnxruntime as ort
                            from tokenizers import Tokenizer
                            import numpy as np
                            
                            tokenizer_path = quantized_path / "tokenizer.json"
                            tokenizer = Tokenizer.from_file(str(tokenizer_path))
                            
                            tokenizer.enable_padding(pad_id=1, pad_token="<pad>", length=512)
                            tokenizer.enable_truncation(max_length=512)
                            
                            model_path = quantized_path / "model_quantized.onnx"
                            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                            
                            print("DEBUG: Pure ONNX loader success!")
                            return PureOnnxEmbeddingWrapper(session, tokenizer)

                        logger.info("Quantized model not found. Using Standard SentenceTransformer.")
                        from sentence_transformers import SentenceTransformer
                        return SentenceTransformer(model_name)
                        
                    except Exception as e:
                        logger.error(f"Failed to load model {model_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        print(f"DEBUG: Fallback initiated due to error: {e}")
                        # Fallback
                        from sentence_transformers import SentenceTransformer
                        return SentenceTransformer(model_name)

                self.encoder = await loop.run_in_executor(
                    None, 
                    _load_model_safe, 
                    settings.EMBEDDING_MODEL
                )

            # Initialize Retriever
            self.retriever = PPRRetriever(self.extractor, self.storage, self.encoder)
                
            logger.info("Background model loading complete.")
        except Exception as e:
            logger.error(f"Failed to load models in background: {e}")
            raise e

    async def _ensure_models(self) -> None:
        """Wait for models to be ready. Lazily triggers initialization if needed."""
        if not self.retriever and not self._model_loading_task:
            await self.initialize()
            
        if self._model_loading_task:
            if not self._model_loading_task.done():
                logger.info("Waiting for models to load...")
            await self._model_loading_task

    def _chunk_text(self, text: str, chunk_size: int = 512) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Input text.
            chunk_size: Target size in characters (approx).
            
        Returns:
            List of text chunks.
        """
        words = text.split()
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len: int = 0
        
        for word in words:
            current_chunk.append(word)
            current_len += len(word) + 1
            if current_len >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    async def add_document(self, text: str, source: str = "user") -> None:
        """
        Ingest a document into the graph.
        
        Pipeline:
        1. Chunk Text
        2. Create Passage Nodes (SQLite)
        3. Extract Entities/Phrases (GLiNER)
        4. Create Phrase Nodes & Edges (Passage <-> Phrase)
        
        Args:
            text: The full text content.
            source: Identifier for the source.
        """
        await self._ensure_models()
        chunks = self._chunk_text(text, settings.CHUNK_SIZE)
        
        for i, chunk in enumerate(chunks):
            entities = await self.extractor.extract_entities(chunk)
            
            unique_phrases = list(set(ent['text'] for ent in entities))
            embeddings_map = {}
            
            if self.encoder and unique_phrases:
                loop = asyncio.get_running_loop()
                prefixed_phrases = ["passage: " + p for p in unique_phrases]
                embeddings_np = await loop.run_in_executor(
                    None, 
                    self.encoder.encode, 
                    prefixed_phrases
                )
                for phrase, emb in zip(unique_phrases, embeddings_np):
                    embeddings_map[phrase] = emb.tolist()

            passage_embedding = None
            if self.encoder:
                loop = asyncio.get_running_loop()
                p_emb_np = await loop.run_in_executor(
                    None,
                    self.encoder.encode,
                    "passage: " + chunk
                )
                passage_embedding = p_emb_np.tolist()

            passage_id = await self.storage.add_node(
                node_type="passage", 
                name=f"p_{source}_{i}",
                content=chunk,
                metadata={"source": source, "index": i},
                embedding=passage_embedding
            )

            for ent in entities:
                phrase_text = ent['text']
                embedding = embeddings_map.get(phrase_text)

                phrase_id = await self.storage.add_node(
                    node_type="phrase",
                    name=phrase_text,
                    content=phrase_text,
                    metadata={"label": ent['label']},
                    embedding=embedding
                )
                
                await self.storage.add_edge(passage_id, phrase_id, weight=1.0)
                await self.storage.add_edge(phrase_id, passage_id, weight=1.0)


    async def add_documents(self, texts: List[str], source: str = "batch") -> None:
        """
        Batch ingest documents.
        Optimized for throughput.
        """
        await self._ensure_models()
        
        all_chunks_map = []
        
        for doc_idx, text in enumerate(texts):
            chunks = self._chunk_text(text, settings.CHUNK_SIZE)
            for chunk_i, chunk in enumerate(chunks):
                all_chunks_map.append((chunk, doc_idx, chunk_i))
        
        if not all_chunks_map:
            return

        all_chunk_texts = [c[0] for c in all_chunks_map]
        
        logger.info(f"Extracting entities from {len(all_chunk_texts)} chunks...")
        entities_batch = await self.extractor.extract_entities_batch(all_chunk_texts)
        
        unique_phrases = set()
        for batch_ents in entities_batch:
            for ent in batch_ents:
                unique_phrases.add(ent['text'])
        
        unique_phrases_list = list(unique_phrases)
        embeddings_map = {}
        
        if self.encoder and unique_phrases_list:
            logger.info(f"Encoding {len(unique_phrases_list)} unique phrases...")
            loop = asyncio.get_running_loop()
            
            def _encode_sbert(phrases):
                 # Add prefix
                 prefixed = ["passage: " + p for p in phrases]
                 return self.encoder.encode(prefixed)

            embeddings_np = await loop.run_in_executor(
                    None, 
                    _encode_sbert, 
                    unique_phrases_list
                )
             
            for phrase, emb in zip(unique_phrases_list, embeddings_np):
                embeddings_map[phrase] = emb.tolist()

        # Encode Passages Batch
        passage_embeddings_map = {} # (doc_idx, chunk_idx) -> embedding
        if self.encoder:
            logger.info(f"Encoding {len(all_chunk_texts)} passages...")
            loop = asyncio.get_running_loop()
            
            def _encode_passages_sbert(texts):
                 prefixed = ["passage: " + t for t in texts]
                 return self.encoder.encode(prefixed)
            
            p_embs_np = await loop.run_in_executor(None, _encode_passages_sbert, all_chunk_texts)
            
            for i, emb in enumerate(p_embs_np):
                 # all_chunk_texts corresponds to all_chunks_map order
                 # all_chunks_map[i] = (text, doc_idx, chunk_idx)
                 _, d_idx, c_idx = all_chunks_map[i]
                 passage_embeddings_map[(d_idx, c_idx)] = emb.tolist()

        # 4. Storage Operations
        # Sequential insert for now due to SQLite lock contention on write
        
        async with self.storage._get_conn() as db:
             pass

        for i, (chunk_text, doc_idx, chunk_idx) in enumerate(all_chunks_map):
            ents = entities_batch[i]
            
            # Add Passage
            p_emb = passage_embeddings_map.get((doc_idx, chunk_idx))
            passage_id = await self.storage.add_node(
                node_type="passage", 
                name=f"p_{source}_{doc_idx}_{chunk_idx}",
                content=chunk_text,
                metadata={"source": source, "doc_index": doc_idx, "chunk_index": chunk_idx},
                embedding=p_emb
            )
            
            for ent in ents:
                phrase_text = ent['text']
                embedding = embeddings_map.get(phrase_text)
                
                phrase_id = await self.storage.add_node(
                    node_type="phrase",
                    name=phrase_text,
                    content=phrase_text,
                    metadata={"label": ent['label']},
                    embedding=embedding
                )
                
                await self.storage.add_edge(passage_id, phrase_id, weight=1.0)
                await self.storage.add_edge(phrase_id, passage_id, weight=1.0)

        self._graph_dirty = True
        logger.info(f"Batch ingestion complete. {len(texts)} docs, {len(all_chunk_texts)} chunks.")

    async def finalize_index(self) -> None:
        """
        Run post-indexing optimizations.
        1. Identify Global Hub Nodes (Top 1% degree).
        2. Optimize Synonyms (Optional Lazy linking).
        """
        logger.info("Finalizing index (identifying Hub Nodes)...")
        await self.storage.flag_hub_nodes(percentile=0.99)
        logger.info("Index optimization complete.")
        
    async def optimize_synonyms(self, threshold: float = 0.85) -> int:
        """
        Background optimization: Link similar Phrases (Synonyms).
        1. Fetch all Phrase nodes (ids, embeddings).
        2. Use vector search or pairwise comparison to find matches > threshold.
        3. Add bidirectional edges: PHRASE <-> PHRASE (weight 1.0).
        """
        logger.info("Starting Synonym Optimization...")
        links_added = 0
        
        # Strategy:
        # We can iterate all Phrases and search.
        # But this is O(N) searches.
        # For 'background' job, it's acceptable.
        # We should only link UNLINKED phrases?
        # Specification says: "Iterate unconnected Phrase nodes" (Problem 4).
        # But even connected ones might need synonyms.
        # Let's iterate all phrases.
        
        async with self.storage._get_conn() as db:
            # Fetch all phrases with embeddings
            async with db.execute("""
                SELECT n.id, n.name, v.embedding
                FROM nodes n
                JOIN vec_nodes v ON n.id = v.rowid
                WHERE n.type = 'phrase'
            """) as cursor:
                phrases = await cursor.fetchall()

        if not phrases:
             return 0

        import struct
        
        for pid, name, emb_blob in phrases:
            # Deserialize embedding
            try:
                count = len(emb_blob) // 4
                emb_list = list(struct.unpack(f'{count}f', emb_blob))
            except Exception as e:
                logger.warning(f"Failed to unpack embedding for {name}: {e}")
                continue
                
            # Search for similar
            # Use storage.search_vectors but we specifically want PHRASE nodes
            # search_vectors returns ANY rowid match. Since `vec_items` has rowid=node.id, 
            # and nodes table has types.
            # We can filter results.
            
            results = await self.storage.search_vectors(emb_list, top_k=5)
            
            for other_id, dist in results:
                if other_id == pid:
                    continue
                
                # sqlite-vec MATCH uses L2 distance by default.
                # Threshold < 0.55 corresponds roughly to Cosine Similarity > 0.85
                # (assuming normalized embeddings where L2 = sqrt(2*(1-cos)))
                
                if dist < 0.55: # Corresponds roughly to Cosine Sim > 0.85
                    await self.storage.add_edge(pid, other_id, weight=1.0)
                    await self.storage.add_edge(other_id, pid, weight=1.0)
                    links_added += 1
        
        logger.info(f"Synonym Optimization: Added {links_added} links.")
        return links_added


    async def search(self, query: str, session_id: str = "default", top_k: int = 5) -> str:
        """
        Search using Ego-Graph PPR with retrieval-time expansion.
        Uses PPRRetriever with Persistent Context.
        """
        await self._ensure_models()
        self.retriever.encoder = self.encoder # Ensure encoder is updated if late-loaded
        
        # 1. Get History Context
        history_entities = session_manager.get_context(session_id)
        
        # 2. Perform Search
        # Retriever handles drift check and decay logic
        result_text, current_entities = await self.retriever.search(
            query=query,
            top_k=top_k,
            history_entities=history_entities,
            decay_factor=0.2 # Default decay if no drift
        )
        
        # 3. Update History Context
        # Only update if we found new entities?
        # If current_entities is empty (no result), maybe keep old context?
        # But if the query was valid but yielded no results, maybe context should be kept or cleared?
        # If retriever found seeds but no path -> current_entities has seeds.
        # If retriever found NO seeds -> current_entities empty.
        # We rely on retriever output.
        
        if current_entities:
            session_manager.update_context(session_id, current_entities)
            
        return result_text
