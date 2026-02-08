# Technical Specification: Edge-Hippo Optimizations

## Overview
This document outlines the architecture and implementation details of the "Edge-Hippo" RAG engine, a variant of HippoRAG 2 optimized for resource-constrained edge devices (e.g., Raspberry Pi 5).

## Core Principles
1.  **Entity-Centric**: The knowledge graph is primarily composed of Entities (Phrases) and Passages (Chunks). Relationships are implicit (co-occurrence) or unlabeled to save space.
2.  **Edge-First**: Minimal RAM usage. No full-graph loading. Heavy processing (PPR) is done on on-demand subgraphs.
3.  **Lightweight Vectors**: Uses `sqlite-vec` for in-process vector storage and similarity search, avoiding heavy external vector databases.

## Architecture Components

### 1. Data Schema (`edge_hippo/storage.py`)
The data model is implemented in SQLite with the following tables:

*   **`nodes` Table**
    *   `id` (INTEGER PK): Unique ID.
    *   `type` (TEXT): 'passage' or 'phrase'.
    *   `name` (TEXT): Entity name (for phrases) or unique ID (for passages).
    *   `content` (TEXT): Text content (for passages).
    *   `metadata` (TEXT): JSON string.
    *   `is_hub` (INTEGER): Boolean flag (0/1). 1 indicates a high-degree 'Hub Node'.

*   **`vec_items` Table (Virtual)**
    *   Uses `sqlite-vec` (`vec0` module).
    *   `rowid` matches `nodes.id`.
    *   `embedding`: Float array (dim=384 for `all-MiniLM-L6-v2`).

*   **`edges` Table**
    *   `source` (INTEGER): FK to `nodes.id`.
    *   `target` (INTEGER): FK to `nodes.id`.
    *   `weight` (REAL): Edge weight (default 1.0).
    *   **Note**: No `relation` column. Edges represent undirected co-occurrence or containment.

### 2. HippoEngine (`edge_hippo/hippo_engine.py`)
The central coordinator for ingestion### 2.1 Core Components
1.  **Ingestion Engine:**
    *   **Chunking:** `RecursiveCharacterTextSplitter` ( LangChain).
    *   **Entity Extraction:** `GLiNER` (NuZero/Gliner-small).
        *   **Runtime:** **Hybrid** (Native `gliner` loader backed by ONNX `gliner_small-v2.1`).
        *   **Optimization:** FP32 ONNX Export (Correctness Priority).
        *   **Performance:** ~250ms latency, ~1.5GB RAM contribution.
2.  **Graph Construction:**
    *   **Storage:** `Graph Store` (SQLite - Node Table, Edge Table).
    *   **Vector Integration:** "Dense-Sparse" hybrid.
        *   **Model:** `intfloat/multilingual-e5-small` (Quantized INT8).
        *   **Runtime:** **Pure ONNX Runtime** (No `transformers` dependency for Embeddings).
        *   **Performance:** <50ms latency, ~50MB RAM contribution.
    *   **Synonym Linking:** `mass_similarity_check` during optimization.
3.  **Retrieval Engine:**
    *   **Algorithm:** HippoRAG (Modified PPR).
    *   **Runtime:** `python-igraph` (C-Core).
    *   **Logic:** Hub Trapping, Sink Nodes, Dynamic Damping.
*   **Hub Identification**: calculates Global Degree Centrality. Flags top 1% of nodes as `is_hub` to penalize generic terms during retrieval.
*   **Synonym Consolidation** (Lazy): Background process `optimize_synonyms()` identifies unconnected Phrase nodes with high vector similarity (> 0.85) and links them (Bidirectional Edge, Weight 1.0).

#### Retrieval Pipeline (Ego-Graph)
1.  **Seed Identification**:
    *   Extracts entities from the User Query.
    *   Performs **Vector Search** (`sqlite-vec`) to find semantic matches (Cue Entity Expansion).
2.  **Ego-Graph Extraction**:
    *   Executes a `WITH RECURSIVE` SQL query to fetch a k-hop subgraph (Depth 2).
    *   **Safety**: Stops recursion at Hub Nodes (Hubs can be leaves but not bridges).
    *   **Limit**: Caps subgraph size at 1500 nodes to prevent RAM explosion.
    *   Explicitly fetches connected **Passage Nodes**.
3.  **Dynamic Weighting**:
    *   Constructs a Directed `igraph` from the subgraph.
    *   **Hub Trapping Strategy**:
        *   Edges originating from Hubs are dampened (weight x 0.1).
        *   Residual probability (0.9) is trapped in the Hub's **Self-Loop**.
    *   **Edge Weights**:
        *   Phrase <-> Phrase: 1.0
        *   Phrase <-> Passage: **2.0** (Prioritize document context).
4.  **Personalized PageRank (PPR)**:
    *   **Boundary Bias Correction**:
        *   Damping factor reduced to **0.50** (Local Focus).
        *   **Sink Node**: Dangling nodes (out-degree 0) are connected to a virtual SINK node (with self-loop) to absorb leakage.
    *   Runs PPR on the temporary graph, resetting probability mass to Seed Nodes.
5.  **Ranking**: Sorts Passages by their Pagerank score.

## Dependencies
*   `aiosqlite`: Async SQLite interface.
*   `sqlite-vec`: In-process vector search extension.
*   `gliner`: Lightweight Named Entity Recognition.
*   `sentence-transformers`: Text embedding (Fallback / Core for some paths).
*   `onnxruntime`: Quantized Inference (Embeddings).
*   `python-igraph`: Fast C-core graph algorithms.
*   `numpy`: Numerical operations.

## Constraints
*   **RAM**: < 4GB (Peak ~2.5GB due to GLiNER Native Loader).
*   **Storage**: SQLite file.
*   **Concurrency**: AsyncIO based.
