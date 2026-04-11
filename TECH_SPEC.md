# Technical Specification: Seahorse RAG MCP

## Overview
Seahorse RAG MCP is an edge-optimized HippoRAG-style system for Raspberry Pi 5 and other resource-constrained devices. It focuses on low RAM usage and fast retrieval without needing a full-time graph in memory.

The repo, distribution, and MCP preset now prefer the Seahorse branding. Public imports and entrypoints now target the `seahorse` namespace, while the underlying implementation continues to mirror `edge_hippo` through compatibility wrappers.

## Core Principles
- **Entity-Centric**: The knowledge graph uses Phrases (Entities) and Passages (Chunks). Relationships are implicit to keep storage requirements low.
- **Edge-First**: Minimal RAM usage. No full-graph loading. Heavy processing is done on small, on-demand subgraphs.
- **Lightweight Vectors**: Uses `sqlite-vec` for local vector storage and similarity search.

## Architecture Components

### 1. Data Schema (`seahorse/storage.py`)
Stored in SQLite with three main tables:

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

### 2. HippoEngine (`seahorse/hippo_engine.py`)
1.  **Ingestion Engine:**
    *   **Chunking:** `RecursiveCharacterTextSplitter` ( LangChain).
    *   **Entity Extraction:** `GLiNER` (default: `fastino/gliner2-multi-v1`).
        *   **Runtime:** ONNX-first (`lmo3/gliner2-multi-v1-onnx`) with local auto-setup.
        *   **Setup Resilience:** Auto-setup repairs missing ONNX sidecar shards (`*.onnx.data`) before full snapshot sync.
        *   **Tokenizer Path:** `tokenizers` (`tokenizer.json`) is used first to avoid `transformers` init overhead in the hot path.
        *   **Fallback:** If ONNX setup/load fails, falls back to `GLINER_MODEL` via native `gliner`.
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
    *   **Limit**: Uses a dynamic node budget from `resource_manager.calculate_node_budget()`; `1500` is an example, not a fixed constant.
    *   Explicitly fetches connected **Passage Nodes**.
3.  **Dynamic Weighting**:
    *   Constructs a Directed `igraph` from the subgraph.
    *   **Hub Trapping + Residual Teleport**:
        *   Hub self-loop uses degree-based `p_self = 0.8 * degG / (degG + AVG_PHRASE_DEG)`.
        *   Structural propagation mass uses `alpha = clip(deg_sub/degG, 0, 1)`.
        *   Residual mass `(1-p_self)*(1-alpha)` teleports via seed distribution.
        *   For seed-hub nodes, residual teleport excludes self (`t_H` with hub removed).
    *   **Typed Candidate Scoring (for fanout/pruning)**:
        *   phrase→passage: degree-penalized score.
        *   passage→phrase: IDF score.
        *   phrase→phrase: edge weight.
4.  **Personalized PageRank (PPR)**:
    *   Uses `igraph.personalized_pagerank` with dynamic damping scheduling around configured base.
    *   **Sink Node**: Dangling nodes (out-degree 0) are connected to a virtual SINK node (with self-loop).
    *   Runs PPR on the temporary graph, resetting probability mass to Seed Nodes.
5.  **Ranking**: Sorts Passages by their Pagerank score.
6.  **Optional Local INT8 Cross-Encoder Re-ranking**:
    *   Scope: runs only on Top-N candidates from the PPR list.
    *   Composition rule:
        *   `final = fused(top_n) + ppr_only(rest)` then truncate to `top_k`.
    *   Runtime:
        *   ONNX Runtime (`CPUExecutionProvider`) + local `tokenizer.json`.
        *   ORT session is process-level singleton.

### 3.1 Retrieval Mode Policy (Current vs Future)
- **Current default (production):**
  - `python-igraph` + `personalized_pagerank`.
  - Hub trapping (`p_self`), residual teleport (`alpha`), seed-hub self-exclusion (`t_H`), fanout cap/pruning, and dynamic damping are applied before PPR.
- **Future work (optional advanced mode, not default):**
  - **Truncated PPR (K-step) mass policy** with explicit mass accounting (`sum(pi) ~= 1 - d^(K+1)` when unnormalized).
  - **COO one-shot sparse step kernel + fully vectorized 2-pass pruning** (`scipy.sparse`) for very large candidate edge sets.
- **Adoption criteria for future mode:**
  - enable only when profiling shows clear latency/throughput bottleneck under current igraph path
  - keep current igraph path as stable fallback for correctness and operational simplicity

### 3. Cross-Encoder Re-ranker Policy (Fixed by Code + Tests)

#### R1) Candidate Dedup (Before Top-N)
- Stable dedup by `passage_id` before top-n selection.
- First occurrence is preserved (deterministic order).
- Duplicates must not appear in final output.

#### R2) `top_n` Boundary Handling
- `top_n == 0`:
  - Re-ranking/fusion is skipped.
  - Immediate fallback to PPR-only with warning log.
- `top_n == 1`:
  - Weighted-sum PPR normalization uses fixed rule `s_ppr = 1.0` (no zero-division).
  - Re-ranker score may be computed, but ordering remains deterministic/stable.
- `top_n < top_k`:
  - Auto-adjust to `top_k` with warning log.

#### R3) Fusion Weight Normalization
- Inputs: `w_ppr`, `w_rerank`
- Rule:
  - `w_sum = w_ppr + w_rerank`
  - if `w_sum <= 0`: error log + PPR-only fallback
  - else: normalize to `w_ppr/w_sum`, `w_rerank/w_sum`
- If normalized values differ from input values, warning is emitted once.

#### R4) Manual Pair Packing / Truncation (No Auto-Truncation Reliance)
- Pair format:
  - `[CLS] query [SEP] passage [SEP]`
- Enforcement:
  1. tokenize query without special tokens, cut to `query_max_len`
  2. tokenize passage without special tokens, cut to `passage_max_len`
  3. enforce `3 + len(query) + len(passage) <= model_max_len`
  4. if overflow remains, trim passage again (only_second behavior)
  5. pad to `model_max_len`
- `token_type_ids` (BERT convention):
  - `[CLS] query [SEP]` and padding => `0`
  - `passage [SEP]` => `1`

#### R5) ORT DType / Output Shape Rules
- ORT inputs are always `np.int64`:
  - `input_ids`, `attention_mask`, `token_type_ids`
- Output handling:
  - cast to `np.float32` before post-processing
  - supported shapes:
    - `(B,1)` => squeeze
    - `(B,2)` => `out[:,1] - out[:,0]`
  - any other shape => error log + PPR-only fallback

#### R6) NaN/Inf and Deterministic Ordering
- NaN/Inf logits are marked invalid and demoted to bottom within Top-N.
- Weighted-sum and RRF both use deterministic tie-break keys
  (`ppr_rank`, `passage_id`, plus fixed secondary keys for RRF).

#### R7) Fusion Methods
- `weighted_sum`:
  - PPR rank-based score + normalized re-ranker score.
- `rrf`:
  - weighted reciprocal rank fusion with constant `k`.

## Constraints
- **RAM**: Peak usage around 2.5GB (mostly GLiNER).
- **Storage**: Single SQLite file.
- **Concurrency**: Fully asynchronous.

## Memory CRUD v4 Specification

### Scope and Modules
- `memory_crud/`:
  - `schema.py`: DDL/connection setup/transaction retry constants and helpers.
  - `normalize.py`: canonical normalization and pseudo-edge safety filters.
  - `judge.py`: Decision v3 validation (`create/update/delete/noop`) and known key payload generation.
  - `prompts.py`: Judge output contract guide.
  - `store.py`: `apply_decision()` write path with SSOT update.
  - `maintenance.py`: query-time decay, mark_retrieved/mark_used, purge.
- `seahorse/memory_adapter.py`:
  - Orchestration adapter for CRUD, retrieval, maintenance, and entity linking.

### Invariants (Fixed by Code + Tests)

#### A1) Initialization Priority (SSOT)
- On `create`, Judge must emit `init.{trust,strength}`.
- Runtime uses `init` as SSOT and only validates/clamps/applies lower bound:
  - `trust = max(clamp(init.trust, 0, 1), TRUST_EPS)` with NaN/Inf rejection.
  - `strength = max(clamp(init.strength, 0, 1), STRENGTH_EPS)` with NaN/Inf rejection.
- `provenance`/`kind` defaults are prompt guidance only. Runtime must not recompute defaults.

#### A2) No Write-only Memories
- `DEFAULT_MIN_TRUST <= TRUST_EPS`
- `DEFAULT_MIN_STRENGTH <= STRENGTH_EPS`
- Fixed constants:
  - `TRUST_EPS=0.05`
  - `STRENGTH_EPS=0.01`
  - `DEFAULT_MIN_TRUST=0.05`
  - `DEFAULT_MIN_STRENGTH=0.01`

#### A3) Single Decay Model (Renorm OFF)
- DB `strength` is always `strength_base`.
- Effective retrieval-time value only:
  - `strength_eff(now)=strength_base*exp(-λ_kind*(now-last_retrieved_effective))`
- Maintenance never renormalizes or overwrites base from decayed value.
- Numerical floor only at query-time: `max(strength_eff, STRENGTH_EPS)`.

#### A4) Retrieved vs Used
- `retrieved`: update `last_retrieved=now`; `strength_base` unchanged.
- `used`: update
  - `strength_base=min(1.0, max(strength_eff(now), STRENGTH_EPS)+0.1)`
  - `last_seen=now`

#### A5) SSOT Field Separation
- SSOT fields are `mem_kv_index` columns only:
  - `mem_key, scope, kind, status, updated_at, last_seen, last_retrieved, trust, strength`
- `nodes.metadata` is display snapshot.
- On mismatch, SSOT fields are reconstructed from `mem_kv_index`; extra derived metadata may be kept when possible.
- v4 fixed behavior: `apply_decision()` never overwrites metadata `mem.last_retrieved` (rollback prevention).

#### A6) `last_retrieved` NULL Fallback
- If `last_retrieved` is `NULL`:
  - `last_retrieved_effective = created_at`
- If `created_at` missing:
  - `last_retrieved_effective = updated_at`

#### A7) Update/Delete Validation
- All candidate memories (`CANDIDATE_MEMORIES`) must include `node_id`.
- For `update/delete`:
  - `target_node_id` must exist in candidates; else invalid -> noop.
  - `affected_keys` must be non-empty; else invalid -> noop.
  - For `update`, key immutability is enforced:
    - canonical incoming key must match existing canonical key; else noop.

#### A8) Scope + Unique Key Convention
- `mem_key` remains unique-active constrained.
- Scope is encoded in key:
  - `mem_key = f"{scope}:{canonical_key}"`
- `mem_kv_index.scope` is filter column and must stay consistent with encoded scope in `mem_key`.

### DDL (Must Stay Exact)
```sql
CREATE TABLE IF NOT EXISTS mem_kv_index (
    node_id INTEGER PRIMARY KEY,
    mem_key TEXT NOT NULL,
    status TEXT NOT NULL, -- 'active', 'superseded', 'deleted'
    kind TEXT,
    scope TEXT,
    updated_at INTEGER,
    last_seen INTEGER,
    last_retrieved INTEGER,
    trust REAL,
    strength REAL,
    FOREIGN KEY(node_id) REFERENCES nodes(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_mem_key_active
ON mem_kv_index(mem_key)
WHERE status = 'active';

CREATE INDEX IF NOT EXISTS ix_mem_updated_at ON mem_kv_index(updated_at);
CREATE INDEX IF NOT EXISTS ix_mem_last_seen ON mem_kv_index(last_seen);
CREATE INDEX IF NOT EXISTS ix_mem_last_retrieved ON mem_kv_index(last_retrieved);
```

### Transaction, Locking, and Conflict Convergence
- Writer transaction mode: `BEGIN IMMEDIATE`.
- Busy timeout enabled (`PRAGMA busy_timeout`).
- Locked/busy errors retry with backoff (3-5 attempts).
- `ux_mem_key_active` conflict on create converges to update inside the same transaction.

### Query/Scoring Rule
- Assume SQLite does not provide `exp()`.
- `query_memory` pipeline:
  1. Load broad candidates (SSOT + minimal metadata).
  2. Compute `last_retrieved_effective` and `strength_eff` in Python.
  3. Apply `min_trust`/`min_strength` filters and `top_k` sort.

### Pseudo-Edge Linking Rule
- Normalize: NFKC + control char removal + whitespace collapse + strip + casefold.
- Keep only tokens with length 2..64 and with at least one alphabetic character.
- `LIKE` substring matching allowed only for term length >= 4.
- Resolution order:
  1. exact phrase match
  2. hub substring match
  3. create phrase (`is_hub=0`)
- Edge weight is `2.0`; duplicate links are prevented by `edges` uniqueness/upsert.

### Purge Rule
- If `last_seen` is `NULL`, treat as unused.
- Reference timestamp fallback:
  - `created_at`, otherwise `updated_at`.
- Apply kind-based grace periods (for example 7-30 days) before soft delete.

### Test Lock Points
- DDL snapshot exact-match.
- `apply_decision` does not overwrite metadata `last_retrieved`.
- `last_retrieved=NULL` fallback to `created_at`.
- immediate retrieval after creation via EPS floor.
- update key immutability violation -> noop.
- `ux_mem_key_active` conflict -> update convergence (with retry path).
- substring `LIKE` minimum length guard.

## Re-ranker Test Lock Points
- `top_n=0` => immediate PPR fallback.
- `top_n=1` => no zero-division; deterministic output.
- weight normalization:
  - `(7,3)` normalized to `(0.7,0.3)`
  - `(0,0)` => fallback
- manual truncation:
  - query growth trims only passage side
  - model length invariant is always maintained
- dtype:
  - ORT input tensors are `int64`
  - mixed `float16/float32` outputs are normalized to `float32` flow
- dedup:
  - duplicate `passage_id` candidates are removed before top-n/fusion
  - final output has no duplicate passage
