# Edge Hippo RAG MCP

A lightweight, edge-optimized GraphRAG MCP server implementing the **HippoRAG 2** architecture. Initially designed for Raspberry Pi 5 (<8GB RAM).

Reference for implementation: 
1. https://arxiv.org/pdf/2502.14802
2. https://arxiv.org/pdf/2602.01965
3. https://arxiv.org/pdf/2510.08958

## Features
- **HippoRAG 2 Architecture:** Uses "Passage" and "Phrase" nodes with "Dense-Sparse Integration".
- **Logic Hardening & Reliability:**
  - **Hub Trapping:** Penalizes high-degree noise nodes by redirecting flow to self-loops.
  - **Boundary Bias Correction:** Uses Sink Nodes and tuned damping (0.50) to prevent edge reflection.
  - **Recursion Safety:** Blocks explosion through Hub nodes and enforces hard subgraph limits.
- **Edge Optimized:** 
  - **Storage:** SQLite (on SSD) for Graph & Vectors. No heavy in-memory graph loading.
  - **Extraction:** GLiNER (CPU optimized) for entity/triple extraction.
  - **Retrieval:** Personalized PageRank (PPR) via `python-igraph` (C-optimized).
- **Adaptive Budgeting:** Automatically detects system RAM (and zram) to scale retrieval depth. Formulas:
  - $N_{limit} = \min(N_{max}, \frac{M_{effective} \times \alpha}{C_{node}})$
  - $M_{effective} = M_{available} + (M_{zram\_free} \times 0.5)$
- **Contextual Re-ranking:** Maintains narrative context across query turns using "Topological Persistent PPR" with drift control to prevent "Echo Chambers".
- **MCP Interface:** Fully compatible with Model Context Protocol. OpenClaw Skill ready.

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/your-repo/edge-hippo-rag-mcp.git
cd edge-hippo-rag-mcp
pip install -e .
```

### 2. Install as OpenClaw Skill (One-Line)
```bash
uvx --from https://github.com/your-repo/edge-hippo-rag-mcp edge-hippo-server
```
*Note: Requires `uv` installed.*

## ⚠️ Prerequisites for Vector Search
**Vector Search requires SQLite Extension Support.**
Standard Python builds on some platforms (e.g., Ubuntu/Debian system Python) may be compiled without `--enable-loadable-sqlite-extensions`.
- If extensions are **unavailable**, Edge Hippo will safely disable vector search and fall back to pure graph traversal (Graceful Degradation).
- To enable vector search, ensure your Python environment supports `sqlite3.enable_load_extension`.

### 3. Run MCP Server
Start the server directly:
```bash
hippo run
```

### 4. CLI Usage (Optional)
Manage the knowledge graph from your terminal:
```bash
# Show stats
hippo stats

# Index text
hippo index --text "Raspberry Pi 5 is the latest model."

# Link synonyms (Critical for hybrid search)
hippo optimize-graph --threshold 0.6

# Search
hippo search "RPi 5"
```

## ⚙️ Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `HIPPO_PERFORMANCE_PROFILE` | `auto` | `auto`, `low`, `mid`, or `high`. |
| `HIPPO_NODE_MAX` | `None` | Override profile's max nodes limit. |
| `HIPPO_MEMORY_ALPHA` | `None` | Override memory allocation fraction (e.g. 0.1). |

## 🧠 Contextual Re-ranking
Weighted context from previous turns is automatically applied to `search` results if `session_id` is reused.
- **Drift Control:** Automatically detects topic shifts and flushes context if the new query is topologically disconnected from history.
- **Preserves Narrative:** Keeps relevant entities in the "Reset Vector" for PPR to maintain focus.

## 🔌 Integrate with Claude Desktop
Add this to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "edge-hippo": {
      "command": "hippo",
      "args": ["run"]
    }
  }
}
```

## Architecture Details
1.  **Ingestion:** Text is chunked -> GLiNER extracts entities -> Stored in SQLite.
2.  **Graph Construction:** Passages are linked to contained Phrases. Phrases are linked by synonymy (via `SentenceTransformers`) using vector similarity.
3.  **Retrieval:** Use `python-igraph` to run PPR starting from query entities. Implements **Hub Trapping** and **Sink Node** logic for robust local retrieval.

## Raspberry Pi 5 Optimizations
- **RAM:** Graph topology is loaded into `igraph` (C struct). Content stays on disk.
- **Adaptive Node Budgeting:** Dynamically caps subgraph size based on available memory to prevent OOM.
- **CPU:** GLiNER and SentenceTransformers models are small.
- **Disk:** Uses `aiosqlite` for non-blocking DB access.
