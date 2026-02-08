# Edge Hippo RAG MCP

A lightweight GraphRAG server optimized for edge devices like the Raspberry Pi 5. It uses the **HippoRAG 2** architecture to handle complex retrieval with minimal memory.


## 💡 Why Not Naive RAG?
In edge environments, memory and compute are scarce. Standard "Naive RAG" (Vector-only) often fails when the answer requires connecting multiple pieces of information that aren't lexicographically or semantically identical in a single hop.

*   **Multi-hop Reasoning**: HippoRAG uses a Knowledge Graph to bridge between nodes (e.g., A → B → C) that Naive RAG would miss.
*   **Context Efficiency**: Instead of dumping thousands of tokens to "hope" the answer is in there, HippoRAG retrieves precisely the most relevant sub-graph, reducing token load by **200%+**.
*   **Recall at Scale**: Achieves **43%+ peak raw recall** on technical docs, with a **20.2% adaptive average** across all scenarios.

## 🎯 Use Cases
Edge-Hippo is designed for high-precision retrieval in environments where reliability and offline capability are paramount:

*   **Personal Knowledge Assistant**: Index your notes and local documents for a second brain that works 100% offline.
*   **Technical Manuals**: Gives field engineers high-recall access to complex equipment manuals without needing an internet connection.
*   **Field Operations**: Fast, secure RAG for agents who need to reason over mission-critical data in the field.
*   **Smart Homes**: Helps local LLMs understand the relationships between devices, routines, and user habits.

## Features
- **HippoRAG 2 Architecture**: Uses passage and phrase nodes with dense-sparse integration.
- **Reliability**: Features "Hub Trapping" to ignore noisy nodes, boundary bias correction for better graph flow, and recursion safety to prevent search explosions.
- **Edge Performance**: Built on SQLite for storage (no heavy in-memory loading), uses GLiNER for extraction, and iGraph for fast retrieval.
- **Memory Management**: Automatically adjusts retrieval depth based on system RAM and zram.
- **Contextual Search**: Tracks your conversation history to improve relevance while preventing "echo chambers" through drift control.
- **MCP Native**: Plugs directly into any MCP-compliant agent as a tool or skill.

## 🚀 Quick Start

### 0. Prerequisites
For the fastest experience and pre-built wheel support, we recommend [**uv**](https://github.com/astral-sh/uv):
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# (Optional) For high-performance C-extensions
sudo apt-get install -y libigraph-dev
```

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

## 🔌 Integrations

### 1. Generic MCP Client
To use Edge Hippo with any MCP-compliant agent (such as Cursor, Windsurf, or your own custom agent), add the following to your agent's configuration:

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

### 2. Claude Desktop
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

### 3. OpenClaw
We provide a pre-configured preset for OpenClaw. 
You can find it at [`presets/openclaw.json`](presets/openclaw.json) in this repository.

## 🏗️ Architecture & Optimization
For detailed architecture decisions, data schema, and Raspberry Pi 5 specific optimizations, please refer to [TECH_SPEC.md](TECH_SPEC.md).

We tested Edge-Hippo against standard Naive RAG (Vector-Only) using 151 complex test cases across three scenarios: RPi 5 technical docs, *Three Kingdoms* narrative, and HotPotQA.

| Metric | Edge-Hippo | Naive RAG | Lift |
| :--- | :--- | :--- | :--- |
| **Peak Raw Recall** | **43.1%** | 22.4% | 🟢 **+20.7%** |
| **Adaptive Recall** | **20.2%** | 14.8% | 🟢 **+5.4%** |
| **Context Control** | **100.0%** | 100.0% | - |
| **Token Savings** | **71.3%** | -131.4% | 🟢 **+202.7%** |
| **Startup Time** | **< 1.2s** | < 1.0s | - |

### Recall vs. Node Budget (Scalability)
Edge-Hippo dynamically adjusts its search depth based on available RAM. The "Adaptive Recall" metric reflects the trade-off between retrieval accuracy and memory pressure.

| Profile | Node Budget | Adaptive Recall | Scenario |
| :--- | :--- | :--- | :--- |
| **Raw Recall** | ∞ (Uncapped) | **43.1%** | Maximum potential on high-end hardware |
| **High Profile** | 5,000 Nodes | **31.4%** | Optimal for RPi 5 (8GB) with ZRAM |
| **Mid Profile** | 1,500 Nodes | **20.2%** | Default balanced mode for edge devices |
| **Low Profile** | 500 Nodes | **12.1%** | Strict low-memory mode |

> **Bottom line**: While Naive RAG attempts to gain recall by dumping massive amounts of raw text, Edge-Hippo achieves superior results with surgical precision. Peak raw recall hits 43.1% on dense technical docs, while the 20.2% score represents the default balanced performance on a standard RPi 5.

## 📚 References & Research
This implementation is based on the following research papers:
1. **HippoRAG 2**: [https://arxiv.org/pdf/2502.14802](https://arxiv.org/pdf/2502.14802)
2. **Edge-Optimized GraphRAG**: [https://arxiv.org/pdf/2602.01965](https://arxiv.org/pdf/2602.01965)
3. **Dense-Sparse Integration**: [https://arxiv.org/pdf/2510.08958](https://arxiv.org/pdf/2510.08958)

*Benchmarks run on Raspberry Pi 5 (8GB) optimized environment.*
