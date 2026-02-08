import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import igraph
import numpy as np
from .config import settings
from .extraction import EntityExtractor
from .storage import GraphStorage

logger = logging.getLogger(__name__)

from .algorithms import check_drift
from .resource_manager import resource_manager

class PPRRetriever:
    def __init__(self, extractor: EntityExtractor, storage: GraphStorage, encoder=None):
        self.extractor = extractor
        self.storage = storage
        self.encoder = encoder

    async def search(self, query: str, top_k: int = 5, history_entities: List[str] = None, decay_factor: float = 0.2) -> Tuple[str, List[str]]:
        """
        Search using Ego-Graph PPR with retrieval-time expansion.
        Supports Contextual Re-ranking via history_entities.
        """
        t0 = time.time()
        
        query_entities = await self.extractor.extract_entities(query)
        t_extract = time.time()
        
        seed_names = [e['text'] for e in query_entities]
        current_entity_ids = set()
        
        for name in seed_names:
            nid = await self.storage.get_node_by_name(name, 'phrase')
            if nid:
                current_entity_ids.add(nid)
        t_db_seeds = time.time()

        # Vector Search for Query Seeds
        if self.encoder:
            loop = asyncio.get_running_loop()
            query_prefixed = "query: " + query
            # Handle sync/async encode
            if asyncio.iscoroutinefunction(self.encoder.encode):
                 query_vec_np = await self.encoder.encode(query_prefixed)
            else:
                 query_vec_np = await loop.run_in_executor(None, self.encoder.encode, query_prefixed)
            
            if hasattr(query_vec_np, 'tolist'):
                query_vec = query_vec_np.tolist()
            else:
                query_vec = query_vec_np

            vec_results = await self.storage.search_vectors(query_vec, top_k=settings.MAX_NEIGHBORS)
            
            seed_scores = {nid: 1.0 for nid in current_entity_ids}
            
            for nid, dist in vec_results:
                current_entity_ids.add(nid)
                sim = 1.0 / (1.0 + dist)
                seed_scores[nid] = max(seed_scores.get(nid, 0.0), sim)
        else:
             seed_scores = {nid: 1.0 for nid in current_entity_ids}
        
        t_vec_seeds = time.time()
        
        history_entity_ids = set()
        history_scores = {}
        
        # Determine effective decay factor via Drift Check
        effective_decay = 0.0
        
        if history_entities:
            # Perform Topological Drift Check
            # We pass names (history_entities) and names (seed_names)
            # check_drift(storage, current, history)
            # It returns True if Drift Detected (Disconnect)
            is_drift = await check_drift(self.storage, seed_names, history_entities)
            
            if not is_drift:
                effective_decay = decay_factor
                # Load history IDs
                for name in history_entities:
                    nid = await self.storage.get_node_by_name(name, 'phrase')
                    if nid:
                        history_entity_ids.add(nid)
                        history_scores[nid] = 1.0 
            else:
                logger.info("Context Drift Detected! Flushing context.")
                # effective_decay remains 0.0


        all_seeds = list(current_entity_ids.union(history_entity_ids))
        
        if not all_seeds:
            return "No matching entities found.", []

        # 2. Extract Ego-Graph with Dynamic Budget
        node_limit = resource_manager.calculate_node_budget()
        logger.info(f"Retrieval: Using dynamic node budget: {node_limit}")
        
        subgraph = await self.storage.get_ego_subgraph(all_seeds, depth=2, limit=node_limit)
        nodes = subgraph['nodes']
        edges = subgraph['edges']

        t_subgraph = time.time()
        
        if not nodes:
            return "No context found for entities."

        id_to_idx = {n['id']: i for i, n in enumerate(nodes)}
        g = igraph.Graph(n=len(nodes), directed=True)
        g.vs["name"] = [n['name'] for n in nodes]
        g.vs["type"] = [n['type'] for n in nodes]
        g.vs["db_id"] = [n['id'] for n in nodes] # Store real ID
        
        edge_list = []
        weights = []
        
        for src, tgt, w in edges:
            if src not in id_to_idx or tgt not in id_to_idx:
                continue
            
            u, v = id_to_idx[src], id_to_idx[tgt]
            n_src, n_tgt = nodes[u], nodes[v]
            
            base_weight = 2.0 if (n_src['type'] == 'passage' or n_tgt['type'] == 'passage') else 1.0
            
            # u -> v
            w_uv = base_weight * (0.1 if n_src['is_hub'] else 1.0)
            edge_list.append((u, v))
            weights.append(w_uv)
            
            if n_src['is_hub']:
                edge_list.append((u, u))
                weights.append(base_weight * 0.9)

            # v -> u
            w_vu = base_weight * (0.1 if n_tgt['is_hub'] else 1.0)
            edge_list.append((v, u))
            weights.append(w_vu)
            
            if n_tgt['is_hub']:
                edge_list.append((v, v))
                weights.append(base_weight * 0.9)
            
        g.add_edges(edge_list)
        g.es["weight"] = weights
        
        # Sink Node
        dangling = [v.index for v in g.vs if g.degree(v, mode="out") == 0]
        if dangling:
            sink_idx = g.vcount()
            g.add_vertices(1)
            g.vs[sink_idx]["name"] = "SINK"
            sink_edges = [(d, sink_idx) for d in dangling] + [(sink_idx, sink_idx)]
            sink_weights = [1.0] * len(sink_edges)
            g.add_edges(sink_edges)
            g.es[len(g.es)-len(sink_edges):]["weight"] = sink_weights
            
        t_graph_build = time.time()
        
        # Construct Reset Vector
        # v_reset = (1 - lambda) * v_current + lambda * v_history
        
        total_nodes = g.vcount()
        reset_vec = [0.0] * total_nodes
        
        # A. Current Query Vector
        current_sum = sum(seed_scores.values()) if seed_scores else 1.0
        current_map = {} # idx -> prob
        
        if current_entity_ids:
            for nid in current_entity_ids:
                if nid in id_to_idx:
                    idx = id_to_idx[nid]
                    prob = seed_scores.get(nid, 1.0) / current_sum
                    current_map[idx] = prob
        
        # B. History Vector
        history_sum = len(history_entity_ids) if history_entity_ids else 1.0
        history_map = {}
        
        if history_entity_ids:
            for nid in history_entity_ids:
                if nid in id_to_idx:
                    idx = id_to_idx[nid]
                    prob = 1.0 / history_sum # Uniform for history? Or use scores? Uniform for now.
                    history_map[idx] = prob
                    
        # C. Combine
        # If no history, lambda effectively 0 (or caller sets it 0)
        # If no current (unlikely), full history?
        
        actual_decay = effective_decay if history_map else 0.0
        
        # Re-normalize if needed
        # We iterate all nodes and set value
        
        for i in range(total_nodes):
            val_c = current_map.get(i, 0.0)
            val_h = history_map.get(i, 0.0)
            
            val = (1.0 - actual_decay) * val_c + actual_decay * val_h
            reset_vec[i] = val
            
        # Verify L1 Norm (Debug)
        # s = sum(reset_vec)
        # if abs(s - 1.0) > 0.01:
        #     logger.warning(f"PPR Reset vector sum {s} != 1.0")

        ppr_scores = g.personalized_pagerank(
            vertices=None,
            damping=settings.PPR_DAMPING,
            reset=reset_vec,
            weights=g.es["weight"]
        )
        t_ppr = time.time()
        
        ranked = []
        for i, node in enumerate(nodes):
            if node['type'] == 'passage':
                ranked.append((node['id'], ppr_scores[i]))
                
        ranked.sort(key=lambda x: x[1], reverse=True)
        top_results = ranked[:top_k]
        
        total_seeds = len(current_entity_ids)
        header = f"Found {len(seed_names)} seed entities"
        if total_seeds > len(seed_names):
            header += f" (expanded to {total_seeds})"
        header += f". History context: {len(history_entity_ids)} entities (Decay={actual_decay})."
        output = [header]
        output.append("\nRelevant Passages:\n")
        
        for pid, score in top_results:
             content = await self.storage.get_node_content(pid)
             if content:
                 output.append(f"--- [Score: {score:.4f}] ---\n{content}\n")

        # Return Entity IDs for next turn context (Top-K seeds? Or just query seeds?)
        # Spec says: "Context Source (E_{t-1}): Use Previous Query Entities (Input Source)"
        # So we return ids of current query entities.
        
        # We return the output string, but maybe we should return object?
        # Standard implementation returns string.
        # HippoEngine handles session update? Or Retriever?
        # Let's keep Retriever pure and let Engine handle session update using seed_names/ids.
        
        return "\n".join(output), list(seed_names)

