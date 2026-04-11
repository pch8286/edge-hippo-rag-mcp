import logging
import time
import asyncio
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
import igraph
import numpy as np
from .config import settings
from .extraction import EntityExtractor
from .storage import GraphStorage
from .reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

from .algorithms import check_drift
from .resource_manager import resource_manager

class PPRRetriever:
    ET_PP = 0  # phrase -> passage
    ET_DP = 1  # passage -> phrase
    ET_PH = 2  # phrase -> phrase

    def __init__(
        self,
        extractor: EntityExtractor,
        storage: GraphStorage,
        encoder=None,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.extractor = extractor
        self.storage = storage
        self.encoder = encoder
        self.reranker = reranker if reranker is not None else CrossEncoderReranker.from_settings()

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

        if self.encoder:
            loop = asyncio.get_running_loop()
            query_prefixed = "query: " + query
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
        
        effective_decay = 0.0
        
        if history_entities:
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

        node_limit = resource_manager.calculate_node_budget()
        logger.info(f"Retrieval: Using dynamic node budget: {node_limit}")
        
        subgraph = await self.storage.get_ego_subgraph(all_seeds, depth=2, limit=node_limit)
        nodes = subgraph['nodes']
        edges = subgraph['edges']

        t_subgraph = time.time()
        
        if not nodes:
            return "No context found for entities.", list(seed_names)

        id_to_idx = {n['id']: i for i, n in enumerate(nodes)}
        total_base_nodes = len(nodes)

        current_map = {}  # idx -> prob over current seed nodes
        if current_entity_ids:
            idx_scores: Dict[int, float] = {}
            for nid in current_entity_ids:
                if nid in id_to_idx:
                    idx_scores[id_to_idx[nid]] = float(seed_scores.get(nid, 1.0))
            current_map = self._build_seed_distribution(
                idx_scores,
                beta=3.0,
                uniform_mix=0.10,
            )

        passage_indices = [i for i, n in enumerate(nodes) if n["type"] == "passage"]
        seed_hub_indices = {
            idx
            for idx in current_map.keys()
            if nodes[idx]["type"] == "phrase" and bool(nodes[idx].get("is_hub", False))
        }
        current_map = self._suppress_seed_hub_self_mass(
            current_map,
            seed_hub_indices=seed_hub_indices,
            passage_indices=passage_indices,
            total_nodes=total_base_nodes,
        )

        history_sum = len(history_entity_ids) if history_entity_ids else 1.0
        history_map = {}

        if history_entity_ids:
            for nid in history_entity_ids:
                if nid in id_to_idx:
                    idx = id_to_idx[nid]
                    prob = 1.0 / history_sum
                    history_map[idx] = prob

        history_map = self._normalize_map(history_map)
        actual_decay = effective_decay if history_map else 0.0

        node_stats, global_meta = await self._get_transition_stats(nodes)
        scored_edges = self._score_candidate_edges(
            nodes=nodes,
            raw_edges=edges,
            id_to_idx=id_to_idx,
            node_stats=node_stats,
            global_meta=global_meta,
        )
        capped_edges = self._apply_fanout_caps(scored_edges, nodes)
        pruned_edges = self._prune_type_budget(capped_edges)
        seed_dist = current_map or history_map
        if not seed_dist:
            seed_dist = self._fallback_uniform_distribution(passage_indices, total_base_nodes)

        g, sink_idx = self._build_residual_graph(
            nodes=nodes,
            pruned_edges=pruned_edges,
            node_stats=node_stats,
            global_meta=global_meta,
            base_seed_dist=seed_dist,
            seed_hub_indices=seed_hub_indices,
        )
        t_graph_build = time.time()

        total_nodes = g.vcount()
        reset_vec = [0.0] * total_nodes
        for i in range(total_base_nodes):
            val_c = current_map.get(i, 0.0)
            val_h = history_map.get(i, 0.0)
            reset_vec[i] = (1.0 - actual_decay) * val_c + actual_decay * val_h
        if sum(reset_vec[:total_base_nodes]) <= 0.0:
            fallback = self._fallback_uniform_distribution(passage_indices, total_base_nodes)
            for idx, p in fallback.items():
                reset_vec[idx] = float(p)
        if sink_idx is not None:
            reset_vec[sink_idx] = 0.0

        damping = self._schedule_damping(
            base=settings.PPR_DAMPING,
            seed_count=max(1, len(current_map)),
            seed_hub_ratio=(len(seed_hub_indices) / max(len(current_map), 1)),
            node_count=total_base_nodes,
        )

        loop = asyncio.get_running_loop()

        def _run_pagerank():
            return g.personalized_pagerank(
                vertices=None,
                damping=damping,
                reset=reset_vec,
                weights=g.es["weight"]
            )

        ppr_scores = await loop.run_in_executor(None, _run_pagerank)
        t_ppr = time.time()
        
        ranked = []
        for i, node in enumerate(nodes):
            if node['type'] == 'passage':
                ranked.append((node['id'], ppr_scores[i]))
                
        ranked.sort(key=lambda x: x[1], reverse=True)
        ranked = self._stable_dedup_ranked(ranked)

        fetch_limit = top_k
        if self.reranker is not None:
            fetch_limit = max(fetch_limit, max(self.reranker.top_n, 0))
        fetch_limit = min(fetch_limit, len(ranked))

        candidates = await self._build_candidates(ranked[:fetch_limit])
        if self.reranker is not None:
            top_results = self.reranker.rerank_and_fuse(query, candidates, top_k=top_k)
        else:
            top_results = candidates[:top_k]
        
        total_seeds = len(current_entity_ids)
        header = f"Found {len(seed_names)} seed entities"
        if total_seeds > len(seed_names):
            header += f" (expanded to {total_seeds})"
        header += (
            f". History context: {len(history_entity_ids)} entities (Decay={actual_decay})."
            f" Damping={damping:.3f}."
        )
        output = [header]
        output.append("\nRelevant Passages:\n")
        
        for item in top_results:
             score = float(item.get("final_score", item.get("ppr_score", 0.0)))
             content = item.get("text", "")
             if content:
                 output.append(f"--- [Score: {score:.4f}] ---\n{content}\n")

        return "\n".join(output), list(seed_names)

    @staticmethod
    def _build_seed_distribution(
        idx_scores: Dict[int, float],
        *,
        beta: float,
        uniform_mix: float,
    ) -> Dict[int, float]:
        """Stable softmax seed distribution with uniform mixing.

        Returns a sparse map (node_idx -> probability) over seed indices.
        """
        if not idx_scores:
            return {}

        items = sorted(idx_scores.items(), key=lambda x: x[0])
        idx = np.asarray([i for i, _ in items], dtype=np.int32)
        raw = np.asarray([max(float(s), 1e-6) for _, s in items], dtype=np.float32)

        z = beta * (raw - float(raw.max()))
        expz = np.exp(z, dtype=np.float32)
        denom = float(expz.sum())
        if denom <= 0.0:
            soft = np.full_like(expz, 1.0 / max(len(expz), 1), dtype=np.float32)
        else:
            soft = expz / denom

        uniform = np.full_like(soft, 1.0 / max(len(soft), 1), dtype=np.float32)
        lam = float(np.clip(uniform_mix, 0.0, 1.0))
        mixed = (1.0 - lam) * soft + lam * uniform
        mixed /= float(mixed.sum())

        return {int(i): float(p) for i, p in zip(idx, mixed)}

    @staticmethod
    def _normalize_map(dist: Dict[int, float]) -> Dict[int, float]:
        if not dist:
            return {}
        z = float(sum(max(0.0, float(v)) for v in dist.values()))
        if z <= 0.0:
            return {}
        return {int(k): float(max(0.0, float(v)) / z) for k, v in dist.items() if v > 0.0}

    @staticmethod
    def _fallback_uniform_distribution(
        preferred_indices: List[int], total_nodes: int
    ) -> Dict[int, float]:
        if preferred_indices:
            p = 1.0 / float(len(preferred_indices))
            return {int(i): p for i in preferred_indices}
        if total_nodes <= 0:
            return {}
        p = 1.0 / float(total_nodes)
        return {i: p for i in range(total_nodes)}

    @classmethod
    def _exclude_hub_distribution(
        cls,
        base_dist: Dict[int, float],
        hub_idx: int,
        passage_indices: List[int],
        total_nodes: int,
    ) -> Dict[int, float]:
        filtered = {k: v for k, v in base_dist.items() if k != hub_idx and v > 0.0}
        if filtered:
            return cls._normalize_map(filtered)
        fallback_targets = [i for i in passage_indices if i != hub_idx]
        if not fallback_targets and total_nodes > 1:
            fallback_targets = [i for i in range(total_nodes) if i != hub_idx]
        if not fallback_targets:
            return {}
        p = 1.0 / float(len(fallback_targets))
        return {int(i): p for i in fallback_targets}

    @classmethod
    def _suppress_seed_hub_self_mass(
        cls,
        seed_dist: Dict[int, float],
        *,
        seed_hub_indices: set[int],
        passage_indices: List[int],
        total_nodes: int,
    ) -> Dict[int, float]:
        """Redistribute seed-hub reset mass away from hub itself."""
        out = dict(seed_dist)
        if not out:
            return out
        for hub_idx in sorted(seed_hub_indices):
            mass = float(out.get(hub_idx, 0.0))
            if mass <= 0.0:
                continue
            out[hub_idx] = 0.0
            redist = cls._exclude_hub_distribution(out, hub_idx, passage_indices, total_nodes)
            if not redist:
                continue
            for j, p in redist.items():
                out[j] = float(out.get(j, 0.0) + mass * p)
        return cls._normalize_map(out)

    async def _get_transition_stats(
        self, nodes: List[Dict[str, Any]]
    ) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
        node_ids = [int(n["id"]) for n in nodes]
        method = getattr(self.storage, "get_transition_stats", None)
        if callable(method):
            try:
                return await method(node_ids)
            except Exception as e:
                logger.warning("Failed to load transition stats, using defaults: %s", e)

        stats = {
            int(n["id"]): {
                "global_phrase_deg": 1.0,
                "global_passage_phrase_deg": 1.0,
                "idf": 1.0 if n["type"] == "phrase" else 0.0,
            }
            for n in nodes
        }
        return stats, {"AVG_PHRASE_DEG": 1.0, "AVG_PASSAGE_PHRASE_DEG": 1.0}

    def _score_candidate_edges(
        self,
        *,
        nodes: List[Dict[str, Any]],
        raw_edges: List[Any],
        id_to_idx: Dict[int, int],
        node_stats: Dict[int, Dict[str, float]],
        global_meta: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        avg_pasg = float(max(global_meta.get("AVG_PASSAGE_PHRASE_DEG", 1.0), 1.0))
        out: List[Dict[str, Any]] = []
        for edge in raw_edges:
            if isinstance(edge, dict):
                src = edge.get("source")
                dst = edge.get("target")
                w = float(edge.get("weight", 1.0))
            else:
                src, dst, w = edge
                w = float(w)

            if src not in id_to_idx or dst not in id_to_idx:
                continue
            if w <= 0.0:
                continue

            sidx = id_to_idx[src]
            didx = id_to_idx[dst]
            stype = nodes[sidx]["type"]
            dtype = nodes[didx]["type"]

            if stype == "phrase" and dtype == "passage":
                etype = self.ET_PP
                deg_pasg = float(
                    node_stats.get(int(dst), {}).get("global_passage_phrase_deg", 1.0)
                )
                ratio = max(0.0, (deg_pasg / avg_pasg) - 1.0)
                score = 2.0 / (1.0 + (0.75 * ratio))
            elif stype == "passage" and dtype == "phrase":
                etype = self.ET_DP
                score = float(node_stats.get(int(dst), {}).get("idf", 1.0))
            elif stype == "phrase" and dtype == "phrase":
                etype = self.ET_PH
                score = w
            else:
                continue

            if score <= 0.0:
                continue
            out.append(
                {
                    "src_idx": int(sidx),
                    "dst_idx": int(didx),
                    "etype": int(etype),
                    "score": float(score),
                }
            )
        return out

    def _apply_fanout_caps(
        self, edges: List[Dict[str, Any]], nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        for e in edges:
            grouped[(e["src_idx"], e["etype"])].append(e)

        kept: List[Dict[str, Any]] = []
        for (src_idx, etype), grp in grouped.items():
            node = nodes[src_idx]
            cap = 0
            if node["type"] == "phrase":
                if etype == self.ET_PP:
                    cap = 48 if bool(node.get("is_hub", False)) else 64
                elif etype == self.ET_PH:
                    cap = 16 if bool(node.get("is_hub", False)) else 32
            elif node["type"] == "passage" and etype == self.ET_DP:
                cap = 128

            if cap <= 0:
                continue
            if len(grp) <= cap:
                kept.extend(grp)
                continue
            grp_sorted = sorted(grp, key=lambda x: (-x["score"], x["dst_idx"]))
            kept.extend(grp_sorted[:cap])
        return kept

    def _prune_type_budget(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        budgets = {
            self.ET_PP: 12000,
            self.ET_DP: 12000,
            self.ET_PH: 4000,
        }
        by_type: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for e in edges:
            by_type[e["etype"]].append(e)

        kept: List[Dict[str, Any]] = []
        for etype, grp in by_type.items():
            budget = int(budgets.get(etype, len(grp)))
            if len(grp) <= budget:
                kept.extend(grp)
                continue
            grp_sorted = sorted(grp, key=lambda x: (-x["score"], x["src_idx"], x["dst_idx"]))
            kept.extend(grp_sorted[:budget])
        return kept

    def _build_residual_graph(
        self,
        *,
        nodes: List[Dict[str, Any]],
        pruned_edges: List[Dict[str, Any]],
        node_stats: Dict[int, Dict[str, float]],
        global_meta: Dict[str, float],
        base_seed_dist: Dict[int, float],
        seed_hub_indices: set[int],
    ) -> Tuple[igraph.Graph, Optional[int]]:
        n = len(nodes)
        edge_weight_map: Dict[Tuple[int, int], float] = defaultdict(float)
        out_score_sum = np.zeros(n, dtype=np.float32)
        out_count = np.zeros(n, dtype=np.int32)

        for e in pruned_edges:
            src = int(e["src_idx"])
            out_score_sum[src] += float(e["score"])
            out_count[src] += 1

        avg_phrase_deg = float(max(global_meta.get("AVG_PHRASE_DEG", 1.0), 1.0))
        p_self = np.zeros(n, dtype=np.float32)
        alpha = np.ones(n, dtype=np.float32)
        residual = np.zeros(n, dtype=np.float32)
        struct_mass = np.ones(n, dtype=np.float32)

        passage_indices = [i for i, node in enumerate(nodes) if node["type"] == "passage"]
        for i, node in enumerate(nodes):
            if node["type"] != "phrase" or not bool(node.get("is_hub", False)):
                continue
            deg_g = float(node_stats.get(int(node["id"]), {}).get("global_phrase_deg", 0.0))
            if deg_g <= 0.0:
                continue

            p = 0.8 * (deg_g / (deg_g + avg_phrase_deg))
            p = float(np.clip(p, 0.0, 0.95))
            p_self[i] = p
            a = float(np.clip(float(out_count[i]) / deg_g, 0.0, 1.0))
            alpha[i] = a
            residual[i] = (1.0 - p) * (1.0 - a)
            struct_mass[i] = (1.0 - p) * a

        for e in pruned_edges:
            src = int(e["src_idx"])
            dst = int(e["dst_idx"])
            denom = float(out_score_sum[src])
            if denom <= 0.0:
                continue
            w = (float(e["score"]) / denom) * float(struct_mass[src])
            if w <= 0.0:
                continue
            edge_weight_map[(src, dst)] += float(w)

        base_seed_dist = self._normalize_map(base_seed_dist)
        for i, node in enumerate(nodes):
            if node["type"] != "phrase" or not bool(node.get("is_hub", False)):
                continue

            if p_self[i] > 0.0:
                edge_weight_map[(i, i)] += float(p_self[i])

            r = float(residual[i])
            if r <= 0.0:
                continue
            if i in seed_hub_indices:
                dist = self._exclude_hub_distribution(base_seed_dist, i, passage_indices, n)
            else:
                dist = base_seed_dist
            for j, prob in dist.items():
                if prob <= 0.0:
                    continue
                edge_weight_map[(i, int(j))] += r * float(prob)

        g = igraph.Graph(n=n, directed=True)
        g.vs["name"] = [node["name"] for node in nodes]
        g.vs["type"] = [node["type"] for node in nodes]
        g.vs["db_id"] = [node["id"] for node in nodes]

        if edge_weight_map:
            edge_list = list(edge_weight_map.keys())
            weights = [float(edge_weight_map[e]) for e in edge_list]
            g.add_edges(edge_list)
            g.es["weight"] = weights

        dangling = [v.index for v in g.vs if g.degree(v, mode="out") == 0]
        if not dangling:
            return g, None

        sink_idx = g.vcount()
        g.add_vertices(1)
        g.vs[sink_idx]["name"] = "SINK"
        g.vs[sink_idx]["type"] = "sink"
        g.vs[sink_idx]["db_id"] = -1
        sink_edges = [(d, sink_idx) for d in dangling] + [(sink_idx, sink_idx)]
        sink_weights = [1.0] * len(sink_edges)
        g.add_edges(sink_edges)
        g.es[len(g.es) - len(sink_edges):]["weight"] = sink_weights
        return g, sink_idx

    @staticmethod
    def _schedule_damping(
        *, base: float, seed_count: int, seed_hub_ratio: float, node_count: int
    ) -> float:
        d = float(base)
        if seed_hub_ratio >= 0.5:
            d -= 0.08
        if seed_count >= 4:
            d += 0.03
        if node_count <= 200:
            d -= 0.03
        elif node_count >= 2000:
            d += 0.02
        return float(np.clip(d, 0.55, 0.92))

    async def _build_candidates(
        self,
        ranked: List[Tuple[int, float]],
    ) -> List[Dict[str, Any]]:
        if not ranked:
            return []

        tasks = [self.storage.get_node_content(pid) for pid, _ in ranked]
        contents = await asyncio.gather(*tasks)
        candidates: List[Dict[str, Any]] = []
        for idx, ((pid, score), content) in enumerate(zip(ranked, contents), start=1):
            candidates.append(
                {
                    "passage_id": int(pid),
                    "ppr_score": float(score),
                    "text": content or "",
                    "ppr_rank": idx,
                    "final_score": float(score),
                }
            )
        return candidates

    @staticmethod
    def _stable_dedup_ranked(ranked: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        seen = set()
        deduped: List[Tuple[int, float]] = []
        for pid, score in ranked:
            if pid in seen:
                continue
            seen.add(pid)
            deduped.append((pid, score))
        if len(deduped) != len(ranked):
            logger.warning(
                "Detected duplicate passage candidates: %d -> %d after stable dedup.",
                len(ranked),
                len(deduped),
            )
        return deduped
