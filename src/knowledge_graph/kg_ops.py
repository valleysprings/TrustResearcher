"""
Knowledge Graph Operations

Unified KG module combining:
- Construction: Build KG from seed topic + literature
- Community: Detect and query communities
- Path Sampling: DFS paths for idea generation with softmax sampling

Flow:
1. kg.build(topic, papers) - construct KG
2. kg.detect_communities() - auto-called after build
3. kg.get_community_context(i) - for base variants
4. kg.sample_paths_for_got(count) - for GoT variants
"""

import math
import random
import re
import json
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict
from ..utils.llm_interface import LLMInterface
from ..skills.ideagen.graphop import (
    KG_CORE_CONCEPTS_PROMPT, KG_EXPANSION_PROMPT, KG_RELATIONSHIP_PROMPT,
    KG_METHODOLOGY_SYSTEM_PROMPT, KG_METHODOLOGY_USER_PROMPT, KG_ENTITY_EXPANSION_PROMPT
)


class KGOps:
    """
    Unified Knowledge Graph operations.

    Combines KG construction, community detection, and path sampling.
    """

    def __init__(self, config: Dict = None, llm_config: Dict = None, logger=None):
        self.config = config or {}
        self.kg_config = self.config['knowledge_graph']
        self.got_config = self.config.get('graph_of_thought', {})
        self.graph = nx.Graph()
        self.entity_types = {}
        self.logger = logger
        self.llm = LLMInterface(config=llm_config, logger=logger)

        # Community state (populated after build)
        self.communities = {}  # node -> community_id
        self.community_info = {}  # community_id -> {nodes, size, modularity}
        self._scored_paths = []  # cached scored paths for softmax sampling
        self._prev_nodes = 0  # for tracking delta
        self._prev_edges = 0

        # Embedding model (lazy loaded if needed)
        self._embedding_model = None
        self._has_embedding_model = None  # Cache the check result

    def _log_kg_stats(self, stage_name: str):
        """Log KG stats after each stage with delta."""
        nodes = len(self.graph.nodes)
        edges = len(self.graph.edges)
        delta_nodes = nodes - self._prev_nodes
        delta_edges = edges - self._prev_edges
        print(f"    {stage_name}: {nodes} nodes (+{delta_nodes}), {edges} edges (+{delta_edges})")
        self._prev_nodes = nodes
        self._prev_edges = edges

    # ==================== Construction ====================

    def add_entity(self, entity: str, entity_type: str = "concept"):
        """Add an entity to the knowledge graph."""
        if entity not in self.graph:
            self.graph.add_node(entity, type=entity_type, label=entity)
            self.entity_types[entity] = entity_type

    def add_relationship(self, entity1: str, entity2: str, relationship: str, weight: float = 1.0):
        """Add a relationship between two entities."""
        if entity1 in self.graph.nodes and entity2 in self.graph.nodes:
            self.graph.add_edge(entity1, entity2, relationship=relationship, weight=weight)

    async def build(self, seed_topic: str, literature_papers: List = None) -> 'KGOps':
        """
        Build knowledge graph from seed topic and literature.

        Args:
            seed_topic: Research topic
            literature_papers: Papers from literature search

        Returns:
            self for chaining
        """
        print(f"Building knowledge graph for: {seed_topic}")

        # Stage 1: Core domain concepts
        await self._build_core_concepts(seed_topic)
        self._log_kg_stats("Stage: Add Core concepts")

        # Stage 2: Literature integration
        if literature_papers:
            await self._integrate_literature(literature_papers)
            self._log_kg_stats("Stage: Integrate Literature")

        # Stage 3: Expand with related concepts (DISABLED for future update)
        # await self._expand_with_related_concepts(seed_topic)
        # self._log_kg_stats("Stage 3 (Expansion)")

        # Stage 4: Enhance relationships
        await self._enhance_relationships()
        self._log_kg_stats("Stage: Enhance Relationships")

        # Stage 5: Detect communities (auto after build)
        self.detect_communities()

        print(f"KG complete: {len(self.graph.nodes)} entities, {len(self.graph.edges)} edges, "
              f"{len(self.community_info)} communities")

        return self

    async def _build_core_concepts(self, seed_topic: str):
        """Stage: Build core domain concepts."""
        print("  Stage: Core concepts...")
        core_prompt = KG_CORE_CONCEPTS_PROMPT.format(seed_topic=seed_topic)
        initial_context = await self.llm.extract_entities_and_relationships(core_prompt)
        self._parse_and_add_knowledge(initial_context)

    async def _integrate_literature(self, papers: List):
        """Stage: Integrate literature papers."""
        print("  Stage: Literature integration...")
        batch_size = self.kg_config['paper_batch_size']

        for i in range(0, len(papers), batch_size):
            batch = papers[i:i+batch_size]
            await self._process_paper_batch(batch, i // batch_size + 1)

        # Add paper entities and relationships
        await self._add_paper_entities(papers)

    async def _process_paper_batch(self, papers: List, batch_num: int):
        """Process a batch of papers."""
        batch_context = [f"Literature batch {batch_num}:"]

        for paper in papers:
            summary = f"- {paper.title} ({paper.year})"
            if paper.abstract:
                summary += f": {paper.abstract}"
            if paper.keywords:
                summary += f"\n  Keywords: {', '.join(paper.keywords)}"
            batch_context.append(summary)

        batch_context.append("\nExtract entities and relationships from this literature.")
        batch_prompt = "\n".join(batch_context)
        batch_knowledge = await self.llm.extract_entities_and_relationships(batch_prompt)
        self._parse_and_add_knowledge(batch_knowledge)

    async def _add_paper_entities(self, papers: List):
        """Add paper entities and their relationships."""
        for paper in papers:
            paper_entity = f"Paper: {paper.title}"
            self.add_entity(paper_entity, "paper")

            # Authors
            for author in paper.authors:
                author_entity = f"Author: {author}"
                self.add_entity(author_entity, "researcher")
                self.add_relationship(paper_entity, author_entity, "authored_by")

            # Keywords
            for keyword in paper.keywords:
                kw_entity = keyword.lower()
                self.add_entity(kw_entity, "concept")
                self.add_relationship(paper_entity, kw_entity, "studies")

            # Methodologies from abstract
            if paper.abstract:
                methods = await self._extract_methodologies(paper.abstract)
                for method in methods:
                    self.add_entity(method.lower(), "methodology")
                    self.add_relationship(paper_entity, method.lower(), "uses_method")

        # Cross-paper connections
        self._create_paper_cross_connections(papers)

    def _create_paper_cross_connections(self, papers: List):
        """Create connections between papers with shared concepts."""
        default_weight = self.kg_config['default_relationship_weight']
        for i, p1 in enumerate(papers):
            for p2 in papers[i+1:]:
                shared = set(p1.keywords or []) & set(p2.keywords or [])
                if shared:
                    e1, e2 = f"Paper: {p1.title}", f"Paper: {p2.title}"
                    if e1 in self.graph and e2 in self.graph:
                        rel = f"shares_concepts: {', '.join(list(shared))}"
                        self.add_relationship(e1, e2, rel, weight=default_weight)

    # DISABLED for more focused graphs
    # async def _expand_with_related_concepts(self, seed_topic: str):
    #     """Stage: Expand with related concepts."""
    #     print("  Stage: Expanding concepts...")
    #     current_entities = self._select_expansion_seeds()
    #     if not current_entities:
    #         return
    #
    #     expansion_prompt = KG_EXPANSION_PROMPT.format(
    #         seed_topic=seed_topic,
    #         current_entities=', '.join(current_entities)
    #     )
    #     expanded = await self.llm.extract_entities_and_relationships(expansion_prompt)
    #     self._parse_and_add_knowledge(expanded)

    async def _enhance_relationships(self):
        """Stage: Enhance relationships between entities."""
        print("  Stage: Enhancing relationships...")
        entities = list(self.graph.nodes())
        if len(entities) < 2:
            return

        cfg = self.kg_config['relationship_enhancement']
        iterations = max(1, int(cfg['iterations']))
        sampled_batches = set()

        for _ in range(iterations):
            sample = self._sample_for_relationships(entities, cfg, sampled_batches)
            if not sample:
                continue
            rel_prompt = KG_RELATIONSHIP_PROMPT.format(entities=', '.join(sample))
            rel_knowledge = await self.llm.extract_entities_and_relationships(rel_prompt)
            self._parse_and_add_knowledge(rel_knowledge)

        # Optional: Embedding-based similarity detection
        if cfg.get('enable_embedding_similarity', False):
            await self._add_embedding_based_relationships(entities, cfg)

    def _select_expansion_seeds(self) -> List[str]:
        """Select seed entities for expansion."""
        nodes = list(self.graph.nodes())
        if not nodes:
            return []

        cfg = self.kg_config['expansion']
        sample_size = min(int(cfg['seed_sample_size']), len(nodes))

        if cfg['sampling_strategy'].lower() == 'random':
            return random.sample(nodes, sample_size)

        # Degree-based: prefer high-degree nodes
        degrees = dict(self.graph.degree(nodes))
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_nodes[:sample_size]]

    def _sample_for_relationships(self, entities: List[str], cfg: Dict, sampled: Set) -> List[str]:
        """Sample entities for relationship enhancement."""
        if not entities:
            return []

        sample_size = min(int(cfg['sample_size']), len(entities))
        degrees = dict(self.graph.degree(entities))
        sorted_by_deg = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

        # Hybrid: top degree + random
        top_count = max(1, int(sample_size * cfg['top_degree_ratio']))
        top_entities = [n for n, _ in sorted_by_deg[:top_count]]

        remaining = [n for n in entities if n not in top_entities]
        rand_count = min(sample_size - len(top_entities), len(remaining))
        rand_entities = random.sample(remaining, rand_count) if rand_count > 0 else []

        batch = (top_entities + rand_entities)[:sample_size]
        if len(batch) < 2:
            return []

        sig = tuple(sorted(batch))
        if sig in sampled:
            return []
        sampled.add(sig)
        return batch

    async def _extract_methodologies(self, abstract: str) -> List[str]:
        """Extract methodologies from abstract."""
        try:
            # Check if abstract is empty or too short
            if not abstract or len(abstract.strip()) < 10:
                if self.logger:
                    self.logger.log_debug(f"Skipping methodology extraction for empty/short abstract: '{abstract if abstract else 'None'}'", "kg_ops")
                return []

            response = await self.llm.generate_with_system_prompt(
                KG_METHODOLOGY_SYSTEM_PROMPT,
                KG_METHODOLOGY_USER_PROMPT.format(abstract=abstract),
                caller="kg_methodology_extraction"
            )

            # Check if response is empty
            if not response or not response.strip():
                if self.logger:
                    self.logger.log_warning(f"Empty methodology extraction response for abstract: '{abstract if abstract else 'None'}'", "kg_ops")
                return []

            methods = [m.strip().lower() for m in response.split(',')]
            return [m for m in methods if len(m) > 3]
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Methodology extraction error: {e}", "kg_ops")
            return []

    async def _add_embedding_based_relationships(self, entities: List[str], cfg: Dict):
        """Add relationships based on embedding similarity."""
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
        except ImportError:
            if self.logger:
                self.logger.log_info("sentence-transformers not available, skipping embedding-based relationships", "kg_ops")
            return

        try:
            print("    Finding embedding-based relationships...")

            # Load embedding model from config
            model_name = cfg['embedding_model']
            model = SentenceTransformer(model_name)

            # Process in batches to avoid memory issues
            batch_size = cfg['similarity_batch_size']
            threshold = cfg['similarity_threshold']
            added_count = 0

            for i in range(0, len(entities), batch_size):
                batch = entities[i:i+batch_size]

                # Compute embeddings
                embeddings = model.encode(batch, show_progress_bar=False)

                # Compute pairwise similarities
                similarities = cosine_similarity(embeddings)

                # Find similar pairs
                for idx1 in range(len(batch)):
                    for idx2 in range(idx1 + 1, len(batch)):
                        sim = similarities[idx1][idx2]
                        if sim >= threshold:
                            e1, e2 = batch[idx1], batch[idx2]
                            # Only add if not already connected
                            if not self.graph.has_edge(e1, e2):
                                self.add_relationship(e1, e2, "semantically_similar", weight=float(sim))
                                added_count += 1

            if self.logger:
                self.logger.log_info(f"Added {added_count} embedding-based relationships", "kg_ops")
            print(f"    Added {added_count} embedding-based relationships")

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Embedding-based relationship detection failed: {e}", "kg_ops", e)

    def _parse_and_add_knowledge(self, text: str):
        """Parse LLM-extracted entities and relationships."""
        try:
            # Parse entities
            ent_match = re.search(r'ENTITIES:(.*?)(?=RELATIONSHIPS:|$)', text, re.DOTALL)
            if ent_match:
                for entity, etype in self._parse_entities(ent_match.group(1)):
                    self.add_entity(entity, etype)

            # Parse relationships
            rel_match = re.search(r'RELATIONSHIPS:(.*?)$', text, re.DOTALL)
            if rel_match:
                for e1, rel, e2 in self._parse_relationships(rel_match.group(1)):
                    if e1 not in self.graph:
                        self.add_entity(e1, "concept")
                    if e2 not in self.graph:
                        self.add_entity(e2, "concept")
                    self.add_relationship(e1, e2, rel)
        except Exception as e:
            print(f"Parse error: {e}")

    def _parse_entities(self, text: str) -> List[Tuple[str, str]]:
        """Parse entities from text."""
        entities = []
        for line in text.strip().split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('•'):
                info = line[1:].strip()
                if '(type:' in info:
                    match = re.match(r'(.*?)\s*\(type:\s*(.*?)\)', info)
                    if match:
                        entities.append((match.group(1).strip(), match.group(2).strip()))
                else:
                    entities.append((info, "concept"))
        return entities

    def _parse_relationships(self, text: str) -> List[Tuple[str, str, str]]:
        """Parse relationships from text."""
        rels = []
        for line in text.strip().split('\n'):
            line = line.strip()
            if (line.startswith('-') or line.startswith('•')) and '->' in line:
                parts = line[1:].strip().split('->')
                if len(parts) >= 3:
                    rels.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
        return rels

    # ==================== Community Detection ====================

    def detect_communities(self):
        """Detect communities using Louvain modularity."""
        if len(self.graph.nodes) < 2:
            return

        self.communities = {}
        self.community_info = {}

        try:
            from networkx.algorithms.community import louvain_communities, modularity
            seed = self.got_config['random_seed']
            comms = list(louvain_communities(self.graph, seed=seed))
        except ImportError:
            from networkx.algorithms.community import greedy_modularity_communities, modularity
            comms = list(greedy_modularity_communities(self.graph))

        for cid, nodes in enumerate(comms):
            nodes_list = list(nodes)
            for n in nodes_list:
                self.communities[n] = cid

            try:
                mod = modularity(self.graph, [nodes])
            except:
                mod = 0.0

            self.community_info[cid] = {
                "nodes": nodes_list,
                "size": len(nodes_list),
                "modularity": mod
            }

    def get_communities(self) -> List[Dict]:
        """Get communities sorted by size."""
        if not self.community_info:
            return []
        return sorted(self.community_info.values(), key=lambda c: c["size"], reverse=True)

    def get_community_context(self, variant_num: int) -> str:
        """Get context string for a variant from community."""
        comms = self.get_communities()
        if not comms:
            return ""

        comm = comms[variant_num % len(comms)]
        max_display = self.got_config['max_community_nodes_display']
        nodes = comm["nodes"][:max_display]

        parts = [f"Community (size={comm['size']}, modularity={comm['modularity']:.3f}):"]
        parts.append(f"Entities: {', '.join(nodes)}")

        for node in nodes:
            if node in self.graph:
                for _, tgt, data in self.graph.edges(node, data=True):
                    rel = data.get('relationship', 'related_to')
                    parts.append(f"  {node} --{rel}--> {tgt}")

        return "\n".join(parts)

    # ==================== Path Sampling ====================

    def sample_paths(self, got_count: int) -> List[Dict]:
        """
        Sample DFS paths from community nodes, traverse full graph.

        Samples at least paths_per_idea * got_count paths to ensure enough
        diversity for softmax sampling per GoT idea.

        Args:
            got_count: Number of GoT ideas to generate (samples 10x this many paths)
        """
        if not self.community_info:
            return []

        paths_per_idea = self.got_config['paths_per_idea']
        min_total_paths = paths_per_idea * got_count
        max_paths = max(min_total_paths, 100)  # At least 100 or 10*got_count

        paths = []
        min_len = self.got_config['min_path_length']
        max_len = self.got_config['max_path_length']
        total_nodes = sum(c["size"] for c in self.community_info.values())

        for cid, info in self.community_info.items():
            ratio = info["size"] / total_nodes if total_nodes > 0 else 1.0
            paths_for_comm = max(1, int(max_paths * ratio))
            comm_paths = self._sample_from_community(info["nodes"], paths_for_comm, min_len, max_len, cid)
            paths.extend(comm_paths)

        # Score and cache
        self._scored_paths = self._score_paths(paths)
        return self._scored_paths[:max_paths]

    def _sample_from_community(self, nodes: List[str], num_paths: int,
                                min_len: int, max_len: int, cid: int) -> List[Dict]:
        """Sample paths starting from community nodes."""
        if not nodes:
            return []

        paths = []
        random.shuffle(nodes)

        for start in nodes[:num_paths * 2]:
            if len(paths) >= num_paths:
                break
            path = self._dfs_path(start, max_len)
            if len(path) >= min_len:
                paths.append({
                    "nodes": path,
                    "path_string": self._path_to_string(path),
                    "score": 0.0,
                    "start_community": cid
                })
        return paths

    def _dfs_path(self, start: str, max_len: int) -> List[str]:
        """DFS path traversing full graph."""
        path = [start]
        visited = {start}
        current = start

        while len(path) < max_len:
            neighbors = [n for n in self.graph.neighbors(current) if n not in visited]
            if not neighbors:
                break
            # Pick highest degree neighbor
            next_node = max(neighbors, key=lambda n: self.graph.degree(n))
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return path

    def _path_to_string(self, path: List[str]) -> str:
        """Convert path to readable string."""
        truncate = self.got_config['thought_content_truncation']
        parts = []
        for node in path:
            etype = self.entity_types.get(node, "concept")
            content = node[:truncate]
            parts.append(f"[{etype}] {content}")
        return " -> ".join(parts)

    def _score_paths(self, paths: List[Dict]) -> List[Dict]:
        """Score paths by degree and diversity."""
        cfg = self.got_config['community_path_scoring']
        len_div = cfg['length_divisor']
        len_weight = cfg['length_bonus_weight']
        div_weight = cfg['diversity_bonus_weight']

        for p in paths:
            nodes = p.get("nodes", [])
            if not nodes:
                p["score"] = 0.0
                continue

            # Average degree score
            degrees = [self.graph.degree(n) for n in nodes if n in self.graph]
            avg_deg = sum(degrees) / len(degrees) if degrees else 0
            norm_deg = min(avg_deg / 10.0, 1.0)

            # Length bonus
            len_bonus = min(len(nodes) / len_div, 1.0) * len_weight

            # Diversity bonus
            types = set(self.entity_types.get(n, "concept") for n in nodes)
            div_bonus = len(types) * div_weight

            p["score"] = norm_deg + len_bonus + div_bonus

        return sorted(paths, key=lambda x: x.get("score", 0), reverse=True)

    def sample_paths_for_got(self, got_count: int, temperature: float = None) -> List[List[Dict]]:
        """
        Softmax sample paths for GoT variants.

        For each GoT idea, samples paths_per_idea (default 10) non-repeating paths
        using softmax sampling based on path scores.

        Args:
            got_count: Number of GoT ideas
            temperature: Softmax temperature (higher = more uniform)

        Returns:
            List of path lists, one per GoT idea. Each inner list has paths_per_idea paths.
        """
        if temperature is None:
            temperature = self.got_config['softmax_temperature']

        paths_per_idea = self.got_config['paths_per_idea']

        if not self._scored_paths:
            return [[] for _ in range(got_count)]

        result = []
        for _ in range(got_count):
            if len(self._scored_paths) <= paths_per_idea:
                # Not enough paths, return all available
                result.append(list(self._scored_paths))
            else:
                # Softmax sample paths_per_idea paths for this idea
                scores = [p.get("score", 0.0) for p in self._scored_paths]
                sampled = self._softmax_sample(self._scored_paths, scores, paths_per_idea, temperature)
                result.append(sampled)

        return result

    def _softmax_sample(self, items: List, scores: List[float], k: int, temp: float) -> List:
        """Softmax sampling without replacement."""
        if not items or k <= 0:
            return []

        max_s = max(scores) if scores else 0
        exp_scores = [math.exp((s - max_s) / temp) for s in scores]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores] if total > 0 else [1/len(items)] * len(items)

        sampled = []
        remaining = list(items)
        remaining_probs = list(probs)

        for _ in range(min(k, len(items))):
            if not remaining:
                break
            total_p = sum(remaining_probs)
            if total_p <= 0:
                break
            norm_probs = [p / total_p for p in remaining_probs]

            r = random.random()
            cumsum = 0
            idx = 0
            for i, p in enumerate(norm_probs):
                cumsum += p
                if r <= cumsum:
                    idx = i
                    break

            sampled.append(remaining[idx])
            remaining.pop(idx)
            remaining_probs.pop(idx)

        return sampled

    # ==================== Query Methods ====================

    def get_related_entities(self, entity: str, max_distance: int = 2) -> Set[str]:
        """Get entities within max_distance hops."""
        if entity not in self.graph:
            return set()

        related = set()
        current = {entity}
        visited = set()

        for _ in range(max_distance):
            next_level = set()
            for e in current:
                if e not in visited:
                    visited.add(e)
                    neighbors = set(self.graph.neighbors(e))
                    next_level.update(neighbors - visited)
                    related.update(neighbors)
            current = next_level
            if not current:
                break
        return related

    def get_cross_cluster_connections(self) -> List[Tuple[str, str, str]]:
        """Get edges bridging different communities."""
        if not self.communities:
            return []

        cross = []
        for e1, e2, data in self.graph.edges(data=True):
            c1, c2 = self.communities.get(e1), self.communities.get(e2)
            if c1 is not None and c2 is not None and c1 != c2:
                rel = data.get('relationship', 'connected')
                cross.append((e1, rel, e2))
        return cross

    def get_summary(self) -> Dict:
        """Get graph summary."""
        return {
            "num_entities": len(self.graph.nodes()),
            "num_relationships": len(self.graph.edges()),
            "num_communities": len(self.community_info),
            "entity_types": {k: list(self.entity_types.values()).count(k)
                           for k in set(self.entity_types.values())} if self.entity_types else {},
        }

    def save(self, filepath: str):
        """Save graph to file."""
        data = {
            "nodes": [(n, d) for n, d in self.graph.nodes(data=True)],
            "edges": [(u, v, d) for u, v, d in self.graph.edges(data=True)],
            "entity_types": self.entity_types
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load graph from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.graph = nx.Graph()
        for node, attrs in data["nodes"]:
            self.graph.add_node(node, **attrs)
        for u, v, attrs in data["edges"]:
            self.graph.add_edge(u, v, **attrs)
        self.entity_types = data.get("entity_types", {})
        self.detect_communities()

