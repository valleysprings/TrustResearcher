"""
Graph of Thought Module - Path Sampling from Knowledge Graph

Implements path-based reasoning for research idea generation.
The core idea: sample diverse, high-quality reasoning paths from the Knowledge Graph,
then use these paths as grounding context for idea generation.

Key Operations:
1. Initialize: Build thought graph from KG entities and facets
2. Connect: Add minimal cross-connections to bridge isolated components
3. Score: Evaluate path quality
4. Sample: Extract diverse reasoning paths using async DFS/BFS

Architecture:
- Thoughts = KG entities + facets (minimal new generation)
- Edges = KG relationships + minimal cross-connections
- Output = high-quality reasoning paths (nodes + relations)
"""

import asyncio
import networkx as nx
import random
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict
from .base_agent import BaseAgent
from ...knowledge_graph.kg_builder import KGBuilder


class ThoughtNode:
    """Represents a thought (KG entity or facet) in the reasoning graph"""

    def __init__(
        self,
        thought_id: str,
        content: str,
        thought_type: str = "kg_entity",  # kg_entity, facet, bridge
        score: float = 0.8,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.thought_id = thought_id
        self.content = content
        self.thought_type = thought_type
        self.score = score
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thought_id": self.thought_id,
            "content": self.content,
            "thought_type": self.thought_type,
            "score": self.score,
            "metadata": self.metadata,
        }


class GraphOfThought(BaseAgent):
    """
    Path-based reasoning sampler.

    Builds a thought graph from KG and samples diverse reasoning paths
    to ground downstream idea generation.
    """

    def __init__(
        self,
        llm_interface=None,
        llm_config: Dict = None,
        config: Dict = None,
        knowledge_graph: Optional[KGBuilder] = None,
    ):
        super().__init__(llm_interface, llm_config)
        self.config = config or {}
        self.knowledge_graph = knowledge_graph
        self.thought_graph = nx.DiGraph()
        self.completed_paths = []
        self.got_config = self.config.get('graph_of_thought', {})

        # Use global logging level to control debug export
        self.debug = self.config.get('logging', {}).get('level', 'INFO') == 'DEBUG'

        # Setup debug logging directory - use unified logs/kg/ structure
        if self.debug:
            self.kg_logs_dir = Path("logs/kg")
            self.kg_logs_dir.mkdir(parents=True, exist_ok=True)

    def reset_graph(self):
        """Reset thought graph"""
        self.thought_graph = nx.DiGraph()
        self.completed_paths = []

    def set_knowledge_graph(self, kg: Optional[KGBuilder]):
        """Set knowledge graph reference"""
        self.knowledge_graph = kg

    async def build_graph_of_thoughts(
        self,
        seed_topic: str,
        facets: Dict[str, str],
        knowledge_graph: Optional[KGBuilder] = None,
    ) -> Dict[str, Any]:
        """
        Build thought graph and sample reasoning paths.

        Process:
        1. Initialize thoughts from KG entities + facets
        2. (Optional) Add minimal bridge connections
        3. Score paths
        4. Sample diverse paths asynchronously

        Args:
            seed_topic: Research topic
            facets: Decomposed facets
            knowledge_graph: Pre-built KG

        Returns:
            Dict with paths, metadata
        """
        if knowledge_graph:
            self.set_knowledge_graph(knowledge_graph)

        self.reset_graph()

        max_paths = self.got_config.get('max_paths', 100)

        print(f"Building Graph-of-Thought for path sampling (from pre-built KG)")

        # Step 1: Initialize from KG + facets
        await self._initialize_thoughts_from_kg(seed_topic, facets)
        print(f"Sampled {len(self.thought_graph.nodes)} nodes from KG (+ facets), preserving {len(self.thought_graph.edges)} existing KG edges")

        # Step 2: Add minimal bridge connections (if graph is disconnected)
        await self._add_bridge_connections()
        print(f"After bridging: {len(self.thought_graph.edges)} edges")

        # Step 3: Sample paths asynchronously
        paths = await self._sample_paths_async(max_paths)
        print(f"Sampled {len(paths)} reasoning paths")

        # Step 4: Score and filter paths
        scored_paths = self._score_and_filter_paths(paths)
        print(f"Retained {len(scored_paths)} high-quality paths")

        stats = {
            "num_thoughts": len(self.thought_graph.nodes),
            "num_edges": len(self.thought_graph.edges),
            "num_paths": len(scored_paths),
            "avg_path_length": sum(len(p['nodes']) for p in scored_paths) / len(scored_paths) if scored_paths else 0,
            "avg_path_score": sum(p['score'] for p in scored_paths) / len(scored_paths) if scored_paths else 0,
        }

        # Step 5: Export debug logs if enabled
        self._export_debug_logs(seed_topic, scored_paths, stats)

        return {
            "paths": scored_paths,
            "thought_graph": self.thought_graph,
            "facet_nodes": list(facets.keys()),
            "cross_connections": self.knowledge_graph.get_cross_cluster_connections() if self.knowledge_graph else [],
            "graph_summary": stats,
        }

    # ==================== Initialization ====================

    async def _initialize_thoughts_from_kg(self, seed_topic: str, facets: Dict[str, str]):
        """
        Initialize thought graph by SAMPLING from the pre-built KG.

        Strategy (NOT creating a new subgraph):
        1. Add facets as thought nodes
        2. Sample top-K high-degree KG entities (hub nodes)
        3. Import existing edges BETWEEN these sampled entities from KG

        This preserves KG structure - we don't generate new edges, we use what's already there.
        """
        if not self.knowledge_graph or not self.knowledge_graph.knowledge_graph:
            # Fallback: only facets
            for facet_name, facet_content in facets.items():
                thought_id = f"facet_{facet_name.lower().replace(' ', '_')}"
                self.thought_graph.add_node(
                    thought_id,
                    thought=ThoughtNode(
                        thought_id=thought_id,
                        content=f"{facet_name}: {facet_content}",
                        thought_type="facet",
                        score=1.0,
                        metadata={"source": "facet", "facet_name": facet_name}
                    )
                )
            return

        kg = self.knowledge_graph.knowledge_graph

        # 1. Add facet thoughts
        for facet_name, facet_content in facets.items():
            thought_id = f"facet_{facet_name.lower().replace(' ', '_')}"
            self.thought_graph.add_node(
                thought_id,
                thought=ThoughtNode(
                    thought_id=thought_id,
                    content=f"{facet_name}: {facet_content}",
                    thought_type="facet",
                    score=1.0,
                    metadata={"source": "facet", "facet_name": facet_name}
                )
            )

        # 2. Import KG entities (high-degree nodes)
        max_kg_entities = self.got_config.get('max_kg_entities', 20)
        node_degrees = dict(kg.degree())
        sorted_entities = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)

        added_entities = []
        for entity, degree in sorted_entities[:max_kg_entities]:
            entity_attrs = kg.nodes[entity]
            max_neighbors = self.got_config.get('max_neighbors_per_entity', 5)
            neighbors = list(kg.neighbors(entity))[:max_neighbors]

            content = self._format_entity_as_thought(entity, entity_attrs, neighbors, kg)
            thought_id = f"kg_{entity.replace(' ', '_').replace(':', '_').replace('/', '_')}"

            self.thought_graph.add_node(
                thought_id,
                thought=ThoughtNode(
                    thought_id=thought_id,
                    content=content,
                    thought_type="kg_entity",
                    score=min(1.0, 0.5 + (degree / max(node_degrees.values())) * 0.5),  # Score based on degree
                    metadata={"source": "knowledge_graph", "entity": entity, "degree": degree}
                )
            )
            added_entities.append((thought_id, entity))

        # 3. Import KG edges
        for thought_id, entity in added_entities:
            for neighbor in kg.neighbors(entity):
                neighbor_id = f"kg_{neighbor.replace(' ', '_').replace(':', '_').replace('/', '_')}"
                if neighbor_id in self.thought_graph:
                    relationship = kg[entity][neighbor].get('relationship', 'related_to')
                    self.thought_graph.add_edge(
                        thought_id,
                        neighbor_id,
                        relationship=relationship,
                        edge_type="kg_relationship",
                        weight=1.0
                    )

    def _format_entity_as_thought(self, entity: str, attrs: Dict, neighbors: List[str], kg: nx.Graph) -> str:
        """Format KG entity as thought"""
        entity_type = attrs.get('type', 'concept')
        relationships = []
        max_rels = self.got_config.get('max_relationships_display', 3)
        for neighbor in neighbors[:max_rels]:
            rel = kg[entity][neighbor].get('relationship', 'related_to')
            relationships.append(f"{rel} '{neighbor}'")

        thought = f"{entity} ({entity_type})"
        if relationships:
            thought += f" | {'; '.join(relationships)}"
        return thought

    # ==================== Bridge Connections ====================

    async def _add_bridge_connections(self):
        """
        Add minimal cross-connections to bridge disconnected graph components.
        Uses LLM sparingly to connect facets with KG entities.
        """
        # Check if graph is disconnected
        undirected = self.thought_graph.to_undirected()
        components = list(nx.connected_components(undirected))

        if len(components) <= 1:
            # Already connected
            return

        print(f"Graph has {len(components)} disconnected components, adding bridges...")

        # Get facet nodes and KG nodes
        facet_nodes = [n for n in self.thought_graph.nodes
                      if self.thought_graph.nodes[n]['thought'].thought_type == 'facet']
        kg_nodes = [n for n in self.thought_graph.nodes
                   if self.thought_graph.nodes[n]['thought'].thought_type == 'kg_entity']

        # Strategy: Connect each facet to top-3 relevant KG nodes
        max_bridges_per_facet = self.got_config.get('max_bridges_per_facet', 3)

        for facet_id in facet_nodes:
            facet_content = self.thought_graph.nodes[facet_id]['thought'].content.lower()

            # Find KG nodes with semantic overlap
            candidates = []
            for kg_id in kg_nodes:
                kg_entity = self.thought_graph.nodes[kg_id]['thought'].metadata.get('entity', '')
                # Simple keyword matching
                if kg_entity.lower() in facet_content or any(
                    word in kg_entity.lower()
                    for word in facet_content.split()
                    if len(word) > 4
                ):
                    candidates.append(kg_id)

            # Add bridges
            bridge_weight = self.got_config.get('bridge_connection_weight', 0.7)
            for kg_id in candidates[:max_bridges_per_facet]:
                if not self.thought_graph.has_edge(facet_id, kg_id):
                    self.thought_graph.add_edge(
                        facet_id,
                        kg_id,
                        relationship="grounds_in",
                        edge_type="bridge",
                        weight=bridge_weight
                    )

    # ==================== Path Sampling ====================

    async def _sample_paths_async(self, max_paths: int) -> List[Dict[str, Any]]:
        """
        Sample paths asynchronously from multiple starting nodes.
        Uses DFS with degree-based branching.
        """
        # Get starting nodes (facets + high-degree KG nodes)
        facet_nodes = [n for n in self.thought_graph.nodes
                      if self.thought_graph.nodes[n]['thought'].thought_type == 'facet']

        node_degrees = dict(self.thought_graph.degree())
        kg_nodes_sorted = sorted(
            [n for n in self.thought_graph.nodes
             if self.thought_graph.nodes[n]['thought'].thought_type == 'kg_entity'],
            key=lambda n: node_degrees.get(n, 0),
            reverse=True
        )

        max_starting = self.got_config.get('max_starting_nodes', 10)
        starting_nodes = facet_nodes + kg_nodes_sorted[:max_starting]

        # Sample paths concurrently
        max_depth = self.got_config.get('max_path_length', 5)
        branching_factor = self.got_config.get('path_branching_factor', 3)
        paths_per_start = max(1, max_paths // len(starting_nodes)) if starting_nodes else 1

        tasks = [
            self._sample_paths_from_node(start, max_depth, branching_factor, paths_per_start)
            for start in starting_nodes
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten paths
        all_paths = []
        for result in results:
            if isinstance(result, list):
                all_paths.extend(result)

        # Deduplicate
        unique_paths = []
        seen_signatures = set()

        for path in all_paths:
            sig = tuple(path['nodes'])
            if sig not in seen_signatures:
                unique_paths.append(path)
                seen_signatures.add(sig)

        return unique_paths[:max_paths]

    async def _sample_paths_from_node(
        self,
        start_node: str,
        max_depth: int,
        branching_factor: int,
        max_paths_from_this_node: int
    ) -> List[Dict[str, Any]]:
        """Sample paths from a single starting node using DFS"""
        paths = []
        visited_paths = set()
        min_path_length = 2  # Minimum path length to record

        def dfs(current: str, path_nodes: List[str], path_edges: List[Dict], depth: int):
            # Record path if it meets minimum length (do this at every step, not just at leaves)
            if len(path_nodes) >= min_path_length:
                path_sig = tuple(path_nodes)
                if path_sig not in visited_paths and len(paths) < max_paths_from_this_node:
                    paths.append({
                        "nodes": path_nodes.copy(),
                        "edges": path_edges.copy(),
                        "score": 0.0,  # Will be scored later
                        "path_string": ""  # Will be formatted later
                    })
                    visited_paths.add(path_sig)

            # Stop if reached max depth or enough paths
            if depth >= max_depth or len(paths) >= max_paths_from_this_node:
                return

            # Get successors
            successors = list(self.thought_graph.successors(current))
            if not successors:
                return  # Leaf node, already recorded above

            # Select successors by degree (prefer high-degree nodes)
            node_degrees = dict(self.thought_graph.degree())
            successors_sorted = sorted(successors, key=lambda n: node_degrees.get(n, 0), reverse=True)
            selected = successors_sorted[:branching_factor]

            # Explore
            for next_node in selected:
                if next_node not in path_nodes:  # Avoid cycles
                    edge_data = self.thought_graph[current][next_node]
                    edge_record = {
                        "from": current,
                        "to": next_node,
                        "relation": edge_data.get('relationship', 'connects'),
                        "edge_type": edge_data.get('edge_type', 'unknown'),
                        "weight": edge_data.get('weight', 1.0)
                    }

                    dfs(next_node, path_nodes + [next_node], path_edges + [edge_record], depth + 1)

        # Start DFS
        dfs(start_node, [start_node], [], 0)

        return paths

    # ==================== Path Scoring and Filtering ====================

    def _score_and_filter_paths(self, paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score paths based on:
        1. Node quality (average thought scores)
        2. Path diversity (edge type variety)
        3. Length (prefer moderate lengths)
        """
        min_score = self.got_config.get('min_path_score', 0.5)

        # Get scoring weights from config
        node_quality_weight = self.got_config.get('node_quality_weight', 0.6)
        edge_diversity_weight = self.got_config.get('edge_diversity_weight', 0.2)
        path_length_weight = self.got_config.get('path_length_weight', 0.2)

        scored_paths = []
        for path in paths:
            # 1. Node quality score
            node_scores = [
                self.thought_graph.nodes[nid]['thought'].score
                for nid in path['nodes']
            ]
            avg_node_score = sum(node_scores) / len(node_scores) if node_scores else 0.5

            # 2. Edge diversity score
            edge_types = set(e['edge_type'] for e in path['edges'])
            diversity_score = min(1.0, len(edge_types) / 3.0)  # Normalize

            # 3. Length score (prefer 3-5 nodes)
            length = len(path['nodes'])
            if 3 <= length <= 5:
                length_score = 1.0
            elif length < 3:
                length_score = length / 3.0
            else:
                length_score = max(0.5, 1.0 - (length - 5) * 0.1)

            # Combined score with configurable weights
            path_score = (
                node_quality_weight * avg_node_score +
                edge_diversity_weight * diversity_score +
                path_length_weight * length_score
            )

            path['score'] = path_score
            path['path_string'] = self._format_path_string(path['nodes'], path['edges'])

            if path_score >= min_score:
                scored_paths.append(path)

        # Sort by score
        scored_paths.sort(key=lambda p: p['score'], reverse=True)

        return scored_paths

    def _format_path_string(self, nodes: List[str], edges: List[Dict]) -> str:
        """Format path as readable string"""
        if not edges:
            return nodes[0] if nodes else ""

        segments = []
        for i, node_id in enumerate(nodes):
            thought = self.thought_graph.nodes[node_id]['thought']
            content = thought.content[:60] + "..." if len(thought.content) > 60 else thought.content
            segments.append(f"[{thought.thought_type}] {content}")

            if i < len(edges):
                segments.append(f" --{edges[i]['relation']}--> ")

        return "".join(segments)

    # ==================== Utilities ====================

    def get_graph_summary(self) -> Dict[str, Any]:
        """Get graph statistics"""
        thought_types = defaultdict(int)
        for nid in self.thought_graph.nodes:
            t_type = self.thought_graph.nodes[nid]['thought'].thought_type
            thought_types[t_type] += 1

        edge_types = defaultdict(int)
        for u, v, data in self.thought_graph.edges(data=True):
            e_type = data.get('edge_type', 'unknown')
            edge_types[e_type] += 1

        return {
            "num_thoughts": len(self.thought_graph.nodes),
            "num_edges": len(self.thought_graph.edges),
            "thought_types": dict(thought_types),
            "edge_types": dict(edge_types),
        }

    # ==================== Debug Logging ====================

    def _export_debug_logs(self, seed_topic: str, paths: List[Dict], stats: Dict):
        """Export GoT debug logs to logs/kg/ directory"""
        # Early return if debug not enabled
        if not self.debug:
            return

        # Ensure directory exists
        if not hasattr(self, 'kg_logs_dir') or self.kg_logs_dir is None:
            self.kg_logs_dir = Path("logs/kg")
            self.kg_logs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export 1: Paths to JSONL (one path per line) - timestamp only
        paths_file = self.kg_logs_dir / f"{timestamp}_paths.jsonl"
        with open(paths_file, 'w', encoding='utf-8') as f:
            for path in paths:
                json.dump(path, f, ensure_ascii=False)
                f.write('\n')

        # Export 2: KG info to JSON - timestamp only
        kg_file = self.kg_logs_dir / f"{timestamp}_kg.json"
        kg_info = self._serialize_kg_info(stats)
        with open(kg_file, 'w', encoding='utf-8') as f:
            json.dump(kg_info, f, ensure_ascii=False, indent=2)

    def _serialize_kg_info(self, stats: Dict) -> Dict[str, Any]:
        """Serialize KG information for export"""
        # Extract all nodes
        nodes_info = []
        for node_id in self.thought_graph.nodes:
            thought = self.thought_graph.nodes[node_id]['thought']
            nodes_info.append({
                "id": node_id,
                "content": thought.content,
                "type": thought.thought_type,
                "score": thought.score,
                "metadata": thought.metadata,
            })

        # Extract all edges
        edges_info = []
        for u, v, data in self.thought_graph.edges(data=True):
            edges_info.append({
                "from": u,
                "to": v,
                "relation": data.get('relationship', 'unknown'),
                "edge_type": data.get('edge_type', 'unknown'),
                "weight": data.get('weight', 1.0),
            })

        # Cross-connections from KG
        cross_connections = []
        if self.knowledge_graph:
            try:
                cross_conn = self.knowledge_graph.get_cross_cluster_connections()
                cross_connections = [
                    {"entity1": e1, "relation": rel, "entity2": e2}
                    for e1, rel, e2 in cross_conn
                ]
            except Exception as e:
                cross_connections = []

        max_cross_conn = self.got_config.get('max_cross_connections_logged', 20)

        return {
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "nodes": nodes_info,
            "edges": edges_info,
            "cross_connections": cross_connections[:max_cross_conn],
        }
