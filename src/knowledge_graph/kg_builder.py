import networkx as nx
from typing import Dict, List, Tuple, Set
from ..utils.llm_interface import LLMInterface
from ..prompts.idea_generation.kg_builder_prompts import (
    KG_CORE_CONCEPTS_PROMPT, KG_EXPANSION_PROMPT, KG_RELATIONSHIP_PROMPT,
    KG_METHODOLOGY_SYSTEM_PROMPT, KG_METHODOLOGY_USER_PROMPT, KG_ENTITY_EXPANSION_PROMPT
)
import json
import re
import random


class KGBuilder:
    def __init__(self, config: Dict = None, llm_config: Dict = None, logger=None):
        self.config = config or {}
        self.knowledge_graph = nx.Graph()
        self.entity_types = {}
        self.logger = logger
        self.llm = LLMInterface(config=llm_config, logger=logger)

    def add_entity(self, entity: str, entity_type: str = "concept"):
        """Add an entity to the knowledge graph with type information"""
        if entity not in self.knowledge_graph:
            self.knowledge_graph.add_node(entity, type=entity_type)
            self.entity_types[entity] = entity_type

    def add_relationship(self, entity1: str, entity2: str, relationship: str, weight: float = 1.0):
        """Add a relationship between two entities"""
        if entity1 in self.knowledge_graph.nodes and entity2 in self.knowledge_graph.nodes:
            self.knowledge_graph.add_edge(entity1, entity2, 
                                        relationship=relationship, 
                                        weight=weight)

    def get_entities(self) -> List[str]:
        """Get all entities in the knowledge graph"""
        return list(self.knowledge_graph.nodes())

    def get_relationships(self, entity: str) -> List[Tuple[str, str, str]]:
        """Get all relationships for a given entity"""
        relationships = []
        if entity in self.knowledge_graph:
            for neighbor in self.knowledge_graph.neighbors(entity):
                edge_data = self.knowledge_graph[entity][neighbor]
                relationship = edge_data.get('relationship', 'related_to')
                relationships.append((neighbor, relationship, edge_data.get('weight', 1.0)))
        return relationships

    async def build(self, seed_topic: str = None, max_papers: int = 10) -> nx.Graph:
        """Build knowledge graph from seed topic using LLM knowledge"""
        if seed_topic:
            # Build from seed topic using LLM knowledge
            await self.build_from_seed_topic(seed_topic)
        
        return self.knowledge_graph

    async def build_from_seed_topic(self, seed_topic: str, literature_papers: List = None):
        """Build knowledge graph iteratively using multi-stage approach"""
        print(f"Starting iterative knowledge graph construction for: {seed_topic}")

        # Stage 1: Core domain concepts
        await self._build_core_concepts(seed_topic)

        # Stage 2: Literature integration
        if literature_papers:
            await self._integrate_literature_iteratively(literature_papers)
        
        # Stage 3: Expand with related concepts
        await self._expand_with_related_concepts(seed_topic)
        
        # Stage 4: Cross-connections and relationships
        await self._enhance_relationships()
        
        print(f"Iterative KG construction complete. Total entities: {len(self.knowledge_graph.nodes)}, edges: {len(self.knowledge_graph.edges)}")

    async def _build_core_concepts(self, seed_topic: str):
        """Stage 1: Build core domain concepts"""
        print("Stage 1: Building core domain concepts...")
        
        core_prompt = KG_CORE_CONCEPTS_PROMPT.format(seed_topic=seed_topic)
        
        initial_context = await self.llm.extract_entities_and_relationships(core_prompt)
        self.parse_and_add_extracted_knowledge(initial_context)

    async def _integrate_literature_iteratively(self, literature_papers: List):
        """Stage 2: Integrate literature papers iteratively"""
        print("Stage 2: Integrating literature iteratively...")
        
        # Process papers in small batches
        batch_size = self.config.get('paper_batch_size', 3)  # Use config or default to 3
        for i in range(0, len(literature_papers), batch_size):
            batch = literature_papers[i:i+batch_size]
            await self._process_literature_batch(batch, i // batch_size + 1)
        
        # Final integration pass
        await self.integrate_literature_entities(literature_papers)

    async def _process_literature_batch(self, papers: List, batch_num: int):
        """Process a batch of literature papers"""
        print(f"Processing literature batch {batch_num} ({len(papers)} papers)...")
        
        batch_context = []
        batch_context.append(f"Literature batch {batch_num}:")
        
        for paper in papers:
            paper_summary = f"- {paper.title} ({paper.year})"
            if paper.abstract:
                paper_summary += f": {paper.abstract[:300]}..."
            if paper.keywords:
                paper_summary += f"\n  Keywords: {', '.join(paper.keywords[:5])}"
            batch_context.append(paper_summary)
        
        batch_context.append("\nExtract additional entities and relationships from this literature batch.")
        batch_context.append("Focus on novel concepts, methodologies, and relationships not yet captured.")
        
        batch_prompt = "\n".join(batch_context)
        batch_knowledge = await self.llm.extract_entities_and_relationships(batch_prompt)
        self.parse_and_add_extracted_knowledge(batch_knowledge)

    async def _expand_with_related_concepts(self, seed_topic: str):
        """Stage 3: Expand with related and adjacent concepts"""
        print("Stage 3: Expanding with related concepts...")
        
        current_entities = self._select_entities_for_expansion()
        
        expansion_prompt = KG_EXPANSION_PROMPT.format(
            seed_topic=seed_topic, 
            current_entities=', '.join(current_entities)
        )
        
        expanded_knowledge = await self.llm.extract_entities_and_relationships(expansion_prompt)
        self.parse_and_add_extracted_knowledge(expanded_knowledge)

    async def _enhance_relationships(self):
        """Stage 4: Enhance relationships between existing entities"""
        print("Stage 4: Enhancing entity relationships...")
        
        entities = list(self.knowledge_graph.nodes())
        if len(entities) < 2:
            return

        cfg = self.config.get('relationship_enhancement', {})
        iterations = max(1, int(cfg.get('iterations', 1)))
        sampled_batches = set()

        for _ in range(iterations):
            sample_entities = self._sample_entities_for_relationships(entities, cfg, sampled_batches)
            if not sample_entities:
                break

            relationship_prompt = KG_RELATIONSHIP_PROMPT.format(entities=', '.join(sample_entities))

        relationship_knowledge = await self.llm.extract_entities_and_relationships(relationship_prompt)
        self.parse_and_add_extracted_knowledge(relationship_knowledge)

    def _select_entities_for_expansion(self) -> List[str]:
        """Choose seed entities for the expansion stage using configured heuristics."""
        nodes = list(self.knowledge_graph.nodes())
        if not nodes:
            return []

        cfg = self.config.get('expansion', {})
        sample_size = max(1, min(int(cfg.get('seed_sample_size', 10)), len(nodes)))
        strategy = cfg.get('sampling_strategy', 'degree').lower()

        if strategy == 'random':
            return random.sample(nodes, sample_size)

        degrees = dict(self.knowledge_graph.degree(nodes))
        sorted_nodes = sorted(degrees.items(), key=lambda item: item[1], reverse=True)
        return [node for node, _ in sorted_nodes[:sample_size]]

    def _sample_entities_for_relationships(self, entities: List[str], cfg: Dict, sampled_batches: Set[Tuple[str, ...]]) -> List[str]:
        """Select a batch of entities for relationship probing using configured heuristics."""
        if not entities:
            return []

        sample_size = max(2, min(int(cfg.get('sample_size', 20)), len(entities)))
        strategy = cfg.get('sampling_strategy', 'degree').lower()
        top_ratio = min(1.0, max(0.0, cfg.get('top_degree_ratio', 0.6)))
        random_ratio = min(1.0, max(0.0, cfg.get('random_ratio', 0.4)))

        degrees = dict(self.knowledge_graph.degree(entities))
        sorted_by_degree = sorted(degrees.items(), key=lambda item: item[1], reverse=True)

        if strategy == 'random':
            candidate_batch = random.sample(entities, sample_size)
        else:
            top_count = max(1, int(sample_size * top_ratio)) if sorted_by_degree else 0
            top_entities = [node for node, _ in sorted_by_degree[:top_count]]

            if strategy == 'degree':
                candidate_batch = top_entities[:sample_size]
            else:  # hybrid or unspecified
                remaining_entities = [node for node in entities if node not in top_entities]
                max_random_candidates = max(0, sample_size - len(top_entities))
                if random_ratio <= 0 or max_random_candidates == 0:
                    random_entities = []
                else:
                    random_count = min(max_random_candidates, max(1, int(sample_size * random_ratio)))
                    random_entities = random.sample(remaining_entities, min(random_count, len(remaining_entities)))

                candidate_batch = top_entities + [node for node in random_entities if node not in top_entities]
                candidate_batch = candidate_batch[:sample_size]

        if len(candidate_batch) < 2:
            return []

        batch_signature = tuple(sorted(candidate_batch))
        attempt = 0
        max_attempts = cfg.get('max_sampling_attempts', 3)
        while batch_signature in sampled_batches and attempt < max_attempts:
            random.shuffle(candidate_batch)
            batch_signature = tuple(sorted(candidate_batch))
            attempt += 1

        if batch_signature in sampled_batches:
            return []

        sampled_batches.add(batch_signature)
        return candidate_batch

    async def integrate_literature_entities(self, literature_papers: List):
        """Integrate entities and relationships from literature papers into the knowledge graph"""
        if not literature_papers:
            return
        
        # Process each paper to extract domain-specific entities
        for paper in literature_papers: 
            # Add paper as an entity
            paper_entity = f"Paper: {paper.title}"
            self.add_entity(paper_entity, "paper")
            
            # Add authors as entities
            for author in paper.authors[:3]:
                author_entity = f"Author: {author}"
                self.add_entity(author_entity, "researcher")
                self.add_relationship(paper_entity, author_entity, "authored_by")
            
            # Add keywords as concept entities  
            for keyword in paper.keywords:  
                keyword_entity = keyword.lower()
                self.add_entity(keyword_entity, "concept")
                self.add_relationship(paper_entity, keyword_entity, "studies")
            
            # Extract methodological entities from abstract
            if paper.abstract:
                methodologies = await self.extract_methodologies_from_abstract(paper.abstract)
                for methodology in methodologies:
                    method_entity = methodology.lower()
                    self.add_entity(method_entity, "methodology")
                    self.add_relationship(paper_entity, method_entity, "uses_method")
            
            # Add year as temporal entity
            if paper.year:
                year_entity = f"Year: {paper.year}"
                self.add_entity(year_entity, "temporal")
                self.add_relationship(paper_entity, year_entity, "published_in")
        
        # Create cross-paper relationships based on shared concepts
        self.create_literature_cross_connections(literature_papers)

    async def extract_methodologies_from_abstract(self, abstract: str) -> List[str]:
        """Extract methodological terms from paper abstract using LLM"""
        methodology_prompt = KG_METHODOLOGY_USER_PROMPT.format(abstract=abstract)
        
        try:
            response = await self.llm.generate_with_system_prompt(
                KG_METHODOLOGY_SYSTEM_PROMPT,
                methodology_prompt,
                max_tokens=100,
                caller="kg_builder_methodology_extraction"
            )
            
            # Parse methodologies
            methodologies = [method.strip().lower() for method in response.split(',')]
            return [method for method in methodologies if len(method) > 3]
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error extracting methodologies: {e}", "kg_builder")
            return []

    def create_literature_cross_connections(self, literature_papers: List):
        """Create connections between papers based on shared concepts and methodologies"""
        # Find shared keywords between papers
        for i, paper1 in enumerate(literature_papers):
            for paper2 in literature_papers:  # Limit combinations
                shared_keywords = set(paper1.keywords or []) & set(paper2.keywords or [])
                
                if shared_keywords:
                    paper1_entity = f"Paper: {paper1.title}"
                    paper2_entity = f"Paper: {paper2.title}"
                    
                    if paper1_entity in self.knowledge_graph and paper2_entity in self.knowledge_graph:
                        # Create relationship based on shared concepts
                        shared_concepts = list(shared_keywords)
                        relationship = f"shares_concepts: {', '.join(shared_concepts)}"
                        default_weight = self.config.get('knowledge_graph', {}).get('default_relationship_weight', 0.8)
                        self.add_relationship(paper1_entity, paper2_entity, relationship, weight=default_weight)

    async def build_from_documents(self, documents: List[str]):
        """Build knowledge graph from a list of documents"""
        for doc in documents:
            entities_and_relationships = await self.extract_entities_and_relationships(doc)
            self.parse_and_add_extracted_knowledge(entities_and_relationships)

    async def extract_entities_and_relationships(self, document: str) -> str:
        """Extract entities and relationships from a document using LLM"""
        return await self.llm.extract_entities_and_relationships(document)

    def parse_and_add_extracted_knowledge(self, extracted_text: str):
        """Parse LLM-extracted entities and relationships and add to graph"""
        try:
            # Parse entities section
            entities_match = re.search(r'ENTITIES:(.*?)(?=RELATIONSHIPS:|$)', extracted_text, re.DOTALL)
            if entities_match:
                entities_text = entities_match.group(1)
                entities = self.parse_entities(entities_text)
                for entity, entity_type in entities:
                    self.add_entity(entity, entity_type)

            # Parse relationships section
            relationships_match = re.search(r'RELATIONSHIPS:(.*?)$', extracted_text, re.DOTALL)
            if relationships_match:
                relationships_text = relationships_match.group(1)
                relationships = self.parse_relationships(relationships_text)
                for entity1, relationship, entity2 in relationships:
                    # Ensure entities exist before adding relationship
                    if entity1 not in self.knowledge_graph:
                        self.add_entity(entity1, "concept")
                    if entity2 not in self.knowledge_graph:
                        self.add_entity(entity2, "concept")
                    self.add_relationship(entity1, entity2, relationship)
                    
        except Exception as e:
            print(f"Error parsing extracted knowledge: {e}")

    def parse_entities(self, entities_text: str) -> List[Tuple[str, str]]:
        """Parse entities from extracted text"""
        entities = []
        lines = entities_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•'):
                # Remove bullet point and parse entity (name and type)
                entity_info = line[1:].strip()
                if '(type:' in entity_info:
                    match = re.match(r'(.*?)\s*\(type:\s*(.*?)\)', entity_info)
                    if match:
                        entity_name = match.group(1).strip()
                        entity_type = match.group(2).strip()
                        entities.append((entity_name, entity_type))
                else:
                    # Default type if not specified
                    entities.append((entity_info, "concept"))
        return entities

    def parse_relationships(self, relationships_text: str) -> List[Tuple[str, str, str]]:
        """Parse relationships from extracted text"""
        relationships = []
        lines = relationships_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•'):
                # Parse relationship: Entity1 -> relationship -> Entity2
                rel_info = line[1:].strip()
                if '->' in rel_info:
                    parts = rel_info.split('->')
                    if len(parts) >= 3:
                        entity1 = parts[0].strip()
                        relationship = parts[1].strip()
                        entity2 = parts[2].strip()
                        relationships.append((entity1, relationship, entity2))
        return relationships

    def get_related_entities(self, entity: str, max_distance: int = 2) -> Set[str]:
        """Get entities related to a given entity within max_distance hops"""
        if entity not in self.knowledge_graph:
            return set()
        
        related = set()
        current_level = {entity}
        visited = set()
        
        for distance in range(max_distance):
            next_level = set()
            for current_entity in current_level:
                if current_entity not in visited:
                    visited.add(current_entity)
                    neighbors = set(self.knowledge_graph.neighbors(current_entity))
                    next_level.update(neighbors - visited)
                    related.update(neighbors)
            current_level = next_level
            if not current_level:
                break
                
        return related

    def get_entity_cluster(self, entity: str) -> List[str]:
        """Get entities in the same cluster/community as the given entity"""
        try:
            import networkx.algorithms.community as nx_comm
            communities = list(nx_comm.greedy_modularity_communities(self.knowledge_graph))
            for community in communities:
                if entity in community:
                    return list(community)
        except ImportError:
            # Fallback to simple connected component
            if entity in self.knowledge_graph:
                return list(nx.connected_component(self.knowledge_graph, entity))
        return [entity]

    def get_cross_cluster_connections(self) -> List[Tuple[str, str, str]]:
        """Get connections that bridge different clusters for cross-pollination"""
        try:
            import networkx.algorithms.community as nx_comm
            communities = list(nx_comm.greedy_modularity_communities(self.knowledge_graph))
            cross_connections = []
            
            for i, community1 in enumerate(communities):
                for j, community2 in enumerate(communities[i+1:], i+1):
                    for entity1 in community1:
                        for entity2 in community2:
                            if self.knowledge_graph.has_edge(entity1, entity2):
                                relationship = self.knowledge_graph[entity1][entity2].get('relationship', 'connected')
                                cross_connections.append((entity1, relationship, entity2))
            return cross_connections
        except ImportError:
            return []

    def expand_graph(self, entity: str, expansion_depth: int = 1):
        """Expand the knowledge graph by exploring an entity further"""
        if entity not in self.knowledge_graph:
            return
            
        # Generate more knowledge about this entity
        expansion_prompt = KG_ENTITY_EXPANSION_PROMPT.format(entity=entity)
        
        expanded_knowledge = self.llm.extract_entities_and_relationships(expansion_prompt)
        self.parse_and_add_extracted_knowledge(expanded_knowledge)

    def get_graph_summary(self) -> Dict:
        """Get a summary of the knowledge graph structure"""
        return {
            "num_entities": len(self.knowledge_graph.nodes()),
            "num_relationships": len(self.knowledge_graph.edges()),
            "entity_types": {k: list(self.entity_types.values()).count(k) for k in set(self.entity_types.values())} if self.entity_types else {},
            "connected_components": nx.number_connected_components(self.knowledge_graph),
            "average_degree": sum(dict(self.knowledge_graph.degree()).values()) / len(self.knowledge_graph.nodes()) if len(self.knowledge_graph.nodes()) > 0 else 0
        }

    def save_graph(self, filepath: str):
        """Save the knowledge graph to a file"""
        graph_data = {
            "nodes": [(node, data) for node, data in self.knowledge_graph.nodes(data=True)],
            "edges": [(u, v, data) for u, v, data in self.knowledge_graph.edges(data=True)]
        }
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2)

    def load_graph(self, filepath: str):
        """Load the knowledge graph from a file"""
        with open(filepath, 'r') as f:
            graph_data = json.load(f)
        
        self.knowledge_graph = nx.Graph()
        for node, data in graph_data["nodes"]:
            self.knowledge_graph.add_node(node, **data)
            if "type" in data:
                self.entity_types[node] = data["type"]
                
        for u, v, data in graph_data["edges"]:
            self.knowledge_graph.add_edge(u, v, **data)
