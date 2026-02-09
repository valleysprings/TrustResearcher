"""
Graph Operations Prompts

Contains prompts for knowledge graph construction and entity extraction operations.
These prompts support the idea generation process through structured knowledge representation.
"""

# ============================================================================
# Entity and Relationship Extraction Prompts
# ============================================================================

ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are an AI system specialized in knowledge extraction from scientific text.
Extract key entities (concepts, methods, techniques, datasets, metrics, etc.) and their relationships."""

ENTITY_EXTRACTION_USER_PROMPT = """Text to analyze: {text}

Please extract:
1. Key entities (concepts, methods, techniques, datasets, metrics, authors, institutions)
2. Relationships between entities (uses, improves, compares, based_on, etc.)

Format your response as:
ENTITIES:
- Entity1 (type: concept/method/dataset/etc.)
- Entity2 (type: ...)

RELATIONSHIPS:
- Entity1 -> uses -> Entity2
- Entity3 -> improves -> Entity1

Focus on entities and relationships that would be valuable for research idea generation."""


# ============================================================================
# Knowledge Graph Construction Prompts
# ============================================================================

# Core concept extraction prompt
KG_CORE_CONCEPTS_PROMPT = """Research topic: {seed_topic}

Extract the fundamental concepts, key methodologies, and core theories for this research area.
Focus on:
1. Core theoretical concepts and frameworks
2. Primary methodologies and techniques
3. Essential datasets and benchmarks
4. Key evaluation metrics
5. Foundational relationships between concepts

Provide comprehensive knowledge about the foundational aspects of this field."""

# Knowledge graph expansion prompt
KG_EXPANSION_PROMPT = """Research topic: {seed_topic}
Current knowledge graph contains: {current_entities}

Expand the knowledge graph with:
1. Related concepts from adjacent fields
2. Emerging trends and recent developments
3. Cross-disciplinary connections
4. Alternative methodologies and approaches
5. Historical context and evolution

Focus on concepts that complement the existing knowledge graph."""

# Relationship enhancement prompt
KG_RELATIONSHIP_PROMPT = """Knowledge graph entities: {entities}

Analyze these entities and identify missing relationships between them.
Focus on:
1. Semantic relationships (uses, implements, extends)
2. Temporal relationships (precedes, follows, builds_upon)
3. Hierarchical relationships (part_of, category_of, instance_of)
4. Functional relationships (enables, requires, optimizes)
5. Contextual relationships (applies_to, relevant_for, contrasts_with)

Extract only the relationships between these existing entities."""

# Methodology extraction system prompt
KG_METHODOLOGY_SYSTEM_PROMPT = "You are an expert at identifying technical methodologies and algorithms from research abstracts."

# Methodology extraction user prompt
KG_METHODOLOGY_USER_PROMPT = """Extract methodological approaches, algorithms, and techniques mentioned in this research abstract:

Abstract: {abstract}

Return only the specific methodological terms (e.g., "neural networks", "reinforcement learning", "optimization algorithm") as a comma-separated list. Focus on technical methods, not general concepts."""

# Entity expansion prompt
KG_ENTITY_EXPANSION_PROMPT = """
Given the research entity '{entity}', provide detailed information about:
1. Related concepts and subconcepts
2. Associated methods and techniques
3. Relevant datasets and metrics
4. Connected research areas
5. Key relationships and dependencies

Focus on information that would be useful for research idea generation.
"""
