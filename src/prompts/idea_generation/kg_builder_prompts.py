"""
Prompt templates for Knowledge Graph Builder operations.
Contains prompts for entity extraction, relationship discovery, and knowledge graph construction.
"""

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