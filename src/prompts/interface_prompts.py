"""
Prompts for the LLM Interface

This module contains prompts used by the LLMInterface for entity extraction tasks.
"""

# Entity and Relationship Extraction Prompts
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