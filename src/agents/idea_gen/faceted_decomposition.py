"""
Faceted Decomposition Module

Breaks down research ideas into structured facets including problem statements,
methodologies, and validation approaches for systematic analysis and refinement.
"""

import re
from typing import Dict, List
from .base_agent import BaseAgent
from ...utils.text_utils import parse_json_response, create_json_prompt_suffix
from ...prompts.idea_generation.faceted_decomposition_prompts import (
    DECOMPOSE_IDEA_SYSTEM_PROMPT, DECOMPOSE_IDEA_USER_PROMPT,
    REFINE_FACET_SYSTEM_PROMPT, REFINE_FACET_USER_PROMPT,
    RESEARCH_OUTLINE_SYSTEM_PROMPT, RESEARCH_OUTLINE_USER_PROMPT
)


class FacetedDecomposition(BaseAgent):
    def __init__(self, llm_interface=None, llm_config: Dict = None, config: Dict = None):
        super().__init__(llm_interface, llm_config)
        self.config = config or {}
        self.facets = {
            "Problem Statement": "",
            "Proposed Methodology": "",
            "Experimental Validation": ""
        }

    async def decompose_idea(self, seed_topic: str, knowledge_context: str = "") -> Dict[str, str]:
        """Decompose a research idea into structured facets using LLM with JSON format"""
        user_prompt = DECOMPOSE_IDEA_USER_PROMPT.format(
            seed_topic=seed_topic,
            knowledge_context=knowledge_context,
            json_format_suffix=create_json_prompt_suffix()
        )

        response = await self.llm.generate_with_system_prompt(DECOMPOSE_IDEA_SYSTEM_PROMPT, user_prompt, max_tokens=None, caller="faceted_decomposition")  # Use configured faceted_decomposition context window
        return self.parse_facets_response(response)

    def parse_facets_response(self, response: str) -> Dict[str, str]:
        """Parse the LLM response into structured facets using JSON-first parsing with fallback"""
        section_names = ["Problem Statement", "Proposed Methodology", "Experimental Validation"]

        # Try JSON parsing first
        parsed_data = parse_json_response(response, section_names, fallback_type="facets")

        # Handle different JSON structures
        facets = {}
        if isinstance(parsed_data, dict):
            # Check if it's already in the expected format (from fallback)
            if all(name in parsed_data for name in section_names):
                facets = parsed_data
            else:
                # Map JSON fields to facet names
                facets["Problem Statement"] = parsed_data.get("problem_statement", "")
                facets["Proposed Methodology"] = parsed_data.get("proposed_methodology", "")
                facets["Experimental Validation"] = parsed_data.get("experimental_validation", "")

                # Also handle potential topic field
                if "topic" in parsed_data:
                    facets["Topic"] = parsed_data["topic"]

                # Handle direct facet name mappings if they exist (overrides snake_case)
                for section_name in section_names:
                    if section_name in parsed_data and parsed_data[section_name]:
                        facets[section_name] = parsed_data[section_name]

        # Ensure all expected facets are present
        for section_name in section_names:
            if section_name not in facets:
                facets[section_name] = ""

        self.facets = facets
        return facets

    async def refine_facet(self, facet_name: str, refinement_context: str) -> str:
        """Refine a specific facet based on additional context or feedback"""
        if facet_name not in self.facets:
            return ""
            
        system_prompt = REFINE_FACET_SYSTEM_PROMPT.format(facet_name=facet_name)
        user_prompt = REFINE_FACET_USER_PROMPT.format(
            facet_name=facet_name,
            current_facet=self.facets[facet_name],
            refinement_context=refinement_context
        )
        
        refined_content = await self.llm.generate_with_system_prompt(
            system_prompt,
            user_prompt,
            max_tokens=self.config.get('llm', {}).get('max_tokens', 16384),
            caller="faceted_decomposition"
        )
        self.facets[facet_name] = refined_content
        return refined_content

    def integrate_knowledge_graph_context(self, kg_entities: List[str], kg_relationships: List[tuple]) -> Dict[str, str]:
        """Integrate knowledge graph context into facet refinement"""
        kg_context = self.build_kg_context_string(kg_entities, kg_relationships)
        
        for facet_name in self.facets:
            if self.facets[facet_name]:  # Only refine non-empty facets
                refinement_context = f"""
                Relevant Knowledge Graph Context:
                Entities: {', '.join(kg_entities[:self.config.get('faceted_decomposition', {}).get('knowledge_graph', {}).get('max_entities_display', 10)])}  # Limit to prevent context overflow
                Key Relationships: {kg_relationships[:self.config.get('faceted_decomposition', {}).get('knowledge_graph', {}).get('max_relationships_display', 5)]}
                
                Please enhance the {facet_name} by leveraging this knowledge context to:
                - Identify more specific techniques or approaches
                - Reference relevant datasets or benchmarks
                - Connect to established research directions
                """
                self.refine_facet(facet_name, refinement_context)
        
        return self.facets

    def build_kg_context_string(self, entities: List[str], relationships: List[tuple]) -> str:
        """Build a context string from knowledge graph entities and relationships"""
        entity_str = ", ".join(entities[:self.config.get('faceted_decomposition', {}).get('knowledge_graph', {}).get('max_entities_context', 15)])  # Limit for context
        rel_str = "; ".join([f"{r[0]} {r[1]} {r[2]}" for r in relationships[:self.config.get('faceted_decomposition', {}).get('knowledge_graph', {}).get('max_relationships_context', 10)]])
        return f"Entities: {entity_str}\nRelationships: {rel_str}"

    def validate_completeness(self) -> Dict[str, bool]:
        """Validate that all facets are properly filled out"""
        validation = {}
        for facet_name, content in self.facets.items():
            validation[facet_name] = len(content.strip()) > self.config.get('faceted_decomposition', {}).get('validation', {}).get('min_content_threshold', 50)  # Minimum content threshold
        return validation

    def get_facets(self) -> Dict[str, str]:
        """Get all current facets"""
        return self.facets.copy()

    def set_facets(self, facets: Dict[str, str]):
        """Set facets from external source"""
        for key, value in facets.items():
            if key in self.facets:
                self.facets[key] = value

    def get_facet(self, facet_name: str) -> str:
        """Get a specific facet"""
        return self.facets.get(facet_name, "")