"""
Planning Module

Provides literature-informed planning and stepwise research planning capabilities.
Integrates with knowledge graphs and faceted decomposition for structured idea development.
"""

import re
import json
from typing import Dict, List, Tuple
from .base_agent import BaseAgent
from ...knowledge_graph.kg_builder import KGBuilder
from .faceted_decomposition import FacetedDecomposition
from .graph_of_thought import GraphOfThought
from ...prompts.idea_generation.planning_module_prompts import (
    STEPWISE_PLAN_SYSTEM_PROMPT, STEPWISE_PLAN_USER_PROMPT,
    GENERAL_PLAN_REFINEMENT_SYSTEM_PROMPT, GENERAL_PLAN_REFINEMENT_USER_PROMPT
)


class PlanningModule(BaseAgent):
    """
    Enhanced planning module that coordinates faceted decomposition and graph-of-thought reasoning
    to create structured research plans from seed topics.
    """

    def __init__(self, knowledge_graph: KGBuilder = None, llm_interface=None, llm_config: Dict = None, config: Dict = None):
        super().__init__(llm_interface, llm_config)
        self.config = config or {}
        self.knowledge_graph = knowledge_graph
        self.faceted_decomposer = FacetedDecomposition(self.llm, config=self.config)
        self.graph_of_thought = GraphOfThought(self.llm, config=self.config)
        self.current_plan = {}
        
    async def plan_research_steps(self, seed_topic: str, context: str = "", literature_papers: List = None, literature_summary: str = "") -> Dict:
        """Create a comprehensive research plan from seed topic using literature-informed faceted decomposition"""
        
        # Step 1: Build knowledge context (including literature)
        knowledge_context = self.build_knowledge_context(seed_topic)
        
        # Step 2: Integrate literature findings
        literature_context = self.build_literature_context(literature_papers, literature_summary)
        
        # Step 3: Perform literature-informed faceted decomposition
        combined_context = f"{knowledge_context}\n\n{literature_context}\n\n{context}".strip()
        facets = await self.faceted_decomposer.decompose_idea(seed_topic, combined_context)
        
        # Step 4: Identify research gaps from literature
        literature_gaps = await self.identify_literature_gaps(seed_topic, literature_papers)

        # Step 5: Create stepwise plan from literature-informed facets
        plan = await self.create_stepwise_plan(facets)

        # Step 6: GoT exploration is now handled by IdeaGenerator, not here
        # (Facets are passed directly to graph_of_thought.build_graph_of_thoughts)

        self.current_plan = {
            'seed_topic': seed_topic,
            'facets': facets,
            'plan': plan,
            'knowledge_context': knowledge_context,
            'literature_context': literature_context,
            'literature_gaps': literature_gaps,
            'referenced_papers': literature_papers[:self.config.get('planning_module', {}).get('literature', {}).get('max_referenced_papers', 5)] if literature_papers else []
        }
        
        return self.current_plan

    def build_knowledge_context(self, seed_topic: str) -> str:
        """Build knowledge context using the knowledge graph"""
        if not self.knowledge_graph:
            return ""
            
        # Get related entities and relationships from KG
        try:
            related_entities = self.knowledge_graph.get_related_entities(seed_topic, max_distance=2)
            kg_summary = self.knowledge_graph.get_graph_summary()
            
            context_parts = []
            
            if related_entities:
                context_parts.append(f"Related entities: {', '.join(list(related_entities)[:self.config.get('planning_module', {}).get('knowledge_graph', {}).get('max_related_entities', 10)])}")
            
            # Get cross-cluster connections for novel insights
            cross_connections = self.knowledge_graph.get_cross_cluster_connections()
            if cross_connections:
                context_parts.append(f"Cross-domain connections: {cross_connections[:self.config.get('planning_module', {}).get('knowledge_graph', {}).get('max_cross_connections', 3)]}")
                
            return "\n".join(context_parts)
            
        except Exception as e:
            print(f"Error building knowledge context: {e}")
            return ""

    def build_literature_context(self, literature_papers: List = None, literature_summary: str = "") -> str:
        """Build comprehensive literature context for informed planning"""
        if not literature_papers and not literature_summary:
            return ""
        
        context_parts = []
        
        # Add literature summary if provided
        if literature_summary:
            context_parts.append("LITERATURE ANALYSIS:")
            context_parts.append(literature_summary)
        
        # Process papers for key insights
        if literature_papers:
            context_parts.append("\nKEY PAPERS AND METHODOLOGIES:")
            
            for i, paper in enumerate(literature_papers[:self.config.get('planning_module', {}).get('literature', {}).get('max_referenced_papers', 5)], 1):
                paper_info = f"{i}. \"{paper.title}\" ({paper.year})"
                if paper.authors:
                    paper_info += f" - {', '.join(paper.authors[:self.config.get('planning_module', {}).get('literature', {}).get('max_displayed_authors', 2)])}"
                context_parts.append(paper_info)
                
                if paper.abstract:
                    # Extract methodology hints from abstract
                    methodology_keywords = self._extract_methodology_keywords(paper.abstract)
                    if methodology_keywords:
                        context_parts.append(f"   Methodology: {', '.join(methodology_keywords[:self.config.get('planning_module', {}).get('literature', {}).get('max_methodology_keywords', 5)])}")
                
                if paper.keywords:
                    context_parts.append(f"   Keywords: {', '.join(paper.keywords[:self.config.get('planning_module', {}).get('literature', {}).get('max_keywords_display', 5)])}")
            
            # Extract common limitations and future work hints
            limitations = self._extract_limitations_from_papers(literature_papers)
            if limitations:
                context_parts.append("\nCOMMON LIMITATIONS IN EXISTING WORK:")
                for limitation in limitations[:self.config.get('planning_module', {}).get('literature', {}).get('max_limitations_display', 3)]:
                    context_parts.append(f"- {limitation}")
        
        return "\n".join(context_parts)

    def _extract_methodology_keywords(self, abstract: str) -> List[str]:
        """Extract methodology-related keywords from paper abstract"""
        methodology_terms = [
            'neural network', 'deep learning', 'machine learning', 'algorithm',
            'optimization', 'framework', 'model', 'approach', 'method',
            'technique', 'system', 'architecture', 'pipeline', 'workflow'
        ]
        
        found_methods = []
        abstract_lower = abstract.lower()
        for term in methodology_terms:
            if term in abstract_lower:
                found_methods.append(term)
        
        return found_methods[:self.config.get('planning_module', {}).get('literature', {}).get('max_methodology_keywords', 5)]  # Limit to 5 most relevant

    def _extract_limitations_from_papers(self, papers: List) -> List[str]:
        """Extract common limitations and gaps from paper abstracts"""
        limitation_indicators = [
            'however', 'but', 'limitation', 'challenge', 'difficult',
            'future work', 'further research', 'remains to be', 'open problem'
        ]
        
        limitations = []
        for paper in papers[:self.config.get('planning_module', {}).get('literature', {}).get('max_papers_for_limitations', 3)]:  # Check top 3 papers
            if paper.abstract:
                abstract_lower = paper.abstract.lower()
                for indicator in limitation_indicators:
                    if indicator in abstract_lower:
                        # Extract sentence containing limitation
                        sentences = paper.abstract.split('.')
                        for sentence in sentences:
                            if indicator in sentence.lower() and len(sentence.strip()) > 20:
                                limitations.append(sentence.strip())
                                break
        
        return limitations

    async def identify_literature_gaps(self, seed_topic: str, literature_papers: List = None) -> List[str]:
        """Use LLM to identify research gaps from literature analysis"""
        if not literature_papers:
            return []
        
        # Create literature summary for gap analysis
        papers_summary = []
        for paper in literature_papers[:self.config.get('planning_module', {}).get('literature', {}).get('max_referenced_papers', 5)]:
            paper_summary = f"- {paper.title} ({paper.year})"
            if paper.abstract:
                paper_summary += f": {paper.abstract[:self.config.get('planning_module', {}).get('literature', {}).get('abstract_truncate_length', 150)]}..."
            papers_summary.append(paper_summary)
        
        gap_analysis_prompt = f"""Analyze these recent papers on "{seed_topic}" and identify research gaps:

RECENT LITERATURE:
{chr(10).join(papers_summary)}

Based on this literature review, identify 3-5 specific research gaps or limitations that represent opportunities for new research. Focus on:
1. Methodological limitations in current approaches
2. Unexplored application domains or use cases  
3. Scalability or performance issues
4. Lack of theoretical understanding
5. Missing empirical evaluations

Return each gap as a single clear sentence."""
        
        try:
            response = await self.llm.generate_with_system_prompt(
                "You are an expert research analyst specializing in identifying research gaps and opportunities from literature reviews.",
                gap_analysis_prompt,
                max_tokens=None,  # Use configured gap_analysis context window
                caller="planning_module_gap_analysis"
            )
            
            # Parse gaps from response
            gaps = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or any(char.isdigit() for char in line[:3])):
                    # Clean up the gap description
                    gap = line.lstrip('- •0123456789. ').strip()
                    if len(gap) > 10:  # Ensure meaningful content
                        gaps.append(gap)
            
            return gaps[:self.config.get('planning_module', {}).get('literature', {}).get('max_gaps_identified', 5)]  # Limit to top 5 gaps
            
        except Exception as e:
            print(f"Error identifying literature gaps: {e}")
            return []

    async def create_stepwise_plan(self, facets: Dict[str, str]) -> List[Dict[str, str]]:
        """Create a detailed stepwise plan from the facets"""
        
        user_prompt = STEPWISE_PLAN_USER_PROMPT.format(
            problem_statement=facets.get('Problem Statement', ''),
            proposed_methodology=facets.get('Proposed Methodology', ''),
            experimental_validation=facets.get('Experimental Validation', '')
        )
        
        plan_response = await self.llm.generate_with_system_prompt(STEPWISE_PLAN_SYSTEM_PROMPT, user_prompt, max_tokens=None, caller="planning_module")  # Use configured planning context window
        return self.parse_plan_response(plan_response)

    def parse_plan_response(self, response: str) -> List[Dict[str, str]]:
        """Parse the LLM response into structured plan steps"""
        steps = []
        
        # Split by numbered sections
        step_matches = re.findall(r'(\d+)\.\s*([^:]+):(.*?)(?=\d+\.|$)', response, re.DOTALL)
        
        for step_num, step_title, step_content in step_matches:
            step_dict = {
                'step_number': step_num,
                'title': step_title.strip(),
                'content': step_content.strip(),
                'status': 'pending'
            }
            
            # Extract specific components if present
            tasks_match = re.search(r'tasks?[:\-]\s*(.*?)(?=deliverables?|dependencies?|timeline|$)', step_content, re.IGNORECASE | re.DOTALL)
            if tasks_match:
                step_dict['tasks'] = tasks_match.group(1).strip()
            
            deliverables_match = re.search(r'deliverables?[:\-]\s*(.*?)(?=dependencies?|timeline|tasks?|$)', step_content, re.IGNORECASE | re.DOTALL)
            if deliverables_match:
                step_dict['deliverables'] = deliverables_match.group(1).strip()
                
            timeline_match = re.search(r'timeline[:\-]\s*(.*?)(?=deliverables?|dependencies?|tasks?|$)', step_content, re.IGNORECASE | re.DOTALL)
            if timeline_match:
                step_dict['timeline'] = timeline_match.group(1).strip()
            
            steps.append(step_dict)

        return steps

    # ==================== DEPRECATED METHODS ====================
    # These methods are no longer used with the new GoT implementation
    # GoT exploration is now handled directly by IdeaGenerator

    # def initialize_got_exploration(self, seed_topic: str, facets: Dict[str, str]):
    #     """DEPRECATED - GoT exploration now handled by IdeaGenerator"""
    #     pass

    # async def expand_research_directions(self, max_variants: int = 3) -> List[str]:
    #     """DEPRECATED - GoT exploration now handled by IdeaGenerator"""
    #     pass

    async def refine_plan_with_feedback(self, feedback: str, focus_area: str = None) -> Dict:
        """Refine the research plan based on feedback"""
        
        if not self.current_plan:
            return {}
        
        # Refine specific facet if focus_area is specified
        if focus_area and focus_area in self.current_plan['facets']:
            refined_facet = await self.faceted_decomposer.refine_facet(focus_area, feedback)
            self.current_plan['facets'][focus_area] = refined_facet
            
            # Update plan based on refined facet
            updated_plan = await self.create_stepwise_plan(self.current_plan['facets'])
            self.current_plan['plan'] = updated_plan
        else:
            # General plan refinement
            await self.refine_general_plan(feedback)
        
        return self.current_plan

    async def refine_general_plan(self, feedback: str):
        """Refine the overall research plan based on general feedback"""
        
        current_plan_str = self.format_plan_for_refinement()
        
        user_prompt = GENERAL_PLAN_REFINEMENT_USER_PROMPT.format(
            current_plan=current_plan_str,
            feedback=feedback
        )
        
        refined_plan_response = await self.llm.generate_with_system_prompt(GENERAL_PLAN_REFINEMENT_SYSTEM_PROMPT, user_prompt, max_tokens=None, caller="planning_module")  # Use configured planning context window
        
        # Parse and update the plan
        refined_steps = self.parse_plan_response(refined_plan_response)
        if refined_steps:
            self.current_plan['plan'] = refined_steps

    def format_plan_for_refinement(self) -> str:
        """Format the current plan for LLM processing"""
        if not self.current_plan:
            return ""
        
        formatted = []
        formatted.append(f"Topic: {self.current_plan['seed_topic']}")
        formatted.append("\nFacets:")
        
        for facet_name, facet_content in self.current_plan['facets'].items():
            formatted.append(f"- {facet_name}: {facet_content}")
        
        formatted.append("\nPlan Steps:")
        for step in self.current_plan.get('plan', []):
            formatted.append(f"{step.get('step_number', '')}. {step.get('title', '')}: {step.get('content', '')}")
        
        return "\n".join(formatted)

    def get_research_context_for_kg(self) -> str:
        """Get research context formatted for knowledge graph expansion"""
        if not self.current_plan:
            return ""
        
        context_parts = [
            f"Research Topic: {self.current_plan['seed_topic']}"
        ]
        
        for facet_name, facet_content in self.current_plan['facets'].items():
            context_parts.append(f"{facet_name}: {facet_content}")
        
        return "\n".join(context_parts)

    def integrate_kg_insights(self):
        """Integrate new insights from knowledge graph back into the plan"""
        if not self.knowledge_graph or not self.current_plan:
            return
        
        # Get entities and relationships related to current research
        current_entities = self.knowledge_graph.get_entities()
        if current_entities:
            # Find related entities to current topic
            topic_related = self.knowledge_graph.get_related_entities(
                self.current_plan['seed_topic'], max_distance=2
            )
            
            # Update facets with KG context
            if topic_related:
                kg_entities = list(topic_related)[:10]
                kg_relationships = []
                
                for entity in kg_entities[:self.config.get('planning_module', {}).get('knowledge_graph', {}).get('max_referenced_papers', 5)]:
                    relationships = self.knowledge_graph.get_relationships(entity)
                    kg_relationships.extend(relationships[:self.config.get('planning_module', {}).get('knowledge_graph', {}).get('max_relationships_per_entity', 2)])  # Limit to avoid context overflow
                
                # Integrate into faceted decomposition
                updated_facets = self.faceted_decomposer.integrate_knowledge_graph_context(
                    kg_entities, kg_relationships
                )
                
                self.current_plan['facets'] = updated_facets
                
                # Regenerate plan with updated facets
                self.current_plan['plan'] = self.create_stepwise_plan(updated_facets)

    def validate_plan_completeness(self) -> Dict[str, bool]:
        """Validate that the research plan is complete and actionable"""
        validation = {
            'has_seed_topic': bool(self.current_plan.get('seed_topic')),
            'has_all_facets': all(
                content.strip() for content in self.current_plan.get('facets', {}).values()
            ),
            'has_structured_plan': bool(self.current_plan.get('plan')),
            'plan_has_concrete_steps': len(self.current_plan.get('plan', [])) >= 3
        }
        
        validation['overall_complete'] = all(validation.values())
        return validation

    def get_next_action_recommendations(self) -> List[str]:
        """Get recommendations for next actions based on current plan state"""
        if not self.current_plan:
            return ["Create initial research plan"]
        
        recommendations = []
        
        # Check plan completeness
        validation = self.validate_plan_completeness()
        
        if not validation['has_all_facets']:
            recommendations.append("Complete facet decomposition with more detailed content")
        
        if not validation['plan_has_concrete_steps']:
            recommendations.append("Develop more detailed implementation steps")

        # Check for GoT exploration opportunities
        if len(self.graph_of_thought.thought_graph.nodes) < 5:
            recommendations.append("Explore more research direction variants")

        # Check for knowledge integration opportunities
        if not self.current_plan.get('knowledge_context'):
            recommendations.append("Integrate more domain knowledge from literature")

        if not recommendations:
            recommendations.append("Plan appears complete - ready for implementation")

        return recommendations

    def save_plan(self, filepath: str):
        """Save the current research plan to a file"""
        with open(filepath, 'w') as f:
            json.dump(self.current_plan, f, indent=2)

    def load_plan(self, filepath: str):
        """Load a research plan from a file"""
        with open(filepath, 'r') as f:
            self.current_plan = json.load(f)