"""
Novelty Agent

Evaluates research ideas for originality and novelty by identifying novel aspects,
comparing against existing work, and providing enhancement suggestions.
"""

from typing import Dict, List, Optional
from .base_agent import BaseAgent
from ..utils.llm_interface import LLMInterface
from ..utils.text_utils import (
    extract_bullet_points, extract_scores_from_text, extract_overall_score,
    extract_sections
)
from .idea_generator import ResearchIdea
from ..prompts.detailed_review.novelty_agent_prompts import (
    COMPREHENSIVE_NOVELTY_SYSTEM_PROMPT, COMPREHENSIVE_NOVELTY_USER_PROMPT,
    NOVELTY_DIMENSIONS
)
import json


class NoveltyAgent(BaseAgent):
    """
    Novelty agent specialized in assessing the originality and novelty of research ideas
    by comparing against existing work and identifying novel contributions.
    """
    
    def __init__(self, config: Dict = None, logger=None, llm_config: Dict = None):
        super().__init__("NoveltyAgent")
        self.config = config or {}
        self.logger = logger
        self.llm = LLMInterface(config=llm_config, logger=logger)
        self.novelty_assessments = []
        
        # Novelty evaluation criteria
        self.novelty_dimensions = NOVELTY_DIMENSIONS

    async def check_novelty(self, idea: ResearchIdea, existing_work_context: str = "") -> Dict:
        """
        Main method to assess the novelty of a research idea
        """
        print(f"Assessing novelty for: {idea.topic}")
        
        # Perform comprehensive novelty assessment
        novelty_result = await self.assess_comprehensive_novelty(idea, existing_work_context)
        
        # Store assessment
        self.novelty_assessments.append({
            'idea_topic': idea.topic,
            'novelty_result': novelty_result
        })
        
        # Update the idea with novelty score
        idea.novelty_score = novelty_result['overall_novelty_score']
        
        return novelty_result

    async def assess_comprehensive_novelty(self, idea: ResearchIdea, existing_work_context: str) -> Dict:
        """Perform comprehensive novelty assessment across multiple dimensions"""
        
        dimensions_description = []
        for dim, desc in self.novelty_dimensions.items():
            dimensions_description.append(f"- {dim.replace('_', ' ').title()}: {desc}")
        
        user_prompt = COMPREHENSIVE_NOVELTY_USER_PROMPT.format(
            idea=idea,
            existing_work_context=existing_work_context,
            dimensions_description=chr(10).join(dimensions_description)
        )
        
        response = await self.llm.generate_with_system_prompt(
            COMPREHENSIVE_NOVELTY_SYSTEM_PROMPT,
            user_prompt,
            max_tokens=self.config.get('llm', {}).get('max_tokens', 16384),
            caller="novelty_agent"
        )
        
        return self.parse_novelty_assessment(response)

    def parse_novelty_assessment(self, response: str) -> Dict:
        """Parse the LLM novelty assessment into structured results"""
        from ..utils.text_utils import safe_json_parse

        novelty_result = {
            'overall_novelty_score': self.config.get('scoring', {}).get('default_score', 3.0),
            'dimension_scores': {},
            'contribution_potential': 'moderate'
        }

        # Try JSON parsing first (since prompt requests JSON format)
        json_data = safe_json_parse(response)

        if json_data and isinstance(json_data, dict):
            # Extract dimension_scores from JSON
            if 'dimension_scores' in json_data and isinstance(json_data['dimension_scores'], dict):
                novelty_result['dimension_scores'] = {
                    k: float(v) for k, v in json_data['dimension_scores'].items()
                    if isinstance(v, (int, float))
                }

            # Extract other fields from JSON if present
            if 'novel_aspects' in json_data:
                novelty_result['novel_aspects'] = json_data['novel_aspects']
            if 'similar_work' in json_data:
                novelty_result['similar_work'] = json_data['similar_work']
            if 'key_differences' in json_data:
                novelty_result['key_differences'] = json_data['key_differences']
            if 'enhancement_areas' in json_data:
                novelty_result['enhancement_areas'] = json_data['enhancement_areas']
            if 'positioning_suggestions' in json_data:
                novelty_result['positioning_suggestions'] = json_data['positioning_suggestions']

        # Fallback to text parsing if JSON parsing failed
        if not novelty_result['dimension_scores']:
            novelty_result['dimension_scores'] = extract_scores_from_text(
                response, list(self.novelty_dimensions.keys()), default_score=3.0
            )

        # Calculate overall score as average of dimension scores
        if novelty_result['dimension_scores']:
            avg_score = sum(novelty_result['dimension_scores'].values()) / len(novelty_result['dimension_scores'])
            novelty_result['overall_novelty_score'] = avg_score

        # Determine contribution potential based on overall score
        score = novelty_result['overall_novelty_score']
        scoring_config = self.config.get('scoring', {})
        high_threshold = scoring_config.get('high_contribution_threshold', 4.0)
        moderate_high_threshold = scoring_config.get('moderate_high_threshold', 3.5)
        moderate_threshold = scoring_config.get('moderate_threshold', 2.5)
        
        if score >= high_threshold:
            novelty_result['contribution_potential'] = 'high'
        elif score >= moderate_high_threshold:
            novelty_result['contribution_potential'] = 'moderate-high'
        elif score >= moderate_threshold:
            novelty_result['contribution_potential'] = 'moderate'
        else:
            novelty_result['contribution_potential'] = 'low'

        return novelty_result


    def get_novelty_trends(self) -> Dict:
        """Analyze trends in novelty assessments"""
        
        if not self.novelty_assessments:
            return {'message': 'No novelty assessments conducted yet'}
        
        # Aggregate statistics
        total_assessments = len(self.novelty_assessments)
        scores = [assessment['novelty_result']['overall_novelty_score'] 
                 for assessment in self.novelty_assessments 
                 if 'overall_novelty_score' in assessment['novelty_result']]
        
        dimension_averages = {}
        for assessment in self.novelty_assessments:
            result = assessment['novelty_result']
            for dim, score in result.get('dimension_scores', {}).items():
                if dim not in dimension_averages:
                    dimension_averages[dim] = []
                dimension_averages[dim].append(score)
        
        # Calculate averages
        for dim in dimension_averages:
            dimension_averages[dim] = sum(dimension_averages[dim]) / len(dimension_averages[dim])
        
        return {
            'total_assessments': total_assessments,
            'average_novelty_score': sum(scores) / len(scores) if scores else 0,
            'highest_novelty_score': max(scores) if scores else 0,
            'dimension_averages': dimension_averages,
            'high_novelty_ideas': sum(1 for score in scores if score >= 4.0),
            'assessment_history': self.novelty_assessments
        }

    # Implementation of BaseAgent abstract methods
    def gather_information(self):
        """Gather novelty criteria and assessment framework"""
        return self.novelty_dimensions

    def generate_ideas(self):
        """Novelty agent doesn't generate ideas, returns gap analysis"""
        return self.get_novelty_trends()

    def critique_ideas(self):
        """Main function - assess novelty of ideas"""
        return self.novelty_assessments

    def refine_ideas(self):
        """Provide novelty enhancement suggestions"""
        return []