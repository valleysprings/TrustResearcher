"""
Reviewer Agent

Provides comprehensive peer review and evaluation of research ideas across multiple criteria.
Supports detailed reviews, comparative analysis, and multi-dimensional scoring.
"""

from typing import Dict, List, Optional, Any
from .base_agent import BaseAgent
from ..utils.llm_interface import LLMInterface
from ..utils.text_utils import safe_json_parse, extract_scores_from_response, extract_sections_from_response, extract_score_from_response, extract_bullet_points_from_response
from .idea_generator import ResearchIdea
from ..prompts.detailed_review.reviewer_agent_prompts import (
    DETAILED_REVIEW_SYSTEM_PROMPT, DETAILED_REVIEW_USER_PROMPT, DETAILED_REVIEW_EXTENDED_PROMPT,
    COMPARATIVE_REVIEW_SYSTEM_PROMPT, COMPARATIVE_REVIEW_USER_PROMPT,
    ORIGINALITY_SYSTEM_PROMPT, ORIGINALITY_USER_PROMPT,
    CLARITY_SYSTEM_PROMPT, CLARITY_USER_PROMPT,
    FEASIBILITY_SYSTEM_PROMPT, FEASIBILITY_USER_PROMPT
)
import json
import re


class ReviewerAgent(BaseAgent):
    """
    Enhanced reviewer agent that provides detailed peer-review style feedback
    on research ideas, evaluating multiple criteria and providing actionable suggestions.
    """
    
    def __init__(self, config: Dict = None, logger=None, llm_config: Dict = None):
        super().__init__("ReviewerAgent")
        self.config = config or {}
        self.logger = logger
        self.llm = LLMInterface(config=llm_config, logger=logger)
        self.feedback_history = []
        
        # Review criteria with weights - configurable via config
        default_review_criteria = {
            'novelty': {'weight': 0.25, 'description': 'Originality and innovation of the approach'},
            'feasibility': {'weight': 0.20, 'description': 'Technical and practical implementability'},
            'clarity': {'weight': 0.20, 'description': 'Clear problem definition and methodology'},
            'impact': {'weight': 0.25, 'description': 'Potential significance and contribution'},
            'methodology_soundness': {'weight': 0.10, 'description': 'Scientific rigor of proposed methods'}
        }
        
        # Load review criteria from config, with defaults as fallback
        self.review_criteria = self.config.get('review_criteria', default_review_criteria)
        
        # Validate that weights sum to approximately 1.0
        total_weight = sum(criterion.get('weight', 0) for criterion in self.review_criteria.values())
        weight_tolerance = self.config.get('weight_tolerance', 0.01)  # Allow small floating point errors
        if abs(total_weight - 1.0) > weight_tolerance:
            if self.logger:
                self.logger.log_warning(f"Review criteria weights sum to {total_weight:.3f}, not 1.0. Consider adjusting weights.", "reviewer_agent")
        
        if self.logger:
            criteria_names = list(self.review_criteria.keys())
            self.logger.log_info(f"Initialized ReviewerAgent with criteria: {criteria_names}", "reviewer_agent")

    async def review(self, idea: ResearchIdea, criteria: List[str] = None, 
              detailed: bool = True) -> Dict:
        """
        Main review method that provides comprehensive feedback on a research idea
        """
        if criteria is None:
            criteria = list(self.review_criteria.keys())
        
        print(f"Reviewing research idea: {idea.topic}")
        
        # Perform detailed review
        review_result = await self.perform_detailed_review(idea, criteria, detailed)
        
        # Store feedback history
        self.feedback_history.append({
            'idea_topic': idea.topic,
            'review_result': review_result,
            'criteria_used': criteria
        })
        
        # Add feedback to the idea
        idea.add_feedback(review_result)
        
        return review_result

    async def perform_detailed_review(self, idea: ResearchIdea, criteria: List[str], 
                               detailed: bool) -> Dict:
        """Perform a comprehensive review of the research idea"""
        
        criteria_descriptions = []
        for criterion in criteria:
            if criterion in self.review_criteria:
                desc = self.review_criteria[criterion]['description']
                criteria_descriptions.append(f"- {criterion.title()}: {desc}")
        
        user_prompt = DETAILED_REVIEW_USER_PROMPT.format(
            criteria_descriptions=chr(10).join(criteria_descriptions),
            idea=idea
        )
        
        if detailed:
            user_prompt += DETAILED_REVIEW_EXTENDED_PROMPT

        max_tokens = self.config.get('llm', {}).get('max_tokens', 16384)
        response = await self.llm.generate_with_system_prompt(DETAILED_REVIEW_SYSTEM_PROMPT, user_prompt, max_tokens=max_tokens, caller="reviewer_agent")
        
        return self.parse_review_response(response, criteria)

    def parse_review_response(self, response: str, criteria: List[str]) -> Dict:
        """Parse the LLM review response into structured feedback"""

        review_result = {
            'overall_score': 0.0,
            'criteria_scores': {},
            'strengths': "",
            'weaknesses': "",
            'suggestions': "",
            'recommendation': 'revise',
            'priority_revisions': ""
        }

        # Parse scores from JSON or extract from text
        review_result['criteria_scores'] = extract_scores_from_response(response, criteria, default_score=3.0)

        # Calculate overall score from criteria scores using weighted average
        total_weighted_score = 0
        total_weight = 0

        for criterion, score in review_result['criteria_scores'].items():
            if criterion in self.review_criteria:
                weight = self.review_criteria[criterion]['weight']
                total_weighted_score += score * weight
                total_weight += weight

        if total_weight > 0:
            review_result['overall_score'] = total_weighted_score / total_weight
        else:
            # Fallback: try to extract overall_score from JSON if calculation failed
            json_data = safe_json_parse(response)
            if json_data and isinstance(json_data, dict) and 'overall_score' in json_data:
                review_result['overall_score'] = float(json_data['overall_score'])

        # Extract sections as raw text from JSON or text parsing
        json_data = safe_json_parse(response)
        if json_data and isinstance(json_data, dict):
            # Get raw text from JSON structure
            for key in ['strengths', 'weaknesses', 'suggestions', 'priority_revisions']:
                if key in json_data:
                    if isinstance(json_data[key], list):
                        # Join list items with newlines and bullet points
                        review_result[key] = '\n'.join(f"- {item}" for item in json_data[key])
                    elif isinstance(json_data[key], str):
                        review_result[key] = json_data[key]
                else:
                    review_result[key] = ""
        else:
            # Fallback to text parsing for raw sections
            section_patterns = {
                'strengths': ['strengths', 'strength'],
                'weaknesses': ['weakness', 'concerns', 'weaknesses'],
                'suggestions': ['suggestions', 'suggestion'],
                'priority_revisions': ['priority revision', 'priority revisions']
            }

            for key in ['strengths', 'weaknesses', 'suggestions', 'priority_revisions']:
                review_result[key] = ""
                pattern_names = section_patterns.get(key, [key])
                for pattern_name in pattern_names:
                    pattern = rf"{pattern_name}[:\-](.*?)(?={'|'.join(sum(section_patterns.values(), []))}|$)"
                    match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                    if match:
                        content = match.group(1).strip()
                        if content:
                            review_result[key] = content
                            break

        # Determine recommendation based on overall score using config thresholds
        thresholds = self.config.get('thresholds', {})
        accept_threshold = thresholds.get('accept', 4.0)
        minor_revise_threshold = thresholds.get('minor_revise', 3.5)
        major_revise_threshold = thresholds.get('major_revise', 2.5)

        if review_result['overall_score'] >= accept_threshold:
            review_result['recommendation'] = 'accept'
        elif review_result['overall_score'] >= minor_revise_threshold:
            review_result['recommendation'] = 'minor_revise'
        elif review_result['overall_score'] >= major_revise_threshold:
            review_result['recommendation'] = 'major_revise'
        else:
            review_result['recommendation'] = 'reject'

        return review_result


    async def comparative_review(self, ideas: List[ResearchIdea], 
                          ranking_criteria: List[str] = None) -> Dict:
        """Compare and rank multiple research ideas"""
        
        if ranking_criteria is None:
            ranking_criteria = ['novelty', 'impact', 'feasibility']
        
        # Review each idea individually first
        reviews = []
        for idea in ideas:
            review = self.review(idea, criteria=ranking_criteria, detailed=False)
            reviews.append({
                'idea': idea,
                'review': review,
                'overall_score': review['overall_score']
            })
        
        # Sort by overall score
        reviews.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Generate comparative analysis
        ideas_summary = []
        for i, review_data in enumerate(reviews):
            idea = review_data['idea']
            score = review_data['overall_score']
            summary_length = self.config.get('max_summary_length', 100)
            problem_statement = idea.facets.get('Problem Statement', '')
            summary = problem_statement[:summary_length] + '...' if len(problem_statement) > summary_length else problem_statement
            ideas_summary.append(f"Idea {i+1} (Score: {score:.2f}): {idea.topic}\n{summary}")
        
        user_prompt = COMPARATIVE_REVIEW_USER_PROMPT.format(
            ideas_summary=chr(10).join(ideas_summary)
        )

        max_tokens = self.config.get('llm', {}).get('max_tokens', 16384)
        comparative_analysis = await self.llm.generate_with_system_prompt(COMPARATIVE_REVIEW_SYSTEM_PROMPT, user_prompt, max_tokens=max_tokens, caller="reviewer_agent")
        
        return {
            'ranked_ideas': reviews,
            'comparative_analysis': comparative_analysis,
            'ranking_criteria': ranking_criteria
        }

    async def check_originality(self, idea: ResearchIdea, context_knowledge: str = "") -> Dict:
        """Check the originality of an idea against existing knowledge"""
        
        user_prompt = ORIGINALITY_USER_PROMPT.format(
            idea=idea,
            context_knowledge=context_knowledge
        )

        max_tokens = self.config.get('llm', {}).get('max_tokens', 16384)
        response = await self.llm.generate_with_system_prompt(ORIGINALITY_SYSTEM_PROMPT, user_prompt, max_tokens=max_tokens, caller="reviewer_agent")
        
        # Parse originality assessment
        originality_result = {
            'originality_score': 3,
            'similar_work': [],
            'novel_aspects': [],
            'differentiation_suggestions': [],
            'related_work_to_investigate': [],
            'assessment': response
        }
        
        # Extract score
        import re
        score_match = re.search(r'originality.*?score.*?(\d)', response, re.IGNORECASE)
        if score_match:
            originality_result['originality_score'] = int(score_match.group(1))
        
        return originality_result

    async def check_clarity(self, idea: ResearchIdea) -> Dict:
        """Assess the clarity of problem definition, methodology, and validation"""
        
        user_prompt = CLARITY_USER_PROMPT.format(idea=idea)

        max_tokens = self.config.get('llm', {}).get('max_tokens', 16384)
        response = await self.llm.generate_with_system_prompt(CLARITY_SYSTEM_PROMPT, user_prompt, max_tokens=max_tokens, caller="reviewer_agent")
        
        return {
            'clarity_assessment': response,
            'overall_clarity_score': extract_score_from_response(response),
            'improvement_suggestions': extract_bullet_points_from_response(response, self.config.get('max_bullet_points', 5))
        }

    async def check_feasibility(self, idea: ResearchIdea) -> Dict:
        """Assess the technical and practical feasibility of the research idea"""
        
        user_prompt = FEASIBILITY_USER_PROMPT.format(idea=idea)

        max_tokens = self.config.get('llm', {}).get('max_tokens', 16384)
        response = await self.llm.generate_with_system_prompt(FEASIBILITY_SYSTEM_PROMPT, user_prompt, max_tokens=max_tokens, caller="reviewer_agent")
        
        return {
            'feasibility_assessment': response,
            'overall_feasibility_score': extract_score_from_response(response),
            'risks_identified': extract_bullet_points_from_response(response, self.config.get('max_bullet_points', 5)),
            'resource_requirements': self.parse_requirements(response)
        }


    def parse_requirements(self, text: str) -> Dict:
        """Parse resource requirements from text"""
        requirements = {
            'computational': 'Not specified',
            'data': 'Not specified', 
            'expertise': 'Not specified',
            'timeline': 'Not specified'
        }
        
        # Simple parsing - could be enhanced
        if 'data' in text.lower():
            requirements['data'] = 'Dataset required'
        if 'compute' in text.lower() or 'gpu' in text.lower():
            requirements['computational'] = 'High compute resources'
        if 'expert' in text.lower():
            requirements['expertise'] = 'Domain expertise needed'
            
        return requirements

    def get_review_summary(self, ideas: List[ResearchIdea]) -> Dict:
        """Generate a summary of all reviews conducted"""
        
        if not self.feedback_history:
            return {'message': 'No reviews conducted yet'}
        
        # Aggregate statistics
        total_reviews = len(self.feedback_history)
        avg_scores = {}
        recommendations = {'accept': 0, 'minor_revise': 0, 'major_revise': 0, 'reject': 0}
        
        for feedback in self.feedback_history:
            review = feedback['review_result']
            
            # Aggregate criterion scores
            for criterion, score in review.get('criteria_scores', {}).items():
                if criterion not in avg_scores:
                    avg_scores[criterion] = []
                avg_scores[criterion].append(score)
            
            # Count recommendations
            rec = review.get('recommendation', 'revise')
            if rec in recommendations:
                recommendations[rec] += 1
        
        # Calculate averages
        for criterion in avg_scores:
            avg_scores[criterion] = sum(avg_scores[criterion]) / len(avg_scores[criterion])
        
        return {
            'total_reviews': total_reviews,
            'average_scores': avg_scores,
            'recommendations_distribution': recommendations,
            'most_common_issues': self.identify_common_issues(),
            'review_history': self.feedback_history
        }

    def identify_common_issues(self) -> List[str]:
        """Identify common issues across reviews"""
        all_weaknesses = []
        for feedback in self.feedback_history:
            weaknesses = feedback['review_result'].get('weaknesses', [])
            all_weaknesses.extend(weaknesses)
        
        # Simple frequency analysis (could be more sophisticated)
        issue_keywords = ['clarity', 'novelty', 'feasibility', 'validation', 'methodology']
        common_issues = []
        
        for keyword in issue_keywords:
            count = sum(1 for weakness in all_weaknesses if keyword.lower() in weakness.lower())
            if count > len(self.feedback_history) * 0.3:  # Appears in >30% of reviews
                common_issues.append(f"{keyword.title()} issues (mentioned {count} times)")
        
        return common_issues

    # Implementation of BaseAgent abstract methods
    def gather_information(self):
        """Gather review criteria and standards"""
        return self.review_criteria

    def generate_ideas(self):
        """Reviewer doesn't generate ideas, returns review summary"""
        return self.get_review_summary([])

    def critique_ideas(self):
        """Main function - critique/review ideas"""
        return self.feedback_history

    def refine_ideas(self):
        """Provide refinement suggestions based on reviews"""
        return [feedback['review_result'].get('suggestions', []) for feedback in self.feedback_history]