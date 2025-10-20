"""
Aggregator Agent

Collects and synthesizes feedback from multiple agents (reviewer, novelty)
to provide consolidated insights and final recommendations.
"""

from typing import Dict, List, Any
from .idea_generator import ResearchIdea


class Aggregator:
    """
    Aggregator that collects and synthesizes feedback from multiple agents
    (reviewer, novelty) to provide consolidated insights.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.collected_feedback = {}
        self.ideas_processed = []
        self.feedbacks = []  # Keep for compatibility
    
    def collect_feedback(self, idea: ResearchIdea, review_feedback: Dict, 
                        novelty_check: Dict, additional_feedback: Dict = None):
        """Collect feedback from multiple agents for an idea"""
        
        feedback_entry = {
            'idea_topic': idea.topic,
            'idea': idea,
            'review_feedback': review_feedback,
            'novelty_check': novelty_check,
            'additional_feedback': additional_feedback or {},
            'timestamp': self.get_timestamp()
        }
        
        self.collected_feedback[idea.topic] = feedback_entry
        
        if idea not in self.ideas_processed:
            self.ideas_processed.append(idea)
    
    def get_timestamp(self) -> str:
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def synthesize_feedback(self, idea_topic: str) -> Dict:
        """Return complete feedback from both agents with extracted scores"""

        if idea_topic not in self.collected_feedback:
            return {'error': f'No feedback collected for {idea_topic}'}

        feedback_entry = self.collected_feedback[idea_topic]
        review_feedback = feedback_entry['review_feedback']
        novelty_check = feedback_entry['novelty_check']

        # Extract scores from the feedback
        review_score = review_feedback.get('overall_score', 0)
        novelty_score = novelty_check.get('overall_novelty_score', 0)

        # Calculate overall score as average of review and novelty scores
        overall_score = (review_score + novelty_score) / 2 if (review_score or novelty_score) else 0

        # Extract recommendation from review feedback
        overall_recommendation = review_feedback.get('recommendation', 'No recommendation available')

        return {
            'idea_topic': idea_topic,
            'review_feedback': review_feedback,
            'novelty_check': novelty_check,
            'review_score': review_score,
            'novelty_score': novelty_score,
            'overall_score': overall_score,
            'overall_recommendation': overall_recommendation
        }
    
    # Compatibility methods for existing interface
    def integrate_feedback(self):
        """Logic to integrate feedback from various agents"""
        integrated_result = {}
        for feedback in self.feedbacks:
            for key, value in feedback.items():
                if key not in integrated_result:
                    integrated_result[key] = []
                integrated_result[key].append(value)
        
        # Example of averaging numerical feedback
        for key in integrated_result:
            if integrated_result[key] and isinstance(integrated_result[key][0], (int, float)):
                integrated_result[key] = sum(integrated_result[key]) / len(integrated_result[key])
        
        return integrated_result
    
    def clear_feedback(self):
        """Clear all feedback"""
        self.feedbacks = []
        self.collected_feedback = {}
        self.ideas_processed = []
    
    def get_portfolio_recommendation(self) -> Dict[str, Any]:
        """
        Analyze the overall portfolio of ideas and provide recommendations
        """
        if not self.collected_feedback:
            return {
                'portfolio_quality': 'unknown',
                'recommended_action': 'No ideas to analyze',
                'summary': 'No ideas have been processed yet.',
                'top_ideas': [],
                'top_3_ideas': [],  # For main.py compatibility
                'average_scores': {'review': 0, 'novelty': 0, 'overall': 0},
                'total_ideas': 0,
                'high_quality_ideas': 0,
                'distribution': {'excellent': 0, 'good': 0, 'moderate': 0, 'weak': 0}
            }
        
        # Analyze all collected ideas
        all_syntheses = []
        review_scores = []
        novelty_scores = []
        overall_scores = []

        for idea_topic in self.collected_feedback.keys():
            synthesis = self.synthesize_feedback(idea_topic)
            all_syntheses.append(synthesis)

            # Use the correct keys from synthesize_feedback
            review_scores.append(synthesis.get('review_score', 0))
            novelty_scores.append(synthesis.get('novelty_score', 0))
            overall_scores.append(synthesis.get('overall_score', 0))
        
        # Calculate averages
        avg_review = sum(review_scores) / len(review_scores) if review_scores else 0
        avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0
        avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0
        
        # Sort ideas by overall score
        max_top_ideas = self.config.get('portfolio_assessment', {}).get('max_top_ideas', 3)
        top_ideas = sorted(all_syntheses, key=lambda x: x.get('overall_score', 0), reverse=True)[:max_top_ideas]
        
        # Determine portfolio quality and recommendations
        thresholds = self.config.get('portfolio_assessment', {})
        excellent_threshold = thresholds.get('excellent_threshold', 4.0)
        good_threshold = thresholds.get('good_threshold', 3.5)
        moderate_threshold = thresholds.get('moderate_threshold', 3.0)
        weak_threshold = thresholds.get('weak_threshold', 2.0)
        
        if avg_overall >= excellent_threshold:
            portfolio_quality = 'excellent'
            recommended_action = 'Portfolio ready for implementation - focus on top ideas'
        elif avg_overall >= good_threshold:
            portfolio_quality = 'good'
            recommended_action = 'Strong portfolio - refine top ideas and implement'
        elif avg_overall >= moderate_threshold:
            portfolio_quality = 'moderate'
            recommended_action = 'Mixed portfolio - develop strongest ideas, reconsider weaker ones'
        elif avg_overall >= weak_threshold:
            portfolio_quality = 'weak'
            recommended_action = 'Portfolio needs significant work - focus on refinement'
        else:
            portfolio_quality = 'poor'
            recommended_action = 'Portfolio requires major revision or new idea generation'
        
        # Generate summary
        good_threshold = self.config.get('portfolio_assessment', {}).get('good_threshold', 3.5)
        high_quality_count = sum(1 for score in overall_scores if score >= good_threshold)
        summary = f"Portfolio contains {len(all_syntheses)} ideas with {high_quality_count} high-quality candidates. "
        summary += f"Average scores: Review {avg_review:.2f}/5, Novelty {avg_novelty:.2f}/5, Overall {avg_overall:.2f}/5."
        
        return {
            'portfolio_quality': portfolio_quality,
            'recommended_action': recommended_action,
            'summary': summary,
            'top_ideas': [
                {
                    'topic': idea['idea_topic'],
                    'score': idea['overall_score'],
                    'recommendation': idea['overall_recommendation']
                }
                for idea in top_ideas
            ],
            'top_3_ideas': [idea['idea_topic'] for idea in top_ideas],  # For main.py compatibility
            'average_scores': {
                'review': round(avg_review, 2),
                'novelty': round(avg_novelty, 2),
                'overall': round(avg_overall, 2)
            },
            'total_ideas': len(all_syntheses),
            'high_quality_ideas': high_quality_count,
            'distribution': {
                'excellent': sum(1 for score in overall_scores if score >= excellent_threshold),
                'good': sum(1 for score in overall_scores if good_threshold <= score < excellent_threshold),
                'moderate': sum(1 for score in overall_scores if moderate_threshold <= score < good_threshold),
                'weak': sum(1 for score in overall_scores if score < moderate_threshold)
            }
        }