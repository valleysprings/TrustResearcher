"""
Research Idea Model

Represents a complete research idea with all facets, feedback, and refinement history.
"""

from typing import Dict, List, Optional


class ResearchIdea:
    """Represents a complete research idea with all facets"""
    def __init__(self, topic: str, facets: Dict[str, str], source: str = "unknown", method: str = "unknown"):
        self.topic = topic
        self.facets = facets  # Problem Statement, Methodology, Validation
        self.source = source  # Which agent generated this (e.g., "idea_generator", "cross_pollination", "kg_extraction")
        self.method = method  # Which specific method (e.g., "got_reasoning", "variant_generation", "literature_expansion")
        self.refinement_history = []
        self.review_feedback = []
        self.novelty_score = None
        self.reasoning_context = {}

    def add_feedback(self, feedback: Dict):
        """Add review feedback to the idea"""
        self.review_feedback.append(feedback)

    def refine(self, new_facets: Dict[str, str], reasoning: str = ""):
        """Refine the idea facets"""
        self.refinement_history.append({
            'old_facets': self.facets.copy(),
            'new_facets': new_facets,
            'reasoning': reasoning
        })
        self.facets = new_facets

    def to_dict(self):
        result = {
            'topic': self.topic,
            'facets': self.facets,
            'source': self.source,
            'method': self.method,
            'review_feedback': self.review_feedback,
            'novelty_score': self.novelty_score,
            'reasoning_context': self.reasoning_context,
        }

        # Handle literature_context if it exists
        if hasattr(self, 'literature_context') and self.literature_context:
            result['literature_context'] = {}
            for key, value in self.literature_context.items():
                if key == 'referenced_papers' and isinstance(value, list):
                    # Convert Paper objects to dictionaries
                    result['literature_context'][key] = [
                        paper.to_dict() if hasattr(paper, 'to_dict') else paper
                        for paper in value
                    ]
                else:
                    result['literature_context'][key] = value

        return result

    def __str__(self):
        return f"""
            Research Idea: {self.topic}

            Problem Statement: {self.facets.get('Problem Statement', 'Not defined')}

            Proposed Methodology: {self.facets.get('Proposed Methodology', 'Not defined')}

            Experimental Validation: {self.facets.get('Experimental Validation', 'Not defined')}

            Source: {self.source} | Method: {self.method}
        """.strip()

    @property
    def problem_statement(self) -> str:
        """Get the problem statement from facets"""
        value = self.facets.get('Problem Statement', '')
        return str(value) if value else ''

    @property
    def proposed_methodology(self) -> str:
        """Get the proposed methodology from facets"""
        value = self.facets.get('Proposed Methodology', '')
        return str(value) if value else ''

    @property
    def experimental_validation(self) -> str:
        """Get the experimental validation from facets"""
        value = self.facets.get('Experimental Validation', '')
        return str(value) if value else ''

    @property
    def potential_impact(self) -> str:
        """Get the potential impact from facets"""
        value = self.facets.get('Potential Impact', '')
        return str(value) if value else ''

    @property
    def summary(self) -> str:
        """Get summary text for similarity comparison"""
        components = [
            self.topic or '',
            self.problem_statement or '',
            self.proposed_methodology or ''
        ]
        return ' '.join(comp for comp in components if comp).strip()
