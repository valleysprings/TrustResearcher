"""
Text processing utilities for agents
"""

import re
import json
from typing import List, Set, Tuple, Dict, Any, Optional


def extract_bullet_points(text: str, max_points: int = 5) -> List[str]:
    """
    Extract bullet points from text

    Args:
        text: Text containing bullet points
        max_points: Maximum number of points to return

    Returns:
        List of extracted bullet points
    """
    lines = text.split('\n')
    bullet_points = []

    for line in lines:
        line = line.strip()
        if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or
                    line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
            # Remove bullet indicator
            cleaned = re.sub(r'^[-•*\d\.]\s*', '', line)
            if cleaned:
                bullet_points.append(cleaned)
        elif line and not bullet_points and len(line) > 20:
            # If no bullet points, treat meaningful sentences as points
            sentences = re.split(r'[.!?]+', line)
            bullet_points.extend([s.strip() for s in sentences if s.strip()])

    return bullet_points[:max_points]


def extract_score_from_text(text: str, default_score: float = 3.0) -> float:
    """
    Extract a numeric score from text, defaulting to provided value if not found

    Args:
        text: Text containing score information
        default_score: Default score if none found

    Returns:
        Extracted or default score
    """
    score_matches = re.findall(r'score.*?(\d(?:\.\d)?)', text, re.IGNORECASE)
    if score_matches:
        scores = [float(s) for s in score_matches]
        return sum(scores) / len(scores)  # Average if multiple scores
    return default_score


def extract_scores_from_text(text: str, criteria: List[str], default_score: float = 3.0) -> Dict[str, float]:
    """
    Extract scores for multiple criteria from text

    Args:
        text: Text containing score information
        criteria: List of criteria to extract scores for
        default_score: Default score if not found

    Returns:
        Dictionary mapping criteria to scores
    """
    scores = {}

    for criterion in criteria:
        # Try multiple patterns for score extraction
        patterns = [
            rf"{criterion}.*?score.*?(\d(?:\.\d)?)",
            rf"{criterion}.*?(\d(?:\.\d)?)/5",
            rf"{criterion}.*?(\d(?:\.\d)?)\s*$"
        ]

        found_score = None
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                found_score = float(matches[0])
                break

        scores[criterion] = found_score if found_score is not None else default_score

    return scores


def extract_overall_score(text: str, default_score: float = 3.0) -> float:
    """
    Extract overall score from text using common patterns

    Args:
        text: Text containing overall score
        default_score: Default score if not found

    Returns:
        Overall score
    """
    patterns = [
        r"overall.*?score.*?(\d(?:\.\d)?)",
        r"overall.*?(\d(?:\.\d)?)/5",
        r"total.*?score.*?(\d(?:\.\d)?)",
        r"final.*?score.*?(\d(?:\.\d)?)"
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return float(matches[0])

    return default_score


def extract_sections(text: str, section_patterns: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Extract named sections from text using regex patterns

    Args:
        text: Text containing sections
        section_patterns: Dict mapping section names to list of possible pattern names

    Returns:
        Dictionary mapping section names to extracted content
    """
    sections = {}

    for section_name, pattern_names in section_patterns.items():
        for pattern_name in pattern_names:
            # Create regex pattern that looks for the pattern name followed by content
            pattern = rf"{pattern_name}[:\-](.*?)(?={'|'.join(sum(section_patterns.values(), []))}|$)"
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

            if match:
                content = match.group(1).strip()
                if content:
                    sections[section_name] = content
                    break  # Found content for this section, move to next

    return sections


def extract_numbered_sections(text: str, section_names: List[str]) -> Dict[str, str]:
    """
    Extract numbered sections (1., 2., 3., etc.) from text

    Args:
        text: Text containing numbered sections
        section_names: List of expected section names in order

    Returns:
        Dictionary mapping section names to content
    """
    sections = {}

    for i, section_name in enumerate(section_names, 1):
        # Look for pattern like "1. Problem Statement:" or "1. Problem Statement"
        patterns = [
            rf"{i}\.\s*{re.escape(section_name)}[:\-]?\s*(.*?)(?=\d+\.|$)",
            rf"{i}\.\s*(.*?)(?=\d+\.|$)"  # Fallback pattern
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()
                if content and len(content) > 10:  # Ensure meaningful content
                    sections[section_name] = content
                    break

    return sections



def extract_field_from_response(text: str, field_names: List[str]) -> str:
    """
    Extract a specific field from structured LLM response

    Args:
        text: LLM response text
        field_names: List of possible field names to look for

    Returns:
        Extracted field content or empty string
    """
    for field_name in field_names:
        patterns = [
            rf'{field_name}:\s*(.+?)(?:\n\n|\n[A-Z]|\n-|$)',
            rf'-\s*{field_name}[:\s]+(.+?)(?:\n\n|\n[A-Z]|\n-|$)',
            rf'{field_name}\s*\(([^)]+)\)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()
                # Clean up the content
                content = re.sub(r'\n+', ' ', content)
                content = re.sub(r'\s+', ' ', content)
                if content and len(content) > 10:
                    return content

    return ""


def safe_json_parse(text: str, fallback=None) -> Any:
    """
    Safely parse JSON from text with error handling

    Args:
        text: Text that may contain JSON
        fallback: Value to return if parsing fails

    Returns:
        Parsed JSON object or fallback value
    """
    # Clean the text first
    text = text.strip()

    # Try parsing the entire text as JSON first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix for LLMs that return JSON content without braces
    # Check if text starts with a quote (indicating a JSON key) but no opening brace
    if text and text[0] == '"' and '{' not in text[:20]:
        try:
            fixed_text = '{' + text + '}'
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            # Try adding closing brace if missing
            try:
                fixed_text = '{' + text
                if not fixed_text.rstrip().endswith('}'):
                    fixed_text = fixed_text.rstrip().rstrip(',') + '}'
                return json.loads(fixed_text)
            except json.JSONDecodeError:
                pass

    # Try to find JSON-like content in the text
    json_patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested braces
        r'\{.*?\}',  # Simple braces
        r'\[.*?\]'   # Arrays
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # Try to extract JSON from markdown code blocks
    code_block_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```'
    ]

    for pattern in code_block_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    return fallback


def parse_json_response(text: str, expected_fields: List[str] = None, fallback_type: str = "dict") -> Dict[str, Any]:
    """
    Parse JSON response with fallback to text parsing for expected fields

    Args:
        text: Response text that should contain JSON
        expected_fields: List of expected field names for fallback parsing
        fallback_type: Type of fallback ("dict" or "facets")

    Returns:
        Dictionary with parsed content
    """
    # Try JSON parsing first
    json_result = safe_json_parse(text)
    if json_result and isinstance(json_result, dict):
        # If we have expected fields and this is facets type, map the fields
        if expected_fields and fallback_type == "facets":
            mapped_result = {}

            # Map JSON fields to expected facet names
            for facet_name in expected_fields:
                snake_case_key = facet_name.lower().replace(" ", "_")
                if snake_case_key in json_result:
                    mapped_result[facet_name] = str(json_result[snake_case_key])
                elif facet_name in json_result:
                    mapped_result[facet_name] = str(json_result[facet_name])
                else:
                    mapped_result[facet_name] = ""

            # Also include any extra fields that might be useful (like topic)
            # Build comprehensive list of field names to exclude (both original and snake_case versions)
            excluded_field_names = set()
            for field in expected_fields:
                excluded_field_names.add(field)  # Original form like "Problem Statement"
                excluded_field_names.add(field.lower().replace(" ", "_"))  # Snake case form like "problem_statement"

            for key, value in json_result.items():
                if key not in excluded_field_names:
                    # Only add if it's not already in mapped_result and convert to string if needed
                    if key not in mapped_result:
                        if isinstance(value, (str, int, float, bool)):
                            mapped_result[key] = str(value) if not isinstance(value, str) else value
                        elif isinstance(value, dict):
                            # Skip nested dictionaries to avoid overwriting facet fields
                            continue
                        else:
                            mapped_result[key] = str(value)

            return mapped_result
        else:
            return json_result

    # Fallback to text parsing if JSON fails
    if expected_fields:
        if fallback_type == "facets":
            return parse_facets_from_text(text, expected_fields)
        else:
            return parse_fields_from_text(text, expected_fields)

    return {}


def parse_facets_from_text(text: str, facet_names: List[str]) -> Dict[str, str]:
    """
    Fallback parser for faceted responses when JSON parsing fails

    Args:
        text: Response text
        facet_names: List of expected facet names

    Returns:
        Dictionary mapping facet names to content
    """
    facets = {}

    # Try JSON parsing first (for when this is called as a fallback)
    json_result = safe_json_parse(text)
    if json_result and isinstance(json_result, dict):
        # Map JSON fields to facet names
        for facet_name in facet_names:
            snake_case_key = facet_name.lower().replace(" ", "_")
            if snake_case_key in json_result:
                facets[facet_name] = str(json_result[snake_case_key])
            elif facet_name in json_result:
                facets[facet_name] = str(json_result[facet_name])

        # Return if we got meaningful content from JSON
        if any(facets.values()):
            # Ensure all expected facets are present
            for facet_name in facet_names:
                if facet_name not in facets:
                    facets[facet_name] = ""
            return facets

    # Try numbered sections if JSON didn't work
    numbered_facets = extract_numbered_sections(text, facet_names)
    if numbered_facets:
        facets.update(numbered_facets)

    # Try field extraction for missing facets
    for facet_name in facet_names:
        if facet_name not in facets or not facets[facet_name]:
            field_content = extract_field_from_response(text, [facet_name])
            if field_content:
                facets[facet_name] = field_content

    # Ensure all expected facets are present
    for facet_name in facet_names:
        if facet_name not in facets:
            facets[facet_name] = ""

    return facets


def parse_fields_from_text(text: str, field_names: List[str]) -> Dict[str, str]:
    """
    Parse specific fields from text using field extraction

    Args:
        text: Response text
        field_names: List of field names to extract

    Returns:
        Dictionary mapping field names to extracted content
    """
    fields = {}

    for field_name in field_names:
        content = extract_field_from_response(text, [field_name])
        fields[field_name] = content

    return fields


def validate_json_response(json_data: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that JSON response contains all required fields

    Args:
        json_data: Parsed JSON data
        required_fields: List of required field names

    Returns:
        Tuple of (is_valid, missing_fields)
    """
    if not isinstance(json_data, dict):
        return False, required_fields

    missing_fields = []
    for field in required_fields:
        if field not in json_data or not json_data[field]:
            missing_fields.append(field)

    return len(missing_fields) == 0, missing_fields


def safe_json_dumps(obj: Any, fallback: str = "") -> str:
    """
    Safely convert object to JSON string with error handling

    Args:
        obj: Object to convert to JSON
        fallback: String to return if conversion fails

    Returns:
        JSON string or fallback
    """
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return fallback


def normalize_text_for_comparison(text: str, remove_articles: bool = True) -> str:
    """
    Normalize text for comparison by removing punctuation and extra whitespace

    Args:
        text: Text to normalize
        remove_articles: Whether to remove common articles

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Convert to lowercase and remove punctuation
    normalized = text.lower()
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.strip()

    if remove_articles:
        # Remove common articles and prepositions
        common_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = normalized.split()
        filtered_words = [word for word in words if word not in common_words]
        normalized = ' '.join(filtered_words)

    return normalized


def merge_text_sections(texts: List[str], prefix: str = "", max_sentences: int = 5) -> str:
    """
    Intelligently merge multiple text sections, avoiding redundancy

    Args:
        texts: List of text sections to merge
        prefix: Optional prefix for merged text
        max_sentences: Maximum sentences to include

    Returns:
        Merged text
    """
    if not texts:
        return ""

    # Filter out empty texts
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return ""

    if len(texts) == 1:
        return texts[0]

    # Split into sentences and deduplicate
    all_sentences = []
    seen_sentences = set()

    for text in texts:
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        for sentence in sentences:
            # Normalize for comparison
            normalized = normalize_text_for_comparison(sentence, remove_articles=False)

            # Check for substantial overlap with existing sentences
            is_unique = True
            for existing_norm in seen_sentences:
                # Calculate word overlap
                words1 = set(normalized.split())
                words2 = set(existing_norm.split())
                if len(words1) > 0 and len(words2) > 0:
                    overlap = len(words1 & words2) / len(words1 | words2)
                    if overlap > 0.7:  # High overlap threshold
                        is_unique = False
                        break

            if is_unique and len(sentence) > 10:
                all_sentences.append(sentence)
                seen_sentences.add(normalized)

    # Reconstruct merged text
    merged_sentences = all_sentences[:max_sentences]
    merged_text = '. '.join(merged_sentences)
    if merged_text and not merged_text.endswith('.'):
        merged_text += '.'

    if prefix and merged_text:
        merged_text = f"{prefix} {merged_text}"

    return merged_text


def extract_technical_terms(text: str, min_length: int = 4) -> Set[str]:
    """
    Extract technical terms from text based on capitalization and patterns

    Args:
        text: Text to analyze
        min_length: Minimum length for terms

    Returns:
        Set of technical terms
    """
    if not text:
        return set()

    words = text.split()
    technical_terms = set()

    for word in words:
        # Clean word
        clean_word = ''.join(c for c in word if c.isalnum())

        # Keep terms that are:
        # - Longer than min_length characters
        # - Capitalized or contain capital letters
        # - Not common words
        if (len(clean_word) >= min_length and
            (word[0].isupper() or any(c.isupper() for c in clean_word[1:]))):
            # Filter out common non-technical words
            if clean_word.lower() not in {
                'this', 'that', 'these', 'those', 'when', 'where', 'what', 'which',
                'method', 'approach', 'system', 'model', 'algorithm', 'study',
                'analysis', 'research', 'paper', 'work', 'using', 'based'
            }:
                technical_terms.add(clean_word.lower())

    return technical_terms


def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two texts

    Args:
        text1: First text
        text2: Second text

    Returns:
        Jaccard similarity score (0.0 to 1.0)
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def calculate_idea_similarity(idea1, idea2) -> float:
    """
    Calculate similarity between two ResearchIdea objects using combined content

    Args:
        idea1: First ResearchIdea object
        idea2: Second ResearchIdea object

    Returns:
        Jaccard similarity score (0.0 to 1.0)
    """
    content1 = extract_content_from_idea(idea1)
    content2 = extract_content_from_idea(idea2)
    return calculate_jaccard_similarity(content1, content2)


def find_most_similar_idea(target_idea, candidate_ideas: List, min_threshold: float = 0.0) -> Tuple:
    """
    Find the most similar idea from a list of candidates

    Args:
        target_idea: Target ResearchIdea to compare against
        candidate_ideas: List of candidate ResearchIdea objects
        min_threshold: Minimum similarity threshold to consider

    Returns:
        Tuple of (most_similar_idea, similarity_score) or (None, max_similarity)
    """
    if not candidate_ideas:
        return None, 0.0

    max_similarity = 0.0
    best_candidate = None

    for candidate in candidate_ideas:
        similarity = calculate_idea_similarity(target_idea, candidate)
        if similarity > max_similarity:
            max_similarity = similarity
            best_candidate = candidate

    if best_candidate and max_similarity >= min_threshold:
        return best_candidate, max_similarity
    else:
        return None, max_similarity


def extract_content_from_idea(idea) -> str:
    """
    Extract combined text content from a ResearchIdea object for analysis

    Args:
        idea: ResearchIdea object

    Returns:
        Combined text content
    """
    components = [
        getattr(idea, 'topic', '') or '',
        getattr(idea, 'problem_statement', '') or '',
        getattr(idea, 'proposed_methodology', '') or ''
    ]

    # Handle facets dictionary if available
    if hasattr(idea, 'facets') and idea.facets:
        components.extend([
            idea.facets.get('Problem Statement', '') or '',
            idea.facets.get('Proposed Methodology', '') or '',
            idea.facets.get('Experimental Validation', '') or ''
        ])

    return ' '.join(comp for comp in components if comp).strip()


def extract_score_from_response(response: str, default_score: float = 3.0) -> float:
    """Extract score from response using JSON or text parsing"""
    json_data = safe_json_parse(response)
    if json_data and isinstance(json_data, dict):
        # Try common score field names
        for key in ['score', 'overall_score', 'clarity_score', 'feasibility_score']:
            if key in json_data:
                return float(json_data[key])

    # Fallback to text parsing
    score_match = re.search(r'score.*?(\d(?:\.\d)?)', response, re.IGNORECASE)
    return float(score_match.group(1)) if score_match else default_score


def extract_bullet_points_from_response(response: str, max_points: int = 5) -> List[str]:
    """Extract bullet points from response using JSON or text parsing"""
    json_data = safe_json_parse(response)
    if json_data and isinstance(json_data, dict):
        # Try common list field names
        for key in ['suggestions', 'risks', 'points', 'improvements']:
            if key in json_data and isinstance(json_data[key], list):
                return json_data[key][:max_points]
            elif key in json_data and isinstance(json_data[key], str):
                items = [item.strip() for item in re.split(r'[•\-*\n]', json_data[key]) if item.strip()]
                return items[:max_points]

    # Fallback to text parsing
    lines = response.split('\n')
    bullet_points = []
    for line in lines:
        line = line.strip()
        if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
            cleaned = re.sub(r'^[-•*]\s*', '', line)
            if cleaned:
                bullet_points.append(cleaned)
    return bullet_points[:max_points]


def extract_scores_from_response(response: str, criteria: List[str], default_score: float = 3.0) -> Dict[str, float]:
    """Extract scores for multiple criteria from response using JSON or text parsing"""
    json_data = safe_json_parse(response)
    if json_data and isinstance(json_data, dict):
        # Try to get scores from JSON structure
        scores = {}

        # First check if there's a criteria_scores object
        if 'criteria_scores' in json_data and isinstance(json_data['criteria_scores'], dict):
            criteria_scores = json_data['criteria_scores']
            for criterion in criteria:
                if criterion in criteria_scores:
                    scores[criterion] = float(criteria_scores[criterion])
                else:
                    scores[criterion] = default_score
        else:
            # Fallback to direct key lookup
            for criterion in criteria:
                score_key = f"{criterion}_score"
                if score_key in json_data:
                    scores[criterion] = float(json_data[score_key])
                elif criterion in json_data:
                    scores[criterion] = float(json_data[criterion])
                else:
                    scores[criterion] = default_score
        return scores
    else:
        # Fallback to text parsing
        scores = {}
        for criterion in criteria:
            pattern = rf"{criterion}.*?score.*?(\d(?:\.\d)?)"
            match = re.search(pattern, response, re.IGNORECASE)
            scores[criterion] = float(match.group(1)) if match else default_score
        return scores


def extract_sections_from_response(response: str, section_keys: List[str], max_points: int = 5) -> Dict[str, List[str]]:
    """Extract sections with bullet points from response using JSON or text parsing"""
    json_data = safe_json_parse(response)
    if json_data and isinstance(json_data, dict):
        # Try to get sections from JSON structure
        result = {}
        for key in section_keys:
            if key in json_data and isinstance(json_data[key], list):
                result[key] = json_data[key][:max_points]
            elif key in json_data and isinstance(json_data[key], str):
                # Convert string to list by splitting on common delimiters
                items = [item.strip() for item in re.split(r'[•\-*\n]', json_data[key]) if item.strip()]
                result[key] = items[:max_points]
            else:
                result[key] = []
        return result
    else:
        # Fallback to text parsing
        section_patterns = {
            'strengths': ['strengths', 'strength'],
            'weaknesses': ['weakness', 'concerns', 'weaknesses'],
            'suggestions': ['suggestions', 'suggestion'],
            'priority_revisions': ['priority revision', 'priority revisions']
        }

        result = {}
        for key in section_keys:
            result[key] = []
            pattern_names = section_patterns.get(key, [key])
            for pattern_name in pattern_names:
                pattern = rf"{pattern_name}[:\-](.*?)(?={'|'.join(sum(section_patterns.values(), []))}|$)"
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    if content:
                        # Extract bullet points from content
                        lines = content.split('\n')
                        bullet_points = []
                        for line in lines:
                            line = line.strip()
                            if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                                cleaned = re.sub(r'^[-•*]\s*', '', line)
                                if cleaned:
                                    bullet_points.append(cleaned)
                        result[key] = bullet_points[:max_points]
                        break
        return result
