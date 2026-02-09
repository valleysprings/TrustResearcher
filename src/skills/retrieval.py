"""
Prompts for RetrievalAgent - literature search and query generation system
"""

# ==================== Topic Decomposition ====================
# Breaks down a research topic into searchable concepts

TOPIC_DECOMPOSITION_SYSTEM_PROMPT = """You are an expert research librarian and computational linguist specializing in academic literature search optimization."""

TOPIC_DECOMPOSITION_USER_PROMPT = """Decompose this research topic into strategically optimized search concepts for academic paper discovery.

Research Topic: "{topic}"

ANALYSIS STRATEGY:
1. DOMAIN ANALYSIS: Identify the primary research domain(s) and subfields
2. MULTI-LEVEL DECOMPOSITION: Extract concepts at different levels of abstraction
3. TERMINOLOGY VARIANTS: Include both established and emerging terminology
4. METHODOLOGICAL ASPECTS: Consider approaches, techniques, and tools
5. INTERDISCIPLINARY CONNECTIONS: Identify relevant adjacent fields

CONCEPT SELECTION CRITERIA:
- Core theoretical concepts and frameworks
- Methodological approaches and algorithms
- Application domains and use cases
- Technical terms that appear in paper titles/abstracts
- Both broad umbrella terms and specific technical terminology
- Consider synonym variations and field-specific jargon
- Include measurement/evaluation concepts when relevant

OPTIMIZATION RULES:
- Each concept: 1-8 words, academically precise
- Mix of broad (high recall) and specific (high precision) terms
- Prioritize terms that distinguish this topic from similar ones
- Include both noun phrases and adjective-noun combinations
- Consider temporal aspects (recent vs. established concepts)

REQUIRED ANALYSIS STEPS:
1. **Domain Classification**: What primary research field(s) and subfields does this belong to?
2. **Core Concepts**: What are the 2-3 most fundamental concepts?
3. **Methodological Terms**: What approaches, techniques, or algorithms are relevant?
4. **Application Context**: What domains or use cases apply?
5. **Technical Specifications**: What specific technical terminology would researchers use?
6. **Interdisciplinary Links**: What adjacent fields might have relevant work?

CONCEPT GENERATION REQUIREMENTS:
- Generate 8-15 concepts covering different abstraction levels
- Include both established academic terms and emerging terminology
- Consider how different research communities might phrase the same concepts
- Include evaluation/measurement terms when applicable
- Mix broad umbrella terms with specific technical phrases

EXAMPLES OF CONCEPT TYPES TO INCLUDE:
- Theoretical frameworks: "reinforcement learning", "causal inference"
- Methodological approaches: "deep neural networks", "bayesian optimization"
- Application domains: "autonomous vehicles", "drug discovery"
- Technical components: "attention mechanisms", "graph neural networks"
- Evaluation concepts: "sample efficiency", "generalization bounds"

OUTPUT FORMAT:
Return ONLY the final comma-separated concept list, ordered from most fundamental to most specific.
Do not include explanations or analysis text in your response."""


# ==================== Query Generation ====================
# Generates optimized search queries for academic paper databases

QUERY_GENERATION_SYSTEM_PROMPT = """You are an expert in academic search query optimization, specializing in crafting effective queries for scholarly databases like Semantic Scholar, Google Scholar, and PubMed."""

QUERY_GENERATION_USER_PROMPT = """Generate optimized search queries for finding relevant academic papers on this research topic.

Research Topic: "{topic}"
Extracted Concepts: {concepts}

QUERY GENERATION STRATEGY:
1. KEYWORD SELECTION: Choose the most distinctive and searchable terms
2. SIMPLICITY: Keep queries simple - complex boolean operators often fail
3. TERMINOLOGY VARIANTS: Include different phrasings and synonyms
4. COVERAGE: Generate diverse queries covering different aspects
5. SPECIFICITY BALANCE: Mix broad and specific queries

QUERY CONSTRUCTION RULES:
- Each query: 2-6 keywords, simple and natural
- Prefer simple keyword combinations over complex boolean logic
- Use natural language phrases that appear in paper titles
- Avoid excessive use of quotes, AND, OR operators
- Keep queries focused but not overly restrictive

REQUIRED QUERY TYPES:
1. **Core Query**: Direct search using main topic terms
2. **Methodological Query**: Focus on techniques and approaches
3. **Application Query**: Focus on use cases and domains
4. **Broad Query**: Capture survey and review papers
5. **Variant Queries**: Use synonyms and alternative phrasings

QUERY GENERATION REQUIREMENTS:
- Generate 5-10 distinct queries for comprehensive coverage
- Each query should target a different aspect or angle
- Queries should complement each other (minimize overlap)
- Include at least one broad query and one specific query
- Use terminology from the concept list when relevant
- Cover different methodological approaches and application domains

EXAMPLES OF GOOD QUERIES (SIMPLE AND EFFECTIVE):
- graph neural networks molecular prediction
- community detection dynamic networks
- transfer learning low-resource NLP
- adversarial robustness computer vision
- causal inference observational data
- reinforcement learning robotics manipulation
- attention mechanisms transformer efficiency
- k-truss decomposition algorithms
- incremental graph maintenance
- scalable graph algorithms

EXAMPLES OF QUERY VARIATIONS:
- Core: k-truss breaking problem
- Methodological: incremental k-truss maintenance
- Application: large-scale graph algorithms
- Broad: graph decomposition survey
- Variant: truss-based community detection
- Specific: localized k-truss updates
- Algorithmic: approximate k-truss algorithms

OUTPUT FORMAT:
Return ONLY a comma-separated list of search queries.
Each query should be simple keywords (2-6 words), no complex boolean operators.
Do not include explanations, numbering, or query labels in your response."""