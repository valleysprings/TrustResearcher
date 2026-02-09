"""
Planning Module

ARCHITECTURE:
- GLOBAL GROUNDING (once at start): Comprehensive context from KG + literature
- PER-IDEA GENERATION (Base/GoT/Cross each independently):
  - Own decomposition based on strategy-specific context
  - Own directions with gaps merged in
"""

from typing import Dict, List, Any
from ...utils.llm_interface import LLMInterface
from ...utils.text_utils import safe_json_parse
from ...knowledge_graph.kg_ops import KGOps
from ...skills.ideagen.planning import (
    GLOBAL_GROUNDING_SYSTEM_PROMPT, GLOBAL_GROUNDING_USER_PROMPT,
    STRATEGIC_DIRECTIONS_BASE_SYSTEM_PROMPT, STRATEGIC_DIRECTIONS_BASE_USER_PROMPT,
    STRATEGIC_DIRECTIONS_GOT_SYSTEM_PROMPT, STRATEGIC_DIRECTIONS_GOT_USER_PROMPT,
    STRATEGIC_DIRECTIONS_CROSS_SYSTEM_PROMPT, STRATEGIC_DIRECTIONS_CROSS_USER_PROMPT,
)


class PlanningModule:
    """
    Research planning module.

    Responsibilities:
    - Global grounding: comprehensive context from KG + literature (once at start)
    - Per-idea decomposition: facets based on strategy-specific context
    - Per-idea directions: with gaps merged in
    """

    def __init__(self, knowledge_graph: KGOps = None, llm_interface=None, planning_config: Dict = None):
        self.llm = llm_interface or LLMInterface()
        self.planning_config = planning_config or {}
        self.pm_config = self.planning_config
        self.knowledge_graph = knowledge_graph

    async def create_global_grounding(self, seed_topic: str, literature_papers: List = None) -> Dict:
        """
        Create global grounding context (once at start).

        Returns:
        - global_grounding: Comprehensive context (field_overview, key_findings, kg_insights, landscape_gaps)
        - cluster_analysis: KG cluster structure for downstream use
        - raw contexts: kg_context, literature_context for reference
        """
        # Build raw contexts
        kg_context = self._build_kg_context(seed_topic)
        literature_context = self._build_literature_context(literature_papers)

        # Generate global grounding via LLM
        grounding = await self._generate_global_grounding(seed_topic, kg_context, literature_context)

        # Analyze clusters for downstream use
        cluster_analysis = self._analyze_clusters(seed_topic)

        return {
            'seed_topic': seed_topic,
            'global_grounding': grounding,
            'cluster_analysis': cluster_analysis,
            'kg_context': kg_context,
            'literature_context': literature_context,
        }

    async def create_plan(self, seed_topic: str, literature_papers: List = None) -> Dict:
        """
        Create global research plan (shared context only).

        Returns:
            Plan dict with: global_grounding (shared across all ideas)
        """
        grounding_result = await self.create_global_grounding(seed_topic, literature_papers)

        return {
            'seed_topic': seed_topic,
            'global_grounding': grounding_result['global_grounding'],
            'cluster_analysis': grounding_result['cluster_analysis'],
        }

    def _build_kg_context(self, seed_topic: str) -> str:
        """Build context from knowledge graph."""
        if not self.knowledge_graph:
            return ""

        try:
            parts = []

            # Related entities
            related = self.knowledge_graph.get_related_entities(seed_topic, max_distance=2)
            if related:
                max_entities = self.pm_config['knowledge_graph']['max_related_entities']
                parts.append(f"Related entities: {', '.join(list(related)[:max_entities])}")

            # Cross-cluster connections
            cross_conn = self.knowledge_graph.get_cross_cluster_connections()
            if cross_conn:
                max_conn = self.pm_config['knowledge_graph']['max_cross_connections']
                parts.append(f"Cross-domain connections: {cross_conn[:max_conn]}")

            return "\n".join(parts)
        except Exception:
            return ""

    def _build_literature_context(self, papers: List = None) -> str:
        """Build context from literature papers."""
        if not papers:
            return ""

        parts = ["KEY PAPERS:"]
        max_papers = self.pm_config['literature']['max_referenced_papers']

        for i, paper in enumerate(papers[:max_papers], 1):
            info = f"{i}. \"{paper.title}\" ({paper.year})"
            if paper.authors:
                max_authors = self.pm_config['literature']['max_displayed_authors']
                info += f" - {', '.join(paper.authors[:max_authors])}"
            parts.append(info)

            if paper.keywords:
                max_kw = self.pm_config['literature']['max_keywords_display']
                parts.append(f"   Keywords: {', '.join(paper.keywords[:max_kw])}")

        return "\n".join(parts)

    def _analyze_clusters(self, seed_topic: str) -> Dict[str, Any]:
        """Analyze KG cluster structure."""
        if not self.knowledge_graph:
            return {}

        try:
            clusters = self.knowledge_graph.get_entity_cluster(seed_topic)
            if not clusters:
                return {}

            cluster_map = clusters if isinstance(clusters, dict) else {0: list(clusters)}
            cross_bridges = self.knowledge_graph.get_cross_cluster_connections()

            return {
                'num_clusters': len(cluster_map),
                'clusters': [
                    {'cluster_id': cid, 'size': len(entities), 'sample': list(entities)[:5]}
                    for cid, entities in cluster_map.items()
                ],
                'cross_cluster_bridges': cross_bridges[:10] if cross_bridges else []
            }
        except Exception:
            return {}

    async def _generate_global_grounding(self, seed_topic: str, kg_context: str, literature_context: str) -> Dict:
        """Generate comprehensive global grounding via LLM. Returns parsed dict."""
        user_prompt = GLOBAL_GROUNDING_USER_PROMPT.format(
            seed_topic=seed_topic,
            kg_context=kg_context or "No knowledge graph context available.",
            literature_context=literature_context or "No literature context available."
        )

        response = await self.llm.generate_with_system_prompt(
            GLOBAL_GROUNDING_SYSTEM_PROMPT,
            user_prompt,
            caller="planning_global_grounding"
        )

        grounding = safe_json_parse(response)
        if not grounding or not isinstance(grounding, dict):
            raise ValueError(f"Failed to parse global grounding response: {response[:100]}")

        return grounding

    # =========================================================================
    # PER-IDEA DIRECTION GENERATION (called by variant generators)
    # =========================================================================

    async def generate_base_directions(self, seed_topic: str, global_grounding: Dict, cluster_context: str) -> Dict:
        """Generate directions for base variant (cluster-focused)."""
        grounding_str = self._format_global_grounding(global_grounding)

        user_prompt = STRATEGIC_DIRECTIONS_BASE_USER_PROMPT.format(
            seed_topic=seed_topic,
            global_grounding=grounding_str,
            cluster_context=cluster_context
        )

        response = await self.llm.generate_with_system_prompt(
            STRATEGIC_DIRECTIONS_BASE_SYSTEM_PROMPT,
            user_prompt,
            caller="planning_directions_base"
        )

        return self._parse_directions_with_gaps(response, "cluster_gaps")

    async def generate_got_directions(self, seed_topic: str, global_grounding: Dict, path_context: str) -> Dict:
        """Generate directions for GoT variant (path-focused)."""
        grounding_str = self._format_global_grounding(global_grounding)

        user_prompt = STRATEGIC_DIRECTIONS_GOT_USER_PROMPT.format(
            seed_topic=seed_topic,
            global_grounding=grounding_str,
            path_context=path_context
        )

        response = await self.llm.generate_with_system_prompt(
            STRATEGIC_DIRECTIONS_GOT_SYSTEM_PROMPT,
            user_prompt,
            caller="planning_directions_got"
        )

        return self._parse_directions_with_gaps(response, "path_gaps")

    async def generate_cross_directions(
        self, seed_topic: str, global_grounding: Dict,
        bridge_context: str, sampled_ideas_summary: str
    ) -> Dict:
        """Generate directions for cross-pollination (bridge-focused)."""
        grounding_str = self._format_global_grounding(global_grounding)

        user_prompt = STRATEGIC_DIRECTIONS_CROSS_USER_PROMPT.format(
            seed_topic=seed_topic,
            global_grounding=grounding_str,
            bridge_context=bridge_context,
            sampled_ideas_summary=sampled_ideas_summary
        )

        response = await self.llm.generate_with_system_prompt(
            STRATEGIC_DIRECTIONS_CROSS_SYSTEM_PROMPT,
            user_prompt,
            caller="planning_directions_cross"
        )

        return self._parse_directions_with_gaps(response, "bridge_gaps")

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _format_global_grounding(self, grounding: Dict) -> str:
        """Format global grounding dict as string for prompts."""
        parts = []
        if grounding.get("field_overview"):
            parts.append(f"FIELD OVERVIEW:\n{grounding['field_overview']}")
        if grounding.get("key_findings"):
            parts.append(f"KEY FINDINGS:\n{grounding['key_findings']}")
        if grounding.get("kg_insights"):
            parts.append(f"KG INSIGHTS:\n{grounding['kg_insights']}")
        if grounding.get("landscape_gaps"):
            parts.append(f"LANDSCAPE GAPS:\n{grounding['landscape_gaps']}")
        return "\n\n".join(parts) if parts else "No global grounding available."

    def _parse_directions_with_gaps(self, response: str, gaps_key: str) -> Dict:
        """Parse LLM response with gaps and directions."""
        # Handle empty response
        if not response or not response.strip():
            print(f"    Warning: Empty LLM response for {gaps_key}, using defaults")
            return {
                "gaps": ["Explore novel approaches in this domain"],
                "directions": ["Investigate underexplored methodologies"]
            }

        data = safe_json_parse(response)
        if data and isinstance(data, dict):
            return {
                "gaps": data.get(gaps_key, []),
                "directions": data.get("directions", [])
            }

        # Fallback: parse as bullet list (directions only)
        directions = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                direction = line.lstrip('- •0123456789. ').strip()
                if len(direction) > 10:
                    directions.append(direction)
        return {"gaps": [], "directions": directions}

    def _format_clusters(self, cluster_analysis: Dict) -> str:
        """Format cluster analysis for LLM."""
        if not cluster_analysis or not cluster_analysis.get('clusters'):
            return "No cluster analysis available"

        lines = []
        for cluster in cluster_analysis.get('clusters', [])[:5]:
            lines.append(f"Cluster {cluster['cluster_id']}: {cluster['size']} entities")
            lines.append(f"  Sample: {', '.join(cluster['sample'][:3])}")
        return '\n'.join(lines)
