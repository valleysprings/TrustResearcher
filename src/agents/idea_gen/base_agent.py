"""
Base Agent for Idea Generation

Shared base class for idea generation modules with common LLM interface initialization.
Provides standardized access to language models for all idea generation components.
"""

from typing import Dict
from ...utils.llm_interface import LLMInterface


class BaseAgent:
    """Base class for idea generation agents with shared LLM initialization"""

    def __init__(self, llm_interface: LLMInterface = None, llm_config: Dict = None):
        self.llm = llm_interface or LLMInterface(config=llm_config)