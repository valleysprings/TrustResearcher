"""
Base Agent

Abstract base class providing common interface for all research agents.
Defines standard methods for information gathering, idea generation, and evaluation.
"""


class BaseAgent:
    def __init__(self, name):
        self.name = name

    def gather_information(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def generate_ideas(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def critique_ideas(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def refine_ideas(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_name(self):
        return self.name