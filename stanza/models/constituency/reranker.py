"""
Interface for a reranker

This module doesn't actually do anything
"""

from abc import ABC, abstractmethod

class Reranker(ABC):
    @abstractmethod
    def score_trees(trees):
        """
        Do something model specific to return scores for the trees
        """

    @abstractmethod
    def score_parse_results(parse_results):
        """
        Update the ParseResult objects with scores
        """
