"""Base agent interface for extraction workflows."""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Abstract base class for extraction agents.
    
    All agents should implement the `run` method which executes the
    extraction workflow and returns structured results.
    """
    
    @abstractmethod
    def run(self, text: str, **kwargs) -> dict[str, Any]:
        """Execute the extraction workflow.
        
        Args:
            text: Input text to process
            **kwargs: Additional parameters specific to the agent
            
        Returns:
            Dictionary containing extraction results
        """
        pass
