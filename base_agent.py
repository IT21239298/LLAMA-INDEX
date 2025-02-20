import abc
from typing import Dict, Any, Awaitable

class BaseAgent(abc.ABC):
    """
    Abstract base class for all agents in the multi-agent system.
    Defines the core interface for agent interaction.
    """
    
    def __init__(self, name: str, priority: int):
        """
        Initialize a base agent.
        
        Args:
            name: Unique name of the agent
            priority: Priority level (lower number means higher priority)
        """
        self.name = name
        self.priority = priority
    
    @abc.abstractmethod
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Abstract method to process a query.
        
        Args:
            query: User's input query
        
        Returns:
            Dictionary containing query processing results
        """
        pass
    
    def evaluate_confidence(self, response: Dict[str, Any]) -> float:
        """
        Evaluate the confidence of the agent's response.
        
        Args:
            response: Agent's response dictionary
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Default confidence evaluation
        return response.get('confidence', 0.0)