# db/base.py
from abc import ABC, abstractmethod
from typing import Any, List, Optional

class DBBase(ABC):
    """
    Abstract base class for database connectors.
    """

    @abstractmethod
    def add_agent(self, agent_id: str, agent_info: dict) -> None:
        pass

    @abstractmethod
    def get_agent(self, agent_id: str) -> Optional[dict]:
        pass

    @abstractmethod
    def add_interaction(self, interaction_data: dict) -> None:
        pass

    @abstractmethod
    def list_interactions(self, agent_id: str) -> List[dict]:
        pass

    @abstractmethod
    def update_profile(self, agent_id: str, profile_vec: List[float]) -> None:
        pass

    @abstractmethod
    def get_profile(self, agent_id: str) -> Optional[List[float]]:
        pass
