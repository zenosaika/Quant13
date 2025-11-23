from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class Agent(ABC):
    """Base class for all agents."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def run(self, state: Dict[str, Any]) -> Any:
        analysis = self._think(state)
        return self._generate_report(analysis, state)

    @abstractmethod
    def _think(self, state: Dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _generate_report(self, analysis: Any, state: Dict[str, Any]) -> Any:
        raise NotImplementedError
