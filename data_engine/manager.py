from abc import ABC, abstractmethod
from typing import List, Dict, Any

class DataProvider(ABC):
    """Abstract base class for all data providers (MT5, Bookmap, etc.)."""
    
    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def get_latest_data(self, symbol: str) -> Dict[str, Any]:
        pass

class DataManager:
    """Manages multiple data sources and feeds the feature engine."""
    
    def __init__(self):
        self.providers: Dict[str, DataProvider] = {}

    def add_provider(self, name: str, provider: DataProvider):
        self.providers[name] = provider

    def fetch_all(self, symbols: List[str]) -> Dict[str, Any]:
        results = {}
        for name, provider in self.providers.items():
            # results[name] = provider.get_latest_data(...)
            pass
        return results
