from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseLoader(ABC):
    """
    Abstract base class for all dataset loaders.
    It defines a standard interface for loading, splitting, and sampling data.
    """
    def __init__(self, path: Optional[str] = None):
        """
        Initialize the loader.

        Args:
            path (Optional[str]): Local path to the dataset. If None, try to load from Hub.
        """
        self.path = path
        self.data = None
        self._load_data()

    @abstractmethod
    def _load_data(self):
        """
        Load dataset from source (local or Hub).
        Subclasses must implement this method to load data into self.data.
        self.data format is typically a dict with keys like 'train', 'test', values are data lists.
        """
        pass

    def get_split(self, split_name: str, num_samples: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get data for specified split

        Args:
            split_name: Split name ('train', 'test', etc.)
            num_samples: Number of samples needed
            offset: Offset to ensure validation and test sets don't overlap
        """
        if self.data is None:
            self._load_data()
        
        split_data = self.data.get(split_name, [])
        
        if num_samples is not None:
            # No longer reshuffle every time, data has been shuffled during loading
            # Split different datasets through offset
            return split_data[offset:offset + num_samples]
        else:
            # If num_samples is None, return all data
            return split_data 