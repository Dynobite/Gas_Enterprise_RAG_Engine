"""Base Parser Interface."""
from abc import ABC, abstractmethod
from typing import List
from src.chunker import TextChunk

class BaseDocumentParser(ABC):
    """Abstract base class for all file parsers."""
    @abstractmethod
    def parse(self, file_path: str) -> List[TextChunk]:
        """Parses a document file and returns a list of TextChunks."""
        pass
