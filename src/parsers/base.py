"""
Base classes and data models for document parsing and chunking.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import os

@dataclass
class DocumentChunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        return cls(text=data["text"], metadata=data.get("metadata", {}))

class BaseParser:
    """Abstract interface for all document parsers."""
    
    def can_parse(self, file_path: str) -> bool:
        raise NotImplementedError

    def parse(self, file_path: str) -> List[DocumentChunk]:
        raise NotImplementedError
