"""Base protocol for embedding providers."""

from pathlib import Path
from typing import Protocol


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers. Implement this to add new models."""

    @property
    def dimensions(self) -> int:
        """The dimensionality of the output embeddings."""
        ...

    def embed_file(self, file_path: Path, mime_type: str) -> list[float]:
        """Generate an embedding for a file (image, video, audio, PDF, etc.)."""
        ...

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a text document (for indexing)."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a search query."""
        ...
