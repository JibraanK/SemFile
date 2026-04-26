"""Search module: query embedding + vector search with filters."""

import logging
from pathlib import Path

from semfile.embeddings.base import EmbeddingProvider
from semfile.store.chromadb_store import ChromaStore, SearchResult

logger = logging.getLogger(__name__)


class Searcher:
    """Semantic search across indexed files."""

    def __init__(self, provider: EmbeddingProvider, store: ChromaStore):
        self.provider = provider
        self.store = store

    def search(
        self,
        query: str,
        limit: int = 20,
        file_types: list[str] | None = None,
        directories: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search for files matching a text query.

        Args:
            query: Natural language search query.
            limit: Maximum number of results.
            file_types: Optional filter by file type (image, video, audio, document, text).
            directories: Optional filter to scope search to specific directories.

        Returns:
            List of SearchResult sorted by similarity (best first).
        """
        # Resolve directory paths
        resolved_dirs = None
        if directories:
            resolved_dirs = [str(Path(d).expanduser().resolve()) for d in directories]

        logger.info(
            "Searching: %r (limit=%d, types=%s, dirs=%s)",
            query, limit, file_types, resolved_dirs,
        )

        query_embedding = self.provider.embed_query(query)

        results = self.store.search(
            query_embedding=query_embedding,
            limit=limit,
            file_types=file_types,
            directories=resolved_dirs,
        )

        logger.info("Found %d results", len(results))
        return results
