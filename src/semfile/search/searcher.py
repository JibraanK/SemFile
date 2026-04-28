"""Search module: query embedding + vector search with filters."""

import logging
from pathlib import Path

from semfile.embeddings.base import EmbeddingProvider
from semfile.rerank.base import Reranker
from semfile.store.chromadb_store import ChromaStore, SearchResult

logger = logging.getLogger(__name__)


class Searcher:
    """Semantic search across indexed files."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        store: ChromaStore,
        reranker: Reranker | None = None,
    ):
        self.provider = provider
        self.store = store
        self.reranker = reranker

    def search(
        self,
        query: str,
        limit: int = 20,
        file_types: list[str] | None = None,
        directories: list[str] | None = None,
        rerank: bool = False,
        rerank_top_n: int = 20,
    ) -> list[SearchResult]:
        """Search for files matching a text query.

        Args:
            query: Natural language search query.
            limit: Maximum number of results to return.
            file_types: Optional filter by file type (image, video, audio, document, text).
            directories: Optional filter to scope search to specific directories.
            rerank: If True and a reranker is configured, run a multimodal
                second pass over the top candidates.
            rerank_top_n: Number of candidates to fetch and rerank when
                rerank is enabled. The wider candidate pool gives the
                reranker room to surface deeper matches.

        Returns:
            List of SearchResult sorted by similarity (best first), or by
            rerank score when reranking is active.
        """
        use_rerank = rerank and self.reranker is not None
        fetch_n = max(rerank_top_n, limit) if use_rerank else limit

        logger.info(
            "Searching: %r (limit=%d, fetch=%d, rerank=%s, types=%s, dirs=%s)",
            query, limit, fetch_n, use_rerank, file_types, directories,
        )
        query_embedding = self.provider.embed_query(query)
        results = self._run(query_embedding, fetch_n, file_types, directories)

        if use_rerank and results:
            results = self.reranker.rerank(query, results, top_n=limit)
        else:
            results = results[:limit]
        return results

    def search_by_file(
        self,
        file_path: Path,
        mime_type: str,
        limit: int = 20,
        file_types: list[str] | None = None,
        directories: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search for files similar to a given file (image, video, audio, doc, text).

        Args:
            file_path: Path to the query file.
            mime_type: MIME type of the query file.
            limit: Maximum number of results.
            file_types: Optional filter by file type.
            directories: Optional filter to scope search to specific directories.

        Returns:
            List of SearchResult sorted by similarity (best first).
        """
        logger.info(
            "Searching by file: %s (mime=%s, limit=%d, types=%s, dirs=%s)",
            file_path, mime_type, limit, file_types, directories,
        )
        query_embedding = self.provider.embed_file(file_path, mime_type)
        return self._run(query_embedding, limit, file_types, directories)

    def _run(
        self,
        query_embedding: list[float],
        limit: int,
        file_types: list[str] | None,
        directories: list[str] | None,
    ) -> list[SearchResult]:
        resolved_dirs = None
        if directories:
            resolved_dirs = [str(Path(d).expanduser().resolve()) for d in directories]

        results = self.store.search(
            query_embedding=query_embedding,
            limit=limit,
            file_types=file_types,
            directories=resolved_dirs,
        )
        logger.info("Found %d results", len(results))
        return results
