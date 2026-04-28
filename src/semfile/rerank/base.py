"""Base protocol for rerankers."""

from typing import Protocol

from semfile.store.chromadb_store import SearchResult


class Reranker(Protocol):
    """Protocol for a second-pass reranker over initial vector-search results."""

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_n: int,
    ) -> list[SearchResult]:
        """Reorder candidates by relevance to the query.

        Implementations should populate `rerank_score` on returned results
        and return at most `top_n` items, best first. On failure, implementations
        should fall back to the original order rather than raising.
        """
        ...
