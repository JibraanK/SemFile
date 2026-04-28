"""ChromaDB vector store for file embeddings."""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb

logger = logging.getLogger(__name__)

COLLECTION_NAME = "files"


@dataclass
class FileRecord:
    """A file record stored in the vector database. One record per chunk for
    long videos; chunk_index=0 / total_chunks=1 for everything else."""

    file_path: str
    parent_dir: str
    filename: str
    file_type: str  # image, video, audio, document, text
    mime_type: str
    file_size: int
    modified_at: float  # Unix timestamp
    indexed_at: float  # Unix timestamp
    thumbnail_path: str
    chunk_index: int = 0
    chunk_start_seconds: float = 0.0
    total_chunks: int = 1


@dataclass
class SearchResult:
    """A search result from the vector database."""

    file_path: str
    parent_dir: str
    filename: str
    file_type: str
    mime_type: str
    file_size: int
    thumbnail_path: str
    distance: float
    chunk_index: int = 0
    chunk_start_seconds: float = 0.0
    total_chunks: int = 1


def _file_id(file_path: str, chunk_index: int = 0) -> str:
    """Generate a stable ID for a (file path, chunk index) pair."""
    return hashlib.sha256(f"{file_path}#chunk_{chunk_index}".encode()).hexdigest()


class ChromaStore:
    """Persistent ChromaDB store for file embeddings."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(db_path))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, record: FileRecord, embedding: list[float]) -> None:
        """Add or update a file record with its embedding."""
        file_id = _file_id(record.file_path, record.chunk_index)
        metadata = {
            "file_path": record.file_path,
            "parent_dir": record.parent_dir,
            "filename": record.filename,
            "file_type": record.file_type,
            "mime_type": record.mime_type,
            "file_size": record.file_size,
            "modified_at": record.modified_at,
            "indexed_at": record.indexed_at,
            "thumbnail_path": record.thumbnail_path,
            "chunk_index": record.chunk_index,
            "chunk_start_seconds": record.chunk_start_seconds,
            "total_chunks": record.total_chunks,
        }
        self._collection.upsert(
            ids=[file_id],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def get_modified_at(self, file_path: str) -> float | None:
        """Get the stored modified_at timestamp for a file, or None if not indexed.
        Reads any chunk's mtime since all chunks of a file share the same value."""
        result = self._collection.get(
            where={"file_path": file_path},
            limit=1,
            include=["metadatas"],
        )
        if result["metadatas"]:
            return result["metadatas"][0].get("modified_at")
        return None

    def remove(self, file_path: str) -> None:
        """Remove all records (including every chunk) for a file path."""
        try:
            self._collection.delete(where={"file_path": file_path})
        except Exception:
            pass  # Already removed

    def search(
        self,
        query_embedding: list[float],
        limit: int = 20,
        file_types: list[str] | None = None,
        directories: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search for similar files with optional filters."""
        where = self._build_where(file_types, directories)

        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": limit,
            "include": ["metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        search_results = []
        if results["metadatas"] and results["distances"]:
            for metadata, distance in zip(
                results["metadatas"][0], results["distances"][0]
            ):
                search_results.append(
                    SearchResult(
                        file_path=metadata["file_path"],
                        parent_dir=metadata["parent_dir"],
                        filename=metadata["filename"],
                        file_type=metadata["file_type"],
                        mime_type=metadata["mime_type"],
                        file_size=metadata["file_size"],
                        thumbnail_path=metadata["thumbnail_path"],
                        distance=distance,
                        chunk_index=metadata.get("chunk_index", 0),
                        chunk_start_seconds=metadata.get("chunk_start_seconds", 0.0),
                        total_chunks=metadata.get("total_chunks", 1),
                    )
                )
        return search_results

    def _build_where(
        self,
        file_types: list[str] | None,
        directories: list[str] | None,
    ) -> dict | None:
        """Build a ChromaDB where filter from file types and directories."""
        conditions = []

        if file_types and len(file_types) == 1:
            conditions.append({"file_type": file_types[0]})
        elif file_types and len(file_types) > 1:
            conditions.append({"file_type": {"$in": file_types}})

        if directories:
            dir_conditions = [
                {"file_path": {"$contains": d}} for d in directories
            ]
            if len(dir_conditions) == 1:
                conditions.append(dir_conditions[0])
            else:
                conditions.append({"$or": dir_conditions})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def get_all_paths(self) -> set[str]:
        """Get all indexed file paths."""
        result = self._collection.get(include=["metadatas"])
        return {m["file_path"] for m in result["metadatas"]} if result["metadatas"] else set()

    def count(self) -> int:
        """Return the number of distinct indexed files (chunks of one file count once)."""
        return len(self.get_all_paths())

    def count_by_type(self) -> dict[str, int]:
        """Count indexed files grouped by type. Dedupes chunks of the same file."""
        result = self._collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        if result["metadatas"]:
            seen: set[str] = set()
            for m in result["metadatas"]:
                path = m.get("file_path", "")
                if path in seen:
                    continue
                seen.add(path)
                ft = m.get("file_type", "unknown")
                counts[ft] = counts.get(ft, 0) + 1
        return counts

    def reset(self) -> None:
        """Delete all data in the store."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
