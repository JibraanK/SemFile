"""Indexing pipeline: discover files, embed, and store."""

import logging
import time
from pathlib import Path

from semfile.config import Config, WatchDirectory
from semfile.embeddings.base import EmbeddingProvider
from semfile.indexer.scanner import scan_directory, scan_path
from semfile.store.chromadb_store import ChromaStore, FileRecord
from semfile.thumbnails.generator import generate_thumbnail

logger = logging.getLogger(__name__)


class Indexer:
    """Indexes files by scanning directories, generating embeddings, and storing them."""

    def __init__(
        self,
        config: Config,
        provider: EmbeddingProvider,
        store: ChromaStore,
    ):
        self.config = config
        self.provider = provider
        self.store = store

    def index_all(self, file_types: set[str] | None = None) -> dict[str, int]:
        """Index all configured watch directories. Returns stats."""
        stats = {"scanned": 0, "indexed": 0, "skipped": 0, "failed": 0, "removed": 0}

        all_files: list[tuple[Path, str, str]] = []
        for watch_dir in self.config.watch_dirs:
            all_files.extend(scan_directory(watch_dir))

        if file_types:
            all_files = [(p, ft, mt) for p, ft, mt in all_files if ft in file_types]

        stats["scanned"] = len(all_files)
        self._index_files(all_files, stats)
        self._cleanup_removed(all_files, stats)

        return stats

    def index_path(self, path: Path, extensions: set[str] | None = None, file_types: set[str] | None = None) -> dict[str, int]:
        """Index a specific path. Returns stats."""
        stats = {"scanned": 0, "indexed": 0, "skipped": 0, "failed": 0, "removed": 0}

        files = scan_path(path, extensions)
        if file_types:
            files = [(p, ft, mt) for p, ft, mt in files if ft in file_types]

        stats["scanned"] = len(files)
        self._index_files(files, stats)

        return stats

    def _index_files(
        self, files: list[tuple[Path, str, str]], stats: dict[str, int]
    ) -> None:
        """Embed and store files, skipping unchanged ones."""
        for i, (file_path, file_type, mime_type) in enumerate(files, 1):
            try:
                path_str = str(file_path)
                mtime = file_path.stat().st_mtime

                # Check if already indexed and unchanged
                stored_mtime = self.store.get_modified_at(path_str)
                if stored_mtime is not None and abs(stored_mtime - mtime) < 1.0:
                    stats["skipped"] += 1
                    continue

                # Generate embedding
                logger.info("[%d/%d] Embedding: %s", i, len(files), file_path.name)
                embedding = self.provider.embed_file(file_path, mime_type)

                # Generate thumbnail
                thumbnail_path = generate_thumbnail(
                    file_path, file_type, self.config.thumbnail_dir
                )

                # Store in vector DB
                record = FileRecord(
                    file_path=path_str,
                    parent_dir=str(file_path.parent),
                    filename=file_path.name,
                    file_type=file_type,
                    mime_type=mime_type,
                    file_size=file_path.stat().st_size,
                    modified_at=mtime,
                    indexed_at=time.time(),
                    thumbnail_path=thumbnail_path,
                )
                self.store.add(record, embedding)
                stats["indexed"] += 1

            except Exception:
                logger.error("Failed to index %s", file_path, exc_info=True)
                stats["failed"] += 1

    def _cleanup_removed(
        self, current_files: list[tuple[Path, str, str]], stats: dict[str, int]
    ) -> None:
        """Remove entries for files that no longer exist on disk."""
        current_paths = {str(f[0]) for f in current_files}
        indexed_paths = self.store.get_all_paths()

        for path_str in indexed_paths - current_paths:
            if not Path(path_str).exists():
                self.store.remove(path_str)
                stats["removed"] += 1
                logger.info("Removed deleted file from index: %s", path_str)
