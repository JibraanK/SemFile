"""Indexing pipeline: discover files, embed, and store."""

import concurrent.futures
import logging
import threading
import time
from collections import deque
from pathlib import Path

from semfile.config import Config, WatchDirectory
from semfile.embeddings.base import EmbeddingProvider
from semfile.indexer.scanner import scan_directory, scan_path
from semfile.store.chromadb_store import ChromaStore, FileRecord
from semfile.thumbnails.generator import generate_thumbnail

logger = logging.getLogger(__name__)

# Concurrency tuned for the Gemini API limits (100 req/min peak, 1000 req/day)
# and an 8GB M1 Air: more workers means more files held in memory at once.
DEFAULT_INDEX_WORKERS = 4
DEFAULT_RATE_LIMIT_PER_MINUTE = 90  # 10% headroom under the 100 RPM peak
DAILY_REQUEST_LIMIT = 1000


class _RateLimiter:
    """Sliding-window rate limiter shared across indexer worker threads."""

    def __init__(self, max_per_minute: int):
        self._max = max_per_minute
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= 60.0:
                    self._times.popleft()
                if len(self._times) < self._max:
                    self._times.append(now)
                    return
                wait = 60.0 - (now - self._times[0])
            time.sleep(max(0.05, wait))


class Indexer:
    """Indexes files by scanning directories, generating embeddings, and storing them."""

    def __init__(
        self,
        config: Config,
        provider: EmbeddingProvider,
        store: ChromaStore,
        max_workers: int = DEFAULT_INDEX_WORKERS,
        rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE,
    ):
        self.config = config
        self.provider = provider
        self.store = store
        self._max_workers = max_workers
        self._rate_limiter = _RateLimiter(rate_limit_per_minute)

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
        """Embed and store files concurrently, skipping unchanged ones."""
        todo: list[tuple[Path, str, str, float]] = []
        for file_path, file_type, mime_type in files:
            try:
                mtime = file_path.stat().st_mtime
            except OSError:
                logger.error("Failed to stat %s", file_path, exc_info=True)
                stats["failed"] += 1
                continue

            stored_mtime = self.store.get_modified_at(str(file_path))
            if stored_mtime is not None and abs(stored_mtime - mtime) < 1.0:
                stats["skipped"] += 1
                continue

            todo.append((file_path, file_type, mime_type, mtime))

        if not todo:
            return

        total = len(todo)
        if total > DAILY_REQUEST_LIMIT:
            logger.warning(
                "%d files queued but Gemini daily limit is %d requests/day; "
                "expect some failures past the cap.",
                total,
                DAILY_REQUEST_LIMIT,
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="semfile-index",
        ) as pool:
            futures = {
                pool.submit(self._embed_one, idx, total, task): task
                for idx, task in enumerate(todo, 1)
            }

            for future in concurrent.futures.as_completed(futures):
                file_path, file_type, mime_type, mtime = futures[future]
                try:
                    embedding, thumbnail_path = future.result()
                except Exception:
                    logger.error("Failed to embed %s", file_path, exc_info=True)
                    stats["failed"] += 1
                    continue

                try:
                    record = FileRecord(
                        file_path=str(file_path),
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
                    logger.error("Failed to store %s", file_path, exc_info=True)
                    stats["failed"] += 1

    def _embed_one(
        self,
        idx: int,
        total: int,
        task: tuple[Path, str, str, float],
    ) -> tuple[list[float], str]:
        """Worker: rate-limited embed + thumbnail. Runs in a pool thread."""
        file_path, file_type, mime_type, _ = task
        self._rate_limiter.acquire()
        logger.info("[%d/%d] Embedding: %s", idx, total, file_path.name)
        embedding = self.provider.embed_file(file_path, mime_type)
        thumbnail_path = generate_thumbnail(
            file_path, file_type, self.config.thumbnail_dir
        )
        return embedding, thumbnail_path

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
