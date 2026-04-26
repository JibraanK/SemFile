"""File system scanner for discovering files to index."""

import logging
from pathlib import Path

from semfile.config import WatchDirectory, get_file_type, get_mime_type

logger = logging.getLogger(__name__)


def scan_directory(watch_dir: WatchDirectory) -> list[tuple[Path, str, str]]:
    """Scan a directory for supported files.

    Returns a list of (file_path, file_type, mime_type) tuples.
    """
    results = []
    root = watch_dir.path

    if not root.exists():
        logger.warning("Watch directory does not exist: %s", root)
        return results

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lstrip(".").lower()
        if watch_dir.extensions and ext not in watch_dir.extensions:
            continue

        file_type = get_file_type(ext)
        mime_type = get_mime_type(ext)
        if file_type is None or mime_type is None:
            continue

        results.append((file_path, file_type, mime_type))

    logger.info("Found %d files in %s", len(results), root)
    return results


def scan_path(path: Path, extensions: set[str] | None = None) -> list[tuple[Path, str, str]]:
    """Scan an arbitrary path (used by `semfile index --path`).

    Returns a list of (file_path, file_type, mime_type) tuples.
    """
    path = path.resolve()

    if path.is_file():
        ext = path.suffix.lstrip(".").lower()
        file_type = get_file_type(ext)
        mime_type = get_mime_type(ext)
        if file_type and mime_type:
            return [(path, file_type, mime_type)]
        return []

    watch_dir = WatchDirectory(path=path, extensions=extensions or set())
    return scan_directory(watch_dir)
