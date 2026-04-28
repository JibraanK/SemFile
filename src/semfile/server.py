"""FastAPI server for Raycast and other HTTP clients."""

import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse

from semfile.config import load_config

logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="SemFile", version="0.1.0")

# Lazy-loaded singletons (initialized on first request)
_state: dict = {}


def _get_state() -> dict:
    """Lazy-initialize config, provider, store, searcher."""
    if not _state:
        from semfile.embeddings.gemini import GeminiEmbeddingProvider
        from semfile.search.searcher import Searcher
        from semfile.store.chromadb_store import ChromaStore

        config = load_config()
        config.ensure_dirs()
        provider = GeminiEmbeddingProvider(dimensions=config.embedding_dimensions)
        store = ChromaStore(db_path=config.db_path)
        searcher = Searcher(provider, store)
        _state.update(
            config=config,
            provider=provider,
            store=store,
            searcher=searcher,
        )
    return _state


@app.get("/search")
def search(
    q: str = Query(..., description="Search query"),
    type: list[str] | None = Query(None, description="File type filter"),
    dir: list[str] | None = Query(None, description="Directory filter"),
    limit: int = Query(20, ge=1, le=100),
) -> dict:
    """Semantic search across indexed files."""
    state = _get_state()
    results = state["searcher"].search(
        query=q,
        limit=limit,
        file_types=type,
        directories=dir,
    )

    config = state["config"]
    return {
        "results": [
            {
                "file_path": r.file_path,
                "filename": r.filename,
                "file_type": r.file_type,
                "mime_type": r.mime_type,
                "file_size": r.file_size,
                "similarity": round(1 - r.distance, 4),
                "thumbnail_url": (
                    f"/thumbnails/{Path(r.thumbnail_path).name}"
                    if r.thumbnail_path
                    else None
                ),
            }
            for r in results
        ],
        "query": q,
        "count": len(results),
    }


@app.get("/status")
def status() -> dict:
    """Index statistics and storage info."""
    state = _get_state()
    store = state["store"]
    config = state["config"]

    db_size = _dir_size(config.db_path)
    thumb_size = _dir_size(config.thumbnail_dir)

    return {
        "total": store.count(),
        "by_type": store.count_by_type(),
        "storage": {
            "total_bytes": db_size + thumb_size,
            "db_bytes": db_size,
            "thumbnail_bytes": thumb_size,
        },
        "embedding_dimensions": config.embedding_dimensions,
    }


@app.get("/thumbnails/{filename}")
def thumbnail(filename: str) -> FileResponse:
    """Serve a thumbnail image."""
    state = _get_state()
    thumb_path = state["config"].thumbnail_dir / filename
    if not thumb_path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(thumb_path, media_type="image/jpeg")


@app.post("/index")
def trigger_index(body: dict | None = None) -> dict:
    """Trigger indexing (synchronous for now)."""
    from semfile.indexer.indexer import Indexer

    state = _get_state()
    indexer = Indexer(state["config"], state["provider"], state["store"])

    path = body.get("path") if body else None
    file_types = body.get("file_types") if body else None
    type_filter = set(file_types) if file_types else None

    if path:
        stats = indexer.index_path(Path(path), file_types=type_filter)
    else:
        stats = indexer.index_all(file_types=type_filter)

    return stats


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
