"""FastAPI server for Raycast and other HTTP clients."""

import logging
import threading
import time
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
            reranker=None,
        )
    return _state


def _get_reranker(state: dict):
    """Lazy-initialize and cache the Gemini reranker."""
    if state.get("reranker") is None:
        from semfile.rerank.gemini import GeminiReranker

        state["reranker"] = GeminiReranker()
        state["searcher"].reranker = state["reranker"]
    return state["reranker"]


@app.get("/search")
def search(
    q: str = Query(..., description="Search query"),
    type: list[str] | None = Query(None, description="File type filter"),
    dir: list[str] | None = Query(None, description="Directory filter"),
    limit: int = Query(20, ge=1, le=100),
    rerank: bool = Query(False, description="Run multimodal reranker over candidates"),
    rerank_top_n: int = Query(20, ge=1, le=100, description="Candidate pool size when reranking"),
) -> dict:
    """Semantic search across indexed files."""
    state = _get_state()
    if rerank:
        try:
            _get_reranker(state)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    results = state["searcher"].search(
        query=q,
        limit=limit,
        file_types=type,
        directories=dir,
        rerank=rerank,
        rerank_top_n=rerank_top_n,
    )

    return {
        "results": [
            {
                "file_path": r.file_path,
                "filename": r.filename,
                "file_type": r.file_type,
                "mime_type": r.mime_type,
                "file_size": r.file_size,
                "similarity": round(1 - r.distance, 4),
                "rerank_score": r.rerank_score,
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
        "reranked": rerank,
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


_index_lock = threading.Lock()


def _empty_job() -> dict:
    return {
        "running": False,
        "started_at": None,
        "finished_at": None,
        "path": None,
        "file_types": None,
        "stats": None,
        "error": None,
        "count_at_start": None,
    }


def _run_index_job(path: str | None, file_types: list[str] | None) -> None:
    from semfile.indexer.indexer import Indexer

    state = _get_state()
    job = state["index_job"]
    indexer = Indexer(state["config"], state["provider"], state["store"])
    type_filter = set(file_types) if file_types else None

    try:
        if path:
            stats = indexer.index_path(Path(path), file_types=type_filter)
        else:
            stats = indexer.index_all(file_types=type_filter)
        job["stats"] = stats
    except Exception as exc:
        logger.exception("Index job failed")
        job["error"] = str(exc)
    finally:
        job["finished_at"] = time.time()
        job["running"] = False


@app.post("/index")
def trigger_index(body: dict | None = None) -> JSONResponse:
    """Kick off an index job in the background. Returns immediately."""
    state = _get_state()
    state.setdefault("index_job", _empty_job())
    job = state["index_job"]

    with _index_lock:
        if job["running"]:
            return JSONResponse(
                {"error": "index already running", "started_at": job["started_at"]},
                status_code=409,
            )

        path = body.get("path") if body else None
        file_types = body.get("file_types") if body else None

        job.update(
            running=True,
            started_at=time.time(),
            finished_at=None,
            path=path,
            file_types=file_types,
            stats=None,
            error=None,
            count_at_start=state["store"].count(),
        )

    thread = threading.Thread(
        target=_run_index_job,
        args=(path, file_types),
        daemon=True,
    )
    thread.start()

    return JSONResponse({"started": True, "started_at": job["started_at"]})


@app.get("/index/status")
def index_status() -> dict:
    """Current index-job status plus the store's live file count."""
    state = _get_state()
    state.setdefault("index_job", _empty_job())
    job = state["index_job"]

    return {
        **job,
        "count": state["store"].count(),
    }


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
