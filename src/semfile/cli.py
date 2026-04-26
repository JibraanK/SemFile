"""CLI entry point for SemFile."""

import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from semfile.config import (
    Config,
    WatchDirectory,
    create_default_config,
    load_config,
    DEFAULT_CONFIG_PATH,
)
from semfile.embeddings.gemini import GeminiEmbeddingProvider
from semfile.indexer.indexer import Indexer
from semfile.search.searcher import Searcher
from semfile.store.chromadb_store import ChromaStore


def _dir_size(path: Path) -> int:
    """Return total size in bytes of all files under a directory."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _fmt_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def _load(config_path: str | None) -> tuple[Config, GeminiEmbeddingProvider, ChromaStore]:
    """Load config, create provider and store."""
    path = Path(config_path) if config_path else None
    config = load_config(path)
    config.ensure_dirs()
    provider = GeminiEmbeddingProvider(dimensions=config.embedding_dimensions)
    store = ChromaStore(db_path=config.db_path)
    return config, provider, store


@click.group()
@click.option("--config", "config_path", default=None, help="Path to config file.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.pass_context
def main(ctx: click.Context, config_path: str | None, verbose: bool) -> None:
    """SemFile - Semantic file search using multimodal embeddings."""
    load_dotenv()
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


@main.command()
@click.option("--path", "target_path", default=None, help="Index a specific directory or file.")
@click.option("--type", "file_types", multiple=True, help="Only index these file types (image, video, audio, document, text).")
@click.pass_context
def index(ctx: click.Context, target_path: str | None, file_types: tuple[str, ...]) -> None:
    """Index files for semantic search."""
    config, provider, store = _load(ctx.obj["config_path"])

    type_filter = set(file_types) if file_types else None

    if not config.watch_dirs and not target_path:
        click.echo("No watch directories configured. Use --path or edit config.")
        click.echo(f"Config file: {DEFAULT_CONFIG_PATH}")
        config_path = create_default_config()
        click.echo(f"Created default config at: {config_path}")
        click.echo("Edit it, then run `semfile index` again.")
        return

    indexer = Indexer(config, provider, store)

    if target_path:
        click.echo(f"Indexing: {target_path}")
        stats = indexer.index_path(Path(target_path), file_types=type_filter)
    else:
        click.echo(f"Indexing {len(config.watch_dirs)} watch directories...")
        stats = indexer.index_all(file_types=type_filter)

    click.echo(
        f"Done. Scanned: {stats['scanned']}, Indexed: {stats['indexed']}, "
        f"Skipped: {stats['skipped']}, Failed: {stats['failed']}, "
        f"Removed: {stats['removed']}"
    )


@main.command()
@click.argument("query")
@click.option("--type", "file_type", multiple=True, help="Filter by file type (image, video, audio, document, text).")
@click.option("--dir", "directories", multiple=True, help="Scope search to specific directories.")
@click.option("--limit", default=20, help="Maximum number of results.")
@click.pass_context
def search(ctx: click.Context, query: str, file_type: tuple[str, ...], directories: tuple[str, ...], limit: int) -> None:
    """Search indexed files by natural language query."""
    config, provider, store = _load(ctx.obj["config_path"])

    if store.count() == 0:
        click.echo("No files indexed yet. Run `semfile index` first.")
        return

    searcher = Searcher(provider, store)
    results = searcher.search(
        query=query,
        limit=limit,
        file_types=list(file_type) if file_type else None,
        directories=list(directories) if directories else None,
    )

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"Found {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        similarity = 1 - r.distance  # cosine distance to similarity
        size_mb = r.file_size / 1_000_000
        click.echo(f"  {i}. [{r.file_type}] {r.filename}")
        click.echo(f"     Path: {r.file_path}")
        click.echo(f"     Similarity: {similarity:.3f}  Size: {size_mb:.1f} MB")
        click.echo()


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show index statistics."""
    config, _, store = _load(ctx.obj["config_path"])

    total = store.count()
    click.echo(f"Indexed files: {total}")

    if total > 0:
        counts = store.count_by_type()
        for file_type, count in sorted(counts.items()):
            click.echo(f"  {file_type}: {count}")

    # Storage sizes
    db_size = _dir_size(config.db_path)
    thumb_size = _dir_size(config.thumbnail_dir)
    total_size = db_size + thumb_size
    click.echo(f"\nStorage: {_fmt_bytes(total_size)}")
    click.echo(f"  Database: {_fmt_bytes(db_size)} ({config.db_path})")
    click.echo(f"  Thumbnails: {_fmt_bytes(thumb_size)} ({config.thumbnail_dir})")
    click.echo(f"Embedding dimensions: {config.embedding_dimensions}")

    if config.watch_dirs:
        click.echo(f"\nWatch directories:")
        for wd in config.watch_dirs:
            exts = ", ".join(sorted(wd.extensions)) if wd.extensions else "*"
            click.echo(f"  {wd.path} [{exts}]")


@main.command("config")
@click.pass_context
def show_config(ctx: click.Context) -> None:
    """Show or create configuration file."""
    config_path = Path(ctx.obj["config_path"]) if ctx.obj["config_path"] else DEFAULT_CONFIG_PATH

    if config_path.exists():
        click.echo(f"Config file: {config_path}")
        click.echo(config_path.read_text())
    else:
        path = create_default_config(config_path)
        click.echo(f"Created default config at: {path}")
        click.echo("Edit it to configure your watch directories.")


@main.command()
@click.confirmation_option(prompt="This will delete all indexed data. Continue?")
@click.pass_context
def reset(ctx: click.Context) -> None:
    """Clear the database and thumbnails."""
    config, _, store = _load(ctx.obj["config_path"])

    store.reset()
    click.echo("Database cleared.")

    # Clean thumbnails
    if config.thumbnail_dir.exists():
        import shutil
        shutil.rmtree(config.thumbnail_dir)
        config.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        click.echo("Thumbnails cleared.")

    click.echo("Reset complete.")
