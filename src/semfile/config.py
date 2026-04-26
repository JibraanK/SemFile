"""Configuration management for SemFile."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_CONFIG_PATH = Path.home() / ".config" / "semfile" / "config.toml"
DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "semfile" / "db"
DEFAULT_THUMBNAIL_DIR = Path.home() / ".local" / "share" / "semfile" / "thumbnails"
DEFAULT_EMBEDDING_DIMENSIONS = 768

SUPPORTED_EXTENSIONS = {
    "image": {"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"},
    "video": {"mp4", "mov", "avi", "mkv", "webm"},
    "audio": {"mp3", "wav", "flac", "aac", "ogg", "m4a"},
    "document": {"pdf"},
    "text": {"txt", "md", "rst", "csv", "json", "xml", "html"},
}

MIME_TYPES = {
    # Images
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
    # Video
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "avi": "video/x-msvideo",
    "mkv": "video/x-matroska",
    "webm": "video/webm",
    # Audio
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "ogg": "audio/ogg",
    "m4a": "audio/mp4",
    # Documents
    "pdf": "application/pdf",
    # Text (embedded as text, not bytes)
    "txt": "text/plain",
    "md": "text/markdown",
    "rst": "text/x-rst",
    "csv": "text/csv",
    "json": "application/json",
    "xml": "application/xml",
    "html": "text/html",
}


@dataclass
class WatchDirectory:
    path: Path
    extensions: set[str]


@dataclass
class Config:
    db_path: Path = field(default_factory=lambda: DEFAULT_DB_PATH)
    thumbnail_dir: Path = field(default_factory=lambda: DEFAULT_THUMBNAIL_DIR)
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    watch_dirs: list[WatchDirectory] = field(default_factory=list)

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)


def get_file_type(extension: str) -> str | None:
    """Return the file type category for a given extension."""
    ext = extension.lower().lstrip(".")
    for file_type, exts in SUPPORTED_EXTENSIONS.items():
        if ext in exts:
            return file_type
    return None


def get_mime_type(extension: str) -> str | None:
    """Return the MIME type for a given extension."""
    ext = extension.lower().lstrip(".")
    return MIME_TYPES.get(ext)


def load_config(config_path: Path | None = None) -> Config:
    """Load config from TOML file, falling back to defaults."""
    path = config_path or DEFAULT_CONFIG_PATH

    if not path.exists():
        return Config()

    with open(path, "rb") as f:
        data = tomllib.load(f)

    semfile = data.get("semfile", {})

    watch_dirs = []
    for entry in semfile.get("watch", []):
        watch_dirs.append(
            WatchDirectory(
                path=Path(entry["path"]).expanduser().resolve(),
                extensions=set(entry.get("extensions", [])),
            )
        )

    return Config(
        db_path=Path(semfile.get("db_path", str(DEFAULT_DB_PATH))).expanduser(),
        thumbnail_dir=Path(
            semfile.get("thumbnail_dir", str(DEFAULT_THUMBNAIL_DIR))
        ).expanduser(),
        embedding_dimensions=semfile.get(
            "embedding_dimensions", DEFAULT_EMBEDDING_DIMENSIONS
        ),
        watch_dirs=watch_dirs,
    )


DEFAULT_CONFIG_TEMPLATE = """\
[semfile]
db_path = "~/.local/share/semfile/db"
thumbnail_dir = "~/.local/share/semfile/thumbnails"
embedding_dimensions = 768

[[semfile.watch]]
path = "~/Pictures"
extensions = ["png", "jpg", "jpeg", "webp"]

[[semfile.watch]]
path = "~/Videos"
extensions = ["mp4", "mov"]

[[semfile.watch]]
path = "~/Documents"
extensions = ["pdf", "txt", "md"]
"""


def create_default_config(config_path: Path | None = None) -> Path:
    """Create a default config file if one doesn't exist. Returns the path."""
    path = config_path or DEFAULT_CONFIG_PATH
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_CONFIG_TEMPLATE)
    return path
