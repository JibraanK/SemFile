"""Thumbnail generation for various file types."""

import hashlib
import logging
import subprocess
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

THUMBNAIL_MAX_SIZE = 256
THUMBNAIL_QUALITY = 80


def _thumbnail_filename(file_path: Path) -> str:
    """Generate a deterministic thumbnail filename from the source path."""
    path_hash = hashlib.sha256(str(file_path).encode()).hexdigest()[:16]
    return f"{path_hash}.jpg"


def generate_thumbnail(
    file_path: Path, file_type: str, thumbnail_dir: Path
) -> str:
    """Generate a thumbnail for a file. Returns the thumbnail path as a string.

    Returns empty string if thumbnail generation fails or is not applicable.
    """
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    thumb_name = _thumbnail_filename(file_path)
    thumb_path = thumbnail_dir / thumb_name

    # Skip if thumbnail already exists
    if thumb_path.exists():
        return str(thumb_path)

    try:
        if file_type == "image":
            return _thumbnail_image(file_path, thumb_path)
        elif file_type == "video":
            return _thumbnail_video(file_path, thumb_path)
        else:
            # Audio, text, documents don't get thumbnails
            return ""
    except Exception:
        logger.warning("Failed to generate thumbnail for %s", file_path, exc_info=True)
        return ""


def _thumbnail_image(file_path: Path, thumb_path: Path) -> str:
    """Generate a thumbnail for an image using Pillow."""
    with Image.open(file_path) as img:
        img.thumbnail((THUMBNAIL_MAX_SIZE, THUMBNAIL_MAX_SIZE))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(thumb_path, "JPEG", quality=THUMBNAIL_QUALITY)
    return str(thumb_path)


def _thumbnail_video(file_path: Path, thumb_path: Path) -> str:
    """Generate a thumbnail for a video using ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-i", str(file_path),
            "-ss", "00:00:01",
            "-frames:v", "1",
            "-vf", f"scale={THUMBNAIL_MAX_SIZE}:-1",
            "-y",
            str(thumb_path),
        ],
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0 or not thumb_path.exists():
        logger.warning("ffmpeg thumbnail failed for %s: %s", file_path, result.stderr.decode(errors="replace")[:200])
        return ""
    return str(thumb_path)
