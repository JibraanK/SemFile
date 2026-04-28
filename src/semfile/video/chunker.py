"""Probe and split videos to fit Gemini's 120-second embedding limit.

Gemini Embedding 2 rejects videos longer than 120s, so long videos are split
into segments before embedding. Each segment is stored as its own record with
a chunk_start_seconds offset, letting search surface specific moments.

Audio has an analogous 180s limit; the same architecture (chunk_index,
total_chunks, chunk_start_seconds in FileRecord) generalizes — TODO when needed.
"""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

SEGMENT_TIME = 110
CHUNK_THRESHOLD = 115
HARD_LIMIT = 120
FFMPEG_TIMEOUT = 600


def probe_duration(path: Path) -> float | None:
    """Return video duration in seconds via ffprobe, or None on failure."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        logger.warning(
            "ffprobe failed for %s: %s",
            path,
            result.stderr.decode(errors="replace")[:200],
        )
        return None
    try:
        return float(result.stdout.decode().strip())
    except (ValueError, AttributeError):
        return None


def _segment(
    src: Path, out_dir: Path, segment_time: int, reencode: bool
) -> subprocess.CompletedProcess:
    template = str(out_dir / f"chunk_%03d{src.suffix}")
    base = ["ffmpeg", "-i", str(src)]
    if reencode:
        codec = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-c:a", "copy"]
    else:
        codec = ["-c", "copy"]
    cmd = [
        *base,
        *codec,
        "-map", "0",
        "-segment_time", str(segment_time),
        "-f", "segment",
        "-reset_timestamps", "1",
        "-avoid_negative_ts", "make_zero",
        "-y",
        template,
    ]
    return subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT)


def chunk_video(
    src: Path, out_dir: Path, segment_time: int = SEGMENT_TIME
) -> list[tuple[Path, float]]:
    """Split src into segments under HARD_LIMIT seconds.

    Returns [(chunk_path, start_seconds), ...]. Uses stream copy for speed; if a
    chunk still exceeds HARD_LIMIT (sparse keyframes), retries with re-encode.
    Raises RuntimeError if ffmpeg fails outright.
    """
    result = _segment(src, out_dir, segment_time, reencode=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg segment failed for {src}: "
            f"{result.stderr.decode(errors='replace')[:300]}"
        )

    chunks = sorted(out_dir.glob(f"chunk_*{src.suffix}"))
    durations = [probe_duration(c) or 0.0 for c in chunks]

    if any(d > HARD_LIMIT - 1 for d in durations):
        logger.warning(
            "Stream-copy chunks for %s exceeded %ds limit (max %.1fs); re-encoding.",
            src.name, HARD_LIMIT, max(durations),
        )
        for c in chunks:
            c.unlink()
        result = _segment(src, out_dir, segment_time=100, reencode=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg re-encode segment failed for {src}: "
                f"{result.stderr.decode(errors='replace')[:300]}"
            )
        chunks = sorted(out_dir.glob(f"chunk_*{src.suffix}"))
        durations = [probe_duration(c) or 0.0 for c in chunks]

    starts: list[float] = []
    cumulative = 0.0
    for d in durations:
        starts.append(cumulative)
        cumulative += d

    return list(zip(chunks, starts))
