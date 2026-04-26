"""Gemini Embedding 2 provider."""

import logging
import os
from pathlib import Path

from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Gemini Embedding 2 limits
MAX_VIDEO_DURATION_SECONDS = 120
MAX_AUDIO_DURATION_SECONDS = 180
MAX_IMAGES_PER_REQUEST = 6
MAX_PDF_PAGES = 6

# Text file types that should be read as text, not bytes
TEXT_MIME_PREFIXES = ("text/", "application/json", "application/xml")


class GeminiEmbeddingProvider:
    """Embedding provider using Gemini Embedding 2 (multimodal)."""

    def __init__(self, dimensions: int = 768, api_key: str | None = None):
        self._dimensions = dimensions
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it in .env or pass api_key."
            )
        self._client = genai.Client(api_key=key)
        self._model = "gemini-embedding-2"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
        before_sleep=lambda retry_state: logger.warning(
            "Retrying embedding request (attempt %d): %s",
            retry_state.attempt_number,
            retry_state.outcome.exception() if retry_state.outcome else "unknown",
        ),
    )
    def _embed(
        self, contents: list | str, config: types.EmbedContentConfig | None = None
    ) -> list[float]:
        """Call the Gemini embed_content API with retry logic."""
        if config is None:
            config = types.EmbedContentConfig(
                output_dimensionality=self._dimensions
            )
        result = self._client.models.embed_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        return result.embeddings[0].values

    def embed_file(self, file_path: Path, mime_type: str) -> list[float]:
        """Embed a file (image, video, audio, PDF)."""
        if any(mime_type.startswith(prefix) for prefix in TEXT_MIME_PREFIXES):
            return self.embed_text(file_path.read_text(errors="replace"))

        file_bytes = file_path.read_bytes()
        logger.info("Embedding file: %s (%s, %.1f MB)", file_path.name, mime_type, len(file_bytes) / 1_000_000)

        contents = [
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
        ]
        return self._embed(contents)

    def embed_text(self, text: str) -> list[float]:
        """Embed text content (for indexing documents)."""
        # For document indexing, use the document format
        formatted = f"title: none | text: {text}"
        return self._embed(formatted)

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query with task prefix."""
        formatted = f"task: search result | query: {query}"
        return self._embed(formatted)
