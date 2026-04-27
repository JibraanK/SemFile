"""Gemini multimodal reranker."""

import json
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

from semfile.store.chromadb_store import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"
VISUAL_TYPES = ("image", "video")


class GeminiReranker:
    """Multimodal reranker using a Gemini chat model.

    Sends the query plus candidate metadata (and thumbnails for image/video
    files) to Gemini and asks for a 0–10 relevance score per candidate.
    """

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it in .env or pass api_key."
            )
        self._client = genai.Client(api_key=key)
        self._model = model

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
        before_sleep=lambda retry_state: logger.warning(
            "Retrying rerank request (attempt %d): %s",
            retry_state.attempt_number,
            retry_state.outcome.exception() if retry_state.outcome else "unknown",
        ),
    )
    def _generate(self, contents: list, config: types.GenerateContentConfig) -> str:
        """Call Gemini generate_content with retry, return raw text."""
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        return response.text

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_n: int,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        attached_indices: list[int] = []
        contents: list = [self._build_prompt(query, candidates, attached_indices)]
        for i in attached_indices:
            thumb = Path(candidates[i].thumbnail_path)
            contents.append(
                types.Part.from_bytes(
                    data=thumb.read_bytes(),
                    mime_type="image/jpeg",
                )
            )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        )

        logger.info(
            "Reranking %d candidates with %s (%d thumbnails attached)",
            len(candidates), self._model, len(attached_indices),
        )

        try:
            raw = self._generate(contents, config)
            scores = self._parse_scores(raw, len(candidates))
        except Exception as exc:
            logger.warning(
                "Rerank failed (%s); returning original semantic order.", exc
            )
            return candidates[:top_n]

        return self._apply_scores(candidates, scores, top_n)

    def _build_prompt(
        self,
        query: str,
        candidates: list[SearchResult],
        attached_indices: list[int],
    ) -> str:
        """Build the prompt text and record which candidate indices have images attached."""
        lines = [
            "You are reranking files for a semantic search system.",
            "Score each candidate from 0 to 10 by how well it matches the user's query.",
            "10 = perfect match. 0 = unrelated. Use the full range.",
            "Consider visual content (when an image is attached) and metadata together.",
            "",
            f"USER QUERY: {query}",
            "",
            "CANDIDATES:",
        ]

        attach_counter = 0
        for i, c in enumerate(candidates):
            has_image = (
                c.file_type in VISUAL_TYPES
                and c.thumbnail_path
                and Path(c.thumbnail_path).exists()
            )
            tag = ""
            if has_image:
                attached_indices.append(i)
                attach_counter += 1
                tag = f" (image attachment #{attach_counter} below)"
            lines.append(
                f"[{i}] type={c.file_type} filename={c.filename} "
                f"dir={c.parent_dir}{tag}"
            )

        lines.extend([
            "",
            f"Image attachments follow this prompt in order, one per "
            f"\"image attachment #N\" reference above (total: {attach_counter}).",
            "",
            "Respond with ONLY a JSON array, no prose, with one object per "
            "candidate in the form:",
            '[{"index": 0, "score": 7.5}, {"index": 1, "score": 3.0}, ...]',
            "Include every candidate index exactly once. Scores are floats 0–10.",
        ])
        return "\n".join(lines)

    @staticmethod
    def _parse_scores(raw: str, n_candidates: int) -> dict[int, float]:
        """Parse the JSON response into {index: score}. Tolerant to fenced output."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")
        scores: dict[int, float] = {}
        for item in data:
            idx = int(item["index"])
            score = float(item["score"])
            if 0 <= idx < n_candidates:
                scores[idx] = max(0.0, min(10.0, score))
        return scores

    @staticmethod
    def _apply_scores(
        candidates: list[SearchResult],
        scores: dict[int, float],
        top_n: int,
    ) -> list[SearchResult]:
        """Attach scores, sort, and slice. Unscored candidates fall to the bottom."""
        for i, c in enumerate(candidates):
            c.rerank_score = scores.get(i)

        def sort_key(c: SearchResult) -> tuple[float, float]:
            primary = c.rerank_score if c.rerank_score is not None else -1.0
            secondary = 1.0 - c.distance
            return (primary, secondary)

        ordered = sorted(candidates, key=sort_key, reverse=True)
        return ordered[:top_n]
