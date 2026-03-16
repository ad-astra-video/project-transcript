"""Chunked transcript formatting task using the fast LLM."""

import difflib
import re
from collections import Counter
from typing import Any, Dict, List

from pydantic import BaseModel

from ..llm_manager import LLMClient
from .prompts import BASE_TRANSCRIPT_FORMAT_PROMPT, FORMAT_TEMPLATES, STYLE_LABELS


class TranscriptFormatChunkSchema(BaseModel):
    """Structured response for one formatted chunk."""

    text: str


class TranscriptFormatTask:
    """Formats raw transcript text into polished prose using fast LLM in chunks."""

    _FORMAT_RESPONSE_SCHEMA: Dict[str, Any] = TranscriptFormatChunkSchema.model_json_schema()

    def __init__(
        self,
        llm_client: LLMClient,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        model_context_tokens: int = 16000,
        input_token_budget: int = 12000,
        chunk_overlap_chars: int = 500,
    ) -> None:
        self._llm_client = llm_client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_context_tokens = model_context_tokens
        self.input_token_budget = min(input_token_budget, model_context_tokens - max_tokens - 1000)
        self.chunk_overlap_chars = max(0, chunk_overlap_chars)

        # Guardrails against over-paraphrasing
        self.min_source_token_coverage = 0.82
        self.min_sequence_similarity = 0.55
        self.min_length_ratio = 0.75
        self.max_length_ratio = 1.30

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    @staticmethod
    def _normalize_token_for_compare(token: str) -> str:
        """Normalize token for repetition checks while preserving original output tokens."""
        if not token:
            return ""
        normalized = token.lower().strip()
        normalized = re.sub(r"^[^\w']+", "", normalized)
        normalized = re.sub(r"[^\w']+$", "", normalized)
        return normalized

    @classmethod
    def _phrases_equivalent(cls, phrase1: List[str], phrase2: List[str]) -> bool:
        if len(phrase1) != len(phrase2):
            return False

        for left_token, right_token in zip(phrase1, phrase2):
            left_norm = cls._normalize_token_for_compare(left_token)
            right_norm = cls._normalize_token_for_compare(right_token)
            if not left_norm or not right_norm:
                return False
            if left_norm != right_norm:
                return False

        return True

    @classmethod
    def dedupe_transcript(cls, text: str, max_phrase_len: int = 6) -> str:
        """
        Remove repeated words and repeated phrases from a transcript.

        max_phrase_len:
            maximum phrase length to check for repetition
        """
        text = re.sub(r"\s+", " ", text or "").strip()
        if not text:
            return ""

        words = text.split()
        cleaned: List[str] = []
        index = 0

        while index < len(words):
            removed = False

            for size in range(max_phrase_len, 0, -1):
                if index + size * 2 > len(words):
                    continue

                phrase1 = words[index : index + size]
                phrase2 = words[index + size : index + size * 2]

                if cls._phrases_equivalent(phrase1, phrase2):
                    cleaned.extend(phrase2)
                    index += size * 2
                    removed = True
                    break

            if not removed:
                cleaned.append(words[index])
                index += 1

        return " ".join(cleaned)

    @classmethod
    def _dedupe_preserving_timing_markers(cls, text: str, max_phrase_len: int = 6) -> str:
        """Apply transcript dedupe without removing timing markers like [00:10]."""
        if not text:
            return ""

        marker_pattern = re.compile(r"(\[[^\]]+\])")
        parts = marker_pattern.split(text)
        processed_parts: List[str] = []

        for part in parts:
            if not part:
                continue
            if marker_pattern.fullmatch(part):
                processed_parts.append(part.strip())
            else:
                deduped = cls.dedupe_transcript(part, max_phrase_len=max_phrase_len)
                if deduped:
                    processed_parts.append(deduped)

        return re.sub(r"\s+", " ", " ".join(processed_parts)).strip()

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p and p.strip()]

    @staticmethod
    def _tokenize_words(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9']+", (text or "").lower())

    @classmethod
    def _source_token_coverage(cls, source: str, candidate: str) -> float:
        """How much of candidate wording is grounded in source wording."""
        source_tokens = cls._tokenize_words(source)
        candidate_tokens = cls._tokenize_words(candidate)
        if not candidate_tokens:
            return 0.0

        source_counts = Counter(source_tokens)
        candidate_counts = Counter(candidate_tokens)
        covered = 0
        for token, cand_count in candidate_counts.items():
            covered += min(cand_count, source_counts.get(token, 0))
        return covered / max(1, len(candidate_tokens))

    @classmethod
    def _sequence_similarity(cls, source: str, candidate: str) -> float:
        source_norm = cls._normalize_text(source).lower()
        candidate_norm = cls._normalize_text(candidate).lower()
        if not source_norm or not candidate_norm:
            return 0.0
        return difflib.SequenceMatcher(None, source_norm, candidate_norm).ratio()

    @classmethod
    def _minimal_cleanup_fallback(cls, text: str) -> str:
        """Fallback used when model output diverges too much from source text."""
        cleaned = cls._dedupe_preserving_timing_markers(text or "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        return cleaned

    def _validate_or_fallback(self, source_chunk: str, candidate_text: str) -> str:
        candidate = (candidate_text or "").strip()
        if not candidate:
            return self._minimal_cleanup_fallback(source_chunk)

        source_norm = self._normalize_text(source_chunk)
        candidate_norm = self._normalize_text(candidate)
        if not source_norm:
            return candidate_norm

        source_len = len(source_norm)
        candidate_len = len(candidate_norm)
        length_ratio = candidate_len / max(1, source_len)
        coverage = self._source_token_coverage(source_norm, candidate_norm)
        seq_sim = self._sequence_similarity(source_norm, candidate_norm)

        if (
            coverage < self.min_source_token_coverage
            or seq_sim < self.min_sequence_similarity
            or length_ratio < self.min_length_ratio
            or length_ratio > self.max_length_ratio
        ):
            return self._minimal_cleanup_fallback(source_chunk)

        return candidate_norm

    @classmethod
    def _remove_boundary_overlap(
        cls,
        previous_chunk_text: str,
        current_chunk_text: str,
        max_overlap_words: int = 80,
    ) -> tuple:
        """Remove repeated leading words at the boundary between two consecutive blocks.

        Returns ``(fixed_prev, fixed_curr)`` so callers can update both sides.

        Three cases are handled:

        1. **Exact-phrase overlap** – ``curr`` starts with words that already end
           ``prev``.  The duplicate words are stripped from ``curr``; ``prev`` is
           unchanged.  For 1–2 word matches the words must average >= 3 chars so
           trivial function words are not accidentally removed.

        2. **Numeric / hyphenated continuation** – ``prev`` ends with a truncated
           token that is a prefix of the first token in ``curr``
           (e.g. ``"24."`` -> ``"24-7"``).  The partial token is stripped from the
           tail of ``prev``; ``curr`` is returned unchanged.

        3. **Mid-word cutoff** – ``prev`` ends with a partial word preceded by
           context words that repeat at the start of ``curr``, followed by the full
           form of that word (e.g. ``"gets a random"`` -> ``"gets a randomized"``).
           The partial token is stripped from the tail of ``prev`` and the repeated
           context words are stripped from the head of ``curr``, exposing the full word.
        """
        prev = (previous_chunk_text or "").strip()
        curr = (current_chunk_text or "").strip()
        if not prev or not curr:
            return prev, curr

        prev_tokens = re.findall(r"\S+", prev)
        curr_tokens = re.findall(r"\S+", curr)
        if not prev_tokens or not curr_tokens:
            return prev, curr

        max_n = min(max_overlap_words, len(prev_tokens), len(curr_tokens))

        # ── Case 1: exact-phrase overlap ────────────────────────────────────────
        overlap_words = 0
        for n in range(max_n, 0, -1):
            prev_tail = prev_tokens[-n:]
            curr_head = curr_tokens[:n]
            if cls._phrases_equivalent(prev_tail, curr_head):
                if n <= 2:
                    total_len = sum(
                        len(cls._normalize_token_for_compare(t)) for t in prev_tail
                    )
                    if (total_len / n) < 3:
                        continue
                overlap_words = n
                break

        if overlap_words:
            return prev, " ".join(curr_tokens[overlap_words:]).strip()

        # ── Case 2: numeric / hyphenated continuation ────────────────────────────
        prev_last_norm = cls._normalize_token_for_compare(prev_tokens[-1])
        curr_first_norm = cls._normalize_token_for_compare(curr_tokens[0])
        if (
            prev_last_norm
            and curr_first_norm
            and len(prev_last_norm) >= 2
            and curr_first_norm != prev_last_norm
            and curr_first_norm.startswith(prev_last_norm)
        ):
            # Strip the truncated token from the tail of prev; curr is already correct.
            fixed_prev = " ".join(prev_tokens[:-1]).strip()
            return fixed_prev, curr

        # ── Case 3: mid-word cutoff with context ────────────────────────────────
        prev_partial_norm = cls._normalize_token_for_compare(prev_tokens[-1])
        if len(prev_partial_norm) >= 3:
            max_context = min(max_overlap_words, len(prev_tokens) - 1, len(curr_tokens) - 1)
            for k in range(1, max_context + 1):
                curr_full_norm = cls._normalize_token_for_compare(curr_tokens[k])
                if not (
                    curr_full_norm
                    and curr_full_norm != prev_partial_norm
                    and curr_full_norm.startswith(prev_partial_norm)
                ):
                    continue
                if len(prev_tokens) < k + 1:
                    continue
                prev_context_words = prev_tokens[-(k + 1) : -1]
                curr_context_words = curr_tokens[:k]
                if cls._phrases_equivalent(prev_context_words, curr_context_words):
                    # Strip partial from prev tail; strip repeated context from curr.
                    fixed_prev = " ".join(prev_tokens[: -(k + 1)]).strip()
                    fixed_curr = " ".join(curr_tokens[k:]).strip()
                    return fixed_prev, fixed_curr

        return prev, curr

    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks under token-aware char budget for a 16k context model."""
        normalized = self._normalize_text(text)
        if not normalized:
            return []

        max_chars = max(2000, self.input_token_budget * 4)
        if len(normalized) <= max_chars:
            return [normalized]

        sentences = self._split_sentences(normalized)
        if not sentences:
            return [normalized[i:i + max_chars] for i in range(0, len(normalized), max_chars)]

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for sentence in sentences:
            sent_len = len(sentence)
            if sent_len > max_chars:
                if current:
                    chunks.append(" ".join(current).strip())
                    current = []
                    current_len = 0
                for i in range(0, sent_len, max_chars):
                    piece = sentence[i:i + max_chars].strip()
                    if piece:
                        chunks.append(piece)
                continue

            if current_len + sent_len + 1 <= max_chars:
                current.append(sentence)
                current_len += sent_len + 1
            else:
                chunks.append(" ".join(current).strip())
                overlap_prefix = ""
                if self.chunk_overlap_chars > 0 and chunks[-1]:
                    overlap_prefix = chunks[-1][-self.chunk_overlap_chars :].strip()
                current = [part for part in [overlap_prefix, sentence] if part]
                current_len = sum(len(part) + 1 for part in current)

        if current:
            chunks.append(" ".join(current).strip())

        return [chunk for chunk in chunks if chunk]

    async def _format_single_chunk(
        self,
        chunk_text: str,
        content_type: str,
        prior_tail: str,
        chunk_index: int,
        total_chunks: int,
    ) -> Dict[str, Any]:
        style_key = (content_type or "UNKNOWN").upper()
        style_instructions = FORMAT_TEMPLATES.get(style_key, FORMAT_TEMPLATES["UNKNOWN"])
        style_label = STYLE_LABELS.get(style_key, STYLE_LABELS["UNKNOWN"])

        system_prompt = f"{BASE_TRANSCRIPT_FORMAT_PROMPT}\n\n{style_instructions}"
        user_content = (
            f"Content type: {style_key}\n"
            f"Target style: {style_label}\n"
            f"Chunk: {chunk_index}/{total_chunks}\n\n"
            f"Prior formatted context tail (may be empty):\n{prior_tail or '(none)'}\n\n"
            f"Raw transcript chunk:\n{chunk_text}\n\n"
            "IMPORTANT: Keep wording very close to source. Perform light edits only.\n"
            "Do NOT significantly paraphrase or summarize.\n\n"
            "Return JSON with:\n"
            "- text: polished paragraph(s)"
        )

        _, content, input_tokens, output_tokens, _ = await self._llm_client.create_completion(
            system_prompt=system_prompt,
            user_content=user_content,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "transcript_format_chunk",
                    "schema": self._FORMAT_RESPONSE_SCHEMA,
                },
            },
        )

        parsed = TranscriptFormatChunkSchema.model_validate_json(
            content.replace("```json", "").replace("```", "").strip()
        )
        return {
            "text": self._validate_or_fallback(chunk_text, parsed.text),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    async def format_text(
        self,
        raw_text: str,
        content_type: str,
        prior_formatted_tail: str = "",
    ) -> Dict[str, Any]:
        """Format text in token-aware chunks and stitch output."""
        preprocessed_text = self._dedupe_preserving_timing_markers(raw_text)
        chunks = self.split_into_chunks(preprocessed_text)
        if not chunks:
            return {
                "formatted_text": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "chunk_count": 0,
            }

        formatted_chunks: List[str] = []
        total_input_tokens = 0
        total_output_tokens = 0
        rolling_tail = prior_formatted_tail[-1000:] if prior_formatted_tail else ""

        for index, chunk in enumerate(chunks, start=1):
            result = await self._format_single_chunk(
                chunk_text=chunk,
                content_type=content_type,
                prior_tail=rolling_tail,
                chunk_index=index,
                total_chunks=len(chunks),
            )
            chunk_text = result["text"]
            if formatted_chunks:
                fixed_prev, chunk_text = self._remove_boundary_overlap(
                    formatted_chunks[-1], chunk_text
                )
                formatted_chunks[-1] = fixed_prev
            if chunk_text:
                formatted_chunks.append(chunk_text)
            total_input_tokens += int(result.get("input_tokens", 0))
            total_output_tokens += int(result.get("output_tokens", 0))
            rolling_tail = (rolling_tail + "\n" + (chunk_text or ""))[-1000:]

        return {
            "formatted_text": "\n\n".join(part for part in formatted_chunks if part).strip(),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "chunk_count": len(chunks),
        }


__all__ = ["TranscriptFormatTask", "TranscriptFormatChunkSchema"]
