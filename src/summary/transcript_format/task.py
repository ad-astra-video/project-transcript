"""Chunked transcript formatting task using the fast LLM."""

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from ..llm_manager import LLMClient
from .prompts import BASE_TRANSCRIPT_FORMAT_PROMPT, FORMAT_TEMPLATES, STYLE_LABELS


class TranscriptFormatChunkSchema(BaseModel):
    """Structured response for one formatted chunk."""

    heading: str
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

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p and p.strip()]

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
            "Return JSON with:\n"
            "- heading: concise heading for this chunk\n"
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
            "heading": parsed.heading.strip() or "Formatted Transcript",
            "text": parsed.text.strip(),
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
        chunks = self.split_into_chunks(raw_text)
        if not chunks:
            return {
                "heading": "",
                "formatted_text": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "chunk_count": 0,
            }

        headings: List[str] = []
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
            headings.append(result["heading"])
            formatted_chunks.append(result["text"])
            total_input_tokens += int(result.get("input_tokens", 0))
            total_output_tokens += int(result.get("output_tokens", 0))
            rolling_tail = (rolling_tail + "\n" + result["text"])[-1000:]

        chosen_heading = next((h for h in headings if h), "Formatted Transcript")
        return {
            "heading": chosen_heading,
            "formatted_text": "\n\n".join(part for part in formatted_chunks if part).strip(),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "chunk_count": len(chunks),
        }


__all__ = ["TranscriptFormatTask", "TranscriptFormatChunkSchema"]
