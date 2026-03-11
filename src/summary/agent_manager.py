"""
AgentManager - Provides autonomous PydanticAI agent with transcript knowledge store.

Components:
- TranscriptKnowledgeStore: in-memory Faiss index backed by a local TEI
  embedding model (default port 6060, mirrors the vLLM container pattern).
- PydanticAI agent with search_transcript and web_search tools.

Usage:
  agent = AgentManager()
  await agent.initialize(llm_manager)  # Inject LLMManager at initialize time
  await agent.index_transcript_segment(text, timestamp, speaker)
  response = await agent.ask_agent(query)

Web search tool priority (first key found wins):
  TAVILY_API_KEY  → Tavily Search API
  EXA_API_KEY     → Exa Search API
  (fallback)      → DuckDuckGo (no key required)

Embedding client configuration:
  LOCAL_EMBEDDING_BASE_URL   – defaults to http://byoc-transcription-tei-embeddings:6060/v1
  LOCAL_EMBEDDING_API_KEY    – defaults to "dummy"
  LOCAL_EMBEDDING_MODEL      – defaults to BAAI/bge-small-en-v1.5
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Callable, Awaitable

import numpy as np
from openai import AsyncOpenAI

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from .llm_manager import LLMManager, MessageFormatMode
from .window_manager import WindowManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_EMBEDDING_BASE_URL = "http://byoc-transcription-tei-embeddings:6060/v1"
_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
_TOP_K = 15  # Number of transcript chunks returned per search


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_hms(seconds: float) -> str:
    """Format *seconds* as HH:MM:SS (or MM:SS when < 1 hour)."""
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


# ---------------------------------------------------------------------------
# Transcript chunk stored alongside its vector
# ---------------------------------------------------------------------------
@dataclass
class TranscriptChunk:
    text: str
    timestamp: Optional[float] = None          # seconds from stream start
    duration: Optional[float] = None           # seconds this chunk covers
    speaker: Optional[str] = None
    window_id: Optional[int] = None


# ---------------------------------------------------------------------------
# In-memory Faiss-backed knowledge store
# ---------------------------------------------------------------------------
class TranscriptKnowledgeStore:
    """
    Stores transcript chunks as dense vectors in a Faiss flat index.

    Vectors are produced by the local TEI embedding service whose base_url
    and model mirror the same configuration pattern used for fast/reasoning
    vLLM clients.

    The store is intentionally ephemeral – it lives for the lifetime of the
    SummaryClient session and is cleared via reset().
    """

    def __init__(
        self,
        embedding_base_url: str = _DEFAULT_EMBEDDING_BASE_URL,
        embedding_api_key: str = "dummy",
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    ):
        self._embedding_base_url = embedding_base_url.rstrip("/")
        self._embedding_api_key = embedding_api_key
        self._embedding_model = embedding_model

        self._client = AsyncOpenAI(
            base_url=self._embedding_base_url,
            api_key=self._embedding_api_key or "dummy",
        )

        # Faiss index – initialised lazily on first insert so we can infer dim
        self._index = None
        self._dim: Optional[int] = None
        self._chunks: List[TranscriptChunk] = []
        self._lock = asyncio.Lock()
        # Speaker remap: source_speaker_id -> target_speaker_id
        self._speaker_remap: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add(self, chunk: TranscriptChunk) -> None:
        """Embed *chunk* and add it to the Faiss index."""
        # Apply any existing speaker remaps to new chunks
        if chunk.speaker in self._speaker_remap:
            chunk.speaker = self._speaker_remap[chunk.speaker]
        vector = await self._embed(chunk.text)
        async with self._lock:
            self._ensure_index(len(vector))
            vec_array = np.array([vector], dtype="float32")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._index.add, vec_array)
            self._chunks.append(chunk)

    async def search(self, query: str, top_k: int = _TOP_K) -> List[TranscriptChunk]:
        """Return the *top_k* most relevant chunks for *query*."""
        if not self._chunks:
            return []
        vector = await self._embed(query)
        async with self._lock:
            k = min(top_k, len(self._chunks))
            vec_array = np.array([vector], dtype="float32")
            loop = asyncio.get_event_loop()
            distances, indices = await loop.run_in_executor(
                None, self._index.search, vec_array, k
            )
            return [self._chunks[i] for i in indices[0] if i >= 0]

    def remap_speaker(self, old_speaker: str, new_speaker: str) -> int:
        """Remap all chunks with old_speaker to new_speaker.

        Args:
            old_speaker: The speaker ID to replace.
            new_speaker: The new speaker ID to use.

        Returns:
            Number of chunks updated.
        """
        count = 0
        for chunk in self._chunks:
            if chunk.speaker == old_speaker:
                chunk.speaker = new_speaker
                count += 1
        self._speaker_remap[old_speaker] = new_speaker
        logger.info(f"Remapped speaker '{old_speaker}' -> '{new_speaker}': {count} chunks updated")
        return count

    def remap_speakers(self, merges: List[Dict[str, str]]) -> int:
        """Apply multiple speaker merges.

        Args:
            merges: List of {"source": "speaker_1", "target": "speaker_0"} dicts.

        Returns:
            Total number of chunks updated.
        """
        total = 0
        for merge in merges:
            source = merge.get("source")
            target = merge.get("target")
            if source and target:
                total += self.remap_speaker(source, target)
        return total

    def reset(self) -> None:
        """Clear all stored vectors and chunks (call on stream end)."""
        self._index = None
        self._dim = None
        self._chunks = []
        self._speaker_remap = {}
        logger.info("TranscriptKnowledgeStore reset")

    @property
    def size(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _embed(self, text: str) -> List[float]:
        response = await self._client.embeddings.create(
            model=self._embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def _ensure_index(self, dim: int) -> None:
        """Create the Faiss index on first insert."""
        if self._index is None:
            import faiss  # deferred import so the module can load without faiss
            self._dim = dim
            self._index = faiss.IndexFlatL2(dim)
            logger.info(f"Faiss IndexFlatL2 created with dim={dim}")


# ---------------------------------------------------------------------------
# Web search helpers
# ---------------------------------------------------------------------------

async def _web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for *query* and return a formatted string of results.

    Priority:
      1. Tavily  (TAVILY_API_KEY)
      2. Exa     (EXA_API_KEY)
      3. DuckDuckGo (fallback, no key required)
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    exa_key = os.getenv("EXA_API_KEY")

    if tavily_key:
        return await _search_tavily(query, tavily_key, max_results)
    elif exa_key:
        return await _search_exa(query, exa_key, max_results)
    else:
        return await _search_duckduckgo(query, max_results)


async def _search_tavily(query: str, api_key: str, max_results: int) -> str:
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        # Run sync client in executor to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.search(query, max_results=max_results)
        )
        results = response.get("results", [])
        formatted = "\n\n".join(
            f"**{r.get('title', 'No title')}**\n{r.get('url', '')}\n{r.get('content', '')}"
            for r in results
        )
        return formatted or "No results found."
    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")
        return f"Tavily search error: {e}"


async def _search_exa(query: str, api_key: str, max_results: int) -> str:
    try:
        from exa_py import Exa
        loop = asyncio.get_event_loop()
        exa = Exa(api_key=api_key)
        response = await loop.run_in_executor(
            None,
            lambda: exa.search_and_contents(
                query,
                num_results=max_results,
                text=True,
            )
        )
        formatted = "\n\n".join(
            f"**{r.title}**\n{r.url}\n{r.text[:500] if r.text else ''}"
            for r in response.results
        )
        return formatted or "No results found."
    except Exception as e:
        logger.warning(f"Exa search failed: {e}")
        return f"Exa search error: {e}"


async def _search_duckduckgo(query: str, max_results: int) -> str:
    try:
        from ddgs import DDGS
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: list(DDGS().text(query, max_results=max_results))
        )
        formatted = "\n\n".join(
            f"**{r.get('title', 'No title')}**\n{r.get('href', '')}\n{r.get('body', '')}"
            for r in results
        )
        return formatted or "No results found."
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return f"DuckDuckGo search error: {e}"


# ---------------------------------------------------------------------------
# AgentManager
# ---------------------------------------------------------------------------

class AgentManager:
    """
    Provides autonomous PydanticAI agent with transcript knowledge store.

    Components:
    - TranscriptKnowledgeStore: semantic search over indexed transcript chunks.
    - PydanticAI agent with `search_transcript` and `web_search` tools.

    Usage:
        agent = AgentManager()
        await agent.initialize(llm_manager)  # Inject LLMManager
        await agent.index_transcript_segment(text, timestamp, speaker)
        response = await agent.ask_agent(query)

    Configuration (env vars):
        EMBEDDING_BASE_URL   Base URL for the TEI service  (default: http://byoc-transcription-tei-embeddings:6060/v1)
        EMBEDDING_API_KEY    API key for TEI               (default: "dummy")
        EMBEDDING_MODEL      Model name served by TEI      (default: BAAI/bge-small-en-v1.5)
    """

    def __init__(
        self,
        # ---- Embedding / knowledge store args ----
        embedding_base_url: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        # ---- Result callback for status updates ----
        result_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ):
        # ---- LLM manager (injected at initialize time) ----
        self.llm: Optional[LLMManager] = None

        # ---- Embedding client configuration ----
        _emb_url = (
            embedding_base_url
            or os.getenv("EMBEDDING_BASE_URL", _DEFAULT_EMBEDDING_BASE_URL)
        ).rstrip("/")
        _emb_key = embedding_api_key or os.getenv("EMBEDDING_API_KEY", "dummy")
        _emb_model = (
            embedding_model
            or os.getenv("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)
        )

        self.knowledge_store = TranscriptKnowledgeStore(
            embedding_base_url=_emb_url,
            embedding_api_key=_emb_key,
            embedding_model=_emb_model,
        )

        # ---- Result callback for pushing agent_status payloads ----
        self._result_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = result_callback

        # ---- PydanticAI agent (built lazily after initialize()) ----
        self._agent: Optional[Agent] = None
        self._agent_model_name: Optional[str] = None
        self._agent_chat_query: Optional[str] = None  # Custom system prompt for agent queries

        # ---- Stream position tracking (updated on each indexed segment) ----
        self._current_stream_ts: float = 0.0       # latest end-of-chunk timestamp seen
        self._total_chunks: int = 0                # total chunks indexed
        self._total_indexed_duration: float = 0.0  # sum of all chunk durations

        # ---- WindowManager reference (injected from SummaryClient) ----
        self._window_manager: Optional[WindowManager] = None

    def set_window_manager(self, window_manager: WindowManager) -> None:
        """Inject the WindowManager so temporal-range tools can access transcription windows."""
        self._window_manager = window_manager

    async def initialize(self, llm_manager: LLMManager) -> Optional[str]:
        """Initialize with LLM manager, auto-detect models, and build the PydanticAI agent."""
        self.llm = llm_manager
        # Build the PydanticAI agent using the reasoning client
        self._build_agent(
            reasoning_client=self.llm.reasoning_client,
            reasoning_model=self.llm.reasoning_model
        )
        return None

    # ------------------------------------------------------------------
    # Knowledge store helpers (called by the summary pipeline)
    # ------------------------------------------------------------------

    async def index_transcript_segment(
        self,
        text: str,
        timestamp: Optional[float] = None,
        duration: Optional[float] = None,
        speaker: Optional[str] = None,
        window_id: Optional[int] = None,
    ) -> None:
        """
        Embed *text* and add it to the in-memory Faiss knowledge store.

        Should be called from the summary pipeline whenever a new
        TranscriptionWindow or SummaryWindow is finalised so that recent
        content is always available to the autonomous agent.
        """
        if not text.strip():
            return
        chunk = TranscriptChunk(
            text=text,
            timestamp=timestamp,
            duration=duration,
            speaker=speaker,
            window_id=window_id,
        )
        try:
            await self.knowledge_store.add(chunk)
            # Update stream position tracking
            if timestamp is not None:
                self._current_stream_ts = max(
                    self._current_stream_ts, timestamp + (duration or 0.0)
                )
            if duration is not None:
                self._total_indexed_duration += duration
            self._total_chunks += 1
        except Exception as e:
            logger.exception(f"Failed to index transcript segment: {e}")

    def reset_knowledge_store(self) -> None:
        """Clear all stored vectors and stream position tracking. Call when a stream ends."""
        self.knowledge_store.reset()
        self._current_stream_ts = 0.0
        self._total_chunks = 0
        self._total_indexed_duration = 0.0

    def remap_speakers(self, merges: List[Dict[str, str]]) -> int:
        """Apply speaker merge updates to the knowledge store.

        Args:
            merges: List of {"source": "speaker_1", "target": "speaker_0"} dicts.

        Returns:
            Total number of chunks updated.
        """
        return self.knowledge_store.remap_speakers(merges)

    def on_update_params(
        self,
        agent_chat_query: Optional[str] = None,
    ) -> None:
        """Handle on_update_params event from SummaryClient.

        Args:
            agent_chat_query: Custom system prompt for the agent chat query.
        """
        if agent_chat_query is not None:
            # We no longer store this as system prompt since update_params is passing
            # the user's actual question in this parameter.
            logger.info(f"Received proxy question via on_update_params (ignoring for state)")

    # ------------------------------------------------------------------
    # Autonomous agent
    # ------------------------------------------------------------------

    def _build_agent(
        self,
        reasoning_client: AsyncOpenAI,
        reasoning_model: str
    ) -> None:
        """
        Construct the PydanticAI Agent backed by the reasoning LLM.

        The agent is given two tools:
          - search_transcript: semantic search over the indexed video transcript
          - web_search: live web search (Tavily > Exa > DuckDuckGo)
        """
        if not reasoning_model:
            logger.warning("AgentManager: reasoning model not set, skipping agent build")
            return

        # Use the reasoning client directly instead of creating a new provider
        model = OpenAIModel(reasoning_model, provider=OpenAIProvider(openai_client=reasoning_client))

        agent: Agent[None, str] = Agent(
            model,
            system_prompt=(
                "You are an intelligent assistant anchored to a live video transcript. "
                "Every user question is about the current video stream and its transcript "
                "unless explicitly stated otherwise — never ask for clarification about this.\n\n"

                "SEARCH STRATEGY — follow all four steps before answering:\n\n"

                "1. HYDE (Hypothetical Document Embedding)\n"
                "   Before calling search_transcript, mentally compose a short hypothetical "
                "transcript passage (1-3 sentences of spoken, informal language) that would "
                "directly answer the user's question. Use that hypothetical passage — not the "
                "bare question — as your first search_transcript query. Spoken text retrieves "
                "spoken text far more accurately than a formal question does.\n\n"

                "2. QUERY DECOMPOSITION — call search_transcript incrementally\n"
                "   If the question has multiple parts or sub-topics, split it into separate "
                "sub-queries and issue a distinct search_transcript call for each one. Do NOT "
                "bundle everything into a single call. Synthesise the results afterwards.\n\n"

                "3. KEYWORD EXPANSION\n"
                "   For each sub-query, broaden coverage by including likely synonyms, "
                "alternative phrasings, speaker names, entity names, and domain terminology "
                "that might appear in spoken transcript text. If an initial search returns "
                "weak or no matches, retry with an expanded or reworded query.\n\n"

                "4. TEMPORAL QUERIES\n"
                "   When the user asks about a specific time period, use get_transcript_time_range\n"
                "   to retrieve the complete verbatim transcript for that range — do NOT use\n"
                "   search_transcript for time-bounded questions.\n"
                "   Before calling get_transcript_time_range, resolve the time reference:\n"
                "   - Relative: 'last/past/previous X min' → call get_stream_info() to get\n"
                "     current position, then start = current - X*60, end = current.\n"
                "     'first X minutes' → start = 0, end = X*60.\n"
                "   - Absolute: 'at 2:30' → 150s; 'between 1:00 and 3:00' → 60–180s;\n"
                "     'from 2:00 to 4:00' → 120–240s; 'after/before X' → boundary-based.\n"
                "   - Context-based: 'beginning/start' → 0–10% of current position;\n"
                "     'end/closing' → 90–100%; 'middle' → 40–60%.\n"
                "   - Event-based: 'when X happened' → use search_transcript first to locate\n"
                "     the timestamp, then optionally get_transcript_time_range for context.\n"
                "   Always state the time range you searched in your response\n"
                "   (e.g. 'Searching from 05:00 to 10:00...').\n"
                "   Edge cases:\n"
                "   - If requested time exceeds current position: note the stream is at\n"
                "     [current] and search up to that point.\n"
                "   - If no content exists in range: report it clearly.\n\n"

                "TOOL SELECTION GUIDE:\n"
                "- get_transcript_time_range: user asks about a time period → complete verbatim\n"
                "  text, ideal for summarisation. Use for 'what happened in the last 5 minutes',\n"
                "  'summarise from 2:00 to 5:00', etc.\n"
                "- search_transcript: user asks about a topic/event → semantic similarity\n"
                "  search, ideal for finding relevant mentions regardless of time.\n"
                "- get_stream_info: call when resolving relative time references ('last X\n"
                "  minutes') to obtain the current stream position and chunk duration metrics.\n"
                "- web_search: supplement with external context when transcript is thin or\n"
                "  user explicitly requests it.\n"
                "- Tools can be combined: e.g. search_transcript to find when something\n"
                "  occurred, then get_transcript_time_range for surrounding context.\n\n"

                "KNOWLEDGE BASE RULES — CRITICAL:\n"
                "The search_transcript tool provides access to the COMPLETE transcript "
                "knowledge base. The results it returns are not excerpts or previews — "
                "they ARE all the relevant content that exists in the transcript. The "
                "transcript is the primary and authoritative source of truth for this "
                "stream.\n"
                "- NEVER describe results as 'truncated', 'partial', 'limited', or 'snippets'.\n"
                "- NEVER say you 'only have access to portions' or that you 'lack full context'.\n"
                "- NEVER suggest information might be 'missing' from the knowledge base.\n"
                "- If the transcript contains little content, that means the stream is short "
                "or recently started — not that data is hidden or unavailable. Treat whatever "
                "is indexed as the complete picture.\n\n"

                "TOOL USAGE RULES:\n"
                "5. Always call search_transcript or get_transcript_time_range at least once "
                "before composing any answer.\n"
                "6. Use web_search proactively in two situations:\n"
                "   a) When the user explicitly asks to look something up online.\n"
                "   b) When the transcript results are thin or don't fully address the question "
                "— use web_search to supplement with relevant external information, context, "
                "documentation, definitions, or background knowledge related to the topic.\n"
                "7. Never use web_search as a substitute for search_transcript — always search "
                "the transcript first.\n\n"

                "RESPONSE RULES:\n"
                "- Ground your answers primarily in transcript content from search_transcript.\n"
                "- Cite speaker names and timestamps (in hh:mm:ss) where available.\n"
                "- If search_transcript returns no relevant results after retrying with expanded "
                "queries, state that the topic was not discussed in the transcript — do NOT "
                "suggest the answer might exist but was not retrieved.\n"
                "- When web_search is used, clearly distinguish web-sourced information from "
                "transcript-sourced information.\n"
                "- FURTHER READING SECTION: Whenever you call web_search (whether proactively "
                "or upon explicit request), always end your response with a '## Further Reading' "
                "section containing a markdown list of the most relevant URLs returned by "
                "web_search. Include the page title as the link text and the URL as the href. "
                "Only include URLs that are genuinely relevant to the question."
            ),
        )

        knowledge_store = self.knowledge_store  # local ref for closures
        result_callback = self._result_callback  # local ref for closures
        window_manager = self._window_manager     # local ref for closures
        agent_self = self                         # local ref for stream position state

        @agent.system_prompt
        async def inject_stream_position() -> str:  # type: ignore[misc]
            """Dynamically inject the current stream position into the system prompt."""
            ts = agent_self._current_stream_ts
            if ts > 0:
                return f"\nCURRENT STREAM POSITION: {ts:.1f}s ({_format_hms(ts)})"
            return "\nCURRENT STREAM POSITION: stream not yet started or no content indexed."

        @agent.tool_plain
        async def get_stream_info() -> str:  # type: ignore[misc]
            """Return current stream position and indexing statistics.

            Call this before resolving relative time references such as
            'last 5 minutes' or 'past 30 seconds' so you can compute the
            correct start_time / end_time values.
            """
            ts = agent_self._current_stream_ts
            total_chunks = agent_self._total_chunks
            total_dur = agent_self._total_indexed_duration
            avg_dur = (total_dur / total_chunks) if total_chunks > 0 else 0.0
            if ts <= 0:
                return "Stream has not started or no transcript content has been indexed yet."
            lines = [
                f"Current stream position : {ts:.1f}s ({_format_hms(ts)})",
                f"Total indexed chunks    : {total_chunks}",
                f"Total indexed duration  : {total_dur:.1f}s ({_format_hms(total_dur)})",
                f"Average chunk duration  : {avg_dur:.2f}s",
            ]
            return "\n".join(lines)

        @agent.tool_plain
        async def get_transcript_time_range(start_time: float, end_time: float) -> str:  # type: ignore[misc]
            """Retrieve the complete verbatim deduplicated transcript for a time range.

            Returns all transcription content between start_time and end_time
            (both in seconds from stream start). Use this when the user asks
            about a specific time period. For topical/semantic queries without
            a specific time range, use search_transcript instead.

            Args:
                start_time: Start of the range in seconds from stream start.
                end_time:   End of the range in seconds from stream start.
            """
            if result_callback is not None:
                try:
                    await result_callback({
                        "type": "agent_status",
                        "tool": "get_transcript_time_range",
                        "display_text": f"reading transcript {_format_hms(start_time)} to {_format_hms(end_time)}",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    })
                except Exception:
                    pass

            if window_manager is None:
                return "Transcript time-range retrieval is unavailable (WindowManager not configured)."

            current_ts = agent_self._current_stream_ts

            # Collect all transcription windows that overlap [start_time, end_time]
            with window_manager._lock:
                windows = [
                    w for w in window_manager._transcription_windows.values()
                    if w.timestamp_start < end_time and w.timestamp_end > start_time
                ]

            if not windows:
                msg = (
                    f"No transcript content exists between "
                    f"{_format_hms(start_time)} ({start_time:.0f}s) and "
                    f"{_format_hms(end_time)} ({end_time:.0f}s)."
                )
                if end_time > current_ts > 0:
                    msg += (
                        f" Note: the stream is currently at {_format_hms(current_ts)} "
                        f"({current_ts:.0f}s) — content beyond that point has not occurred yet."
                    )
                return msg

            windows.sort(key=lambda w: w.timestamp_start)

            header_note = ""
            if end_time > current_ts > 0:
                header_note = (
                    f" (stream is at {_format_hms(current_ts)}, "
                    f"showing content up to that point)"
                )

            lines = [
                f"[Transcript {_format_hms(start_time)} to {_format_hms(end_time)}"
                f" — {len(windows)} transcription window(s){header_note}]",
                "",
            ]
            for w in windows:
                ts_label = f"[{_format_hms(w.timestamp_start)} – {_format_hms(w.timestamp_end)}]"
                lines.append(ts_label)
                lines.append(w.new_text)
                lines.append("---")

            return "\n".join(lines)

        @agent.tool_plain
        async def search_transcript(query: str) -> str:  # type: ignore[misc]
            """Search the video transcript for content relevant to *query*.

            Returns the most semantically similar transcript segments.
            """
            if result_callback is not None:
                try:
                    await result_callback({
                        "type": "agent_status",
                        "tool": "search_transcript",
                        "display_text": "searching transcript",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    })
                except Exception:
                    pass
            chunks = await knowledge_store.search(query)
            if not chunks:
                return "No transcript content has been spoken yet. The stream may have just started or nothing has been said."
            lines = []
            for chunk in chunks:
                ts = f"[{chunk.timestamp:.1f}s] " if chunk.timestamp is not None else ""
                sp = f"({chunk.speaker}) " if chunk.speaker else ""
                lines.append(f"{ts}{sp}{chunk.text}")
            header = f"[{len(chunks)} transcript segment(s) — this is the complete relevant content from the transcript]\n"
            return header + "\n---\n".join(lines)

        @agent.tool_plain
        async def web_search(query: str) -> str:  # type: ignore[misc]
            """Search the web for up-to-date information about *query*.

            Uses Tavily if TAVILY_API_KEY is set, Exa if EXA_API_KEY is set,
            otherwise falls back to DuckDuckGo.
            """
            if result_callback is not None:
                try:
                    await result_callback({
                        "type": "agent_status",
                        "tool": "web_search",
                        "display_text": "searching web",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    })
                except Exception:
                    pass
            return await _web_search(query)

        self._agent = agent
        self._agent_model_name = reasoning_model
        logger.info(f"AgentManager: autonomous agent built on model '{reasoning_model}'")

    async def ask_agent(self, query: str) -> Dict[str, Any]:
        """
        Run the autonomous agent loop for *query*.

        The agent will autonomously decide whether to call search_transcript,
        web_search, or both before composing its final answer.

        Args:
            query: Natural language question from the user / UI.

        Returns:
            Dict with keys:
              - ``response``: The agent's synthesised text response (str).
              - ``error``: Content extracted from <error>…</error> tags in the
                model output, or ``None`` if no error tag was present.

        Raises:
            RuntimeError: If initialize() has not been called yet.
        """
        if self._agent is None:
            raise RuntimeError(
                "AgentManager not initialised. Call initialize() before ask_agent()."
            )

        result = await self._agent.run(query)
        output_data = getattr(result, 'data', None) or getattr(result, 'output', str(result))

        # Detect whether PydanticAI actually executed any tool calls during the run.
        # If the model leaked <tool_call> XML as plain text but PydanticAI recorded no
        # tool calls, the model is emitting tool-call syntax that the framework did not
        # intercept.  Retry once with an explicit hint to use the built-in tools.
        has_leaked_tool_call = isinstance(output_data, str) and bool(
            re.search(r'<tool_call>', output_data, re.I)
        )
        if has_leaked_tool_call:
            tool_calls_made = False
            try:
                for msg in result.all_messages():
                    parts = getattr(msg, 'parts', [])
                    if any(getattr(p, 'tool_name', None) for p in parts):
                        tool_calls_made = True
                        break
            except Exception:
                tool_calls_made = True  # can't tell — don't retry

            if not tool_calls_made:
                logger.warning(
                    "AgentManager: model leaked <tool_call> XML without PydanticAI "
                    "intercepting it — retrying with tool-use hint."
                )
                result = await self._agent.run(
                    query + "\n\n(Use the search_transcript and web_search tools available to you.)"
                )
                output_data = getattr(result, 'data', None) or getattr(result, 'output', str(result))

        # Detect whether the entire response is wrapped in XML-like tags.
        # If the response (after trimming whitespace) starts with '<' and ends with '>',
        # the model likely returned a raw structured/tool-call response instead of plain
        # text — retry the query once.
        if isinstance(output_data, str):
            _stripped = output_data.strip()
            if _stripped.startswith('<') and _stripped.endswith('>'):
                logger.warning(
                    "AgentManager: response appears XML-wrapped "
                    "(first char='<', last char='>') — retrying query."
                )
                result = await self._agent.run(query)
                output_data = getattr(result, 'data', None) or getattr(result, 'output', str(result))

        # Clean up any raw tool_call XML blocks leaked by the model
        # Handles both well-formed </tool_call> and malformed /tool_call> closing tags
        if isinstance(output_data, str):
            output_data = re.sub(r'(?si)<tool_call>.*?</?tool_call>', '', output_data).strip()

        # Unwrap <answer>…</answer> tags, keeping the inner content in place
        if isinstance(output_data, str):
            output_data = re.sub(r'(?si)<answer>(.*?)</answer>', r'\1', output_data).strip()

        # Extract <error>…</error> tags emitted by the model
        error_value: Optional[str] = None
        if isinstance(output_data, str):
            error_match = re.search(r'(?si)<error>(.*?)</error>', output_data)
            if error_match:
                error_value = error_match.group(1).strip()
                output_data = re.sub(r'(?si)<error>.*?</error>', '', output_data).strip()

        return {"response": output_data, "error": error_value}
