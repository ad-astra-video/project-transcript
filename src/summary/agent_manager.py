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
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Awaitable

import numpy as np
from openai import AsyncOpenAI

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from .llm_manager import LLMManager, MessageFormatMode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_EMBEDDING_BASE_URL = "http://byoc-transcription-tei-embeddings:6060/v1"
_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
_TOP_K = 5  # Number of transcript chunks returned per search


# ---------------------------------------------------------------------------
# Transcript chunk stored alongside its vector
# ---------------------------------------------------------------------------
@dataclass
class TranscriptChunk:
    text: str
    timestamp: Optional[float] = None          # seconds from stream start
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
            self._index.add(np.array([vector], dtype="float32"))
            self._chunks.append(chunk)

    async def search(self, query: str, top_k: int = _TOP_K) -> List[TranscriptChunk]:
        """Return the *top_k* most relevant chunks for *query*."""
        if not self._chunks:
            return []
        vector = await self._embed(query)
        async with self._lock:
            k = min(top_k, len(self._chunks))
            distances, indices = self._index.search(
                np.array([vector], dtype="float32"), k
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
        from duckduckgo_search import DDGS
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

        # ---- PydanticAI agent (built lazily after initialize()) ----
        self._agent: Optional[Agent] = None
        self._agent_model_name: Optional[str] = None
        self._agent_chat_query: Optional[str] = None  # Custom system prompt for agent queries

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
            speaker=speaker,
            window_id=window_id,
        )
        try:
            await self.knowledge_store.add(chunk)
        except Exception as e:
            logger.exception(f"Failed to index transcript segment: {e}")

    def reset_knowledge_store(self) -> None:
        """Clear all stored vectors. Call when a stream ends."""
        self.knowledge_store.reset()

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
            self._agent_chat_query = agent_chat_query
            logger.info(f"Updated agent_chat_query to: {agent_chat_query[:50]}..." if len(agent_chat_query) > 50 else f"Updated agent_chat_query to: {agent_chat_query}")

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
                "You are an intelligent assistant with access to the live video "
                "transcript and the web. Use the search_transcript tool to find "
                "relevant segments from the video, and use web_search to fetch "
                "up-to-date information from the internet. "
                "Always cite sources and timestamps where available."
            ),
        )

        knowledge_store = self.knowledge_store  # local ref for closures

        @agent.tool_plain
        async def search_transcript(query: str) -> str:  # type: ignore[misc]
            """Search the video transcript for content relevant to *query*.

            Returns the most semantically similar transcript segments.
            """
            chunks = await knowledge_store.search(query)
            if not chunks:
                return "No transcript content indexed yet."
            lines = []
            for chunk in chunks:
                ts = f"[{chunk.timestamp:.1f}s] " if chunk.timestamp is not None else ""
                sp = f"({chunk.speaker}) " if chunk.speaker else ""
                lines.append(f"{ts}{sp}{chunk.text}")
            return "\n---\n".join(lines)

        @agent.tool_plain
        async def web_search(query: str) -> str:  # type: ignore[misc]
            """Search the web for up-to-date information about *query*.

            Uses Tavily if TAVILY_API_KEY is set, Exa if EXA_API_KEY is set,
            otherwise falls back to DuckDuckGo.
            """
            return await _web_search(query)

        self._agent = agent
        self._agent_model_name = reasoning_model
        logger.info(f"AgentManager: autonomous agent built on model '{reasoning_model}'")

    async def ask_agent(self, query: str) -> str:
        """
        Run the autonomous agent loop for *query*.

        The agent will autonomously decide whether to call search_transcript,
        web_search, or both before composing its final answer.

        Args:
            query: Natural language question from the user / UI.

        Returns:
            The agent's synthesised text response.

        Raises:
            RuntimeError: If initialize() has not been called yet.
        """
        if self._agent is None:
            raise RuntimeError(
                "AgentManager not initialised. Call initialize() before ask_agent()."
            )
        # Use custom system prompt if set via update_params, otherwise use default
        system_prompt = self._agent_chat_query if self._agent_chat_query else None
        result = await self._agent.run(query, system_prompt=system_prompt)
        return result.output
