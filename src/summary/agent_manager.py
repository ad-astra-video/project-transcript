"""
AgentManager - Extends LLMManager with an autonomous PydanticAI agent.

Adds:
- TranscriptKnowledgeStore: in-memory Faiss index backed by a local TEI
  embedding model (default port 6060, mirrors the vLLM container pattern).
- AgentManager: wraps LLMManager and exposes:
    * index_transcript_segment()  – embed and store a transcript chunk
    * ask_agent()                 – run the autonomous agent loop

Web search tool priority (first key found wins):
  TAVILY_API_KEY  → Tavily Search API
  EXA_API_KEY     → Exa Search API
  (fallback)      → DuckDuckGo (no key required)

Embedding client configuration (mirrors fast/reasoning pattern):
  EMBEDDING_BASE_URL   – defaults to http://tei-embeddings:80/v1
  EMBEDDING_API_KEY    – defaults to "dummy"
  EMBEDDING_MODEL      – defaults to BAAI/bge-small-en-v1.5
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
_DEFAULT_EMBEDDING_BASE_URL = "http://tei-embeddings:6060/v1"
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add(self, chunk: TranscriptChunk) -> None:
        """Embed *chunk* and add it to the Faiss index."""
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

    def reset(self) -> None:
        """Clear all stored vectors and chunks (call on stream end)."""
        self._index = None
        self._dim = None
        self._chunks = []
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
    Wraps LLMManager and adds:
    - A TranscriptKnowledgeStore for semantic search over indexed transcript chunks.
    - A PydanticAI autonomous agent with `search_transcript` and `web_search` tools.

    The existing LLMManager / LLMClient / HealthMetrics interface is unchanged –
    all existing plugins continue to work via `self.llm`.

    Configuration (mirrors the vLLM env-var pattern):
        EMBEDDING_BASE_URL   Base URL for the TEI service  (default: http://tei-embeddings:80/v1)
        EMBEDDING_API_KEY    API key for TEI               (default: "dummy")
        EMBEDDING_MODEL      Model name served by TEI      (default: BAAI/bge-small-en-v1.5)
    """

    def __init__(
        self,
        # ---- LLMManager constructor args (pass-through) ----
        fast_base_url: str,
        fast_api_key: str,
        reasoning_base_url: str,
        reasoning_api_key: str,
        rapid_model: Optional[str] = None,
        reasoning_model: Optional[str] = None,
        message_format_mode: Optional[MessageFormatMode] = None,
        request_timeout_seconds: float = 240.0,
        health_result_callback: Optional[Callable[[dict], Awaitable[None]]] = None,
        health_monitoring_callback: Optional[Callable[[dict, str], Awaitable[None]]] = None,
        # ---- Embedding / knowledge store args ----
        embedding_base_url: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        # ---- Core LLM manager (unchanged interface for existing plugins) ----
        self.llm = LLMManager(
            fast_base_url=fast_base_url,
            fast_api_key=fast_api_key,
            reasoning_base_url=reasoning_base_url,
            reasoning_api_key=reasoning_api_key,
            rapid_model=rapid_model,
            reasoning_model=reasoning_model,
            message_format_mode=message_format_mode,
            request_timeout_seconds=request_timeout_seconds,
            health_result_callback=health_result_callback,
            health_monitoring_callback=health_monitoring_callback,
        )

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

    # ------------------------------------------------------------------
    # Delegated LLMManager properties / methods
    # ------------------------------------------------------------------

    @property
    def fast_client(self):
        return self.llm.fast_client

    @property
    def reasoning_client(self):
        return self.llm.reasoning_client

    @property
    def rapid_llm_client(self):
        return self.llm.rapid_llm_client

    @property
    def reasoning_llm_client(self):
        return self.llm.reasoning_llm_client

    @property
    def rapid_model(self):
        return self.llm.rapid_model

    @property
    def reasoning_model(self):
        return self.llm.reasoning_model

    def set_stream_id(self, stream_id: Optional[str]):
        self.llm.set_stream_id(stream_id)

    def start_scheduler(self):
        return self.llm.start_scheduler()

    def stop_scheduler(self):
        return self.llm.stop_scheduler()

    def update_params(self, **kwargs):
        return self.llm.update_params(**kwargs)

    def __getattr__(self, name: str):
        # Proxy any attribute not defined on AgentManager through to the inner
        # LLMManager so that existing code (including tests) that accesses
        # private attributes like _rapid_llm_client or _reasoning_base_url
        # continues to work transparently.
        try:
            return getattr(self.llm, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    async def initialize(self) -> Optional[str]:
        """Initialize LLM clients, auto-detect models, and build the PydanticAI agent."""
        detected = await self.llm.initialize()
        self._build_agent()
        return detected

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
            logger.warning(f"Failed to index transcript segment: {e}")

    def reset_knowledge_store(self) -> None:
        """Clear all stored vectors. Call when a stream ends."""
        self.knowledge_store.reset()

    # ------------------------------------------------------------------
    # Autonomous agent
    # ------------------------------------------------------------------

    def _build_agent(self) -> None:
        """
        Construct the PydanticAI Agent backed by the reasoning LLM.

        The agent is given two tools:
          - search_transcript: semantic search over the indexed video transcript
          - web_search: live web search (Tavily > Exa > DuckDuckGo)
        """
        model_name = self.llm.reasoning_model
        if not model_name:
            logger.warning("AgentManager: reasoning model not set, skipping agent build")
            return

        reasoning_base_url = self.llm._reasoning_base_url
        reasoning_api_key = self.llm._reasoning_api_key

        provider = OpenAIProvider(
            base_url=reasoning_base_url,
            api_key=reasoning_api_key or "dummy",
        )
        model = OpenAIModel(model_name, provider=provider)

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
        self._agent_model_name = model_name
        logger.info(f"AgentManager: autonomous agent built on model '{model_name}'")

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
        result = await self._agent.run(query)
        return result.output
