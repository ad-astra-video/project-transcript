# Context Summary Plugin

Plugin for generating context-aware summaries from transcribed content.

## Overview

The Context Summary Plugin generates detailed summaries of transcription content, incorporating context from previous windows, insights extraction, and content-type-specific rules.

## Events Emitted

Currently, this plugin does not emit any events.

## Events Subscribed To

The plugin subscribes to the following events from other plugins:

### `on_content_type_detected`

Emitted by the Content Type Detection Plugin when a content type is detected.

**Handler:** `handle_content_type_detected`

**Payload:**

| Field | Type | Description |
|-------|------|-------------|
| `content_type` | `str` | The detected content type (e.g., "TECHNICAL_TALK", "PODCAST") |
| `confidence` | `float` | Detection confidence score (0.0 to 1.0) |
| `source` | `str` | Source of the detection (e.g., "AUTO_DETECTED") |
| `reasoning` | `str` | LLM's reasoning for the detection |

**Behavior:** When this event is received, the plugin updates the content type state in the SummaryClient so that subsequent summarization uses the correct content-type-specific rules and prompts.

### `summary_window_available`

Triggered when a new summary window is ready for processing.

## Usage

The plugin is automatically initialized by the SummaryClient. No manual subscription is required.

## Configuration

The plugin is initialized with the following parameters:

- `window_manager`: WindowManager instance for accessing transcription windows
- `llm`: LLMManager instance for accessing LLM clients
- `result_callback`: Callback to send results back to the summary client
- `summary_client`: Reference to the SummaryClient
- `initial_summary_delay_seconds`: Delay before first summary (default: 15.0)

## Content Type Integration

The plugin uses content type information to customize:

- **System prompts**: Different prompts for different content types
- **Insight extraction**: Content-type-specific insight types
- **Rule modifiers**: Content-type-specific rules for summarization

When a content type is detected via the `on_content_type_detected` event, the plugin automatically updates its internal state to use the appropriate rules.

## API Reference

### `ContextSummaryPlugin`

```python
class ContextSummaryPlugin:
    def __init__(self, window_manager, llm, result_callback, summary_client=None,
                 initial_summary_delay_seconds: float = 15.0, **kwargs)
    
    async def process(self, summary_window_id: int, **kwargs)
    
    async def handle_content_type_detected(self, content_type: str, confidence: float, 
                                            source: str, reasoning: str)
```

### `ContextSummaryTask`

```python
class ContextSummaryTask:
    def __init__(self, client, model, max_tokens=8192, temperature=0.2,
                 system_prompt=None, insights_response_json_schema=None,
                 message_format_mode="system", content_type_state=None,
                 window_manager=None, semaphore=None)
    
    async def process_context_summary(self, text: str, context: str = "") -> Dict[str, Any]
```

## Content Type Rule Modifiers

The plugin uses `CONTENT_TYPE_RULE_MODIFIERS` to customize behavior based on content type:

| Content Type | sentiment_enabled | participants_enabled |
|--------------|-------------------|---------------------|
| GENERAL_MEETING | true | true |
| TECHNICAL_TALK | false | false |
| LECTURE_OR_TALK | false | true |
| INTERVIEW | true | true |
| PODCAST | true | true |
| STREAMER_MONOLOGUE | true | false |
| NEWS_UPDATE | true | false |
| GAMEPLAY_COMMENTARY | false | false |
| CUSTOMER_SUPPORT | true | true |
| DEBATE | true | true |
| UNKNOWN | false | false |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      SummaryClient                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Plugin Event System                     │   │
│  │  _notify_plugins(event_name, **kwargs)             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ ContentType     │ │ ContextSummary  │ │ RapidSummary    │
│ DetectionPlugin │ │ Plugin          │ │ Plugin          │
│                 │ │                 │ │                 │
│ Emits:          │ │ Subscribes:     │ │ Subscribes:     │
│ - on_content_   │ │ - on_content_   │ │ - summary_      │
│   type_detected │ │   type_detected │ │   window_       │
│                 │ │ - summary_      │ │   available     │
│ Subscribes:     │ │   window_       │ │                 │
│ - transcription_│ │   available     │ │                 │
│   window_       │ │                 │ │                 │
│   available     │ │                 │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘