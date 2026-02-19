# Content Type Detection Plugin

Plugin for automatically detecting the content type of transcribed audio/video content.

## Overview

The Content Type Detection Plugin analyzes transcript text to determine the type of content being transcribed (e.g., GENERAL_MEETING, TECHNICAL_TALK, PODCAST, etc.). This detection is used to customize summarization prompts and rules for different content types.

## Supported Content Types

| Content Type | Description |
|--------------|-------------|
| `GENERAL_MEETING` | Standard business or team meetings |
| `TECHNICAL_TALK` | Technical presentations, code reviews, engineering discussions |
| `LECTURE_OR_TALK` | Educational lectures, presentations |
| `INTERVIEW` | Interview format with Q&A |
| `PODCAST` | Podcast discussions |
| `STREAMER_MONOLOGUE` | Single speaker streaming content |
| `NEWS_UPDATE` | News broadcasts or updates |
| `GAMEPLAY_COMMENTARY` | Gaming streams with commentary |
| `CUSTOMER_SUPPORT` | Customer support calls |
| `DEBATE` | Debate or discussion format |
| `UNKNOWN` | Default state before detection |

## Events Emitted

The plugin emits the following events that other plugins can subscribe to:

### `on_content_type_detected`

Emitted after successfully detecting a content type.

**Payload:**

| Field | Type | Description |
|-------|------|-------------|
| `content_type` | `str` | The detected content type (one of the values listed above) |
| `confidence` | `float` | Detection confidence score (0.0 to 1.0) |
| `source` | `str` | Source of the detection. Currently always `"AUTO_DETECTED"` |
| `reasoning` | `str` | LLM's reasoning for the detection |

**Example:**

```python
{
    "content_type": "TECHNICAL_TALK",
    "confidence": 0.85,
    "source": "AUTO_DETECTED",
    "reasoning": "Discussion of API endpoints, database schemas, and code implementation details suggests technical content."
}
```

## Events Subscribed To

The plugin subscribes to the following events from other plugins:

### `transcription_window_available`

Triggered when a new transcription window is available for processing.

## Usage

### Subscribing to `on_content_type_detected`

Other plugins can subscribe to receive notifications when content type is detected:

```python
# In your plugin's init function
summary_client.register_plugin_event_sub(
    plugin_name="my_plugin",
    plugin_instance=my_plugin_instance,
    events={
        "on_content_type_detected": my_plugin_instance.handle_content_type_detected
    }
)

# Handler method
async def handle_content_type_detected(self, content_type: str, confidence: float, source: str, reasoning: str):
    print(f"Content type detected: {content_type} (confidence: {confidence})")
    # Update your plugin's state or behavior based on the content type
```

## Configuration

The plugin is initialized with the following parameters:

- `window_manager`: WindowManager instance for accessing transcription windows
- `llm`: LLMManager instance for accessing LLM clients
- `result_callback`: Callback to send results back to the summary client
- `summary_client`: Reference to the SummaryClient (optional, for event emission)
- `initial_summary_delay_seconds`: Delay before first detection attempt (default: 15.0)

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
│ Subscribes:     │ │ Subscribes:     │ │ Subscribes:     │
│ - transcription_│ │ - summary_      │ │ - summary_      │
│   window_       │ │   window_       │ │   window_       │
│   available     │ │   available     │ │   available     │
│                 │ │                 │ │                 │
│ Emits:          │ │                 │ │                 │
│ - on_content_   │ │                 │ │                 │
│   type_detected │ │                 │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

## API Reference

### `ContentTypeDetectionPlugin`

```python
class ContentTypeDetectionPlugin:
    def __init__(self, window_manager, llm, result_callback, summary_client=None,
                 initial_summary_delay_seconds: float = 15.0, **kwargs)
    
    async def process(self, transcription_window_id: int, **kwargs)
```

### `ContentType`

```python
class ContentType(str, Enum):
    GENERAL_MEETING = "GENERAL_MEETING"
    TECHNICAL_TALK = "TECHNICAL_TALK"
    LECTURE_OR_TALK = "LECTURE_OR_TALK"
    INTERVIEW = "INTERVIEW"
    PODCAST = "PODCAST"
    STREAMER_MONOLOGUE = "STREAMER_MONOLOGUE"
    NEWS_UPDATE = "NEWS_UPDATE"
    GAMEPLAY_COMMENTARY = "GAMEPLAY_COMMENTARY"
    CUSTOMER_SUPPORT = "CUSTOMER_SUPPORT"
    DEBATE = "DEBATE"
    UNKNOWN = "UNKNOWN"
```

### `ContentTypeState`

```python
class ContentTypeState(BaseModel):
    content_type: str = ContentType.UNKNOWN.value
    previous_content_type: str = ""
    confidence: float = 0.0
    source: str = ContentTypeSource.INITIAL.value
    last_detection_text: str = ""
    context_length: int = 2000
    sentiment_enabled: bool = False
    participants_enabled: bool = False