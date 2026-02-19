# Rapid Summary Plugin

Plugin for generating quick summaries of transcribed content.

## Overview

The Rapid Summary Plugin provides fast, lightweight summarization of transcription content. It's designed for scenarios where quick summaries are needed with minimal latency.

## Events Emitted

Currently, this plugin does not emit any events.

## Events Subscribed To

The plugin subscribes to the following events from other plugins:

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

## API Reference

### `RapidSummaryPlugin`

```python
class RapidSummaryPlugin:
    def __init__(self, window_manager, llm, result_callback, summary_client=None, **kwargs)
    
    async def process(self, summary_window_id: int, **kwargs)
```

### `RapidSummaryTask`

```python
class RapidSummaryTask:
    def __init__(self, client, model, max_tokens=500, temperature=0.3)
    
    async def summarize(self, text: str) -> Dict[str, Any]
```

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
│   type_detected │ │   type_detected  │ │   window_       │
│                 │ │ - summary_       │ │   available     │
│ Subscribes:     │ │   window_        │ │                 │
│ - transcription_│ │   available     │ │                 │
│   window_       │ │                 │ │                 │
│   available     │ │                 │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

## Differences from Context Summary Plugin

| Feature | Rapid Summary | Context Summary |
|---------|---------------|-----------------|
| Latency | Low | Higher |
| Context usage | Current window only | Previous windows + insights |
| Use case | Quick updates | Detailed summaries |
| Token limit | Lower | Higher |