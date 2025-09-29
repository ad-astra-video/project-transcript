# Real-time Video Transcription Pipeline

A Python-based real-time audio transcription pipeline built on the pytrickle framework. The application processes audio streams in real-time, transcribes speech using faster-whisper, and generates live SRT subtitles that are sent back through the trickle protocol.

## Quick Start

```bash
# Build and run with Docker (recommended)
docker build -t livepeer/byoc-transcipt .
docker run -it --gpus all -p 8000:8000 -v /home/user/models:/modelslivepeer/byoc-transcipt

# The service will start on port 8000 and be ready to process audio streams
# Check logs for "Starting StreamProcessor on port 8000"
```

## Features

- **Real-time Audio Processing**: Processes audio frames as they arrive via pytrickle StreamProcessor
- **Speech Transcription**: Uses faster-whisper for accurate audio transcription with CUDA acceleration
- **Live SRT Generation**: Generates real-time SRT subtitles with proper timing
- **Sliding Window Processing**: Uses configurable audio windows with overlap for continuous transcription

## Architecture

```
Audio Frames (pytrickle) → Buffer → Sliding Window → Whisper Transcription → SRT Generation → Publish to data_url
                              ↓
                        Rolling Buffer (3s window, 1s overlap)
```

## Project Structure

```
src/
├── pipeline/
│   ├── __init__.py           # Pipeline module initialization
│   └── main.py               # Main StreamProcessor-based pipeline
└── transcription/
    ├── __init__.py           # Transcription module
    ├── whisper_client.py     # faster-whisper integration with async support
    └── srt_generator.py      # SRT subtitle generation with timing

requirements.txt              # Python dependencies including pytrickle from git
Dockerfile                    # CUDA-enabled container with uv package manager
.dockerignore                 # Docker build context exclusions
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for faster transcription)
- Docker (for containerized deployment)

### Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd project-transcript
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Docker Setup (Recommended)

1. **Build the Docker image:**
   ```bash
   docker build -t livepeer/byoc-transcipt .
   ```

2. **Run with GPU support:**
   ```bash
   docker run -it --gpus all -p 8000:8000 livepeer/byoc-transcipt
   ```

## Configuration

The pipeline is configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `large` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `WHISPER_DEVICE` | `cuda` | Device for transcription (`cpu`, `cuda`) |
| `MODEL_DIR` | `/models` | Directory for storing Whisper models |
| `PORT` | `8000` | Port for the StreamProcessor server |

### Audio Processing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `audio_sample_rate` | `16000` | Audio sample rate for processing |
| `chunk_window` | `3.0` | Audio window duration in seconds |
| `chunk_overlap` | `1.0` | Overlap between windows in seconds |
| `whisper_language` | `None` | Language code for transcription (auto-detect if None) |
| `compute_type` | `float16` | Compute precision (`float16`, `float32`) |

## Usage

### Local Development

```bash
# Run with default configuration
python -m src.pipeline.main

# Or run directly
python src/pipeline/main.py
```

### Custom Configuration

```bash
# Set environment variables
export WHISPER_MODEL="medium"
export WHISPER_DEVICE="cuda"
export PORT="8000"

# Run pipeline
python -m src.pipeline.main
```

### Docker Usage

```bash
# Build the image
docker build -t livepeer/byoc-transcipt .

# Run with GPU support
docker run -it --gpus all -p 8000:8000 livepeer/byoc-transcipt

# Run with custom environment variables
docker run -it --gpus all -p 8000:8000 \
  -e WHISPER_MODEL=medium \
  -e WHISPER_DEVICE=cuda \
  livepeer/byoc-transcipt

# Run without GPU (CPU only)
docker run -it -p 8000:8000 \
  -e WHISPER_DEVICE=cpu \
  livepeer/byoc-transcipt
```

## API Integration

### StreamProcessor Integration

The pipeline uses pytrickle's StreamProcessor for real-time audio processing:

- **Input**: Receives audio frames via StreamProcessor
- **Processing**: Buffers audio in sliding windows for transcription
- **Output**: Sends subtitle updates via WebSocket data publishing
- **Format**: Audio frames processed in real-time with configurable sample rates

### WebSocket Data Publishing

Subtitle updates are sent via WebSocket in JSON format:

```json
{
  "type": "subtitle_update",
  "timestamp_utc": "2024-01-01T12:00:00.000Z",
  "window": {
    "start": 0.0,
    "end": 3.0
  },
  "srt_content": "1\n00:00:00,000 --> 00:00:02,500\nHello, this is a transcription example.\n\n2\n00:00:02,500 --> 00:00:03,000\nThe audio has been processed successfully."
}
```

### Server Endpoints

The StreamProcessor exposes the following endpoints:

- `GET /health` - Health check endpoint
- `POST /stream` - Audio stream input (handled by pytrickle)

## Performance Tuning

### Whisper Model Selection

| Model | Speed | Accuracy | VRAM Usage | Recommended Use |
|-------|--------|----------|------------|------------------|
| `tiny` | Fastest | Lowest | ~1GB | Testing, low-resource environments |
| `base` | Fast | Good | ~1GB | Development, real-time applications |
| `small` | Medium | Better | ~2GB | Balanced performance |
| `medium` | Slower | High | ~5GB | High accuracy requirements |
| `large` | Slowest | Highest | ~10GB | Production, maximum accuracy |

### Hardware Recommendations

- **CPU**: Multi-core processor (4+ cores recommended)
- **Memory**: 8GB+ RAM (16GB+ for large models)
- **GPU**: CUDA-compatible GPU with 4GB+ VRAM (RTX 3060 or better)
- **Storage**: SSD recommended for model caching and temporary files
- **Network**: Stable connection for real-time streaming

## Troubleshooting

### Common Issues

1. **pytrickle import error**
   ```
   Error: ModuleNotFoundError: No module named 'pytrickle'
   Solution: Ensure pytrickle is installed from git: pip install git+https://github.com/livepeer/pytrickle.git@v0.1.4
   ```

2. **CUDA out of memory**
   ```
   Error: CUDA out of memory
   Solution: Use smaller Whisper model or switch to CPU processing with WHISPER_DEVICE=cpu
   ```

3. **StreamProcessor connection failed**
   ```
   Error: Failed to start StreamProcessor
   Solution: Check port availability and ensure no other services are using port 8000
   ```

4. **Audio processing issues**
   ```
   Error: Audio buffer issues or transcription delays
   Solution: Adjust chunk_window and chunk_overlap parameters for your use case
   ```

### Logging

The pipeline uses Python's logging module. Adjust log level for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

Monitor these metrics for performance optimization:
- Audio buffer processing time
- Transcription latency (should be < window duration)
- Memory usage (especially GPU VRAM)
- WebSocket connection stability
- Real-time factor (processing speed vs. audio speed)


## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [pytrickle](https://github.com/livepeer/pytrickle) for real-time streaming framework
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for efficient speech recognition
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-zone) for GPU acceleration
- [OpenAI Whisper](https://github.com/openai/whisper) for the underlying speech recognition model
