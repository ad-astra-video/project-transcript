# Video Transcription and Subtitle Integration Pipeline

A Python-based real-time video processing pipeline that receives video segments via a trickle server, transcribes audio using faster-whisper, generates SRT subtitles, and integrates them back into the video stream.

## Features

- **Real-time Processing**: Processes 3-second video segments as they arrive
- **Speech Transcription**: Uses faster-whisper for accurate audio transcription
- **Subtitle Integration**: Supports both hard-coded and soft-coded subtitles
- **Trickle Protocol**: Built-in support for segment-based video streaming
- **Optional Data URL**: Send subtitle files to a third-party URL (toggleable)
- **Error Recovery**: Robust error handling and automatic recovery

## Architecture

```
Video Segments (Trickle) → Decode → Extract Audio → Transcribe → Generate SRT → Integrate Subtitles → Re-encode → Output (Trickle)
                                                                                      ↓
                                                                           Optional: Send to Data URL
```

## Project Structure

```
src/
├── pipeline/
│   ├── __init__.py           # Pipeline module initialization
│   ├── config.py             # Configuration management
│   └── main.py               # Main pipeline orchestration
├── video/
│   ├── __init__.py           # Video processing module
│   └── ffmpeg_decoder.py     # FFmpeg-based video decoding and audio extraction
├── transcription/
│   ├── __init__.py           # Transcription module
│   ├── whisper_client.py     # faster-whisper integration 
│   └── srt_generator.py      # SRT subtitle generation
├── subtitles/
│   ├── __init__.py           # Subtitle integration module
│   └── subtitle_integrator.py # Hard/soft subtitle integration
└── trickle/                  # Existing trickle protocol implementation
    ├── trickle_subscriber.py
    ├── trickle_publisher.py
    └── media.py
```

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg (must be installed and available in PATH)
- CUDA-compatible GPU (optional, for faster transcription)

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd /home/user/project-transcript
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg:**
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

## Configuration

The pipeline is configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SUBSCRIBE_URL` | `http://localhost:8080/subscribe` | Trickle subscriber URL for input video |
| `PUBLISH_URL` | `http://localhost:8080/publish` | Trickle publisher URL for output video |
| `DATA_URL` | None | Optional URL for sending subtitle files |
| `SEGMENT_DURATION` | `3.0` | Video segment duration in seconds |
| `WHISPER_MODEL` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `WHISPER_DEVICE` | `cpu` | Device for transcription (`cpu`, `cuda`) |
| `HARD_CODE_SUBTITLES` | `true` | Whether to hard-code subtitles (true) or soft-code (false) |
| `ENABLE_DATA_URL` | `false` | Whether to send subtitles to data URL |

## Usage

### Basic Usage

```bash
# Run with default configuration
python src/pipeline/main.py
```

### Custom Configuration

```bash
# Set environment variables
export SUBSCRIBE_URL="http://your-server:8080/subscribe"
export PUBLISH_URL="http://your-server:8080/publish" 
export WHISPER_MODEL="medium"
export WHISPER_DEVICE="cuda"
export HARD_CODE_SUBTITLES="false"

# Run pipeline
python src/pipeline/main.py
```

### Docker Usage (Optional)

```dockerfile
FROM python:3.9-slim

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY src/ /app/src/

# Set working directory
WORKDIR /app

# Run pipeline
CMD ["python", "src/pipeline/main.py"]
```

## API Integration

### Trickle Protocol

The pipeline integrates with the trickle protocol for video streaming:

- **Input**: Subscribes to video segments via HTTP GET requests
- **Output**: Publishes processed segments via HTTP POST requests
- **Format**: Video segments in `video/mp2t` format
- **Error Handling**: Automatic recovery from 404 (not ready) and 470 (too old) errors

### Data URL (Optional)

When enabled, subtitle files are sent to a third-party URL:

```http
POST /your-data-endpoint
Content-Type: text/plain; charset=utf-8
X-Segment-Id: 123

1
00:00:00,000 --> 00:00:02,500
Hello, this is a transcription example.

2
00:00:02,500 --> 00:00:05,000
The audio has been processed successfully.
```

## Performance Tuning

### Whisper Model Selection

| Model | Speed | Accuracy | Memory |
|-------|--------|----------|---------|
| `tiny` | Fastest | Lowest | ~1GB |
| `base` | Fast | Good | ~1GB |
| `small` | Medium | Better | ~2GB |
| `medium` | Slower | High | ~5GB |
| `large` | Slowest | Highest | ~10GB |

### Hardware Recommendations

- **CPU**: Multi-core processor (4+ cores recommended)
- **Memory**: 8GB+ RAM (16GB+ for large models)
- **GPU**: CUDA-compatible GPU for faster transcription (optional)
- **Storage**: SSD recommended for temporary file operations

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   ```
   Error: FFmpeg not found in PATH
   Solution: Install FFmpeg and ensure it's in your system PATH
   ```

2. **CUDA out of memory**
   ```
   Error: CUDA out of memory
   Solution: Use smaller Whisper model or switch to CPU processing
   ```

3. **Trickle connection failed**
   ```
   Error: Failed to connect to trickle server
   Solution: Verify server URLs and network connectivity
   ```

### Logging

The pipeline uses Python's logging module. Adjust log level for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

Monitor these metrics for performance optimization:
- Segment processing time
- Memory usage
- Transcription accuracy

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/
```

### Code Style

The project follows PEP 8 style guidelines:

```bash
# Install formatting tools
pip install black isort flake8

# Format code
black src/
isort src/

# Check style
flake8 src/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for efficient speech recognition
- [FFmpeg](https://ffmpeg.org/) for video/audio processing
- [aiohttp](https://aiohttp.readthedocs.io/) for async HTTP operations
