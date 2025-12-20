# Video Player Webapp with BYOC SDK Integration

A React-based video player application that integrates with the @muxionlabs/byoc-sdk for WebRTC streaming and real-time subtitle processing. This app publishes video streams via WHIP (WebRTC-HTTP Ingestion Protocol) and consumes subtitle data through Server-Sent Events (SSE).

## Features

- **WebRTC WHIP Streaming**: Publish video streams to AI processing pipelines
- **Real-time Subtitle Processing**: Consume and display subtitles from SSE data
- **WebVTT Integration**: Dynamic subtitle track management with cue addition
- **Configuration Management**: User-friendly modal for gateway and API settings
- **Responsive Design**: Built with Tailwind CSS for modern, responsive UI
- **Error Handling**: Comprehensive error management and user feedback
- **Loading States**: Professional loading indicators and connection status

## Technology Stack

- **Frontend**: React 18 with TypeScript
- **Styling**: Tailwind CSS
- **Streaming**: WebRTC (WHIP/WHEP)
- **SDK**: @muxionlabs/byoc-sdk
- **Build Tool**: Vite
- **Subtitles**: WebVTT format with dynamic track management

## Project Structure

```
webapp/
├── src/
│   ├── components/
│   │   ├── ConfigModal.tsx          # Gateway configuration modal
│   │   ├── VideoPlayer.tsx         # Video preview component
│   │   ├── StreamControls.tsx     # Stream management controls
│   │   ├── SubtitleTrack.tsx     # Subtitle display and management
│   │   ├── LoadingSpinner.tsx    # Loading state component
│   │   └── ErrorDisplay.tsx       # Error handling component
│   ├── types/
│   │   └── index.ts                # TypeScript interfaces
│   ├── App.tsx                    # Main application component
│   ├── main.tsx                   # React entry point
│   └── index.css                   # Tailwind CSS styles
├── package.json                   # Dependencies and scripts
├── tsconfig.json                # TypeScript configuration
├── vite.config.ts                # Vite build configuration
├── tailwind.config.js            # Tailwind CSS configuration
└── index.html                    # HTML entry point
```

## Installation

1. **Install Dependencies**:
   ```bash
   cd webapp
   npm install
   ```

2. **Development Server**:
   ```bash
   npm run dev
   ```

3. **Build for Production**:
   ```bash
   npm run build
   ```

## Configuration

### Gateway Configuration

The app requires a BYOC gateway URL and optional API key for authentication:

- **Default Gateway URL**: `http://localhost:5937`
- **API Key**: Optional authentication token
- **Default Pipeline**: `default` (configurable)

### Configuration Modal

Access the configuration modal by clicking the "Configure" button in the header. The modal allows you to set:

1. **Gateway URL**: The base URL for your BYOC gateway server
2. **API Key**: Authentication token for API requests
3. **Default Pipeline**: Pipeline name for stream processing

## Usage

### Starting a Stream

1. **Configure Gateway**: Click "Configure" and set your gateway URL and API key
2. **Start Publishing**: Click "Start Publishing" to begin streaming
3. **Monitor Status**: Watch the connection status and stream ID display
4. **Video Preview**: See your local video stream in the preview panel

### Subtitle Processing

1. **Automatic Detection**: The app automatically consumes SSE subtitle data
2. **Real-time Display**: Subtitles appear as they're processed by the AI pipeline
3. **WebVTT Format**: Subtitles are converted to WebVTT format for browser compatibility
4. **Dynamic Cues**: New subtitle cues are added to the track in real-time

### SSE Data Format

The app expects subtitle data in the following JSON format:

```json
{
  "type": "subtitle_update",
  "timestamp_utc": "2025-12-19T21:18:49.418525+00:00",
  "window": {
    "start": 13.498666666666665,
    "end": 16.498666666666665
  },
  "srt_content": "1\n00:00:14,238 --> 00:00:16,038\nWe all want to help one another.\n"
}
```

### Stopping a Stream

Click "Stop Publishing" to disconnect from the stream and stop subtitle consumption.

## API Integration

### BYOC SDK Classes Used

- **Stream**: Handles WebRTC WHIP publishing
- **StreamConfig**: Manages gateway configuration
- **DataStreamClient**: Consumes SSE subtitle data

### Key Methods

- `stream.startStream()`: Initiates WHIP streaming
- `stream.getMediaStream()`: Retrieves local media stream
- `dataStream.connect()`: Establishes SSE connection
- `dataStream.on('data')`: Handles subtitle data events

## Error Handling

The app includes comprehensive error handling for:

- **Connection Errors**: Gateway connectivity issues
- **Stream Errors**: WebRTC negotiation failures
- **SSE Errors**: Subtitle data stream problems
- **Configuration Errors**: Invalid gateway or API settings

## UI Components

### Stream Controls
- Start/Stop publishing buttons
- Connection status indicator
- Stream ID display
- Error message display

### Video Player
- Local video preview
- Live streaming indicator
- Video track count display

### Subtitle Track
- Current subtitle display
- WebVTT track integration
- Subtitle statistics
- Recent cues list

### Configuration Modal
- Gateway URL input
- API key input
- Pipeline configuration
- Connection status

## Development

### Adding New Features

1. **Core Logic**: Implement in `src/core` directory
2. **React Integration**: Add hooks in `src/react` directory
3. **UI Components**: Create in `src/components` directory
4. **Type Safety**: Update interfaces in `src/types/index.ts`

### Testing

- **Unit Tests**: Add tests for core functionality
- **Integration Tests**: Test BYOC SDK integration
- **E2E Tests**: Test complete streaming workflows

## Browser Compatibility

- **Modern Browsers**: Chrome, Firefox, Safari, Edge
- **WebRTC Support**: Required for WHIP streaming
- **SSE Support**: Required for subtitle data consumption
- **WebVTT Support**: Required for subtitle display

## Performance Considerations

- **Efficient Rendering**: React optimization for real-time updates
- **Memory Management**: Proper cleanup of WebRTC connections and SSE streams
- **Network Optimization**: Efficient data handling for subtitle streams

## Security

- **API Key Management**: Secure storage and transmission
- **CORS Configuration**: Proper cross-origin resource sharing
- **WebRTC Security**: Encrypted peer connections

## Troubleshooting

### Common Issues

1. **Gateway Connection Failed**
   - Verify gateway URL is correct
   - Check gateway server is running
   - Ensure CORS is properly configured

2. **No Subtitle Data**
   - Verify SSE endpoint is accessible
   - Check API key authentication
   - Confirm subtitle data format matches expected structure

3. **WebRTC Connection Issues**
   - Check firewall settings
   - Verify STUN/TURN server configuration
   - Ensure HTTPS is used in production

### Debug Mode

Enable debug logging by setting:
```typescript
const DEBUG = true;
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## License

This project is licensed under the MIT License.