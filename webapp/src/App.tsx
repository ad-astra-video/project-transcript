import { useState, useRef, useCallback } from 'react';
import { Stream, StreamConfig, DataStreamClient } from '@muxionlabs/byoc-sdk';
import {
  WebappConfig,
  ConnectionState,
  StreamState,
  ParsedSubtitle,
  SubtitleUpdate,
  TranscriptData,
  TranscriptSegment
} from './types';
import ConfigModal from './components/ConfigModal';
import VideoPlayer from './components/VideoPlayer';
import StreamControls from './components/StreamControls';
import SubtitleTrack from './components/SubtitleTrack';
import LoadingSpinner from './components/LoadingSpinner';
import ErrorDisplay from './components/ErrorDisplay';
import SourceSelectionModal from './components/SourceSelectionModal';

const DEFAULT_CONFIG: WebappConfig = {
  gatewayUrl: 'https://gateway-usa.muxion.video/g',
  defaultPipeline: 'transcription',
  apiKey: ''
};

function App() {
  // State management
  const [config, setConfig] = useState<WebappConfig>(DEFAULT_CONFIG);
  const [isConfigModalOpen, setIsConfigModalOpen] = useState(true);
  const [isSourceSelectionOpen, setIsSourceSelectionOpen] = useState(false);
  const [connectionState, setConnectionState] = useState<ConnectionState>({
    isConnected: false,
    isConnecting: false,
    error: null,
    streamId: null
  });
  const [streamState, setStreamState] = useState<StreamState>({
    isPublishing: false,
    isPreviewing: false,
    mediaStream: null,
    videoElement: null
  });
  const [subtitles, setSubtitles] = useState<ParsedSubtitle[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [streamInfo, setStreamInfo] = useState<any>(null);
  const [pendingStreamStart, setPendingStreamStart] = useState<any>(null);

  // Refs
  const streamRef = useRef<Stream | null>(null);
  const dataStreamRef = useRef<DataStreamClient | null>(null);
  const videoElementRef = useRef<HTMLVideoElement | null>(null);

  // Subtitle parser function
  const parseSubtitleData = useCallback((data: string): SubtitleUpdate | TranscriptData | null => {
    try {
      const parsed = JSON.parse(data);
      if (parsed.type === 'subtitle_update') {
        return parsed as SubtitleUpdate;
      } else if (parsed.type === 'transcript') {
        return parsed as TranscriptData;
      }
      return null;
    } catch (error) {
      console.error('Failed to parse subtitle data:', error);
      return null;
    }
  }, []);

  // Convert transcript segments to parsed subtitle
  const parseTranscriptSegments = useCallback((transcriptData: TranscriptData): ParsedSubtitle[] => {
    return transcriptData.segments.map(segment => ({
      id: segment.id,
      startTime: segment.start_ms / 1000, // Convert milliseconds to seconds
      endTime: segment.end_ms / 1000, // Convert milliseconds to seconds
      text: segment.text,
      rawSrt: `Segment: ${segment.text}\nStart: ${segment.start_ms}ms\nEnd: ${segment.end_ms}ms`
    }));
  }, []);

  // Convert SRT content to parsed subtitle (legacy support)
  const parseSRTContent = useCallback((srtContent: string, window: { start: number; end: number }): ParsedSubtitle => {
    const lines = srtContent.trim().split('\n');
    const textLines = lines.slice(2); // Skip index and timing line
    const text = textLines.join('\n').trim();
    
    return {
      id: `subtitle-${Date.now()}-${Math.random()}`,
      startTime: window.start,
      endTime: window.end,
      text,
      rawSrt: srtContent
    };
  }, []);

  // Handle SSE data
  const handleSSEData = useCallback((event: any) => {
    console.log('Received SSE data:', event.data);
    
    const subtitleUpdate = parseSubtitleData(event.data);
    if (subtitleUpdate) {
      console.log('Parsed subtitle update:', subtitleUpdate);
      
      if (subtitleUpdate.type === 'transcript') {
        // Handle transcript data
        const parsedSubtitles = parseTranscriptSegments(subtitleUpdate as TranscriptData);
        console.log('Parsed transcript subtitles:', parsedSubtitles);
        setSubtitles(prev => [...prev, ...parsedSubtitles]);
      } else if (subtitleUpdate.type === 'subtitle_update') {
        // Handle legacy subtitle_update data
        const parsedSubtitle = parseSRTContent(subtitleUpdate.srt_content, subtitleUpdate.window);
        console.log('Parsed SRT subtitle:', parsedSubtitle);
        setSubtitles(prev => [...prev, parsedSubtitle]);
      }
    } else {
      console.log('No valid subtitle update found in data');
    }
  }, [parseSubtitleData, parseSRTContent, parseTranscriptSegments]);

  // Connect to stream
  const handleConnect = useCallback(async () => {
    if (!config.gatewayUrl) {
      setConnectionState(prev => ({ ...prev, error: 'Gateway URL is required' }));
      return;
    }

    setIsLoading(true);
    setConnectionState(prev => ({ ...prev, isConnecting: true, error: null }));

    try {
      // Create StreamConfig for BYOC SDK
      const streamConfig = new StreamConfig({
        gatewayUrl: config.gatewayUrl,
        defaultPipeline: config.defaultPipeline
      });

      // Initialize stream
      const stream = new Stream(streamConfig);
      streamRef.current = stream;

      // Start stream using the correct SDK method
      const startOptions: any = {
        pipeline: config.defaultPipeline,
        streamName: `stream-${Date.now()}`,
        width: 1280,
        height: 720,
        fpsLimit: 30,
        enableVideoIngress: true,
        enableAudioIngress: true,
        enableDataOutput: true
      };

      const startResponse = await stream.start(startOptions);
      setStreamInfo(startResponse);
      
      // Store the stream start response and show source selection modal
      setPendingStreamStart({ stream, streamConfig, startResponse });
      setIsSourceSelectionOpen(true);
      setIsLoading(false);

    } catch (error) {
      console.error('Connection error:', error);
      setConnectionState(prev => ({
        ...prev,
        isConnecting: false,
        error: error instanceof Error ? error.message : 'Connection failed'
      }));
      setIsLoading(false);
    }
  }, [config, handleSSEData]);

  // Handle source selection and complete publishing
  const handleSourceSelection = useCallback(async (source: 'camera' | 'screen') => {
    if (!pendingStreamStart) return;

    setIsLoading(true);
    setIsSourceSelectionOpen(false);

    try {
      const { stream, streamConfig, startResponse } = pendingStreamStart;

      // Create updated start options with source selection
      const startOptions = {
        pipeline: config.defaultPipeline || 'transcription',
        streamName: `stream-${Date.now()}`,
        width: 1280,
        height: 720,
        fpsLimit: 30,
        enableVideoIngress: true,
        enableAudioIngress: true,
        enableDataOutput: true,
        useScreenShare: source === 'screen'
      };

      // Stop the current stream and restart with selected source
      if (stream) {
        await stream.stop();
      }

      // Create new stream with selected source
      const newStream = new Stream(streamConfig);
      streamRef.current = newStream;

      // Start the stream with selected source
      const newStartResponse = await newStream.start(startOptions);
      setStreamInfo(newStartResponse);

      // Publish to the WHIP URL
      await newStream.publish(startOptions);

      // Get local media stream for preview
      const localStream = newStream.getLocalStream();
      setStreamState(prev => ({ ...prev, mediaStream: localStream, isPreviewing: true, isPublishing: true }));

      // Setup video element with the captured stream
      if (videoElementRef.current && localStream) {
        videoElementRef.current.srcObject = localStream;
      }

      // Initialize SSE client for subtitle data using the same StreamConfig as the Stream
      const dataStream = new DataStreamClient(streamConfig);
      dataStreamRef.current = dataStream;

      dataStream.on('data', handleSSEData);
      dataStream.on('error', (error: any) => {
        console.error('SSE Error:', error);
        setConnectionState(prev => ({ ...prev, error: 'Failed to connect to subtitle stream' }));
      });

      // Connect to data stream with stream name
      await dataStream.connect({ streamName: newStartResponse.streamId });

      // Update connection state
      setConnectionState(prev => ({
        ...prev,
        isConnected: true,
        isConnecting: false,
        streamId: newStartResponse.streamId
      }));

      // Clear pending stream start
      setPendingStreamStart(null);

    } catch (error) {
      console.error('Publishing error:', error);
      setConnectionState(prev => ({
        ...prev,
        isConnecting: false,
        error: error instanceof Error ? error.message : 'Publishing failed'
      }));
      setPendingStreamStart(null);
    } finally {
      setIsLoading(false);
    }
  }, [pendingStreamStart, handleSSEData, config]);

  // Disconnect from stream
  const handleDisconnect = useCallback(async () => {
    try {
      // Stop SSE client
      if (dataStreamRef.current) {
        dataStreamRef.current.disconnect();
        dataStreamRef.current = null;
      }

      // Stop stream
      if (streamRef.current) {
        await streamRef.current.stop();
        streamRef.current = null;
      }

      // Reset state
      setConnectionState({
        isConnected: false,
        isConnecting: false,
        error: null,
        streamId: null
      });

      setStreamState({
        isPublishing: false,
        isPreviewing: false,
        mediaStream: null,
        videoElement: null
      });

      setStreamInfo(null);
      setSubtitles([]);
      setPendingStreamStart(null);

    } catch (error) {
      console.error('Disconnect error:', error);
    }
  }, []);

  // Handle config changes
  const handleConfigChange = useCallback((newConfig: WebappConfig) => {
    setConfig(newConfig);
  }, []);

  // Handle video element reference
  const handleVideoElementRef = useCallback((element: HTMLVideoElement | null) => {
    videoElementRef.current = element;
    setStreamState(prev => ({ ...prev, videoElement: element }));
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Video Player Webapp</h1>
              <p className="text-sm text-gray-600 mt-1">
                BYOC SDK Integration with Real-time Subtitle Processing
              </p>
            </div>
            <div className="flex items-center space-x-4">
              {connectionState.isConnected && (
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse-slow"></div>
                  <span className="text-sm text-green-600 font-medium">Connected</span>
                </div>
              )}
              <button
                onClick={() => setIsConfigModalOpen(true)}
                className="btn-secondary"
              >
                Configure
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error Display */}
        {connectionState.error && (
          <ErrorDisplay 
            error={connectionState.error} 
            onDismiss={() => setConnectionState(prev => ({ ...prev, error: null }))}
          />
        )}

        {/* Loading Spinner */}
        {isLoading && <LoadingSpinner />}

        {/* Stream Controls */}
        <StreamControls
          connectionState={connectionState}
          onConnect={handleConnect}
          onDisconnect={handleDisconnect}
          isLoading={isLoading}
        />

        {/* Video Player */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-8">
          {/* WHIP Published Stream Preview */}
          <VideoPlayer
            streamState={streamState}
            onVideoElementRef={handleVideoElementRef}
            subtitles={subtitles}
          />
          
          <SubtitleTrack
            subtitles={subtitles}
            videoElement={videoElementRef.current}
          />
        </div>

        {/* Connection Status */}
        <div className="mt-8 card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Connection Status</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex justify-between">
              <span className="text-gray-600">Status:</span>
              <span className={`font-medium ${connectionState.isConnected ? 'text-green-600' : 'text-gray-600'}`}>
                {connectionState.isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Stream ID:</span>
              <span className="font-medium text-gray-900">
                {connectionState.streamId || 'None'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Subtitles:</span>
              <span className="font-medium text-gray-900">
                {subtitles.length} cues
              </span>
            </div>
          </div>
        </div>

        {/* Stream Information */}
        {streamInfo && (
          <div className="mt-8 card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Stream Information</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex justify-between">
                <span className="text-gray-600">WHIP URL:</span>
                <span className="font-medium text-gray-900 break-all">
                  {streamInfo.whipUrl ? 'Available' : 'None'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Data URL:</span>
                <span className="font-medium text-gray-900 break-all">
                  {streamInfo.dataUrl ? 'Available' : 'None'}
                </span>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Configuration Modal */}
      <ConfigModal
        isOpen={isConfigModalOpen}
        onClose={() => setIsConfigModalOpen(false)}
        config={config}
        onConfigChange={handleConfigChange}
        onConnect={handleConnect}
        isConnecting={connectionState.isConnecting}
      />

      {/* Source Selection Modal */}
      <SourceSelectionModal
        isOpen={isSourceSelectionOpen}
        onClose={() => {
          setIsSourceSelectionOpen(false);
          setPendingStreamStart(null);
        }}
        onSourceSelect={handleSourceSelection}
        isLoading={isLoading}
      />
    </div>
  );
}

export default App;