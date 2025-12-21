// BYOC SDK Types - Updated to match SDK interface
export interface StreamConfig {
  gatewayUrl: string;
  defaultPipeline?: string;
  iceServers?: RTCIceServer[];
  // SDK internal properties
  location?: string;
  playbackUrl?: string;
  link?: string;
  eTag?: string;
  streamStartResponse?: any;
}

// Separated configuration for webapp use
export interface WebappConfig {
  gatewayUrl: string;
  defaultPipeline?: string;
  apiKey?: string;
}

// Utility function to create SDK StreamConfig from webapp config
export const createStreamConfig = (config: WebappConfig): StreamConfig => {
  const { gatewayUrl, defaultPipeline } = config;
  return {
    gatewayUrl,
    defaultPipeline
  };
};

// Utility function to extract apiKey for authentication
export const getApiKey = (config: WebappConfig): string | undefined => {
  return config.apiKey;
};

// Authentication headers utility
export const createAuthHeaders = (apiKey?: string): Record<string, string> => {
  if (!apiKey) return {};
  return {
    'Authorization': `Bearer ${apiKey}`,
    'X-API-Key': apiKey
  };
};

export interface StreamStartOptions {
  pipeline?: string;
  streamName?: string;
  width?: number;
  height?: number;
  fpsLimit?: number;
  enableVideoIngress?: boolean;
  enableAudioIngress?: boolean;
  enableVideoEgress?: boolean;
  enableAudioEgress?: boolean;
  enableDataOutput?: boolean;
  customParams?: Record<string, any>;
}

export interface StreamStartResponse {
  streamId: string;
  whipUrl: string;
  whepUrl?: string;
  rtmpUrl?: string;
  rtmpOutputUrl?: string;
  dataUrl: string;
  stopUrl: string;
  statusUrl: string;
  updateUrl: string;
}

export interface StreamError extends Error {
  code: string;
  service?: 'WHIP' | 'WHEP' | 'SSE';
  details?: any;
}

// Subtitle Data Types
export interface TranscriptData {
  type: 'transcript';
  timestamp_utc: string;
  timing: {
    media_window_start_ms: number;
    media_window_end_ms: number;
  };
  segments: TranscriptSegment[];
  stats: {
    audio_duration_ms: number;
  };
}

export interface TranscriptSegment {
  id: string;
  start_ms: number;
  end_ms: number;
  text: string;
  words?: TranscriptWord[];
}

export interface TranscriptWord {
  start_ms: number;
  end_ms: number;
  text: string;
}

export interface SubtitleUpdate {
  type: 'subtitle_update';
  timestamp_utc: string;
  window: {
    start: number;
    end: number;
  };
  srt_content: string;
}

export interface ParsedSubtitle {
  id: string;
  startTime: number;
  endTime: number;
  text: string;
  rawSrt: string;
}

export interface WebVTTCue {
  identifier: string;
  startTime: number;
  endTime: number;
  text: string;
  settings?: string;
}

// Connection State Types
export interface ConnectionState {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  streamId: string | null;
}

export interface StreamState {
  isPublishing: boolean;
  isPreviewing: boolean;
  mediaStream: MediaStream | null;
  videoElement: HTMLVideoElement | null;
}

// Component Props Types
export interface ConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: StreamConfig;
  onConfigChange: (config: StreamConfig) => void;
  onConnect: () => void;
  isConnecting: boolean;
}

export interface VideoPlayerProps {
  streamState: StreamState;
  onVideoElementRef: (element: HTMLVideoElement | null) => void;
}

export interface SubtitleTrackProps {
  subtitles: ParsedSubtitle[];
  videoElement: HTMLVideoElement | null;
}

// Data Stream Types
export interface DataStreamOptions {
  dataUrl?: string;
  streamName?: string;
  maxLogs?: number;
}