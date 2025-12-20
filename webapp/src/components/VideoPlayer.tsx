import React, { useEffect, useRef } from 'react';
import { StreamState, ParsedSubtitle } from '../types';

// Extend Window interface to include VTTCue
declare global {
  interface Window {
    VTTCue: any;
  }
}

interface VideoPlayerProps {
  streamState: StreamState;
  onVideoElementRef: (element: HTMLVideoElement | null) => void;
  subtitles: ParsedSubtitle[];
  onStartTranscribing: () => void;
  onStopTranscribing: () => void;
  onConfigure: () => void;
  isConnecting: boolean;
  isConnected: boolean;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ 
  streamState, 
  onVideoElementRef, 
  subtitles = [], 
  onStartTranscribing, 
  onStopTranscribing, 
  onConfigure, 
  isConnecting, 
  isConnected 
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const textTrackRef = useRef<TextTrack | null>(null);
  // Track when we started receiving subtitles to calculate relative timestamps
  const subtitleStartTimeRef = useRef<number | null>(null);

  useEffect(() => {
    onVideoElementRef(videoRef.current);
  }, [onVideoElementRef]);

  useEffect(() => {
    if (videoRef.current && streamState.mediaStream) {
      videoRef.current.srcObject = streamState.mediaStream;
    }
  }, [streamState.mediaStream]);

  // Create a JS TextTrack on the video element for dynamic cues
  // Helper to create/return the TextTrack
  const ensureTextTrack = (): TextTrack | null => {
    if (!videoRef.current) return null;
    if (!textTrackRef.current) {
      try {
        const t = videoRef.current.addTextTrack('captions', 'English', 'en');
        t.mode = 'showing';
        textTrackRef.current = t;
        // eslint-disable-next-line no-console
        console.debug('TextTrack created', t);
      } catch (e) {
        // eslint-disable-next-line no-console
        console.warn('addTextTrack failed', e);
        return null;
      }
    }
    return textTrackRef.current;
  };

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const tryCreate = () => {
      if (streamState.isPreviewing) {
        const track = ensureTextTrack();
        if (track) {
          track.mode = 'showing';
          console.debug('TextTrack initialized and set to showing mode');
        }
      }
    };

    // If metadata already loaded, try to create immediately
    if (video.readyState >= 1) {
      tryCreate();
    } else {
      video.addEventListener('loadedmetadata', tryCreate);
    }

    return () => {
      video.removeEventListener('loadedmetadata', tryCreate);
      // reset base when video is removed or preview stops
      subtitleStartTimeRef.current = null;
    };
  }, [streamState.isPreviewing]);

  // Update track when subtitles change
  useEffect(() => {
    // Ensure there's a TextTrack available (may create it)
    const track = ensureTextTrack() ?? textTrackRef.current;
    // Debug
    // eslint-disable-next-line no-console
    console.debug('subtitles effect run; track:', track, 'subtitles:', subtitles);

    if (!track) {
      // No track yet; will run again when isPreviewing changes or when ensureTextTrack is called
      return;
    }

    if (subtitles.length === 0) {
      return;
    }

    // For live subtitles, add cues incrementally
    const existingCues = Array.from((track.cues ?? []) as any);
    const existingIds = new Set(existingCues.map((cue: any) => cue.id));
    const CueCtor = (window as any).VTTCue || (window as any).TextTrackCue;

    // eslint-disable-next-line no-console
    console.debug('existing cues before update:', existingCues.map((c: any) => c.id));

    // Initialize subtitle start time when we first receive subtitles
    if (subtitleStartTimeRef.current == null && subtitles.length > 0 && videoRef.current) {
      subtitleStartTimeRef.current = videoRef.current.currentTime;
      // eslint-disable-next-line no-console
      console.debug('subtitle start time set to', subtitleStartTimeRef.current);
    }

    subtitles.forEach((subtitle) => {
      if (!existingIds.has(subtitle.id)) {
        try {
          const now = videoRef.current?.currentTime ?? 0;
          
          // Calculate adjusted start time: current video time + 1ms
          const adjustedStart = now + 0.001; // 1 millisecond from current time
          
          // Calculate original duration
          const originalDuration = subtitle.endTime - subtitle.startTime;
          
          // Calculate adjusted end time: adjusted start + original duration
          const adjustedEnd = adjustedStart + originalDuration;
          
          // Check for overlap with last cue and adjust if needed
          if (existingCues.length > 0) {
            const lastCue = existingCues[existingCues.length - 1] as any;
            if (lastCue.endTime > adjustedStart) {
              // Overlap detected - adjust last cue end to be 1ms before current cue start
              lastCue.endTime = adjustedStart - 0.001;
              // eslint-disable-next-line no-console
              console.debug('adjusted last cue end time to prevent overlap:', lastCue.id, { newEnd: lastCue.endTime });
            }
          }
          
          // Create new cue with adjusted timing
          const cue = new CueCtor(adjustedStart, adjustedEnd, subtitle.text);
          cue.id = subtitle.id; // Set the ID for tracking
          track.addCue(cue);
          // eslint-disable-next-line no-console
          console.debug('added cue', cue.id, { 
            adjustedStart, 
            adjustedEnd, 
            originalDuration, 
            originalStart: subtitle.startTime, 
            originalEnd: subtitle.endTime,
            text: subtitle.text 
          });
        } catch (e) {
          // addCue may throw for overlapping cues in some browsers
          // eslint-disable-next-line no-console
          console.warn('addCue failed for subtitle', subtitle, e);
        }
      }
    });

    // Remove old cues that are no longer in the subtitles array
    Array.from((track.cues ?? []) as any).forEach((cue: any) => {
      const stillExists = subtitles.some(sub => sub.id === cue.id);
      if (!stillExists) {
        try {
          track.removeCue(cue);
          // eslint-disable-next-line no-console
          console.debug('removed cue', cue.id);
        } catch (e) {
          // ignore
        }
      }
    });

    // Ensure track is visible
    track.mode = 'showing';
    
    // Force a re-render by updating the video element
    if (videoRef.current) {
      videoRef.current.textTracks[0].mode = 'showing';
    }
  }, [subtitles, streamState.isPreviewing]);

  return (
    <div className="card">
      <div className="relative bg-black rounded-lg overflow-hidden">
        {streamState.isPreviewing ? (
          <>
            <video
              ref={videoRef}
              className="w-full h-auto max-h-96 object-contain"
              autoPlay
              muted
              playsInline
              controls
              crossOrigin="anonymous"
            >
              <track
                kind="captions"
                src=""
                srcLang="en"
                label="English"
                default
              />
            </video>
          </>
        ) : (
          <div className="flex items-center justify-center h-96">
            <div className="text-center">
              <div className="w-16 h-16 mx-auto mb-4 bg-gray-200 rounded-full flex items-center justify-center">
                <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </div>
              <p className="text-gray-500">No video stream available</p>
              <p className="text-sm text-gray-400 mt-1">Start transcribing to see preview</p>
            </div>
          </div>
        )}
      </div>
      
      {streamState.isPreviewing && (
        <div className="mt-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-red-600 font-medium">Live</span>
          </div>
          <div className="flex items-center space-x-4">
            <div className="text-sm text-gray-500">
              {streamState.mediaStream?.getAudioTracks().length || 0} audio tracks
            </div>
          </div>
        </div>
      )}
      
      {/* Control Buttons */}
      <div className="mt-4 flex items-center justify-center space-x-4">
        {!isConnected ? (
          <button
            onClick={onStartTranscribing}
            disabled={isConnecting}
            className="btn-primary flex items-center justify-center space-x-2"
          >
            {isConnecting ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Connecting...</span>
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <span>Start Transcribing</span>
              </>
            )}
          </button>
        ) : (
          <button
            onClick={onStopTranscribing}
            className="bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center space-x-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
            </svg>
            <span>Stop Transcribing</span>
          </button>
        )}
        <button
          onClick={onConfigure}
          className="btn-secondary"
        >
          Configure
        </button>
      </div>
    </div>
  );
};

export default VideoPlayer;