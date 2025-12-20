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
  subtitles?: ParsedSubtitle[];
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ streamState, onVideoElementRef, subtitles = [] }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const trackRef = useRef<HTMLTrackElement | null>(null);

  useEffect(() => {
    onVideoElementRef(videoRef.current);
  }, [onVideoElementRef]);

  useEffect(() => {
    if (videoRef.current && streamState.mediaStream) {
      videoRef.current.srcObject = streamState.mediaStream;
    }
  }, [streamState.mediaStream]);

  // Convert parsed subtitle to WebVTT format
  const convertToWebVTT = (subtitles: ParsedSubtitle[]): string => {
    const vttHeader = 'WEBVTT\n\n';
    const vttCues = subtitles.map((subtitle, index) => {
      const startTime = formatTime(subtitle.startTime);
      const endTime = formatTime(subtitle.endTime);
      return `${index + 1}\n${startTime} --> ${endTime}\n${subtitle.text}\n`;
    }).join('\n');
    
    return vttHeader + vttCues;
  };

  // Format time to WebVTT format (HH:MM:SS.mmm)
  const formatTime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    const milliseconds = Math.floor((seconds % 1) * 1000);
    
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${milliseconds.toString().padStart(3, '0')}`;
  };

  // Update track when subtitles change
  useEffect(() => {
    if (trackRef.current && subtitles.length > 0) {
      // Clear existing cues
      while (trackRef.current.track.cues && trackRef.current.track.cues.length > 0) {
        trackRef.current.track.removeCue(trackRef.current.track.cues[0]);
      }
      
      // Add new cues manually
      subtitles.forEach((subtitle) => {
        const cue = new (window.VTTCue || window.TextTrackCue)(
          subtitle.startTime,
          subtitle.endTime,
          subtitle.text
        );
        trackRef.current!.track.addCue(cue);
      });
      
      // Enable the track
      (trackRef.current as any).mode = 'showing';
    }
  }, [subtitles]);

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Video Preview</h3>
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
            />
            {/* Subtitle track for video overlay */}
            <track
              ref={trackRef}
              kind="subtitles"
              src=""
              srcLang="en"
              label="English"
              default
            />
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
              <p className="text-sm text-gray-400 mt-1">Start publishing to see preview</p>
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
          <div className="text-sm text-gray-500">
            {streamState.mediaStream?.getVideoTracks().length || 0} video tracks
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoPlayer;