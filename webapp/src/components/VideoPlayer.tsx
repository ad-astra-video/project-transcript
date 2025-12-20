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
  const textTrackRef = useRef<TextTrack | null>(null);
  // Base offset to map subtitle timestamps to the video's currentTime
  const subtitleBaseRef = useRef<number | null>(null);

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
      if (streamState.isPreviewing) ensureTextTrack();
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
      subtitleBaseRef.current = null;
    };
  }, [streamState.isPreviewing]);

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

    // If we don't yet have a base mapping from subtitle timeline -> video.currentTime,
    // establish it from the earliest incoming subtitle startTime so cues align with the video timeline.
    if (subtitleBaseRef.current == null && subtitles.length > 0 && videoRef.current) {
      const minStart = Math.min(...subtitles.map(s => s.startTime));
      subtitleBaseRef.current = videoRef.current.currentTime - minStart;
      // eslint-disable-next-line no-console
      console.debug('subtitle base set to', subtitleBaseRef.current, 'video.currentTime', videoRef.current.currentTime, 'minStart', minStart);
    }

    subtitles.forEach((subtitle) => {
      if (!existingIds.has(subtitle.id)) {
        try {
          const base = subtitleBaseRef.current ?? 0;
          const start = base + subtitle.startTime;
          const end = base + subtitle.endTime;
          const cue = new CueCtor(start, end, subtitle.text);
          cue.id = subtitle.id; // Set the ID for tracking
          track.addCue(cue);
          // eslint-disable-next-line no-console
          console.debug('added cue', cue.id, start, end, subtitle.text);
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
  }, [subtitles, streamState.isPreviewing]);

  // Dev helper to add sample cues (useful to verify track/cues visually)
  const addTestCues = () => {
    const track = ensureTextTrack();
    if (!track) {
      // eslint-disable-next-line no-console
      console.warn('No text track available');
      return;
    }
    const CueCtor = (window as any).VTTCue || (window as any).TextTrackCue;
    if (!CueCtor) {
      // eslint-disable-next-line no-console
      console.warn('No VTTCue constructor available');
      return;
    }

    const samples = [
      [0, 0.9, 'Hildy!'],
      [1, 1.4, 'How are you?'],
      [1.5, 2.9, "Tell me, is the lord of the universe in?"],
      [3, 4.2, "Yes, he's in - in a bad humor"],
      [4.3, 6, "Somebody must've stolen the crown jewels"],
    ];

    samples.forEach((s, i) => {
      try {
        const cue = new CueCtor(s[0], s[1], s[2]);
        cue.id = `debug-${i}`;
        track.addCue(cue);
      } catch (e) {
        // eslint-disable-next-line no-console
        console.warn('failed to add debug cue', e);
      }
    });

    // eslint-disable-next-line no-console
    console.debug('track cues after adding samples:', track.cues);
  };

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
          <div className="flex items-center space-x-4">
            <div className="text-sm text-gray-500">
              {streamState.mediaStream?.getVideoTracks().length || 0} video tracks
            </div>
            <button
              type="button"
              onClick={addTestCues}
              className="text-sm px-2 py-1 bg-gray-100 rounded border"
            >
              Add test cues
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoPlayer;