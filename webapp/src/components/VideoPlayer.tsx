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
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ streamState, onVideoElementRef, subtitles = [] }) => {
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
          const subtitleStartTime = subtitleStartTimeRef.current ?? now;
          
          // Calculate timestamps relative to when we started receiving subtitles
          const start = subtitleStartTime + subtitle.startTime;
          const end = subtitleStartTime + subtitle.endTime;

          let adjustedStart = start;
          let adjustedEnd = end;

          const minVisibleDuration = 0.6; // seconds
          const extendIfExpiredTo = now + 1.8; // seconds

          if (end <= now) {
            // already expired — extend so it's visible now
            adjustedEnd = Math.max(extendIfExpiredTo, now + minVisibleDuration);
            adjustedStart = Math.max(start, now - 0.1);
          } else if (start <= now && end <= now + 0.1) {
            // almost expired — extend a little
            adjustedEnd = Math.max(end, now + minVisibleDuration);
          } else if (end - start < minVisibleDuration) {
            adjustedEnd = Math.max(end, start + minVisibleDuration);
          }

          const cue = new CueCtor(adjustedStart, adjustedEnd, subtitle.text);
          cue.id = subtitle.id; // Set the ID for tracking
          track.addCue(cue);
          // eslint-disable-next-line no-console
          console.debug('added cue', cue.id, { start: adjustedStart, end: adjustedEnd, originalStart: start, originalEnd: end, now });
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
        const now = videoRef.current?.currentTime ?? 0;
        const start = now + (s[0] as number);
        const end = now + (s[1] as number);
        const cue = new CueCtor(start, end, s[2] as string);
        cue.id = `debug-${i}`;
        track.addCue(cue);
      } catch (e) {
        // eslint-disable-next-line no-console
        console.warn('failed to add debug cue', e);
      }
    });

    // Ensure track is showing and log textTracks for debugging
    try {
      track.mode = 'showing';
    } catch (e) {
      // ignore
    }
    // eslint-disable-next-line no-console
    console.debug('video.textTracks:', videoRef.current?.textTracks);
    // eslint-disable-next-line no-console
    console.debug('track cues after adding samples:', Array.from((track.cues ?? []) as any).map((c: any) => ({ id: c.id, start: c.startTime, end: c.endTime })));
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