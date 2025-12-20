import React, { useEffect, useRef, useState } from 'react';
import { ParsedSubtitle } from '../types';

interface SubtitleTrackProps {
  subtitles: ParsedSubtitle[];
  videoElement: HTMLVideoElement | null;
}

const SubtitleTrack: React.FC<SubtitleTrackProps> = ({ subtitles, videoElement }) => {
  const trackRef = useRef<HTMLTrackElement | null>(null);
  const [currentSubtitle, setCurrentSubtitle] = useState<ParsedSubtitle | null>(null);

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
      const vttContent = convertToWebVTT(subtitles);
      const blob = new Blob([vttContent], { type: 'text/vtt' });
      const url = URL.createObjectURL(blob);
      
      trackRef.current.src = url;
      (trackRef.current as any).mode = 'showing';
      
      return () => {
        URL.revokeObjectURL(url);
      };
    }
  }, [subtitles]);

  // Track current subtitle based on video time
  useEffect(() => {
    if (!videoElement) return;

    const handleTimeUpdate = () => {
      const currentTime = videoElement.currentTime;
      const activeSubtitle = subtitles.find(
        subtitle => currentTime >= subtitle.startTime && currentTime <= subtitle.endTime
      );
      
      setCurrentSubtitle(activeSubtitle || null);
    };

    videoElement.addEventListener('timeupdate', handleTimeUpdate);
    
    return () => {
      videoElement.removeEventListener('timeupdate', handleTimeUpdate);
    };
  }, [videoElement, subtitles]);

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Subtitle Track</h3>
      
      {/* Hidden track element for WebVTT support */}
      <track
        ref={trackRef}
        kind="subtitles"
        src=""
        srcLang="en"
        label="English"
        default
      />
      
      {/* Current subtitle display */}
      <div className="min-h-32">
        {currentSubtitle ? (
          <div className="bg-black bg-opacity-75 text-white p-4 rounded-lg">
            <div className="text-sm text-gray-300 mb-2">
              {formatTime(currentSubtitle.startTime)} - {formatTime(currentSubtitle.endTime)}
            </div>
            <div className="text-lg leading-relaxed">
              {currentSubtitle.text}
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-32 border-2 border-dashed border-gray-300 rounded-lg">
            <div className="text-center">
              <div className="w-12 h-12 mx-auto mb-3 bg-gray-200 rounded-full flex items-center justify-center">
                <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4V2a1 1 0 011-1h8a1 1 0 011 1v2m-9 0h10a2 2 0 012 2v12a2 2 0 01-2 2H7a2 2 0 01-2-2V6a2 2 0 012-2z" />
                </svg>
              </div>
              <p className="text-gray-500">No subtitles available</p>
              <p className="text-sm text-gray-400 mt-1">
                {subtitles.length > 0 ? 'Waiting for video playback' : 'Start streaming to receive subtitles'}
              </p>
            </div>
          </div>
        )}
      </div>
      
      {/* Subtitle statistics */}
      <div className="mt-4 flex justify-between items-center text-sm text-gray-600">
        <div>
          <span className="font-medium">Total Cues:</span> {subtitles.length}
        </div>
        <div>
          <span className="font-medium">Duration:</span> {subtitles.length > 0 ? formatTime(subtitles[subtitles.length - 1].endTime) : '0:00'}
        </div>
      </div>
      
      {/* Subtitle list */}
      {subtitles.length > 0 && (
        <div className="mt-4">
          <h4 className="text-md font-medium text-gray-900 mb-2">Recent Subtitle Cues</h4>
          <div className="max-h-48 overflow-y-auto space-y-2">
            {subtitles.slice(-5).reverse().map((subtitle) => (
              <div key={subtitle.id} className="p-3 bg-gray-50 rounded border">
                <div className="text-xs text-gray-500 mb-1">
                  {formatTime(subtitle.startTime)} - {formatTime(subtitle.endTime)}
                </div>
                <div className="text-sm text-gray-700 line-clamp-2">
                  {subtitle.text}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default SubtitleTrack;