import React, { useEffect, useRef } from 'react';
import { ParsedSubtitle } from '../types';

interface SubtitleTrackProps {
  subtitles: ParsedSubtitle[];
}

const SubtitleTrack: React.FC<SubtitleTrackProps> = ({ subtitles }) => {
  const trackRef = useRef<HTMLTrackElement | null>(null);

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

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Subtitles</h3>      
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
          <div className="max-h-48 overflow-y-auto space-y-2">
            {subtitles.slice(-5).reverse().map((subtitle) => (
              <div key={subtitle.id} className="p-4 bg-gray-50 rounded border h-100">
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