import React, { useEffect, useRef } from 'react';
import { ParsedSubtitle } from '../types';

interface SubtitleTrackProps {
  subtitles: ParsedSubtitle[];
}

const SubtitleTrack: React.FC<SubtitleTrackProps> = ({ subtitles }) => {
  const trackRef = useRef<HTMLTrackElement | null>(null);

  // Export subtitles to CSV format
  const exportToCSV = () => {
    if (subtitles.length === 0) {
      alert('No subtitles to export');
      return;
    }

    const csvHeader = 'Start Time,End Time,Text\n';
    const csvRows = subtitles.map(subtitle => {
      const startTime = formatTime(subtitle.startTime);
      const endTime = formatTime(subtitle.endTime);
      const text = `"${subtitle.text.replace(/"/g, '""')}"`; // Escape quotes in text
      return `${startTime},${endTime},${text}`;
    }).join('\n');

    const csvContent = csvHeader + csvRows;
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    
    if (link.download !== undefined) {
      const url = URL.createObjectURL(blob);
      link.setAttribute('href', url);
      link.setAttribute('download', `subtitles_${new Date().toISOString().split('T')[0]}.csv`);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  // Export subtitles to JSON format
  const exportToJSON = () => {
    if (subtitles.length === 0) {
      alert('No subtitles to export');
      return;
    }

    const jsonContent = JSON.stringify(subtitles, null, 2);
    const blob = new Blob([jsonContent], { type: 'application/json;charset=utf-8;' });
    const link = document.createElement('a');
    
    if (link.download !== undefined) {
      const url = URL.createObjectURL(blob);
      link.setAttribute('href', url);
      link.setAttribute('download', `subtitles_${new Date().toISOString().split('T')[0]}.json`);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

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
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Subtitles</h3>
        {/* Export buttons */}
        {subtitles.length > 0 && (
          <div className="flex space-x-2">
            <button
              onClick={exportToCSV}
              className="px-3 py-1 text-xs bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors flex items-center space-x-1"
              title="Export subtitles as CSV"
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span>CSV</span>
            </button>
            <button
              onClick={exportToJSON}
              className="px-3 py-1 text-xs bg-green-500 text-white rounded hover:bg-green-600 transition-colors flex items-center space-x-1"
              title="Export subtitles as JSON"
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span>JSON</span>
            </button>
          </div>
        )}
      </div>
      
      {/* Subtitle statistics */}
      <div className="flex justify-between items-center text-sm text-gray-600">
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