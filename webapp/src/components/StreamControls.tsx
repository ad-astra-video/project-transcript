import React from 'react';
import { ConnectionState } from '../types';

interface StreamControlsProps {
  connectionState: ConnectionState;
  onConnect: () => void;
  onDisconnect: () => void;
  isLoading: boolean;
}

const StreamControls: React.FC<StreamControlsProps> = ({
  connectionState,
  onConnect,
  onDisconnect,
  isLoading
}) => {
  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Stream Controls</h3>
      
      <div className="flex flex-col sm:flex-row gap-4">
        {!connectionState.isConnected ? (
          <button
            onClick={onConnect}
            disabled={isLoading}
            className="btn-primary flex items-center justify-center space-x-2"
          >
            {isLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Connecting...</span>
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <span>Start Publishing</span>
              </>
            )}
          </button>
        ) : (
          <button
            onClick={onDisconnect}
            className="bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center space-x-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
            </svg>
            <span>Stop Publishing</span>
          </button>
        )}
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${
              connectionState.isConnected ? 'bg-green-500 animate-pulse-slow' : 'bg-gray-300'
            }`}></div>
            <span className={`text-sm font-medium ${
              connectionState.isConnected ? 'text-green-600' : 'text-gray-500'
            }`}>
              {connectionState.isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          
          {connectionState.streamId && (
            <div className="text-sm text-gray-600">
              <span className="font-medium">Stream ID:</span> {connectionState.streamId}
            </div>
          )}
        </div>
      </div>
      
      {connectionState.error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
          <div className="flex items-center space-x-2">
            <svg className="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="text-sm text-red-700 font-medium">Error:</span>
          </div>
          <p className="text-sm text-red-600 mt-1">{connectionState.error}</p>
        </div>
      )}
    </div>
  );
};

export default StreamControls;