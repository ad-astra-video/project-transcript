import React, { useState, useEffect } from 'react';

interface SourceSelectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSourceSelect: (source: 'camera' | 'screen') => void;
  isLoading?: boolean;
}

const SourceSelectionModal: React.FC<SourceSelectionModalProps> = ({
  isOpen,
  onClose,
  onSourceSelect,
  isLoading = false
}) => {
  const [selectedSource, setSelectedSource] = useState<'camera' | 'screen'>('camera');

  useEffect(() => {
    if (isOpen) {
      setSelectedSource('camera');
    }
  }, [isOpen]);

  const handleSourceSelect = () => {
    onSourceSelect(selectedSource);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Select Media Source</h2>
            <button
              onClick={onClose}
              disabled={isLoading}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="space-y-4">
            <p className="text-gray-600 mb-4">
              Choose which media source to use for your stream:
            </p>

            <div className="space-y-3">
              {/* Camera Option */}
              <label className="flex items-start p-4 border rounded-lg cursor-pointer hover:bg-gray-50 transition-colors">
                <input
                  type="radio"
                  name="source"
                  value="camera"
                  checked={selectedSource === 'camera'}
                  onChange={(e) => setSelectedSource(e.target.value as 'camera')}
                  disabled={isLoading}
                  className="mt-1 mr-3"
                />
                <div className="flex-1">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                      <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900">Camera</h3>
                      <p className="text-sm text-gray-600">Use your webcam or external camera</p>
                    </div>
                  </div>
                </div>
              </label>

              {/* Screen Share Option */}
              <label className="flex items-start p-4 border rounded-lg cursor-pointer hover:bg-gray-50 transition-colors">
                <input
                  type="radio"
                  name="source"
                  value="screen"
                  checked={selectedSource === 'screen'}
                  onChange={(e) => setSelectedSource(e.target.value as 'screen')}
                  disabled={isLoading}
                  className="mt-1 mr-3"
                />
                <div className="flex-1">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                      <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900">Screen Share</h3>
                      <p className="text-sm text-gray-600">Share your entire screen or application window</p>
                    </div>
                  </div>
                </div>
              </label>
            </div>

            <div className="flex justify-end space-x-3 mt-6 pt-4 border-t">
              <button
                onClick={onClose}
                disabled={isLoading}
                className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={handleSourceSelect}
                disabled={isLoading}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center space-x-2"
              >
                {isLoading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>Starting...</span>
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    <span>Start Stream</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SourceSelectionModal;