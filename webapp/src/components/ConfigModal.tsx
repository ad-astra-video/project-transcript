import React, { useState, useEffect } from 'react';
import { WebappConfig } from '../types';

interface ConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: WebappConfig;
  onConfigChange: (config: WebappConfig) => void;
  onConnect: () => void;
  isConnecting: boolean;
}

const ConfigModal: React.FC<ConfigModalProps> = ({
  isOpen,
  onClose,
  config,
  onConfigChange,
  onConnect,
  isConnecting
}) => {
  const [localConfig, setLocalConfig] = useState<WebappConfig>(config);

  useEffect(() => {
    setLocalConfig(config);
  }, [config]);

  const handleSaveAndClose = (e: React.FormEvent) => {
    e.preventDefault();
    onConfigChange(localConfig);
    onClose();
  };

  const handleConnect = (e: React.FormEvent) => {
    e.preventDefault();
    onConfigChange(localConfig);
    onConnect();
    onClose();
  };

  const handleInputChange = (field: keyof WebappConfig, value: string) => {
    setLocalConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
        <div className="flex justify-between items-center p-6 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">Configure Gateway</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
            disabled={isConnecting}
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <form onSubmit={handleSaveAndClose} className="p-6 space-y-4">
          <div>
            <label htmlFor="gatewayUrl" className="block text-sm font-medium text-gray-700 mb-2">
              Gateway URL
            </label>
            <input
              id="gatewayUrl"
              type="url"
              value={localConfig.gatewayUrl}
              onChange={(e) => handleInputChange('gatewayUrl', e.target.value)}
              className="input-field"
              placeholder="http://localhost:5937"
              required
              disabled={isConnecting}
            />
            <p className="text-xs text-gray-500 mt-1">
              The base URL for your BYOC gateway server
            </p>
          </div>

          <div>
            <label htmlFor="apiKey" className="block text-sm font-medium text-gray-700 mb-2">
              API Key (Optional)
            </label>
            <input
              id="apiKey"
              type="text"
              value={localConfig.apiKey}
              onChange={(e) => handleInputChange('apiKey', e.target.value)}
              className="input-field"
              placeholder="Enter API key for authorization"
              disabled={isConnecting}
            />
            <p className="text-xs text-gray-500 mt-1">
              API key for authentication header in requests
            </p>
          </div>

          <div>
            <label htmlFor="defaultPipeline" className="block text-sm font-medium text-gray-700 mb-2">
              Default Pipeline
            </label>
            <input
              id="defaultPipeline"
              type="text"
              value={localConfig.defaultPipeline}
              onChange={(e) => handleInputChange('defaultPipeline', e.target.value)}
              className="input-field"
              placeholder="default"
              disabled={isConnecting}
            />
            <p className="text-xs text-gray-500 mt-1">
              Default pipeline name for stream processing
            </p>
          </div>

          <div className="flex justify-end space-x-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="btn-secondary"
              disabled={isConnecting}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn-primary"
              disabled={isConnecting || !localConfig.gatewayUrl}
            >
              Save & Close
            </button>
            <button
              type="button"
              onClick={handleConnect}
              className="bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200"
              disabled={isConnecting || !localConfig.gatewayUrl}
            >
              {isConnecting ? (
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Connecting...</span>
                </div>
              ) : (
                'Save & Connect'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ConfigModal;